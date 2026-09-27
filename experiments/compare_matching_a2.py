"""Matching-stage A/B: production candidates vs production + A2.

Read-only with respect to src/. Uses the existing 24 features and the existing
LogisticMatcher defaults, on the same 20,000-S1 hash subset and the same S1
split as the previous 20k experiment.

Why this is staged
------------------
A single-pass run of this comparison keeps a laptop CPU pinned for roughly an
hour: ~12.5M records parsed and normalised in pure Python, then 7.58M pair
featurisations, then two logistic-regression fits. A previous blocking run on
this machine reached ~96 C and had to be stopped. So the work is split into
three stages, each of which writes its result to disk and exits. Nothing is
recomputed when a later stage runs, and an interrupted stage resumes from its
last checkpoint instead of starting over.

    stage 1  candidate generation   -> s1_records, truth, candidate rows
    stage 2  featurisation          -> X, y, is_prod, is_a2only_true, s1_of_row
    stage 3  fit + evaluate + report

Thermal controls (see experiments/thermal.py) are applied in every stage:
BLAS capped to one thread, below-normal priority, pinned to a few logical
processors, a duty-cycle sleep in the hot loops, and a background temperature
sampler that pauses the run when the CPU gets too hot instead of cooking it.
Stage 2 additionally checkpoints after S2, so a thermal pause in the middle of
the longest stage costs at most one file re-read.

Scale
-----
Production 6,373,030 candidates, production+A2 7,580,413. Too many to hold as
Python feature tuples (~6 GB), so features go into one preallocated float32
array and the production arm is a mask over it. S2 and S3 entity ids are only
unique within their own file, so the candidate lookup is per source; the pool
is not in entity_id order, so the join is a dict lookup, not a sorted merge.

Split
-----
``split_by_reference`` documents that its assignment "is a deterministic
function of the S1 id and salt ... does not shift when the candidate set
changes size". It is therefore called once per unique S1 id (20,000 objects)
rather than 7.58M times, which gives the identical per-entity assignment at
negligible cost.

Usage
-----
    python experiments/compare_matching_a2.py --stage 1
    python experiments/compare_matching_a2.py --stage 2
    python experiments/compare_matching_a2.py --stage 3
    python experiments/compare_matching_a2.py --stage all
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import thermal as TH                                        # noqa: E402

# BLAS must be capped before numpy is imported anywhere.
BLAS_THREADS = TH.preparse_blas_threads(sys.argv[1:], default=1)
TH.cap_blas_threads(BLAS_THREADS)

import numpy as np                                          # noqa: E402

from src import matching as M                              # noqa: E402
from src.blocking import (BlockingConfig, BlockingIndex,  # noqa: E402
                          CandidatePair, RecordRef, generate_candidates)
from src.features import FEATURE_NAMES, featurize          # noqa: E402

DATASET = Path(
    r"C:\Users\BITPATNA\Downloads\6ab10eb3b23ba_student_resource"
    r"\student_resource\dataset")
TRAIN = DATASET / "train"
OUT = REPO / "output" / "experiments" / "match_a2"

N_S1 = 20_000
S1_SALT, S1_MOD = b"exp20k-s1", 110
CFG = BlockingConfig()
CAP = CFG.max_group_size
PREFIX4 = CFG.name_prefix_length

SPLIT_FRACTION = 0.2
SPLIT_SALT = "exp20k-split-v1"

EXPECT_PROD_CANDIDATES = 6_373_030
EXPECT_UNION_CANDIDATES = 7_580_413
EXPECT_A2_ONLY_TRUE = 2_995
EXPECT_PROD_TRUE = 55_325

N_FEATURES = len(FEATURE_NAMES)
CHECKS: list[tuple[str, bool, str]] = []
T0 = time.perf_counter()
GUARD: TH.ThermalGuard | None = None


def step(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" -- {detail}" if detail else ""), flush=True)


def note(msg: str) -> None:
    print(f"  {msg}", flush=True)


def train_path(name: str) -> Path:
    if "test" in name.lower():
        raise AssertionError(f"refusing non-train file: {name}")
    path = (TRAIN / name).resolve()
    if path.parent != TRAIN.resolve():
        raise AssertionError(f"refusing path outside train/: {path}")
    return path


def hashed(eid: str, salt: bytes, mod: int) -> bool:
    d = hashlib.blake2b(eid.encode("utf-8"), key=salt, digest_size=8).digest()
    return int.from_bytes(d, "big") % mod == 0


def a2_key(record):
    if len(record.name_core) < 2:
        return None
    return ("A2_name_2tok_order", record.country,
            record.name_core[0][:PREFIX4], record.name_core[1][:PREFIX4])


# ==========================================================================
# stage 1: subset, ground truth, candidates
# ==========================================================================
def stage1(args) -> None:
    step("STAGE 1: candidate generation")
    s1_records = pickle.loads((OUT / "s1_records.pkl").read_bytes()) \
        if (OUT / "s1_records.pkl").exists() and not args.force else None
    if s1_records is None:
        step("  stream S1 -> hash-select subset")
        s1_records = {}
        scanned = 0
        for rec in M.iter_records(train_path("train_source1.tsv")):
            scanned += 1
            if len(s1_records) < N_S1 and hashed(rec.entity_id, S1_SALT, S1_MOD):
                s1_records[rec.entity_id] = rec
            guard_tick()
            if len(s1_records) >= N_S1:
                break
        step(f"  S1 scanned {scanned:,}, selected {len(s1_records):,}")
        (OUT / "s1_records.pkl").write_bytes(pickle.dumps(s1_records, 4))
    check("S1 subset size", len(s1_records) == N_S1, f"{len(s1_records):,}")
    s1_order = sorted(s1_records)

    truth_path = OUT / "truth.pkl"
    truth = pickle.loads(truth_path.read_bytes()) if truth_path.exists() \
        and not args.force else None
    if truth is None:
        step("  stream ground truth")
        wanted = set(s1_records)
        truth = {}
        with train_path("train_ground_truth.tsv").open(
                "r", encoding="utf-8", errors="replace", newline="") as fh:
            fh.readline()
            for line in fh:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 2 or parts[0].strip() not in wanted:
                    continue
                truth[parts[0].strip()] = frozenset(
                    p.strip() for p in parts[1].split(",") if p.strip())
        total_true = sum(len(v) for v in truth.values())
        step(f"  GT rows {len(truth):,}, true pairs {total_true:,}")
        check("true pairs total", total_true == 69_301, f"{total_true:,}")
        truth_path.write_bytes(pickle.dumps(truth, 4))

    s1_a2_keys = {a2_key(r) for r in s1_records.values()}
    s1_a2_keys.discard(None)

    step("  stream FULL S2 + S3 -> production index + A2 buckets")
    index = BlockingIndex(CFG)
    a2_buckets: dict[tuple, list] = {}
    for fname, src in (("train_source2.tsv", "S2"), ("train_source3.tsv", "S3")):
        n = 0
        for rec in M.iter_records(train_path(fname)):
            n += 1
            index.add(rec, src)
            k = a2_key(rec)
            if k is not None and k in s1_a2_keys:
                b = a2_buckets.get(k)
                if b is None:
                    a2_buckets[k] = [RecordRef(src, rec.entity_id)]
                elif len(b) < CAP:
                    b.append(RecordRef(src, rec.entity_id))
            guard_tick()
        step(f"    {fname}: {n:,} rows indexed")
    step(f"  A2 buckets {len(a2_buckets):,}")

    step("  query both arms")
    want: dict[str, dict] = {"S2": {}, "S3": {}}
    n_prod_seen = 0

    def _add(w: dict, key: str, packed: int) -> None:
        cur = w.get(key)
        if cur is None:
            w[key] = packed
        elif type(cur) is int:
            w[key] = [cur, packed]
        else:
            cur.append(packed)

    s1_index = {eid: i for i, eid in enumerate(s1_order)}
    for eid in s1_order:
        rec = s1_records[eid]
        base = s1_index[eid] * 2
        prod = {(c.candidate.source, c.candidate.entity_id)
                for c in generate_candidates(rec, index, "S1")}
        n_prod_seen += len(prod)
        for src, cid in prod:
            _add(want[src], cid, base | 1)
        k = a2_key(rec)
        if k is not None:
            for r in a2_buckets.get(k, ()):
                if (r.source, r.entity_id) not in prod:
                    _add(want[r.source], r.entity_id, base)
        guard_tick()

    n_union = 0
    for src in ("S2", "S3"):
        w = want[src]
        ids = np.empty(sum(1 if type(v) is int else len(v)
                           for v in w.values()), dtype=object)
        packed = np.empty(ids.shape[0], dtype=np.int64)
        i = 0
        for cid, v in w.items():
            if type(v) is int:
                ids[i] = cid
                packed[i] = v
                i += 1
            else:
                for pk in v:
                    ids[i] = cid
                    packed[i] = pk
                    i += 1
        np.save(OUT / f"cand_{src}_ids.npy", ids, allow_pickle=True)
        np.save(OUT / f"cand_{src}_packed.npy", packed)
        n_union += ids.shape[0]
        step(f"    {src}: {ids.shape[0]:,} candidate rows over {len(w):,} ids")

    del index, a2_buckets, want
    gc.collect()
    check("production candidate count", n_prod_seen == EXPECT_PROD_CANDIDATES,
          f"{n_prod_seen:,} == {EXPECT_PROD_CANDIDATES:,}")
    check("union candidate count", n_union == EXPECT_UNION_CANDIDATES,
          f"{n_union:,} == {EXPECT_UNION_CANDIDATES:,}")
    check("A2-only additions", n_union - n_prod_seen == 1_207_383,
          f"{n_union - n_prod_seen:,} == 1,207,383")
    (OUT / "stage1_meta.json").write_text(json.dumps({
        "n_s1": len(s1_records), "true_pairs": sum(len(v) for v in truth.values()),
        "production_candidates": n_prod_seen, "union_candidates": n_union,
        "blas_threads": BLAS_THREADS,
        "thermal": GUARD.report() if GUARD else None,
    }, indent=2), encoding="utf-8")
    step("STAGE 1 complete")


# ==========================================================================
# stage 2: featurisation
# ==========================================================================
def load_want() -> dict[str, dict]:
    want: dict[str, dict] = {}
    for src in ("S2", "S3"):
        ids = np.load(OUT / f"cand_{src}_ids.npy", allow_pickle=True)
        packed = np.load(OUT / f"cand_{src}_packed.npy")
        w: dict = {}
        for cid, pk in zip(ids.tolist(), packed.tolist()):
            cur = w.get(cid)
            if cur is None:
                w[cid] = pk
            elif type(cur) is int:
                w[cid] = [cur, pk]
            else:
                cur.append(pk)
        want[src] = w
    return want


def stage2(args) -> None:
    step("STAGE 2: featurisation")
    s1_records = pickle.loads((OUT / "s1_records.pkl").read_bytes())
    truth = pickle.loads((OUT / "truth.pkl").read_bytes())
    s1_order = sorted(s1_records)
    want = load_want()
    n_union = sum(sum(1 if type(v) is int else len(v)
                      for v in want[src].values()) for src in ("S2", "S3"))
    step(f"  union rows {n_union:,}, {N_FEATURES} features, float32 "
         f"({n_union * N_FEATURES * 4 / 1e6:.0f} MB)")

    x_path = OUT / "X.npy"
    state_path = OUT / "stage2_state.json"
    pos = 0
    n_prod_true = 0
    done_src: list[str] = []
    if x_path.exists() and state_path.exists() and not args.force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pos = state["pos"]
        n_prod_true = state["n_prod_true"]
        done_src = state["done_src"]
        saved = np.load(OUT / "stage2_rows.npz")
        y = saved["y"].copy()
        is_prod = saved["is_prod"].copy()
        is_a2only_true = saved["is_a2only_true"].copy()
        s1_of_row = saved["s1_of_row"].copy()
        step(f"  resuming from checkpoint: {pos:,} rows done, {done_src} finished")

    X = np.lib.format.open_memmap(
        x_path, mode="r+" if x_path.exists() and not args.force else "w+",
        dtype=np.float32, shape=(n_union, N_FEATURES))
    if pos == 0:
        y = np.zeros(n_union, dtype=np.int8)
        is_prod = np.zeros(n_union, dtype=bool)
        is_a2only_true = np.zeros(n_union, dtype=bool)
        s1_of_row = np.zeros(n_union, dtype=np.int32)

    def checkpoint(src: str) -> None:
        if src not in done_src:
            done_src.append(src)          # must persist in memory too, or the
        X.flush()                        # next checkpoint overwrites this one
        np.savez(OUT / "stage2_rows.npz", y=y, is_prod=is_prod,
                 is_a2only_true=is_a2only_true, s1_of_row=s1_of_row)
        state_path.write_text(json.dumps({
            "pos": int(pos), "n_prod_true": int(n_prod_true),
            "done_src": done_src, "n_union": n_union,
        }), encoding="utf-8")
        step(f"  checkpoint written after {src}: {pos:,} rows "
             f"(completed: {done_src})")

    if pos == n_union:
        step(f"  all {n_union:,} rows already featurised; nothing to do")
    for fname, src in (("train_source2.tsv", "S2"), ("train_source3.tsv", "S3")):
        if pos >= n_union:
            break
        if src in done_src:
            step(f"  {src} already done, skipping")
            continue
        step(f"  featurising {src}")
        w = want[src]
        seen = 0
        started = pos
        for rec in M.iter_records(train_path(fname)):
            v = w.get(rec.entity_id)
            if v is None:
                guard_tick()
                continue
            seen += 1
            for pk in ((v,) if type(v) is int else v):
                si = pk >> 1
                prod_row = bool(pk & 1)
                s1_eid = s1_order[si]
                X[pos] = featurize(s1_records[s1_eid], rec)
                s1_of_row[pos] = si
                is_prod[pos] = prod_row
                hit = rec.entity_id in truth.get(s1_eid, frozenset())
                y[pos] = 1 if hit else 0
                is_a2only_true[pos] = (not prod_row) and hit
                if hit and prod_row:
                    n_prod_true += 1
                pos += 1
            guard_tick()
        step(f"    {src}: joined {seen:,} ids, {pos - started:,} rows "
             f"({pos:,} total)")
        if seen != len(w):
            raise AssertionError(
                f"{src}: {len(w) - seen:,} wanted candidate ids never found")
        checkpoint(src)
        if GUARD is not None and GUARD.temp() is not None and \
                GUARD.temp() >= GUARD.pause_above:
            step("  hot after a file; pausing before the next one")
            GUARD.check(force=True)

    if pos != n_union:
        raise AssertionError(f"featurised {pos:,} of {n_union:,} expected rows")
    X.flush()
    step(f"  feature matrix {pos:,} x {N_FEATURES}; positives {int(y.sum()):,}")
    check("all rows featurised", pos == n_union, f"{pos:,}")
    check("production-arm true pairs", n_prod_true == EXPECT_PROD_TRUE,
          f"{n_prod_true:,} == {EXPECT_PROD_TRUE:,}")
    check("A2-only true pairs", int(is_a2only_true.sum()) == EXPECT_A2_ONLY_TRUE,
          f"{int(is_a2only_true.sum()):,} == {EXPECT_A2_ONLY_TRUE:,}")
    check("total true pairs", int(y.sum()) == EXPECT_PROD_TRUE + EXPECT_A2_ONLY_TRUE,
          f"{int(y.sum()):,}")
    del want
    gc.collect()
    step("STAGE 2 complete")


# ==========================================================================
# stage 3: fit, evaluate, report
# ==========================================================================
def stage3(args) -> None:
    step("STAGE 3: fit + evaluate")
    s1_order = sorted(pickle.loads((OUT / "s1_records.pkl").read_bytes()))
    X = np.load(OUT / "X.npy", mmap_mode="r")
    rows = np.load(OUT / "stage2_rows.npz")
    y, is_prod = rows["y"], rows["is_prod"]
    is_a2only_true, s1_of_row = rows["is_a2only_true"], rows["s1_of_row"]
    n_union = X.shape[0]
    step(f"  loaded {n_union:,} x {X.shape[1]} features")

    probe = [M.LabelledPair(
        CandidatePair(RecordRef("S1", eid), RecordRef("S2", "S2-0")), 0)
        for eid in s1_order]
    train_probe, val_probe = M.split_by_reference(
        probe, validation_fraction=SPLIT_FRACTION, salt=SPLIT_SALT)
    val_s1 = {i.pair.reference.entity_id for i in val_probe}
    train_s1 = {i.pair.reference.entity_id for i in train_probe}
    del probe, train_probe, val_probe
    s1_index = {eid: i for i, eid in enumerate(s1_order)}
    is_val_s1 = np.zeros(len(s1_order), dtype=bool)
    for eid in val_s1:
        is_val_s1[s1_index[eid]] = True
    row_is_val = is_val_s1[s1_of_row]
    step(f"  train S1 {len(train_s1):,}, validation S1 {len(val_s1):,}")

    def run_arm(name: str, mask):
        if mask is None:
            idx = None
            ya, a2t, row_val = y, is_a2only_true, row_is_val
        else:
            idx = np.flatnonzero(mask)
            ya, a2t, row_val = y[idx], is_a2only_true[idx], row_is_val[idx]
        val_idx = np.flatnonzero(row_val)
        trn_idx = np.flatnonzero(~row_val)
        ytr = ya[trn_idx].tolist()
        yva = ya[val_idx].tolist()
        Xtr = X if trn_idx.size == ya.size else X[trn_idx]
        Xva = np.asarray(X[val_idx])
        step(f"  [{name}] rows {ya.size:,} (train {trn_idx.size:,}, "
             f"valid {val_idx.size:,}) positives {int(ya.sum()):,} "
             f"({int(ya.sum()) / max(1, ya.size):.4%})")
        model = M.LogisticMatcher()
        model.fit(Xtr, ytr)
        s = model.summary
        step(f"  [{name}] fitted: n_iter={s.n_iter} converged={s.converged} "
             f"class_weight={s.class_weight}")
        if GUARD is not None:
            GUARD.check(force=True)
        scores = model.predict_proba(Xva)
        reports = M.evaluate_thresholds(yva, scores)
        best = M.best_validation_threshold(reports)
        pred = [sc >= best.threshold for sc in scores]
        a2_val = a2t[val_idx].tolist()
        out = {
            "name": name, "rows": int(ya.size), "positives": int(ya.sum()),
            "pos_rate": float(ya.sum()) / max(1, ya.size),
            "train_rows": int(trn_idx.size), "valid_rows": int(val_idx.size),
            "n_iter": s.n_iter, "converged": s.converged,
            "class_weight": s.class_weight, "reports": reports, "best": best,
            "tp": best.true_positives, "fp": best.false_positives,
            "fn": best.false_negatives, "precision": best.precision,
            "recall": best.recall, "f_beta": best.f_beta,
            "pred_pos": best.predicted_positives,
            "a2_only_true_total": int(a2t.sum()),
            "a2_only_true_valid": sum(a2_val),
            "a2_only_true_caught": sum(1 for a, p in zip(a2_val, pred) if a and p),
        }
        del Xtr, Xva, scores, pred
        gc.collect()
        return out

    arm1 = run_arm("production", is_prod)
    arm2 = run_arm("production+A2", None)

    # ---------------------------------------------------------------- report
    print()
    print("=" * 78)
    print("1. CONTROLS (must match the blocking A/B exactly)")
    print("=" * 78)
    check("union candidate count", n_union == EXPECT_UNION_CANDIDATES,
          f"{n_union:,} == {EXPECT_UNION_CANDIDATES:,}")
    check("production candidate count", int(is_prod.sum()) == EXPECT_PROD_CANDIDATES,
          f"{int(is_prod.sum()):,} == {EXPECT_PROD_CANDIDATES:,}")
    check("feature matrix width", X.shape[1] == N_FEATURES, f"{X.shape[1]}")
    check("no NaN/inf in features",
          bool(np.isfinite(np.asarray(X)).all()), "checked all rows")
    check("every S1 entity on exactly one side",
          len(train_s1) + len(val_s1) == len(s1_order),
          f"{len(train_s1):,} + {len(val_s1):,} = {len(s1_order):,}")
    check("no S1 entity in both sides", not (train_s1 & val_s1))

    print()
    print("=" * 78)
    print("2. ARMS")
    print("=" * 78)
    for a in (arm1, arm2):
        print(f"  {a['name']}:")
        print(f"    candidate rows            {a['rows']:,}")
        print(f"    true pairs                {a['positives']:,} "
              f"({a['pos_rate']:.4%} positive)")
        print(f"    train / validation rows   {a['train_rows']:,} / {a['valid_rows']:,}")
        print(f"    class_weight={a['class_weight']}  n_iter={a['n_iter']}  "
              f"converged={a['converged']}")

    print()
    print("=" * 78)
    print("3. VALIDATION RESULT AT EACH ARM'S BEST F0.5 THRESHOLD")
    print("=" * 78)
    for a in (arm1, arm2):
        b = a["best"]
        print(f"  {a['name']}")
        print(f"    {'threshold':>10} {'precision':>10} {'recall':>9} {'F0.5':>9} "
              f"{'TP':>10} {'FP':>10} {'FN':>10} {'pred+':>11}")
        print(f"    {b.threshold:>10.1f} {b.precision:>10.4f} {b.recall:>9.4f} "
              f"{b.f_beta:>9.4f} {b.true_positives:>10,} "
              f"{b.false_positives:>10,} {b.false_negatives:>10,} "
              f"{b.predicted_positives:>11,}")
    print()
    print(f"  F0.5 change (+A2 minus production): {arm2['f_beta'] - arm1['f_beta']:+.4f}")
    print(f"  precision change: {arm2['precision'] - arm1['precision']:+.4f}")
    print(f"  recall change:    {arm2['recall'] - arm1['recall']:+.4f}")
    print(f"  extra TP:         {arm2['tp'] - arm1['tp']:+,}")
    print(f"  extra FP:         {arm2['fp'] - arm1['fp']:+,}")

    for a in (arm1, arm2):
        print()
        print(f"  full threshold sweep, {a['name']}:")
        print(f"    {'thr':>5} {'prec':>9} {'rec':>9} {'F0.5':>9} {'TP':>9} "
              f"{'FP':>9} {'FN':>9} {'pred+':>10}")
        for r in a["reports"]:
            print(f"    {r.threshold:5.1f} {r.precision:9.4f} {r.recall:9.4f} "
                  f"{r.f_beta:9.4f} {r.true_positives:9,} "
                  f"{r.false_positives:9,} {r.false_negatives:9,} "
                  f"{r.predicted_positives:10,}")

    print()
    print("=" * 78)
    print("4. THE 2,995 A2-RECOVERED TRUE PAIRS: DOES THE MATCHER CATCH THEM?")
    print("=" * 78)
    print(f"  A2-only true pairs in the candidate set   : {arm2['a2_only_true_total']:,}")
    print(f"  ... in the VALIDATION split               : {arm2['a2_only_true_valid']:,}")
    print(f"  ... predicted positive at the +A2 best     : "
          f"{arm2['a2_only_true_caught']:,} "
          f"(threshold {arm2['best'].threshold})")
    rec = arm2["a2_only_true_caught"] / max(1, arm2["a2_only_true_valid"])
    print(f"  recall on the A2-recovered subset         : {rec:.4%}")
    print(f"  (the other "
          f"{arm2['a2_only_true_valid'] - arm2['a2_only_true_caught']:,} "
          f"scored below threshold)")
    print()
    print("  These pairs do not exist in the production arm at all: blocking never")
    print("  retrieved them, so the production model had no opportunity to score them.")

    print()
    print("=" * 78)
    print("5. THERMAL / RESOURCE CONTROLS FOR THIS RUN")
    print("=" * 78)
    if GUARD is not None:
        t = GUARD.report()
        hottest = t["max_cpu_temp_c"]
        hottest_text = "unavailable" if hottest is None else f"{hottest:.1f} C"
        note(f"temperature monitoring : {t['temperature_monitoring']}")
        note(f"max CPU temp observed  : {hottest_text}")
        note(f"pause/resume/abort C  : {t['pause_threshold_c']:.0f} / "
             f"{t['resume_threshold_c']:.0f} / {t['abort_threshold_c']:.0f}")
        note(f"pause events          : {t['pause_events']} "
             f"({t['total_paused_seconds']:.0f}s total)")
        note(f"duty cycle            : {t['duty_sleep_seconds_per']}s every "
             f"{t['duty_every_rows']:,} rows")
    note(f"BLAS thread env       : {TH.describe_threads()}")
    note(f"BLAS threads cap      : {BLAS_THREADS}")
    note(f"process priority      : {PRIORITY}")
    note(f"CPUs allowed          : {CPUS_ALLOWED} of {os.cpu_count()} logical")

    print()
    print("=" * 78)
    print("CHECKS")
    print("=" * 78)
    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail else ""))
    print()
    print("=" * 78)
    print("Validation F0.5 on a bounded S1 subset with the full S2/S3 pool. NOT")
    print("competition performance and NOT a submitted result. The 20k S1 subset is")
    print("a sample, and the competition metric is macro-averaged over S1 entities,")
    print("whereas the numbers above are pair-level on pooled candidates.")
    print("=" * 78)
    step("STAGE 3 complete")


# ==========================================================================
def guard_tick() -> None:
    if GUARD is not None:
        GUARD.tick()


PRIORITY = "not set"
CPUS_ALLOWED = "all"


def main(argv: list[str] | None = None) -> int:
    global GUARD, PRIORITY, CPUS_ALLOWED
    parser = argparse.ArgumentParser(
        description="Matching-stage A/B, staged for thermal safety.")
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="1")
    parser.add_argument("--threads", type=int, default=1,
                        help="BLAS/OpenMP threads (read before numpy import)")
    parser.add_argument("--cpus", type=int, default=3,
                        help="logical processors this process may use")
    parser.add_argument("--pause-above", type=float,
                        default=TH.DEFAULT_PAUSE_ABOVE_C)
    parser.add_argument("--resume-below", type=float,
                        default=TH.DEFAULT_RESUME_BELOW_C)
    parser.add_argument("--abort-above", type=float,
                        default=TH.DEFAULT_ABORT_ABOVE_C)
    parser.add_argument("--duty-sleep", type=float, default=0.05,
                        help="seconds of sleep per --duty-every records")
    parser.add_argument("--duty-every", type=int, default=2000)
    parser.add_argument("--no-thermal-guard", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="ignore existing checkpoints and recompute")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    PRIORITY = TH.set_low_priority(below_normal=True)
    mask = TH.default_cpu_mask(args.cpus)
    CPUS_ALLOWED = TH.pin_to_cpus(mask) or f"failed (mask {mask})"
    time.sleep(1.0)      # let priority/affinity settle before real work

    if not args.no_thermal_guard:
        GUARD = TH.ThermalGuard(
            pause_above=args.pause_above, resume_below=args.resume_below,
            abort_above=args.abort_above, duty_sleep=args.duty_sleep,
            duty_every=args.duty_every, label="cpu",
        )
    step(f"stage={args.stage}  priority={PRIORITY}  "
         f"cpus={CPUS_ALLOWED}/{os.cpu_count()}  blas_threads={BLAS_THREADS}")
    step(f"BLAS env: {TH.describe_threads()}")

    stages = ["1", "2", "3"] if args.stage == "all" else [args.stage]
    for i, st in enumerate(stages):
        try:
            if st == "1":
                stage1(args)
            elif st == "2":
                stage2(args)
            else:
                stage3(args)
        except TH.ThermalAbort as exc:
            step(f"THERMAL STOP during stage {st}: {exc}")
            if GUARD is not None:
                GUARD.stop()
            return 2
        if i + 1 < len(stages):
            step(f"cooling down 60s before stage {stages[i + 1]}")
            time.sleep(60)
    if GUARD is not None:
        GUARD.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
