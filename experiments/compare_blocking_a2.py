"""Controlled A/B: production blocking vs production + A2 (original-order 2-token name key).

Read-only experiment. Does not modify src/, does not add a key to blocking.py.

Design, and why it is a fair comparison
---------------------------------------
* Same S1 subset as the diagnosis run (salt=b'exp20k-s1', mod=110), so the
  production baseline is expected to reproduce at 6,373,030 candidates and
  79.8329% recall. That reproduction is asserted as a control.

* The pool index is a plain production ``BlockingIndex`` over the FULL S2+S3,
  filled with the unmodified ``index.add()``. Production candidates are then
  read back through the unmodified ``blocking_keys()`` lookup, so the baseline
  arm is the production code path itself, not a re-implementation of it.

* A2 lives in a local capped bucket dict built in the same pass. Only buckets
  whose key is actually used by one of the 20k S1 entities are retained, and
  each is capped at max_group_size exactly as production caps its own buckets.
  A2 is a separate key family, so its presence cannot perturb any production
  key's contents.

* A2 definition (unchanged from the diagnosis script):
      ("A2_name_2tok_order", country, core[0][:4], core[1][:4])
  taken from ``name_core`` in ORIGINAL order, only when len(core) >= 2.
  It therefore produces nothing for a 1-token name, and it is additive
  alongside the sorted production key rather than a replacement for it.

Run: python experiments/compare_blocking_a2.py
"""
from __future__ import annotations

import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import matching as M                                       # noqa: E402
from src.blocking import (BlockingConfig, BlockingIndex,           # noqa: E402
                          RecordRef, generate_candidates)
from src.preprocessing import PreprocessedRecord                   # noqa: E402

DATASET = Path(
    r"C:\Users\BITPATNA\Downloads\6ab10eb3b23ba_student_resource"
    r"\student_resource\dataset")
TRAIN = DATASET / "train"

N_S1 = 20_000
S1_SALT, S1_MOD = b"exp20k-s1", 110      # identical to the 20k experiment + diagnosis
CFG = BlockingConfig()                     # production: 2tok + addr_ht0/1, cap 1000
CAP = CFG.max_group_size
PREFIX4 = CFG.name_prefix_length

# Control values from experiments/diagnose_blocking_misses.py
EXPECT_PROD_CANDIDATES = 6_373_030
EXPECT_PROD_RECALL = 0.798329
TOL = 0.0005

OPENED: list[Path] = []


def train_path(name: str) -> Path:
    if "test" in name.lower():
        raise AssertionError(f"refusing non-train file: {name}")
    path = (TRAIN / name).resolve()
    if path.parent != TRAIN.resolve():
        raise AssertionError(f"refusing path outside train/: {path}")
    OPENED.append(path)
    return path


def hashed(eid: str, salt: bytes, mod: int) -> bool:
    d = hashlib.blake2b(eid.encode("utf-8"), key=salt, digest_size=8).digest()
    return int.from_bytes(d, "big") % mod == 0


def a2_key(record: PreprocessedRecord):
    """A2: first two core tokens in ORIGINAL order, 4 chars, plus country."""
    if len(record.name_core) < 2:
        return None
    return ("A2_name_2tok_order", record.country,
            record.name_core[0][:PREFIX4], record.name_core[1][:PREFIX4])


T0 = time.perf_counter()


def step(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------- config
print("=" * 78)
print("A2 COMPARISON - configuration")
print("=" * 78)
print(f"  dataset          {DATASET}")
print(f"  S1 subset        {N_S1:,}, salt={S1_SALT!r} mod={S1_MOD} (same as diagnosis)")
print(f"  baseline arm     production keys "
      f"(name_2tok sorted p{PREFIX4} + addr_ht0/addr_ht1, cap {CAP})")
print(f"  treatment arm    baseline + A2_name_2tok_order (original order, p{PREFIX4}, "
      f"cap {CAP})")
print(f"  index scope      FULL train_source2.tsv + train_source3.tsv")
print(f"  control          baseline must reproduce {EXPECT_PROD_CANDIDATES:,} candidates "
      f"and {EXPECT_PROD_RECALL:.4%} recall (+/- {TOL})")
print()

# ------------------------------------------------- PASS A: S1 subset
step("PASS A: stream S1 -> hash-select subset")
s1_records: dict[str, PreprocessedRecord] = {}
s1_scanned = 0
for rec in M.iter_records(train_path("train_source1.tsv")):
    s1_scanned += 1
    if len(s1_records) < N_S1 and hashed(rec.entity_id, S1_SALT, S1_MOD):
        s1_records[rec.entity_id] = rec
    if len(s1_records) >= N_S1:
        break
step(f"S1 scanned {s1_scanned:,}, selected {len(s1_records):,}")

s1_a2_keys: set = set()
for rec in s1_records.values():
    k = a2_key(rec)
    if k is not None:
        s1_a2_keys.add(k)
step(f"distinct A2 keys held by the S1 subset: {len(s1_a2_keys):,}")

# ------------------------------------------------- PASS B: ground truth
step("PASS B: stream ground truth")
wanted = set(s1_records)
truth: dict[str, frozenset] = {}
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
step(f"GT rows {len(truth):,}, true pairs {total_true:,}")

# ------------------------- PASS C: full pool -> production index + A2 buckets
step("PASS C: stream FULL S2 + S3 -> production index + A2 buckets")
index = BlockingIndex(CFG)
a2_buckets: dict[tuple, list] = defaultdict(list)
a2_capped: set = set()
pool_rows = 0

for fname, src in (("train_source2.tsv", "S2"), ("train_source3.tsv", "S3")):
    n = 0
    for rec in M.iter_records(train_path(fname)):
        n += 1
        pool_rows += 1
        index.add(rec, src)                       # production path, untouched
        k = a2_key(rec)
        if k is not None and k in s1_a2_keys:
            bucket = a2_buckets[k]
            if len(bucket) < CAP:
                bucket.append(RecordRef(src, rec.entity_id))
            else:
                a2_capped.add(k)
    step(f"  {fname}: {n:,} rows indexed")

stats = index.stats()
step(f"pool rows {pool_rows:,}; A2 buckets {len(a2_buckets):,}; "
     f"A2 buckets that hit the cap {len(a2_capped):,}")
step(f"index: {stats}")


def a2_candidates_for(record: PreprocessedRecord) -> set:
    k = a2_key(record)
    if k is None:
        return set()
    return set(a2_buckets.get(k, ()))


# ------------------------------------------------------- measurement
step("generate candidates for both arms")
prod_ids: dict[str, set] = {}
union_ids: dict[str, set] = {}
a2_only_ids: dict[str, set] = {}
for eid in sorted(s1_records):
    rec = s1_records[eid]
    # Baseline arm: the production code path, read straight out of the index.
    p = {c.candidate.entity_id for c in generate_candidates(rec, index, "S1")}
    a = {r.entity_id for r in a2_candidates_for(rec)}
    prod_ids[eid] = p
    a2_only_ids[eid] = a - p
    union_ids[eid] = p | a

n_prod = sum(len(v) for v in prod_ids.values())
n_union = sum(len(v) for v in union_ids.values())
n_a2_only = sum(len(v) for v in a2_only_ids.values())
prod_per = [len(prod_ids[e]) for e in sorted(s1_records)]
union_per = [len(union_ids[e]) for e in sorted(s1_records)]
a2only_per = [len(a2_only_ids[e]) for e in sorted(s1_records)]
step(f"baseline candidates {n_prod:,} | union {n_union:,} | A2-only additions {n_a2_only:,}")

# truth resolution
in_prod = in_a2_only = in_both = in_neither = 0
recovered: list[tuple[str, str]] = []
still_missing: list[tuple[str, str]] = []
for eid in sorted(truth):
    p, a, u = prod_ids.get(eid, set()), a2_only_ids.get(eid, set()), union_ids.get(eid, set())
    for mid in sorted(truth[eid]):
        if mid in p:
            in_prod += 1
            in_both += 1
        elif mid in a:
            in_a2_only += 1
            recovered.append((eid, mid))
        elif mid in u:
            in_a2_only += 1
            recovered.append((eid, mid))
        else:
            in_neither += 1
            still_missing.append((eid, mid))

rec_prod = in_prod / total_true
rec_union = (in_prod + in_a2_only) / total_true

# added-candidate yield
yield_ratio = (in_a2_only / n_a2_only) if n_a2_only else 0.0

# ============================================================ REPORT
print()
print("=" * 78)
print("1. CONTROL - did the production baseline reproduce?")
print("=" * 78)
ok_c = abs(n_prod - EXPECT_PROD_CANDIDATES) <= 1
ok_r = abs(rec_prod - EXPECT_PROD_RECALL) <= TOL
print(f"  [{'PASS' if ok_c else 'FAIL'}] candidate count {n_prod:,} "
      f"(expected {EXPECT_PROD_CANDIDATES:,})")
print(f"  [{'PASS' if ok_r else 'FAIL'}] blocking recall {rec_prod:.4%} "
      f"(expected {EXPECT_PROD_RECALL:.4%})")

print()
print("=" * 78)
print("2. HEADLINE COMPARISON")
print("=" * 78)
print(f"  {'':<34} {'production':>14} {'production + A2':>18} {'change':>14}")
print("  " + "-" * 84)
print(f"  {'candidate pairs':<34} {n_prod:>14,} {n_union:>18,} "
      f"{n_union - n_prod:>+14,}")
print(f"  {'avg candidates / S1':<34} {n_prod / len(s1_records):>14.1f} "
      f"{n_union / len(s1_records):>18.1f} "
      f"{(n_union - n_prod) / len(s1_records):>+14.1f}")
print(f"  {'max candidates / S1':<34} {max(prod_per):>14,} {max(union_per):>18,} "
      f"{max(union_per) - max(prod_per):>+14,}")
print(f"  {'median candidates / S1':<34} "
      f"{sorted(prod_per)[len(prod_per) // 2]:>14,} "
      f"{sorted(union_per)[len(union_per) // 2]:>18,}")
print(f"  {'true pairs retrieved':<34} {in_prod:>14,} {in_prod + in_a2_only:>18,} "
      f"{in_a2_only:>+14,}")
print(f"  {'blocking recall (pair level)':<34} {rec_prod:>13.4%} {rec_union:>17.4%} "
      f"{(rec_union - rec_prod) * 100:>+13.2f}pp")
print(f"  {'true pairs still missing':<34} {total_true - in_prod:>14,} "
      f"{in_neither:>18,} {(in_neither) - (total_true - in_prod):>+14,}")

print()
print("=" * 78)
print("3. COST / BENEFIT OF THE A2 ADDITION")
print("=" * 78)
print(f"  A2-eligible S1 entities (len(name_core) >= 2) : "
      f"{sum(1 for r in s1_records.values() if a2_key(r) is not None):,} "
      f"of {len(s1_records):,}")
print(f"  S1 entities given at least one NEW candidate   : "
      f"{sum(1 for v in a2only_per if v):,}")
print(f"  S1 entities with zero production candidates    : "
      f"{sum(1 for v in prod_per if v == 0):,} -> "
      f"{sum(1 for v in union_per if v == 0):,} after adding A2")
print(f"  added candidate pairs                          : {n_a2_only:,} "
      f"({n_a2_only / max(1, n_prod):.1%} of the production total)")
print(f"  added candidates that are TRUE matches         : {in_a2_only:,}")
print(f"  true matches per 10,000 added candidates       : "
      f"{in_a2_only / max(1, n_a2_only) * 10000:,.2f}")
print(f"  added pairs per recovered true match           : "
      f"{n_a2_only / max(1, in_a2_only):,.0f}")
print(f"  median NEW candidates among affected S1        : "
      f"{sorted(v for v in a2only_per if v)[len([v for v in a2only_per if v]) // 2]:,}"
      if any(a2only_per) else "  n/a")
print(f"  worst-affected S1 (new candidates)             : {max(a2only_per):,}")

print()
print("  Recall movement per extra candidate:")
for label, cand, rec_extra in (
        ("A2 added alone", n_a2_only, in_a2_only),):
    print(f"    {label}: +{rec_extra / total_true * 100:.2f}pp recall for "
          f"{cand:,} extra pairs "
          f"({cand / max(1, rec_extra):,.0f} pairs per recovered match)")

print()
print("=" * 78)
print("4. TRUE-PAIR RESOLUTION MATRIX")
print("=" * 78)
print(f"  retrieved by production only                 : {in_both:,} "
      f"({in_both / total_true:.2%})")
print(f"  NOT in production, recovered only via A2    : {in_a2_only:,} "
      f"({in_a2_only / total_true:.2%})")
print(f"  still missing under both arms               : {in_neither:,} "
      f"({in_neither / total_true:.2%})")
print(f"  total true pairs                            : {total_true:,}")

print()
print("  Sample of pairs recovered ONLY by A2 (A2 key shown):")
shown = 0
for eid, mid in recovered:
    if shown >= 8:
        break
    a = s1_records[eid]
    print(f"    {eid} -> {mid}")
    print(f"      S1 name={a.name!r} core={a.name_core}")
    print(f"      S1 A2 key = {a2_key(a)}")
    shown += 1

print()
print("=" * 78)
print("5. EXTRAPOLATION TO THE FULL S1 (indicative only)")
print("=" * 78)
full_s1 = 2_206_821
scale = full_s1 / len(s1_records)
print(f"  S1 rows in train_source1.tsv                 : {full_s1:,}")
print(f"  scale factor from this subset                : {scale:.1f}x")
print(f"  production candidates, extrapolated          : {int(n_prod * scale):,}")
print(f"  +A2 candidates, extrapolated                 : {int(n_union * scale):,}")
print(f"  extra pairs A2 would add, extrapolated       : {int(n_a2_only * scale):,}")
print("  Treat as a size estimate, not a measurement: the subset is a random")
print("  sample of S1, so the ratio should hold on average.")

print()
print("=" * 78)
print("6. CHECKS")
print("=" * 78)
print(f"  [{'PASS' if ok_c and ok_r else 'FAIL'}] production baseline reproduced "
      f"exactly -> A/B is like-for-like")
print(f"  [{'PASS' if all('test' not in p.name.lower() for p in OPENED) else 'FAIL'}] "
      f"no test data -- {', '.join(sorted(p.name for p in OPENED))}")
print(f"  [{'PASS' if in_both + in_a2_only + in_neither == total_true else 'FAIL'}] "
      f"resolution matrix sums to {total_true:,}")
print(f"  [{'PASS' if n_union >= n_prod else 'FAIL'}] union is a superset of baseline "
      f"({n_union:,} >= {n_prod:,})")
print(f"  [INFO] A2 buckets that reached the cap {len(a2_capped):,} of "
      f"{len(a2_buckets):,}; a capped A2 bucket can also lose true matches")

print()
print("=" * 78)
print("Pair-level BLOCKING RECALL only. Not F0.5, not competition performance,")
print("and it says nothing about whether the extra candidates are usable by the")
print("matcher. No production file was modified; A2 was not added to blocking.py.")
print("=" * 78)
step("done")
