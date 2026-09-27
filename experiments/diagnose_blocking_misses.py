"""Diagnose which true S1<->S2/S3 matches the current blocking keys MISS.

Read-only diagnosis. Does not modify src/, does not add a blocking key, does
not train anything, uses TRAIN files only.

Why the index covers the FULL S2 + S3
--------------------------------------
The previous 20k experiment built a bounded pool that *kept every true match*,
so it could not reveal blocking misses at all. To see a genuine miss the index
must hold every real candidate row. So this script indexes all of S2 and S3,
exactly as production would, and only holds records for (a) the selected S1
entities and (b) their ground-truth matches. Everything else is streamed
through the index and released.

Reproducibility
---------------
Same S1 subset as the 20k experiment (salt=b'exp20k-s1', mod=110), so the
numbers are directly comparable. Alternative key families are PRE-REGISTERED
below, before any result was seen, so the tradeoff table is not cherry-picked.

Run: python experiments/diagnose_blocking_misses.py
"""
from __future__ import annotations

import hashlib
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import matching as M                                        # noqa: E402
from src.blocking import (BlockingConfig, BlockingIndex,             # noqa: E402
                          address_blocking_keys, generate_candidates,
                          name_blocking_key)
from src.preprocessing import PreprocessedRecord                      # noqa: E402

DATASET = Path(
    r"C:\Users\BITPATNA\Downloads\6ab10eb3b23ba_student_resource"
    r"\student_resource\dataset")
TRAIN = DATASET / "train"

N_S1 = 20_000
S1_SALT, S1_MOD = b"exp20k-s1", 110      # identical to the 20k experiment
CFG = BlockingConfig()                     # 2tok/addr_ht0/addr_ht1, cap 1000
PREFIX4 = CFG.name_prefix_length
PREFIX3 = CFG.address_prefix_length

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


T0 = time.perf_counter()


def step(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


# =====================================================================
# PRE-REGISTERED alternative key families (analysis only, never indexed
# for production). Each returns a key or None.
# =====================================================================

def _p4(tok: str) -> str:
    return tok[:PREFIX4]


def _p3(tok: str) -> str:
    return tok[:PREFIX3]


def k_prod_name(r: PreprocessedRecord):
    return name_blocking_key(r, CFG)


def k_A1_name_1tok(r: PreprocessedRecord):
    """First core token in ORIGINAL order, 4 chars, + country."""
    if not r.name_core:
        return None
    return ("A1_name_1tok", r.country, _p4(r.name_core[0]))


def k_A2_name_2tok_order(r: PreprocessedRecord):
    """First two core tokens in ORIGINAL order (production sorts them)."""
    if len(r.name_core) < 2:
        return None
    return ("A2_name_2tok_order", r.country, _p4(r.name_core[0]), _p4(r.name_core[1]))


def k_A3_name_2tok_p3(r: PreprocessedRecord):
    """Production shape but 3-char prefixes instead of 4."""
    t = sorted(r.name_core)
    if not t:
        return None
    if len(t) == 1:
        return ("A3_name_p3", r.country, _p3(t[0]))
    return ("A3_name_p3", r.country, _p3(t[0]), _p3(t[1]))


def k_A4_name_allcore(r: PreprocessedRecord):
    """ALL core tokens, sorted, 4-char prefixes (order-free, full name)."""
    if not r.name_core:
        return None
    return ("A4_name_allcore", r.country,
            *sorted(_p4(t) for t in r.name_core))


def k_A5_addr_exact(r: PreprocessedRecord):
    """Whole normalised address + country."""
    if not r.address:
        return None
    return ("A5_addr_exact", r.country, r.address)


def k_A6_addr_prefix8(r: PreprocessedRecord):
    """First 8 characters of the normalised address + country."""
    if not r.address:
        return None
    return ("A6_addr_prefix8", r.country, r.address[:8])


def k_A7_addr_ht_allp3(r: PreprocessedRecord):
    """House number + 3-char prefixes of ALL address words, order-free."""
    if not r.house_number or not r.address_alpha_tokens:
        return None
    return ("A7_addr_ht_all", r.country, r.house_number,
            *sorted(_p3(w) for w in r.address_alpha_tokens))


def k_A8_name_2tok_nocountry(r: PreprocessedRecord):
    """Production name key with the country component REMOVED."""
    t = sorted(r.name_core)
    if not t:
        return None
    return ("A8_name_nocountry", *(_p4(x) for x in t[:CFG.name_token_count]))


def k_A9_addr_ht_nocountry(r: PreprocessedRecord):
    """Production address keys with the country component REMOVED."""
    if not r.house_number:
        return None
    words = sorted(r.address_alpha_tokens, key=len, reverse=True)[:2]
    return tuple(sorted(
        ("A9_addr_nocountry", r.house_number, _p3(w)) for w in words))


ALT_KEYS = [
    ("A1_name_1tok", k_A1_name_1tok),
    ("A2_name_2tok_order", k_A2_name_2tok_order),
    ("A3_name_2tok_p3", k_A3_name_2tok_p3),
    ("A4_name_allcore", k_A4_name_allcore),
    ("A5_addr_exact", k_A5_addr_exact),
    ("A6_addr_prefix8", k_A6_addr_prefix8),
    ("A7_addr_ht_all", k_A7_addr_ht_allp3),
    ("A8_name_nocountry", k_A8_name_2tok_nocountry),
    ("A9_addr_nocountry", k_A9_addr_ht_nocountry),
]


# =====================================================================
# Script / similarity diagnostics
# =====================================================================

def script_class(name: str) -> str:
    if not name:
        return "empty"
    if name.isascii():
        return "latin"
    scripts = set()
    for ch in name:
        if not ch.isalpha():
            continue
        try:
            scripts.add(unicodedata.name(ch).split()[0])
        except ValueError:
            scripts.add("UNKNOWN")
    # Collapse the common Indic/SE-Asian families into readable buckets.
    joined = " ".join(sorted(scripts))
    if "DEVANAGARI" in joined:
        return "devanagari"
    if "TAMIL" in joined:
        return "tamil"
    if "LATIN" in joined and len(scripts) == 1:
        return "latin+marks"
    if "CJK" in joined or "HIRAGANA" in joined or "KATAKANA" in joined:
        return "cjk"
    if "ARABIC" in joined:
        return "arabic"
    if "CYRILLIC" in joined:
        return "cyrillic"
    if "GREEK" in joined:
        return "greek"
    return "other-script"


def bigram_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ga = {a[i:i + 2] for i in range(len(a) - 1)}
    gb = {b[i:i + 2] for i in range(len(b) - 1)}
    if not ga or not gb:
        return 1.0 if a == b else 0.0
    return len(ga & gb) / len(ga | gb)


# =====================================================================
# PASS A - select the S1 subset (same salt/mod as the 20k experiment)
# =====================================================================
print("=" * 78)
print("BLOCKING MISS DIAGNOSIS - configuration")
print("=" * 78)
print(f"  dataset          {DATASET}")
print(f"  S1 subset        {N_S1:,} entities, salt={S1_SALT!r} mod={S1_MOD} "
      f"(identical to the 20k experiment)")
print(f"  production keys  name_2tok (sorted, p{PREFIX4}) + addr_ht0/addr_ht1 "
      f"(house + p{PREFIX3}), cap {CFG.max_group_size}")
print(f"  index scope      FULL train_source2.tsv + train_source3.tsv")
print(f"  alt key families {', '.join(n for n, _ in ALT_KEYS)} (pre-registered)")
print()

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

# =====================================================================
# PASS B - ground truth
# =====================================================================
step("PASS B: stream ground truth -> matches for the selected S1")
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
matched_ids = {m for v in truth.values() for m in v}
total_true = sum(len(v) for v in truth.values())
step(f"GT rows {len(truth):,}, true pairs {total_true:,}, distinct match ids {len(matched_ids):,}")

# S1 key sets for the alternative-key volume counting (memory-bounded: we
# only remember keys the S1 entities actually use).
s1_alt_keys: dict[str, set] = {name: set() for name, _ in ALT_KEYS}
for rec in s1_records.values():
    for name, fn in ALT_KEYS:
        k = fn(rec)
        if k is not None:
            s1_alt_keys[name].add(k)
step("S1 alternative-key sets: " + ", ".join(
    f"{n}={len(v):,}" for n, v in s1_alt_keys.items()))

# =====================================================================
# PASS C - full S2/S3: build the production index, hold only matched
#          records, and count alternative-key hits per S1 key.
# =====================================================================
step("PASS C: stream FULL S2 + S3 -> production index + alt-key counts")
index = BlockingIndex(CFG)
matched_records: dict[str, PreprocessedRecord] = {}
alt_counts: dict[str, dict] = {name: {} for name, _ in ALT_KEYS}
alt_lookup = {name: s1_alt_keys[name] for name, _ in ALT_KEYS}
alt_fns = dict(ALT_KEYS)
pool_rows = 0
found_in_stream = 0

for fname, src in (("train_source2.tsv", "S2"), ("train_source3.tsv", "S3")):
    n = 0
    for rec in M.iter_records(train_path(fname)):
        n += 1
        pool_rows += 1
        index.add(rec, src)
        if rec.entity_id in matched_ids:
            matched_records[rec.entity_id] = rec
            found_in_stream += 1
        for name, fn in alt_fns.items():
            lookup = alt_lookup[name]
            if not lookup:
                continue
            k = fn(rec)
            if k is not None and k in lookup:
                d = alt_counts[name]
                d[k] = d.get(k, 0) + 1
    step(f"  {fname}: {n:,} rows indexed")

missing_records = sorted(matched_ids - set(matched_records))
stats = index.stats()
step(f"pool rows {pool_rows:,}; matched records recovered {found_in_stream:,}")
step(f"index: {stats}")

# =====================================================================
# Candidate generation with the PRODUCTION keys -> blocked-out true pairs
# =====================================================================
step("generate production candidates for the selected S1 entities")
cand_ids: dict[str, set] = {}
prod_counts: list[int] = []
for eid in sorted(s1_records):
    ids = {c.candidate.entity_id
           for c in generate_candidates(s1_records[eid], index, "S1")}
    cand_ids[eid] = ids
    prod_counts.append(len(ids))
step(f"production candidates total {sum(prod_counts):,}, "
     f"avg {sum(prod_counts) / max(1, len(prod_counts)):.1f}/S1, "
     f"max {max(prod_counts, default=0):,}")

blocked_out: list[tuple[str, str]] = []
found_pairs = 0
for eid in sorted(truth):
    for mid in sorted(truth[eid]):
        if mid in cand_ids.get(eid, ()):
            found_pairs += 1
        else:
            blocked_out.append((eid, mid))
recall = found_pairs / max(1, total_true)
step(f"true pairs found {found_pairs:,}/{total_true:,} -> recall {recall:.4%}")
step(f"BLOCKED-OUT true pairs: {len(blocked_out):,}")

# =====================================================================
# Characterise every blocked-out pair
# =====================================================================
def name_key_equal(a: PreprocessedRecord, b: PreprocessedRecord) -> bool:
    ka, kb = k_prod_name(a), k_prod_name(b)
    return ka is not None and ka == kb


def name_prefixes_equal(a: PreprocessedRecord, b: PreprocessedRecord) -> bool:
    """True if the name key would match if country were ignored."""
    ta, tb = sorted(a.name_core), sorted(b.name_core)
    if not ta or not tb:
        return False
    if len(ta) == 1 or len(tb) == 1:
        return _p4(ta[0]) == _p4(tb[0])
    return (_p4(ta[0]), _p4(ta[1])) == (_p4(tb[0]), _p4(tb[1]))


def addr_key_sets(r: PreprocessedRecord) -> set:
    return set(address_blocking_keys(r, CFG))


def addr_key_equal_nocountry(a: PreprocessedRecord, b: PreprocessedRecord) -> bool:
    return bool(k_A9_addr_ht_nocountry(a)) and \
        bool(set(k_A9_addr_ht_nocountry(a)) & set(k_A9_addr_ht_nocountry(b)))


FLAGS = [
    "name_missing_either",
    "name_no_core_token_either",
    "name_key_differs",
    "name_1tok_vs_multi",
    "name_core_count_differs",
    "name_prefix4_differs",
    "script_differs",
    "latin_both_low_similarity",
    "addr_missing_either",
    "addr_no_key_either",
    "addr_key_differs",
    "house_absent_either",
    "house_differs",
    "street_prefix_differs",
    "country_disagrees",
    "country_only_blocker",
    "both_name_and_addr_weak",
    "SHARED_KEY_BUT_MISSED",
    "shared_key_was_capped",
]
flag_counts: Counter = Counter()
per_pair_flags: list[set] = []
primary_counts: Counter = Counter()
examples: dict[str, list] = defaultdict(list)
shared_key_counter: Counter = Counter()
capped_shared_counter: Counter = Counter()
other_detail: Counter = Counter()


def latin_family(cls: str) -> bool:
    return cls in ("latin", "latin+marks")


def shared_prod_keys(a: PreprocessedRecord, b: PreprocessedRecord) -> set:
    """Production keys the two records have in common.

    A blocked-out pair that shares a production key is a CONTRADICTION unless the
    shared bucket was truncated by the group cap, because the index files every
    record under every key it produces. That is the signal for cap loss.
    """
    shared: set = set()
    na, nb = k_prod_name(a), k_prod_name(b)
    if na is not None and na == nb:
        shared.add(na)
    shared |= (addr_key_sets(a) & addr_key_sets(b))
    return shared


def primary_category(flags: set, a: PreprocessedRecord, b: PreprocessedRecord) -> str:
    """One bucket per pair, fixed priority so the counts sum to the total."""
    # Highest priority: the keys DID match, so nothing about the key design is
    # at fault -- the record was dropped from an over-full bucket.
    if "SHARED_KEY_BUT_MISSED" in flags:
        return "shared key truncated by cap=1000"
    if "name_missing_either" in flags or "name_no_core_token_either" in flags:
        if "addr_no_key_either" in flags or "addr_missing_either" in flags:
            return "no usable key on either side"
        return "name unusable (address keys exist)"
    if "country_only_blocker" in flags:
        return "country blocks an otherwise matching key"
    if "name_key_differs" in flags and "addr_key_differs" in flags:
        return "both name and address keys differ"
    if "name_key_differs" in flags:
        if "name_1tok_vs_multi" in flags:
            return "1-token vs multi-token name (no padding)"
        if "script_differs" in flags:
            return "name key differs: different script"
        return "name key differs: token/prefix mismatch"
    if "addr_key_differs" in flags:
        if "house_differs" in flags:
            return "address key differs: house number"
        return "address key differs: street prefix"
    return "other"


step("characterise blocked-out pairs")
for eid, mid in blocked_out:
    a = s1_records.get(eid)
    b = matched_records.get(mid)
    if a is None or b is None:
        primary_counts["record unavailable"] += 1
        per_pair_flags.append(set())
        continue
    f: set = set()

    # --- name ---
    if a.name_is_missing or b.name_is_missing:
        f.add("name_missing_either")
    if not a.name_core or not b.name_core:
        f.add("name_no_core_token_either")
    if not name_key_equal(a, b):
        f.add("name_key_differs")
    la, lb = len(a.name_core), len(b.name_core)
    if 1 in (la, lb) and max(la, lb) > 1:
        f.add("name_1tok_vs_multi")
    if la != lb:
        f.add("name_core_count_differs")
    ta, tb = sorted(a.name_core), sorted(b.name_core)
    if ta and tb:
        pa4 = (_p4(ta[0]),) if (la == 1 or lb == 1) else (_p4(ta[0]), _p4(ta[1]))
        pb4 = (_p4(tb[0]),) if (la == 1 or lb == 1) else (_p4(tb[0]), _p4(tb[1]))
        if pa4 != pb4:
            f.add("name_prefix4_differs")
    sa, sb = script_class(a.name), script_class(b.name)
    if a.name and b.name and latin_family(sa) != latin_family(sb):
        f.add("script_differs")
    if sa == "latin" and sb == "latin" and a.name and b.name and \
            bigram_jaccard(a.name, b.name) < 0.4:
        f.add("latin_both_low_similarity")

    # --- address ---
    if a.address_is_missing or b.address_is_missing:
        f.add("addr_missing_either")
    ka, kb = addr_key_sets(a), addr_key_sets(b)
    if not ka or not kb:
        f.add("addr_no_key_either")
    elif not (ka & kb):
        f.add("addr_key_differs")
    if not a.house_number or not b.house_number:
        f.add("house_absent_either")
    elif a.house_number != b.house_number:
        f.add("house_differs")
    wa = set(w[:PREFIX3] for w in a.address_alpha_tokens)
    wb = set(w[:PREFIX3] for w in b.address_alpha_tokens)
    if a.address and b.address and wa and wb and not (wa & wb):
        f.add("street_prefix_differs")

    # --- country ---
    if a.country != b.country:
        f.add("country_disagrees")
    if a.country != b.country and (
            name_prefixes_equal(a, b) or addr_key_equal_nocountry(a, b)):
        f.add("country_only_blocker")

    if "name_key_differs" in f and "addr_key_differs" in f:
        f.add("both_name_and_addr_weak")

    # A blocked-out pair that nevertheless shares a production key can only be a
    # cap-truncation loss. Verify rather than assume, by asking the index.
    shared = shared_prod_keys(a, b)
    if shared:
        f.add("SHARED_KEY_BUT_MISSED")
        for key in shared:
            shared_key_counter[key] += 1
            if key in index._capped:
                f.add("shared_key_was_capped")
                capped_shared_counter[key] += 1

    for flag in f:
        flag_counts[flag] += 1
    per_pair_flags.append(f)
    cat = primary_category(f, a, b)
    primary_counts[cat] += 1
    if cat == "other":
        other_detail["+".join(sorted(f)) or "no-flag"] += 1
    if len(examples[cat]) < 6:
        examples[cat].append((eid, mid, a, b, sorted(f)))

# =====================================================================
# Recovery + volume for the pre-registered alternative keys
# =====================================================================
step("test alternative keys against every blocked-out pair")
recovery: dict[str, int] = {n: 0 for n, _ in ALT_KEYS}
recovery_combo: Counter = Counter()
per_pair_recovered: list[set] = []
for (eid, mid), _ in zip(blocked_out, per_pair_flags):
    a, b = s1_records.get(eid), matched_records.get(mid)
    if a is None or b is None:
        per_pair_recovered.append(set())
        continue
    hits = set()
    for name, fn in ALT_KEYS:
        ka, kb = fn(a), fn(b)
        if ka is None or kb is None:
            continue
        if isinstance(ka, tuple) and ka and isinstance(ka[0], str) and \
                ka[0].startswith("A9"):
            if set(ka) & set(kb):
                hits.add(name)
        elif ka == kb:
            hits.add(name)
    for h in hits:
        recovery[h] += 1
    per_pair_recovered.append(hits)
    recovery_combo["+".join(sorted(hits)) if hits else "NONE"] += 1

any_recovered = sum(1 for h in per_pair_recovered if h)
step(f"blocked-out pairs recoverable by >=1 pre-registered key: {any_recovered:,}")

# volume: per S1, the number of pool rows sharing that S1's key
step("measure alternative-key candidate volume")
volume: dict[str, dict] = {}
for name, _ in ALT_KEYS:
    counts = alt_counts[name]
    per_s1 = []
    for rec in s1_records.values():
        k = dict(ALT_KEYS)[name](rec)
        per_s1.append(counts.get(k, 0) if k is not None else 0)
    zero = sum(1 for v in per_s1 if v == 0)
    volume[name] = {
        "total_hits": sum(per_s1),
        "avg_hits": sum(per_s1) / max(1, len(per_s1)),
        "max_hits": max(per_s1, default=0),
        "s1_with_no_hit": zero,
        "recovery": recovery[name],
    }

# =====================================================================
# REPORT
# =====================================================================
n_missed = len(blocked_out)
print()
print("=" * 78)
print("1. HOW MANY TRUE PAIRS ARE BLOCKED OUT")
print("=" * 78)
print(f"  S1 entities analysed              {len(s1_records):,}")
print(f"  true S1->S2/S3 pairs              {total_true:,}")
print(f"  retrieved by production keys      {found_pairs:,}")
print(f"  BLOCKED-OUT true pairs            {n_missed:,}")
print(f"  blocking recall (pair level)      {recall:.4%}")
print(f"  production candidates             {sum(prod_counts):,} "
      f"(avg {sum(prod_counts) / max(1, len(prod_counts)):.1f}/S1, "
      f"max {max(prod_counts, default=0):,})")
print(f"  index stats                       {stats.indexed_records:,} records, "
      f"{stats.distinct_keys:,} keys, largest group {stats.largest_group:,}, "
      f"capped groups {stats.capped_groups:,}, keyless {stats.keyless_records:,}")

print()
print("=" * 78)
print("2. FAILURE-MODE TABLE (primary category, one bucket per pair)")
print("=" * 78)
print(f"  {'category':<48} {'count':>8} {'pct':>8}")
print("  " + "-" * 66)
for cat, cnt in primary_counts.most_common():
    print(f"  {cat:<48} {cnt:>8,} {cnt / max(1, n_missed):>7.2%}")
print(f"  {'TOTAL':<48} {sum(primary_counts.values()):>8,} "
      f"{sum(primary_counts.values()) / max(1, n_missed):>7.2%}")

print()
print("  Multi-label view (categories OVERLAP - a pair can appear in several):")
print(f"  {'flag':<48} {'count':>8} {'pct':>8}")
print("  " + "-" * 66)
for flag in FLAGS:
    cnt = flag_counts.get(flag, 0)
    if cnt:
        print(f"  {flag:<48} {cnt:>8,} {cnt / max(1, n_missed):>7.2%}")

print()
print("  CAP-TRUNCATION ANALYSIS (pairs whose production keys DID match)")
shared_total = sum(1 for f in per_pair_flags if "SHARED_KEY_BUT_MISSED" in f)
capped_total = sum(1 for f in per_pair_flags if "shared_key_was_capped" in f)
print(f"    blocked-out pairs sharing a production key : {shared_total:,} "
      f"({shared_total / max(1, n_missed):.2%} of misses)")
print(f"    ... of which the shared key was CAPPED     : {capped_total:,} "
      f"({capped_total / max(1, n_missed):.2%} of misses)")
if shared_key_counter:
    print(f"    most frequent shared keys (all should be over-full buckets):")
    for key, cnt in shared_key_counter.most_common(12):
        bucket = index._buckets.get(key)
        size = len(bucket) if bucket else 0
        capped = "CAPPED" if key in index._capped else "not-capped"
        print(f"      {str(key):<62} {cnt:>5,}  bucket={size:<5} {capped}")
if other_detail:
    print("    'other' breakdown:")
    for combo, cnt in other_detail.most_common(8):
        print(f"      {combo:<72} {cnt:>6,}")

print()
print("=" * 78)
print("3. REPRESENTATIVE EXAMPLES PER CATEGORY")
print("=" * 78)
for cat, rows in examples.items():
    if not rows:
        continue
    print(f"\n  --- {cat} ({primary_counts[cat]:,} pairs) ---")
    for eid, mid, a, b, flags in rows[:4]:
        print(f"    {eid} -> {mid}   flags: {','.join(flags)}")
        print(f"      S1  name={a.name!r} core={a.name_core} country={a.country!r}")
        print(f"          addr={a.address!r} house={a.house_number!r} "
              f"alpha={a.address_alpha_tokens[:6]}")
        print(f"      Sx  name={b.name!r} core={b.name_core} country={b.country!r} "
              f"[{script_class(b.name)}]")
        print(f"          addr={b.address!r} house={b.house_number!r} "
              f"alpha={b.address_alpha_tokens[:6]}")
        print(f"      name_jaccard={bigram_jaccard(a.name, b.name):.3f}  "
              f"prod_name_keys: {k_prod_name(a)} vs {k_prod_name(b)}")

print()
print("=" * 78)
print("4./5. ALTERNATIVE KEYS: RECOVERY vs COST")
print("=" * 78)
print("  recall_if_added_alone = blocked-out pairs this key would newly retrieve")
print("  total_hits = pool rows reachable per S1 via the key (COUNT only, so a")
print("              row sharing two of an S1's keys is counted twice -> the")
print("              resulting candidate increase is an UPPER BOUND)")
print()
hdr = (f"  {'key family':<24} {'recovers':>9} {'recall':>8} {'cum.recall':>11} "
       f"{'total_hits':>12} {'avg/S1':>9} {'max/S1':>8} {'no-hit S1':>10}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
base_missed = n_missed
print(f"  {'(production baseline)':<24} {'-':>9} {recall:>7.2%} {'-':>11} "
      f"{sum(prod_counts):>12,} {sum(prod_counts) / max(1, len(s1_records)):>9.1f} "
      f"{max(prod_counts, default=0):>8,} {0:>10}")
ranked = sorted(ALT_KEYS, key=lambda nf: -recovery[nf[0]])
running = found_pairs
for name, _ in ranked:
    v = volume[name]
    running += v["recovery"]
    cum = running / max(1, total_true)
    print(f"  {name:<24} {v['recovery']:>9,} "
          f"{v['recovery'] / max(1, base_missed):>7.2%} {cum:>10.2%} "
          f"{v['total_hits']:>12,} {v['avg_hits']:>9.1f} {v['max_hits']:>8,} "
          f"{v['s1_with_no_hit']:>10,}")

print()
print("  Recovery overlap (which key COMBINATIONS cover a missed pair):")
for combo, cnt in recovery_combo.most_common(12):
    print(f"    {combo:<72} {cnt:>8,} "
          f"({cnt / max(1, n_missed):>6.2%})")

print()
print("=" * 78)
print("6. CHECKS")
print("=" * 78)
print(f"  [{'PASS' if all('test' not in p.name.lower() for p in OPENED) else 'FAIL'}] "
      f"no test data loaded -- opened {', '.join(sorted(p.name for p in OPENED))}")
print(f"  [{'PASS' if all(p.parent == TRAIN.resolve() for p in OPENED) else 'FAIL'}] "
      f"all files under train/")
print(f"  [{'PASS' if len(missing_records) == 0 else 'WARN'}] all ground-truth "
      f"match records recovered from S2/S3 ({len(missing_records)} missing)")
print(f"  [{'PASS' if found_pairs + n_missed == total_true else 'FAIL'}] "
      f"found + blocked-out == total true pairs")
print(f"  [INFO] pairs missing from the pool entirely: {len(missing_records):,}")

print()
print("=" * 78)
print("NOT final competition performance. This measures blocking recall on one")
print("20k S1 subset against the full S2/S3 index. It says nothing about the")
print("matcher, and it is not F0.5. No new key has been added to blocking.py.")
print("=" * 78)
step("done")
