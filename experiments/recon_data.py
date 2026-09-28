"""Recon of the competition data: sizes, countries, singleton rate, match multiplicity."""
import collections
import csv
import os
import sys

csv.field_size_limit(10 ** 7)
BASE = os.path.join("experiments", "work_match_a2", "dataset", "student_resource", "dataset")


def source_profile(name):
    path = os.path.join(BASE, name)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        idx = {c.strip(): i for i, c in enumerate(header)}
        print(f"\n=== {name} ===")
        print("header:", header)
        n = 0
        countries = collections.Counter()
        n_name_empty = 0
        n_addr_empty = 0
        name_len = 0
        for row in reader:
            if not row:
                continue
            n += 1
            c = row[idx["country"]].strip() if idx["country"] < len(row) else ""
            countries[c] += 1
            nm = row[idx["business_name"]].strip() if idx["business_name"] < len(row) else ""
            ad = row[idx["business_address"]].strip() if idx["business_address"] < len(row) else ""
            if not nm:
                n_name_empty += 1
            if not ad:
                n_addr_empty += 1
            name_len += len(nm)
    print(f"rows: {n:,}")
    print(f"countries: {dict(countries.most_common(10))}")
    print(f"empty business_name: {n_name_empty:,} ({n_name_empty/max(1,n):.3%})")
    print(f"empty business_address: {n_addr_empty:,} ({n_addr_empty/max(1,n):.3%})")
    print(f"mean name length: {name_len/max(1,n):.1f}")
    return n


def truth_profile():
    path = os.path.join(BASE, "train", "train_ground_truth.tsv")
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        i_id, i_m = header.index("source1_entity_id"), header.index("matched_entity_ids")
        n = 0
        singletons = 0
        total_pairs = 0
        mult = collections.Counter()
        src2 = src3 = 0
        for row in reader:
            if not row or len(row) <= max(i_id, i_m):
                continue
            sid = row[i_id].strip()
            if not sid:
                continue
            n += 1
            ids = [p.strip() for p in row[i_m].split(",") if p.strip()]
            if not ids:
                singletons += 1
            k = len(ids)
            mult[min(k, 10)] += 1
            total_pairs += k
            for x in ids:
                if x.startswith("S2-"):
                    src2 += 1
                elif x.startswith("S3-"):
                    src3 += 1
    print("\n=== train_ground_truth.tsv ===")
    print(f"S1 rows in truth: {n:,}")
    print(f"SINGLETONS (empty match list): {singletons:,} ({singletons/max(1,n):.3%})")
    print(f"total true pairs: {total_pairs:,}")
    print(f"  pointing at S2: {src2:,}   at S3: {src3:,}")
    print("matches-per-entity histogram (10 == 10+):",
          {k: v for k, v in sorted(mult.items())})
    print(f"mean matches per entity: {total_pairs/max(1,n):.3f}")
    return n, singletons, total_pairs


if __name__ == "__main__":
    n1 = source_profile(os.path.join("train", "train_source1.tsv"))
    source_profile(os.path.join("train", "train_source2.tsv"))
    source_profile(os.path.join("train", "train_source3.tsv"))
    source_profile(os.path.join("test", "test_source1.tsv"))
    source_profile(os.path.join("test", "test_source2.tsv"))
    source_profile(os.path.join("test", "test_source3.tsv"))
    truth_profile()
