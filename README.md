# Business Entity Resolution

Linking business records that appear in three independent sources (S1, S2, S3),
so that records describing the same real-world business are grouped together.

## Goal

S1 is the reference source. S2 and S3 are combined internally into a single
candidate dataset while their original entity IDs stay intact. For every S1
record we must decide which S2/S3 record, if any, is the same real-world
business. The final submission contains the required match results together
with the candidate pairs that were generated.

## Pipeline

1. **Preprocessing** - load the three sources, normalise text, parse addresses
   and phone numbers, keep the original IDs untouched.
2. **Blocking** - cut the S1 x (S2 + S3) comparison space down to a manageable
   set of candidate pairs.
3. **Features** - compute similarity features (exact, fuzzy, token overlap,
   address and phone agreement) for each candidate pair.
4. **Matching** - score pairs and convert scores into final entity matches.
5. **Pipeline** - run the steps in order and write the submission file.

Matching relies only on the information contained in the provided S1, S2 and S3
files. No external business databases, lookups or enrichment are used.

## Folder structure

```
business-entity-resolution/
├── src/                  # pipeline package
│   ├── preprocessing.py  # loading and cleaning S1, S2, S3
│   ├── blocking.py       # candidate pair generation
│   ├── features.py       # pairwise similarity features
│   ├── matching.py       # scoring and match decisions
│   └── pipeline.py       # end-to-end run
├── notebooks/            # exploratory analysis
├── data/                 # competition datasets (git-ignored)
├── output/               # generated files and submission (git-ignored)
├── requirements.txt      # project dependencies
├── README.md
└── .gitignore
```

## Status

Work in progress. The folder layout exists and the modules are placeholders;
pipeline logic is implemented stage by stage.
