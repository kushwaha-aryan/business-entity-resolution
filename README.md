# Business Entity Resolution

Competition project for linking business records that appear in three
independent sources (S1, S2, S3).

## Goal

S1 is the reference source. S2 and S3 are combined internally into a single
candidate dataset while their original entity IDs stay intact. For every S1
record we must decide which S2/S3 record, if any, is the same real-world
business. The final submission contains the required match results together
with the candidate pairs that were generated.

## How it works (planned, one phase at a time)

1. **Preprocessing** - load the three sources, normalise text, parse addresses
   and phone numbers, keep the original IDs untouched.
2. **Blocking** - cut the S1 x (S2 + S3) comparison space down to a manageable
   set of candidate pairs.
3. **Features** - compute similarity features (exact, fuzzy, token overlap,
   address and phone agreement) for each candidate pair.
4. **Matching** - score pairs and convert scores into final entity matches.
5. **Pipeline** - run the steps in order and write the submission file.

Only the folder layout exists so far. The modules are empty placeholders and
no pipeline logic has been written yet.

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
├── requirements.txt      # dependencies, added per phase
├── README.md
├── AGENTS.md             # working rules for AI-assisted work
└── .gitignore
```

## Rules

See [AGENTS.md](AGENTS.md). The most important ones: work one phase at a
time with approval, no external business lookup or enrichment of any kind, and
keep original entity IDs plus the exact competition output format.
