# AGENTS.md

Working rules for AI-assisted work on this competition project. Read this
before changing anything.

## 1. Work incrementally, one phase at a time

Phases run in order: preprocessing, blocking, features, matching, pipeline.
Finish and get sign-off on a phase before starting the next one. A phase is
done when its output is inspectable and its choices are explained.

## 2. Explain the purpose of each change before making it

Before editing or creating anything, state in one or two sentences what the
change does and why it is needed now. No silent edits.

## 3. Do not implement later phases without approval

Never write code for a future phase, and never "just add" the next step as a
convenience. Wait for explicit instruction.

## 4. Help me understand the code, do not just generate it

Explain the reasoning behind each decision: why this comparison, why this
threshold, what breaks in the edge case. Prefer showing the logic over handing
over a finished block. Ask before assuming a data property.

## 5. Prefer simple, explainable approaches first

Start with exact matching, simple string normalisation and rule-based
thresholds. Reach for heavier ML only when the simple version is measurably
insufficient, and justify it with numbers.

## 6. No external data or lookup of any kind

Never use business lookup APIs, external databases, geocoding, web search, or
any internet-based entity enrichment. The competition prohibits external data
lookup. Only the provided S1, S2 and S3 files may be used. If a feature seems
to need outside information, say so instead of fetching it.

## 7. Preserve original entity IDs and the exact output format

Every record keeps its source entity ID through all steps. Do not renumber,
reindex or overwrite IDs. The submission must match the competition's required
format exactly, including column names, ordering and the candidate pair output.
Verify the format before generating any final file.

## 8. Never commit datasets or generated submissions

`data/` and `output/` stay git-ignored (see `.gitignore`). Only code,
notebooks, `requirements.txt` and documentation are tracked. Also never commit
secrets, API keys or `.env` files.

## 9. Do not touch competition files

Do not delete, move, rename or modify any provided dataset or competition file.
If a file looks wrong or malformed, report it and let the user decide.
