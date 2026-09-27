# Business Entity Resolution — Project Knowledge

## 1. What is the project?

This project solves **Business Entity Resolution / Entity Matching**.

We have business records from three independent sources:

* S1 — reference source
* S2
* S3

The same real-world business can appear differently across sources because of:

* spelling variations
* abbreviations
* punctuation/formatting
* different address formats
* missing information
* transliteration

For every S1 business, we need to find the corresponding S2/S3 records, if any.

A business can have **zero, one, or multiple matches**.

This is **not missing-value filling**. It is an entity matching problem.

---

## 2. Important data

Main fields:

* `entity_id`
* `business_name`
* `business_address`
* `country`

The original `entity_id` must always be preserved.

S2 and S3 can be merged internally into one candidate dataset because their IDs identify the source (`S2-...` / `S3-...`).

Ground truth tells us which S2/S3 records match each S1 record.

---

## 3. Overall pipeline

```text
Raw S1/S2/S3
     ↓
Preprocessing
     ↓
Blocking / Candidate Generation
     ↓
Similarity Features
     ↓
Matching Model
     ↓
Final Results
```

Each stage has a different job.

---

## 4. Preprocessing

Purpose: make messy text easier to compare.

Typical operations may include:

* lowercase
* whitespace normalization
* punctuation normalization
* handling common formatting variations

Example:

```text
"McDonald's India Pvt. Ltd."
        ↓
"mcdonalds india pvt ltd"
```

Normalization is only for comparison. Original IDs and required output information must remain intact.

---

## 5. Blocking / Candidate Generation

Comparing every S1 record with every S2/S3 record can be extremely expensive.

Example:

```text
10,000 S1 × 100,000 candidates
= 1 billion comparisons
```

Blocking reduces this to a smaller set of **candidate pairs**.

Think of it like a HashMap/index:

```text
blocking key → possible records
```

Blocking answers:

> "Which records are worth comparing?"

It does **not** make the final match decision.

### Most important point

Blocking creates the **recall ceiling**.

If the true match is removed during blocking, the matching model can never recover it.

Therefore:

```text
Strict blocking → fewer candidates but risk missing matches
Loose blocking  → more candidates but better chance of retaining matches
```

---

## 6. Similarity Features

For each candidate pair, we calculate features describing how similar the two records are.

Possible features:

### Name

* exact match
* Levenshtein similarity
* Jaccard/token overlap
* TF-IDF cosine similarity

### Address

* normalized similarity
* token overlap
* Jaccard
* TF-IDF cosine similarity

### Country

Direct agreement/disagreement can be used as a feature.

### Key concepts

**Levenshtein:** measures character edits needed to transform one string into another.

**Jaccard:**

```text
|A ∩ B| / |A ∪ B|
```

Measures overlap between token sets.

**TF-IDF:** converts text into numerical vectors based on word importance.

**Cosine similarity:** measures similarity between those vectors.

---

## 7. Matching Model

After blocking, a candidate pair might look like:

```text
S1 ID
S2/S3 ID
name similarity
address similarity
country agreement
other features
```

The matching model uses these features to determine whether the pair represents the same real-world business.

Training data can provide:

* **positive pairs** → known matches
* **negative pairs** → non-matches

The model produces a score/probability, and a threshold or decision rule can determine the final match.

The exact model will be selected after inspecting the actual dataset.

---

## 8. Blocking vs Matching

This distinction should always be clear:

**Blocking = candidate generation**

> Which records should we compare?

**Matching = final decision**

> Which candidate records are actually the same entity?

A better matching model cannot recover a true match that blocking removed.

---

## 9. Evaluation

The competition uses **macro F0.5**, which gives more importance to precision than recall.

### Precision

Of predicted matches, how many are correct?

```text
correct predictions / all predictions
```

### Recall

Of all true matches, how many did we find?

```text
correct predictions / all true matches
```

Because precision is emphasized, incorrect merges are particularly costly.

A **false merge** means two different real businesses were incorrectly treated as the same business.

Singleton cases also matter.

---

## 10. Final output

The project produces:

### `matching_results.tsv`

Maps each test S1 entity to its matched S2/S3 entity IDs.

Example:

```text
S1-001    S2-019,S3-203
S1-002    S2-155
S1-003
```

An empty list means no match.

### `candidate_pairs.tsv`

Contains the candidate pairs actually passed to the matching stage.

Important:

> Every final match must exist in the candidate set.

---

## 11. Competition constraint

Only the provided data should be used.

Do **not** use:

* commercial ER APIs
* government business databases
* geocoding APIs
* internet business lookup
* external business enrichment

The pipeline must rely on the supplied S1/S2/S3 data.

---

## 12. Interview explanation

### 30-second version

> "This is a business entity resolution project where I match records from S2 and S3 to a reference source S1. I first preprocess and normalize the records, then use blocking to reduce the huge number of possible comparisons to a manageable candidate set. For those candidates, I calculate similarity features such as business-name and address similarity and country agreement. A matching model then scores the candidate pairs and determines the final matches. Finally, I generate the required matching results and candidate-pair files. Since the evaluation uses precision-heavy F0.5, avoiding incorrect business merges is especially important."

---

## 13. Questions I should be able to answer

Before saying I fully understand the project, I should be able to explain:

1. What is Entity Resolution?
2. What are S1, S2 and S3?
3. Why is S1 the reference source?
4. Why can one S1 have multiple matches?
5. What is preprocessing?
6. What is blocking?
7. Why is blocking necessary?
8. Why does blocking determine the recall ceiling?
9. What is a candidate pair?
10. What are positive and negative pairs?
11. What do Levenshtein, Jaccard, TF-IDF and cosine similarity measure?
12. What does the matching model predict?
13. What is a matching threshold?
14. What are precision and recall?
15. Why is F0.5 precision-heavy?
16. What is a false merge?
17. What are `matching_results.tsv` and `candidate_pairs.tsv`?
18. Why can't external business lookup be used?
19. What happens if blocking misses the true match?
20. How does the complete pipeline go from raw data to final output?

If I can explain these in my own words, I understand the core of the project rather than simply knowing that the code was generated.

---

## 14. Amazon ML Challenge 2026 Guidelines

* Challenge window: 25 Sep 2026, 12:00 AM IST to 27 Sep 2026, 11:59 PM IST.
* Dataset is approximately 1 GB compressed and must NOT be committed to GitHub.
* Maximum 5 submissions per day for 3 days.
* Maintain version history of all submissions.
* Final/shortlisted solution may require:
  * 1–2 page ML approach document
  * source code for experiments, training and inference
  * methodology
  * candidate generation / blocking strategy
  * model architecture
  * feature engineering
  * experiments and conclusion
* No cheating, plagiarism, multiple-ID attempts, or other unfair practices.
* External business-data lookup/enrichment is prohibited according to the problem statement.
* The solution should be reproducible from the tracked source code.
