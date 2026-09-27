"""First matching model: an interpretable Logistic Regression baseline.

This module owns three things and nothing else:

* turning blocked candidate pairs into **labels**, using the training ground
  truth;
* turning those labelled pairs into a feature matrix via :mod:`src.features`;
* fitting a Logistic Regression, scoring candidates, and reporting
  precision / recall / F0.5 at a range of thresholds.

It deliberately does **not** contain the end-to-end pipeline, does not write
submission files, and does not claim a competition-ready threshold. The
threshold search below is a validation-set diagnostic for this baseline only.

Label rule
----------
A candidate pair ``(S1-x, S2-y)`` is a positive iff the candidate's
``entity_id`` appears in the comma-separated ``matched_entity_ids`` list of the
ground-truth row for ``S1-x``. Everything else that blocking retrieved is a
negative. Negatives are *never* invented outside the blocking candidate set:
a pair the blocking keys did not retrieve is not evidence of a non-match, it is
simply not a candidate, and treating un-retrieved pairs as negatives would
measure blocking rather than the model.

Leakage
-------
The train/validation split is made at the **S1 entity** level, never per pair.
All candidates belonging to one S1 entity land on the same side. Splitting
per pair would put the same business on both sides, letting the model memorise
a name and address it will then meet again in validation, which inflates the
score for reasons that have nothing to do with generalisation.

The split is a salted BLAKE2b hash of the S1 id rather than a shuffled list, so
it is reproducible across runs, machines and Python versions, and it does not
depend on the iteration order of a set.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from .blocking import CandidatePair, RecordRef
from .features import FEATURE_NAMES, featurize
from .preprocessing import PreprocessedRecord, preprocess_record

__all__ = [
    "DEFAULT_THRESHOLDS",
    "BETA",
    "ThresholdReport",
    "LabelledPair",
    "FitSummary",
    "source_of",
    "load_ground_truth",
    "iter_records",
    "load_records",
    "label_candidates",
    "feature_matrix",
    "split_by_reference",
    "precision_recall_fbeta",
    "evaluate_thresholds",
    "best_validation_threshold",
    "LogisticMatcher",
]


#: Beta for the competition's F-measure. 0.5 < 1.0, so recall is weighted a
#: quarter of precision's weight: false merges are the expensive mistake.
BETA = 0.5

#: The threshold grid to report. Deliberately a fixed literal tuple so a run is
#: reproducible; it is a reporting grid, not a tuned hyper-parameter set.
DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
)


# --------------------------------------------------------------------------
# Input loading
# --------------------------------------------------------------------------

def source_of(entity_id: str) -> str:
    """``"S2-681193310"`` -> ``"S2"``.

    The competition ids already carry their source prefix, so the source is
    recoverable from the id. It is still kept as a separate field on
    :class:`~src.blocking.RecordRef` rather than being re-derived everywhere,
    because two sources may legitimately reuse an id.
    """
    prefix, separator, _ = entity_id.partition("-")
    return prefix if separator else ""


def load_ground_truth(
    path,
    *,
    encoding: str = "utf-8",
) -> dict[str, frozenset[str]]:
    """Read ``train_ground_truth.tsv`` into ``{s1_entity_id: {matched_ids}}``.

    The file is a two-column TSV: ``source1_entity_id`` and a
    comma-separated ``matched_entity_ids`` list. Ids are kept exactly as they
    appear, because that is the form the candidate ids also take, so label
    lookup is a plain set membership test with no normalisation involved.

    The whole file is returned as a dict, which is the one structure here that
    scales with the dataset rather than with the candidate set. If that becomes
    a problem, the only thing to change is this function.
    """
    truth: dict[str, frozenset[str]] = {}
    with open(path, "r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return truth
        # Resolve column positions by name so a reordered file still loads.
        try:
            id_column = header.index("source1_entity_id")
            match_column = header.index("matched_entity_ids")
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"ground truth must have source1_entity_id and matched_entity_ids "
                f"columns, found {header!r}"
            ) from exc

        for row in reader:
            if not row or len(row) <= max(id_column, match_column):
                continue
            entity_id = row[id_column].strip()
            if not entity_id:
                continue
            raw = row[match_column].strip()
            matched = frozenset(part.strip() for part in raw.split(",") if part.strip())
            truth[entity_id] = matched
    return truth


def iter_records(
    path,
    *,
    encoding: str = "utf-8",
) -> Iterator[PreprocessedRecord]:
    """Stream a source TSV as :class:`PreprocessedRecord` objects.

    All normalisation is delegated to :func:`src.preprocessing.preprocess_record`,
    so this function never normalises anything itself. Rows are yielded one at a
    time: the files are far too large to materialise, and callers that need a
    lookup can build a dict over only the ids they were asked about.

    The source is not a parameter because the ids already carry their prefix;
    use :func:`source_of` on ``record.entity_id`` when it is needed.

    Short rows are padded rather than skipped, because a truncated row is a data
    problem the caller should see as a record with missing fields, not as a
    silently absent record.
    """
    with open(path, "r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return
        try:
            id_col = header.index("entity_id")
            name_col = header.index("business_name")
            address_col = header.index("business_address")
            country_col = header.index("country")
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"source file must have entity_id, business_name, "
                f"business_address and country columns, found {header!r}"
            ) from exc

        for row in reader:
            if not row:
                continue

            def field(index: int) -> str:
                return row[index] if index < len(row) else ""

            entity_id = field(id_col).strip()
            if not entity_id:
                continue
            yield preprocess_record(
                entity_id=entity_id,
                business_name=field(name_col),
                business_address=field(address_col),
                country=field(country_col),
            )


def load_records(
    path,
    *,
    encoding: str = "utf-8",
) -> dict[str, PreprocessedRecord]:
    """Eagerly read a whole source file into ``{entity_id: record}``.

    Only appropriate for a slice or a candidate pool. For a full source file use
    :func:`iter_records` and keep just the ids that are actually needed.
    """
    records: dict[str, PreprocessedRecord] = {}
    for record in iter_records(path, encoding=encoding):
        records[record.entity_id] = record
    return records


# --------------------------------------------------------------------------
# Labelling
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelledPair:
    """One candidate pair with its 0/1 label attached."""

    pair: CandidatePair
    label: int


def label_candidates(
    pairs: Iterable[CandidatePair],
    ground_truth: dict[str, frozenset[str]],
) -> list[LabelledPair]:
    """Attach ground-truth labels to candidate pairs.

    A pair is positive when the candidate id is listed for that reference id.
    A pair whose reference id is absent from the ground truth is labelled
    negative, which is the right default: the file enumerates the matches, so
    an absent row means no match is recorded.
    """
    labelled: list[LabelledPair] = []
    for pair in pairs:
        matches = ground_truth.get(pair.reference.entity_id, frozenset())
        labelled.append(LabelledPair(pair, 1 if pair.candidate.entity_id in matches else 0))
    return labelled


def feature_matrix(
    labelled: Iterable[LabelledPair],
    records: dict[str, PreprocessedRecord],
) -> tuple[list[tuple[float, ...]], list[int], list[str]]:
    """Build ``(X, y, reference_ids)`` for labelled pairs.

    ``X`` rows are in :data:`src.features.FEATURE_NAMES` order, produced by
    :func:`src.features.featurize`, so the feature contract lives in exactly one
    place. ``reference_ids[i]`` is the S1 entity of row ``i``, which is what
    lets the caller group rows back to an entity for the S1-level split.

    Both records of every pair must be present in ``records``. A missing id is a
    hard error rather than a skipped row: skipping would quietly change the
    positive and negative counts, and substituting a blank record would score
    the pair against fabricated data and report a confident wrong answer.
    """
    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    references: list[str] = []
    for item in labelled:
        reference_id = item.pair.reference.entity_id
        candidate_id = item.pair.candidate.entity_id
        for role, wanted in (("reference", reference_id), ("candidate", candidate_id)):
            if wanted not in records:
                raise KeyError(
                    f"no record supplied for {role} {wanted!r}; the record pool and "
                    f"the candidate pairs disagree"
                )
        rows.append(featurize(records[reference_id], records[candidate_id]))
        labels.append(item.label)
        references.append(reference_id)
    return rows, labels, references


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def _split_bucket(reference_id: str, salt: str) -> int:
    """Stable bucket in ``[0, 10000)`` for an S1 id.

    A salted BLAKE2b digest, not :func:`hash` (salted per process) and not a
    seeded shuffle (whose result depends on the sequence of everything shuffled
    before it). The same id always lands in the same bucket for a given salt.
    """
    digest = hashlib.blake2b(
        f"{salt}\x00{reference_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % 10_000


def split_by_reference(
    labelled: Sequence[LabelledPair],
    *,
    validation_fraction: float = 0.2,
    salt: str = "v1",
) -> tuple[list[LabelledPair], list[LabelledPair]]:
    """Split labelled pairs into train and validation at the S1-entity level.

    Every candidate of one S1 entity goes to the same side, so no business
    appears in both. The assignment is a deterministic function of the S1 id and
    ``salt``, so the split is reproducible and does not shift when the candidate
    set changes size.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    cut = validation_fraction * 10_000

    train: list[LabelledPair] = []
    validation: list[LabelledPair] = []
    for item in labelled:
        if _split_bucket(item.pair.reference.entity_id, salt) < cut:
            validation.append(item)
        else:
            train.append(item)
    return train, validation


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ThresholdReport:
    """Precision / recall / F-measure at one threshold."""

    threshold: float
    predicted_positives: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f_beta: float


def precision_recall_fbeta(
    y_true: Sequence[int],
    y_pred: Sequence[bool],
    *,
    beta: float = BETA,
) -> tuple[float, float, float]:
    """Return ``(precision, recall, F_beta)``.

    F_beta = ``(1 + b^2) * P * R / (b^2 * P + R)``. With ``beta=0.5`` that is
    ``1.25 * P * R / (0.25 * P + R)``, the competition's measure. An undefined
    ratio (no predicted positives, or no actual positives) is reported as
    ``0.0`` rather than raising, so a threshold that predicts nothing shows up
    as a score of zero instead of crashing the sweep.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    true_positives = false_positives = false_negatives = 0
    for actual, predicted in zip(y_true, y_pred):
        if predicted:
            if actual:
                true_positives += 1
            else:
                false_positives += 1
        elif actual:
            false_negatives += 1

    predicted_positives = true_positives + false_positives
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    actual_positives = true_positives + false_negatives
    recall = true_positives / actual_positives if actual_positives else 0.0

    if precision == 0.0 or recall == 0.0:
        return precision, recall, 0.0
    beta_squared = beta * beta
    f_beta = ((1.0 + beta_squared) * precision * recall
              / (beta_squared * precision + recall))
    return precision, recall, f_beta


def evaluate_thresholds(
    y_true: Sequence[int],
    scores: Sequence[float],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    *,
    beta: float = BETA,
) -> list[ThresholdReport]:
    """Sweep thresholds and report the full confusion matrix at each one.

    A candidate is predicted positive when ``score >= threshold``. The sweep is
    a handful of linear passes rather than anything clever: it is simple, and at
    these sizes simple is fast enough to be worth more than optimal.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must be the same length")
    reports: list[ThresholdReport] = []
    for threshold in thresholds:
        predicted = [score >= threshold for score in scores]
        precision, recall, f_beta = precision_recall_fbeta(y_true, predicted, beta=beta)
        true_positives = sum(1 for a, p in zip(y_true, predicted) if a and p)
        predicted_positives = sum(1 for p in predicted if p)
        reports.append(ThresholdReport(
            threshold=threshold,
            predicted_positives=predicted_positives,
            true_positives=true_positives,
            false_positives=predicted_positives - true_positives,
            false_negatives=sum(y_true) - true_positives,
            precision=precision,
            recall=recall,
            f_beta=f_beta,
        ))
    return reports


def best_validation_threshold(
    reports: Sequence[ThresholdReport],
) -> ThresholdReport:
    """Highest F-measure on the validation set. Ties break to the higher threshold.

    This is a **validation-only diagnostic for this baseline**, not a
    competition threshold. The reported figure is optimistic by construction,
    because the same data chose the threshold and will be used to judge it; a
    real estimate needs a held-out set that took no part in the choice.
    """
    if not reports:
        raise ValueError("no threshold reports to choose from")
    return max(reports, key=lambda report: (report.f_beta, report.threshold))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FitSummary:
    """What the fit actually did, so a run can be checked rather than trusted."""

    n_samples: int
    n_positive: int
    n_negative: int
    positive_rate: float
    class_weight: object
    n_iter: int
    converged: bool
    n_features: int


class LogisticMatcher:
    """Logistic Regression over the :mod:`src.features` vector.

    Class weighting
    ---------------
    ``class_weight`` defaults to ``None``, i.e. no reweighting. The candidate
    set is built to be generous, so positives are a minority, and the obvious
    worry is that an unweighted fit minimises overall error and settles on a
    near-constant low score that looks good on accuracy while being useless.
    The measurement says otherwise here.

    On a bounded real sample (20,000 S1 entities against a 120,000-record
    candidate pool, S1-level split), candidates were 9.1% positive. That is
    mild enough that ``"balanced"`` upweights positives roughly ten-fold, which
    drags the decision boundary toward predicting more positives -- the wrong
    direction for a metric that weights precision four times as heavily as
    recall. Measured on the same validation split:

    ==================  ==============  =========
    weighting            best F0.5       threshold
    ==================  ==============  =========
    ``"balanced"``       0.9677          0.9
    ``None``             0.9712          0.6
    ==================  ==============  =========

    The margin is small and the pool was a head slice, so this is a lean and
    not a settled result. ``None`` is still the default because it is the
    simpler model, and because the one measurement available does not support
    the extra machinery. Pass ``class_weight="balanced"`` to compare.

    Features are used unscaled. Every similarity column already lies in
    ``[0, 1]`` and the two count columns are small integers, so a scaler would
    change the regularisation geometry without changing the information.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        class_weight: object = None,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 0,
    ) -> None:
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self._model = None
        self.summary: FitSummary | None = None

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
    ) -> "LogisticMatcher":
        """Fit on a feature matrix. Deterministic for a fixed solver and input order."""
        from sklearn.linear_model import LogisticRegression

        n_samples = len(X)
        n_positive = sum(1 for label in y if label)
        if n_samples != len(y):
            raise ValueError("X and y must be the same length")
        if n_positive == 0 or n_positive == n_samples:
            raise ValueError(
                f"need both classes to fit, got {n_positive} positive of {n_samples}"
            )

        self._model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        n_iter = int(getattr(self._model, "n_iter_", [self.max_iter])[0])
        self.summary = FitSummary(
            n_samples=n_samples,
            n_positive=n_positive,
            n_negative=n_samples - n_positive,
            positive_rate=n_positive / n_samples,
            class_weight=self.class_weight,
            n_iter=n_iter,
            converged=n_iter < self.max_iter,
            n_features=len(FEATURE_NAMES),
        )
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Probability that each row is a match, in input order."""
        if self._model is None:
            raise RuntimeError("call fit() before predicting")
        return [float(p) for p in self._model.predict_proba(X)[:, 1]]

    def predict(
        self,
        X: Sequence[Sequence[float]],
        threshold: float,
    ) -> list[bool]:
        """Apply a configurable threshold to the scores."""
        return [score >= threshold for score in self.predict_proba(X)]

    def coefficients(self) -> dict[str, float]:
        """Feature name -> fitted coefficient, for inspection.

        Kept because the point of a first baseline is that it can be argued
        with: a sign that contradicts the feature it belongs to is a bug signal,
        not a curiosity.
        """
        if self._model is None:
            raise RuntimeError("call fit() before asking for coefficients")
        return dict(zip(FEATURE_NAMES, (float(c) for c in self._model.coef_[0])))

    def report(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    ) -> list[ThresholdReport]:
        """Score a matrix and sweep thresholds over it in one call."""
        return evaluate_thresholds(y, self.predict_proba(X), thresholds)
