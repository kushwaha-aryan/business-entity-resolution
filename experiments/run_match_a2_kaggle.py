#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Business entity resolution - 20k S1 matching A/B, as a single Kaggle cell.

WHAT THIS IS
============
One self-contained Python file. Paste the whole thing into ONE cell of a Kaggle
Notebook cell and run it. It does not import the project, it does not need a
checkout, and it does not need anything installed beyond what a Kaggle Notebook
already has (numpy, scikit-learn, pandas-free).

WHAT IT DOES
============
Two matching arms are compared over the same 20,000 S1 entities:

    Arm A  production blocking
            name_2tok + addr_ht0 + addr_ht1, per-key cap 1000
    Arm B  production blocking + A2
            A2 = original-order first two core name tokens, cut to 4 chars,
            country included, per-key cap 1000

Each arm becomes a labelled candidate matrix, the existing 24 features are
computed, a Logistic Regression is fitted, and F0.5 is swept over thresholds.
The point of the A/B is blocking recall: how many true pairs Arm B finds that
Arm A misses. No external data, no embeddings, no LLM, no network, no API.

EMBEDDED SOURCES
================
``src/preprocessing.py``, ``src/blocking.py``, ``src/features.py`` and
``src/matching.py`` are embedded verbatim below and written to
``/kaggle/working/src`` at run time, so the experiment is provably the same
code that produced the reference numbers. Their SHA-256 digests are compared on
every run and the outcome is printed and recorded in the report, but a mismatch
does **not** stop the experiment: the embedded copies are the versions this
experiment is defined to run, so they are used as-is and the report says so.
No command-line argument is needed for this - the cell runs as pasted.

KAGGLE BEHAVIOUR
================
* The uploaded ZIP is found by searching ``/kaggle/input``.
* It is extracted to ``/kaggle/working/dataset``.
* Every stage writes an atomic checkpoint, so re-running the cell resumes.
* BLAS is capped to one thread *before* numpy is imported, and the CPU
  affinity is narrowed, because a multi-threaded BLAS is the usual reason a
  process shows 100% CPU. Kaggle does not expose a CPU temperature sensor, so
  there is no thermal guard here; the duty cycle below is what keeps the
  process courteous instead.
* ``--stage`` runs one stage; the default runs all of them in order.

USAGE
=====
    # whole run, resuming whatever is already checkpointed
    # (nothing else needed - defaults are the Kaggle paths)

    # or drive it from a cell by editing ARGS below
    ARGS = ["--stage", "candidates"]
"""

# ``from __future__`` must be the first statement in the file, so it goes here
# rather than in the import block below, which the BLAS cap has to precede.
from __future__ import annotations

# ---------------------------------------------------------------------------
# BLAS thread caps. This block must run BEFORE numpy is imported, because the
# OpenMP/BLAS runtimes read these variables once, at load time. Setting them
# later has no effect on an already-initialised native library, which is why
# --threads cannot be made to work at runtime without threadpoolctl.
# ---------------------------------------------------------------------------
import os as _os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ[_var] = "1"

BLAS_THREADS = int(_os.environ["OMP_NUM_THREADS"])

# ===========================================================================
# IMPORTS
# ===========================================================================
import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# ===========================================================================
# CONFIGURATION - the experiment, exactly as specified
# ===========================================================================
#: The experiment is defined on a 20,000-entity S1 slice. The slice is not a
#: random sample: it is a salted hash of the entity id, so it is identical on
#: every machine, every run and every re-run, and it cannot drift when the file
#: order changes.
N_S1 = 20_000
S1_SALT = b"exp20k-s1"
S1_MOD = 110

#: The train/validation split is made at the S1 *entity* level, never per pair.
#: All candidates of one S1 entity land on the same side, otherwise a business
#: appears in both and the model is graded on a name it has already memorised.
SPLIT_FRACTION = 0.2
SPLIT_SALT = "exp20k-split-v1"

#: Per-key candidate cap. A key that explodes into hundreds of thousands of
#: pairs is a sign the key is not discriminating, and letting it through would
#: swamp the candidate pool with low-quality pairs.
CAP = 1000

#: A2 uses the first two core name tokens in *original order* (not sorted),
#: truncated to this many characters, with country kept in the key. Truncation
#: is what lets an inflection or a suffix ("restaurants" vs "restaurant")
#: still collide.
PREFIX4 = 4

#: The competition's F-measure beta. Declared here as the *expected* value so
#: the embedded module can be checked against it at load time; the runtime
#: ``BETA`` is taken from the module itself once it is imported.
BETA_EXPECTED = 0.5
BETA = BETA_EXPECTED

#: Reference counts for the 20,000-entity subset, computed with these exact
#: settings. They are *sanity checks only*. A mismatch prints a warning and the
#: run continues, because a changed count is far more likely to be a mid-run
#: interruption than a change in the data. The report prints observed counts and
#: never substitutes these.
EXPECT_PROD_CANDIDATES = 6_373_030
EXPECT_UNION_CANDIDATES = 7_580_413
EXPECT_A2_ONLY_CANDIDATES = 1_207_383
EXPECT_PROD_TRUE = 55_325
EXPECT_A2_ONLY_TRUE = 2_995
EXPECT_TOTAL_TRUE = 69_301

STAGES = ("dataset", "subset", "candidates", "features", "model", "report", "all")

S3_HINT = re.compile(r"student_resource", re.IGNORECASE)
ZIP_NAME_HINT = re.compile(r"\.zip$", re.IGNORECASE)
SKIP_NAMES = {"__pycache__"}

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

#: The dataset's own filenames inside the archive. Only train_* is read.
#: test_* is deliberately never opened: it is the blind set, and reading it
#: would make the reported numbers meaningless.
SOURCES = ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv")
TRUTH_FILE = "train_ground_truth.tsv"
TRAIN_FILES = SOURCES + (TRUTH_FILE,)

S1_RECORDS = "s1_records.pkl"
TRUTH_CACHE = "truth.pkl"
#: Candidate pairs, one structured array per arm (see PAIR_DTYPE below).
PROD_PAIRS = "prod_pairs.npy"
UNION_PAIRS = "union_pairs.npy"
#: Feature matrices, one per arm. These files are the Stage 4 checkpoint: they
#: are memory-mapped and written in place, so a re-run reopens them and carries
#: on rather than recomputing rows that are already there.
FEATURES_PROD = "features_prod.npy"
FEATURES_UNION = "features_union.npy"
FEATURE_STATE = "features_state.json"
MODEL_RESULTS = "model_results.json"
REPORT_TXT = "match_a2_report.txt"
REPORT_JSON = "match_a2_report.json"
DATASET_MARKER = "dataset_ready.json"
STAGE3_META = "candidates_meta.json"

T0 = time.perf_counter()
CHECKS: list[tuple[str, bool, str]] = []

WORK: Path = KAGGLE_WORKING
CKPT: Path = KAGGLE_WORKING / "checkpoints" / "match_a2"
DATA_ROOT: Path = KAGGLE_WORKING / "dataset"
SRC_ROOT: Path = KAGGLE_WORKING / "src"
OUT_DIR: Path = CKPT
TRAIN_DIR: Path | None = None

PACE = None

# ===========================================================================
# EMBEDDED PROJECT SOURCES - verbatim
# ===========================================================================
_SRC_PREPROCESSING = r'''
"""Deterministic text normalisation for business entity resolution.

This module has no file I/O and no third-party imports. It only turns the raw
``business_name`` / ``business_address`` / ``country`` fields of one source row
into comparison-ready forms. Reading the TSV files happens elsewhere.

Three rules drive every decision here:

1. **Unicode is preserved.** S2/S3 contain native-script names while S1 is
   effectively Latin, so normalisation must never delete a whole script. Only
   Latin accents are folded, and only when they sit directly on a Latin letter,
   so Devanagari / Tamil / Arabic vowel signs survive untouched.
2. **Country is an open set of strings.** Nothing in this module knows that
   "US" or "India" exist, so an unseen country such as France needs no code
   change. See the challenge statement: "Treat country as an open set of string
   labels: do not hard-code, filter, or one-hot your pipeline."
3. **Every function is pure and deterministic.** No randomness, no reliance on
   set or dict iteration order for output. Blocking keys and cached artefacts
   are only valid while this holds.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "MISSING_PLACEHOLDERS",
    "LEGAL_SUFFIX_TOKENS",
    "PreprocessedRecord",
    "is_missing",
    "normalize_text",
    "normalize_name",
    "normalize_address",
    "normalize_country",
    "core_tokens",
    "address_tokens",
    "address_alpha_tokens",
    "extract_house_number",
    "preprocess_record",
]


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Literal values that stand for "no value" rather than real content. Checked
#: case-insensitively against a whole field. ``n/a`` is listed because the
#: address normalisation turns its slash into a space; the derived normalised
#: set below is generated from these same entries so the two can never drift.
MISSING_PLACEHOLDERS = frozenset({"", "-", "--", "?", "n/a", "na", "nan", "null"})

#: Business / legal / generic tokens dropped when deriving *core* name tokens.
#: This is deliberately the exact list used in the blocking experiments. Adding
#: to it changes every name blocking key, so any extension needs the blocking
#: numbers re-validated before it is adopted.
#:
#: Note two quirks that are part of the validated behaviour: "and", "of" and
#: "the" are removed wherever they occur, and removal is per token rather than
#: only at the end of the name. Both were present when the blocking recall of
#: 76.6% was measured, so they are preserved here on purpose.
LEGAL_SUFFIX_TOKENS = frozenset({
    "inc", "llc", "ltd", "co", "corp", "company", "plc", "gmbh", "sa", "sas",
    "pvt", "private", "limited", "llp", "lp", "the", "and", "of", "center",
    "services", "group", "enterprises",
})

_SEPARATORS_RE = re.compile(r"[\W_]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_APOSTROPHES = ("'", "\u2019", "\u02bc", "\uff07")
_AMPERSAND = "&"


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

def _fold_latin_accents(text: str) -> str:
    """Remove combining accents that sit on a Latin letter, keeping all others.

    ``"caf\\u00e9" -> "cafe"`` but a Devanagari combining sign is preserved
    because the character before it is not a Latin letter. A plain ``Mn``
    filter would strip Indic vowel signs and damage native-script names.

    The string is decomposed to expose accents and then recomposed, so the
    result is always NFC. Without the recomposition, NFKD would leave Indic
    consonants in a decomposed form that is needlessly hard to read and to
    compare byte-wise.
    """
    out: list[str] = []
    on_latin = False
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.category(ch) == "Mn" and on_latin:
            continue
        out.append(ch)
        on_latin = ch.isascii() and ch.isalpha()
    return unicodedata.normalize("NFC", "".join(out))


def _to_separators(text: str) -> str:
    """Replace every character that is not a letter, number or mark with a space.

    Letters and numbers of *any* script are kept, so non-Latin text survives.
    Marks are kept because they carry meaning in Indic scripts. The fast regex
    path is used for ASCII input, where the two implementations are provably
    identical (no mark characters exist in ASCII).
    """
    if text.isascii():
        return _SEPARATORS_RE.sub(" ", text)
    return "".join(
        ch if unicodedata.category(ch)[0] in ("L", "N", "M") else " "
        for ch in text
    )


def normalize_text(value: str | None) -> str:
    """Canonical comparison form of a free-text field.

    Lower-cases, expands ``&`` to ``and``, drops apostrophes without splitting
    the word, folds Latin accents, turns punctuation into spaces and collapses
    whitespace. The result is deterministic and script-preserving.
    """
    if value is None:
        return ""
    text = value.casefold().replace(_AMPERSAND, " and ")
    for mark in _APOSTROPHES:
        text = text.replace(mark, "")
    text = _fold_latin_accents(text)
    return _WHITESPACE_RE.sub(" ", _to_separators(text)).strip()


#: Same normaliser applied to the normalised placeholder spellings, so that
#: ``"N/A"`` is recognised as missing both before and after normalisation.
_MISSING_NORMALISED = frozenset(normalize_text(p) for p in MISSING_PLACEHOLDERS)


def is_missing(value: str | None) -> bool:
    """True when a field carries no information.

    A field counts as missing when it is ``None``, blank, or equal to a literal
    placeholder such as ``null``, ``n/a`` or ``na``. Only whole-field matches are
    treated as missing: a stray ``na`` inside a real address is not a value we
    can confidently discard.
    """
    if value is None:
        return True
    text = value.strip()
    if not text:
        return True
    return text.casefold() in MISSING_PLACEHOLDERS or normalize_text(text) in _MISSING_NORMALISED


# --------------------------------------------------------------------------
# Field-level normalisation
# --------------------------------------------------------------------------

def normalize_name(value: str | None) -> str:
    """Normalised form of ``business_name``. Empty string when missing."""
    return "" if is_missing(value) else normalize_text(value)


def normalize_address(value: str | None) -> str:
    """Normalised form of ``business_address``. Empty string when missing.

    No country-specific structure is assumed: the result is a flat token string,
    because address layouts differ per country and France is unseen in training.
    """
    return "" if is_missing(value) else normalize_text(value)


def normalize_country(value: str | None) -> str:
    """Normalised open-set country label. Never mapped to a fixed vocabulary.

    Only ``None`` and blank count as missing. :data:`MISSING_PLACEHOLDERS` is
    deliberately *not* applied here, because that set is tuned for free-text
    address fields: ``na`` is a legitimate country code (Namibia) as well as a
    common way of writing "not available", and collapsing it to the empty
    string would silently delete a real open-set value. The statement requires
    country to stay an unfiltered open set of string labels, so the only
    country value that is dropped is one that carries no characters at all.
    """
    if value is None or not value.strip():
        return ""
    return normalize_text(value)


def core_tokens(normalized_name: str) -> tuple[str, ...]:
    """Name tokens with legal/generic tokens removed, in original order.

    Takes an already normalised name, so that callers do not pay for
    normalising the same field twice. ``core_tokens(name)`` from the temporary
    probe is equivalent to ``core_tokens(normalize_name(name))`` here.
    """
    if not normalized_name:
        return ()
    return tuple(t for t in normalized_name.split() if t not in LEGAL_SUFFIX_TOKENS)


def address_tokens(normalized_address: str) -> tuple[str, ...]:
    """All address tokens, in original order."""
    if not normalized_address:
        return ()
    return tuple(normalized_address.split())


def address_alpha_tokens(normalized_address: str, min_length: int = 3) -> tuple[str, ...]:
    """Address tokens that contain a letter, for use as blocking/feature words.

    "Contains a letter" rather than :meth:`str.isalpha` on purpose:
    ``str.isalpha`` is false for a Devanagari word carrying a vowel sign, which
    would silently drop native-script address words.
    """
    return tuple(
        t for t in address_tokens(normalized_address)
        if len(t) >= min_length and any(unicodedata.category(c)[0] == "L" for c in t)
    )


def extract_house_number(normalized_address: str) -> str | None:
    """First all-digit address token, or ``None`` when there is none.

    Matches the behaviour of the validated blocking experiment exactly: a
    token qualifies only if it is entirely digits. A house number is therefore
    optional, never assumed, and native-script digits work because
    :meth:`str.isdigit` is Unicode-aware.
    """
    for token in address_tokens(normalized_address):
        if token.isdigit():
            return token
    return None


# --------------------------------------------------------------------------
# Record-level convenience
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessedRecord:
    """One source row after normalisation. ``entity_id`` is never altered."""

    entity_id: str
    country: str
    name: str
    name_core: tuple[str, ...]
    address: str
    address_tokens: tuple[str, ...]
    address_alpha_tokens: tuple[str, ...]
    house_number: str | None
    name_is_missing: bool
    address_is_missing: bool


def preprocess_record(
    entity_id: str,
    business_name: str | None,
    business_address: str | None,
    country: str | None,
) -> PreprocessedRecord:
    """Normalise a single source row, keeping its original ``entity_id``."""
    name = normalize_name(business_name)
    address = normalize_address(business_address)
    return PreprocessedRecord(
        entity_id=entity_id,
        country=normalize_country(country),
        name=name,
        name_core=core_tokens(name),
        address=address,
        address_tokens=address_tokens(address),
        address_alpha_tokens=address_alpha_tokens(address),
        house_number=extract_house_number(address),
        name_is_missing=is_missing(business_name),
        address_is_missing=is_missing(business_address),
    )
'''

_SRC_BLOCKING = r'''
"""Blocking: reduce the S1 x (S2 + S3) space to a manageable candidate set.

Blocking is *candidate generation*, not matching. Nothing here computes a
similarity feature or trains a model. The only job of this module is to make
sure a true match is reachable through at least one cheap key while keeping the
number of pairs handed to the model affordable.

Three key families are built, all from the representations that
:mod:`src.preprocessing` already produced, so no normalisation logic is
duplicated here:

``name_2tok``
    the first two core name tokens sorted alphabetically and cut to four
    characters, plus the country
``addr_ht0``
    house number plus a short prefix of the longest useful address word
``addr_ht1``
    house number plus a short prefix of the second longest address word

A pair survives if *any* of the three routes retrieves it, so results are
unioned and deduplicated: a true match only has to be found once. That is why
the routes are complementary rather than ranked, and why one weak key cannot
cancel out a good one from another route.

Every key embeds the normalised country as an opaque string. There is no
country vocabulary, no one-hot encoding and no country-specific address parsing
anywhere in this file, so a country never seen during development (``France``
in the competition's test split) is handled by the same code path as a familiar
one.

The key *shapes* below were chosen by offline experiments on the training
split. The measured volumes and recall from those experiments are deliberately
not encoded as constants here; only the structure is hard-coded, so the numbers
can be re-derived on new data instead of being trusted blindly.

No file I/O and no third-party imports. Callers pass in
:class:`~src.preprocessing.PreprocessedRecord` objects, which keeps this module
independent of where the TSV files live.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .preprocessing import PreprocessedRecord

__all__ = [
    "BlockingConfig",
    "RecordRef",
    "CandidatePair",
    "BlockingStats",
    "BlockingIndex",
    "name_blocking_key",
    "address_blocking_keys",
    "blocking_keys",
    "build_index",
    "generate_candidates",
]

#: A blocking key is a tuple of strings whose first element names the strategy
#: that produced it. The tag costs nothing and buys three things: keys are
#: readable in a debugger, a name key can never collide with an address key
#: even if a country label or token happens to look like a house number, and
#: the strategies could later share one dict without a migration.
BlockingKey = tuple[str, ...]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockingConfig:
    """Tunable shape of the blocking keys.

    The defaults encode the *structure* that was validated, not the measured
    results. Every field is overridable so the same code can express a
    different trade-off between recall and candidate volume without being
    rewritten.
    """

    #: How many core name tokens make up ``name_2tok``.
    name_token_count: int = 2
    #: Characters kept from each chosen name token in ``name_2tok``. Truncating
    #: is what lets "Restaurant"/"Restaurante" reach the same group.
    name_prefix_length: int = 4
    #: Characters kept from each chosen address word in the ``addr_ht*`` keys.
    address_prefix_length: int = 3
    #: How many of the longest address words produce keys (0 and 1 -> ht0, ht1).
    address_word_count: int = 2
    #: Largest bucket any single key may hold. See :meth:`BlockingIndex.add` for
    #: exactly what happens to the records that exceed it.
    max_group_size: int = 1000

    def __post_init__(self) -> None:
        if self.name_token_count < 1:
            raise ValueError("name_token_count must be at least 1")
        if self.name_prefix_length < 1:
            raise ValueError("name_prefix_length must be at least 1")
        if self.address_prefix_length < 1:
            raise ValueError("address_prefix_length must be at least 1")
        if self.address_word_count < 0:
            raise ValueError("address_word_count cannot be negative")
        if self.max_group_size < 1:
            raise ValueError("max_group_size must be at least 1")


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class RecordRef:
    """A row identified by its source *and* its id, both kept verbatim.

    The source is part of the identity on purpose. S2 and S3 ids are only
    unique within their own file, so two rows from different sources can carry
    the same ``entity_id``; keying on the id alone would silently merge them
    and attribute a candidate to the wrong source. No id is ever stripped,
    renumbered or rewritten.
    """

    source: str
    entity_id: str

    def as_tuple(self) -> tuple[str, str]:
        """Plain ``(source, entity_id)`` pair, handy for writing TSV output."""
        return (self.source, self.entity_id)


@dataclass(frozen=True, order=True)
class CandidatePair:
    """One retrieved pair: a reference row and a candidate row from S2 or S3."""

    reference: RecordRef
    candidate: RecordRef


# --------------------------------------------------------------------------
# Key construction
# --------------------------------------------------------------------------

def name_blocking_key(
    record: PreprocessedRecord,
    config: BlockingConfig | None = None,
) -> BlockingKey | None:
    """The ``name_2tok`` key, or ``None`` when the name has no core token.

    Reproduces the key that was measured on the 20,000-row S1 slice at
    76.6% recall / 192.4 candidates per S1 row. That experiment built the key
    as::

        st = sorted(core_tokens(name))
        key = (country, st[0][:4], st[1][:4]) if len(st) > 1 else (country, st[0][:4])

    and all three parts of that expression matter:

    * **Sorted.** ``Blue Sky Restaurant`` and ``Restaurant Sky Blue`` produce the
      same key, so a business written in a different word order is still
      reachable by name.
    * **Truncated to four characters.** This is what absorbs the small spelling
      variations that separate two records of the same company, and it is why
      the key stays short enough to be selective.
    * **Single-token fallback.** A name with exactly one core token is *not*
      dropped; it gets a one-token key instead. Note the consequence, which is
      easy to mistake for a bug: a one-token key is a different tuple from a
      two-token key, so "Cafe" does not join the group of "Cafe Blue". The
      fallback groups short names with each other, it does not let them match
      longer names. That is the behaviour that was measured, so it is kept
      rather than "fixed".

    A name whose core tokens are all legal suffixes has no core token at all
    and gets ``None``; padding such a name would group every business called
    "Private Ltd" together.

    The tokens come from :mod:`src.preprocessing`, so they are Unicode-safe.
    The experiment's own ``core_tokens`` was ASCII-only and reduced a
    Devanagari or Tamil name to an empty token list, which silently collapsed
    those businesses into one bucket. That behaviour is deliberately *not*
    reproduced: same key shape, script-preserving tokens.
    """
    config = config or BlockingConfig()
    tokens = sorted(record.name_core)
    if not tokens:
        return None
    prefix = config.name_prefix_length
    return ("name_2tok", record.country,
            *(token[:prefix] for token in tokens[: config.name_token_count]))


def _longest_address_words(record: PreprocessedRecord, count: int) -> list[str]:
    """The ``count`` longest useful address words, longest first.

    :meth:`sorted` is stable, so words of equal length keep their address
    order. That makes the ranking deterministic without inventing a tie-break
    rule, and it means the first two words of a two-word address are not
    swapped arbitrarily between the two runs.
    """
    if count <= 0:
        return []
    return sorted(record.address_alpha_tokens, key=len, reverse=True)[:count]


def address_blocking_keys(
    record: PreprocessedRecord,
    config: BlockingConfig | None = None,
) -> tuple[BlockingKey, ...]:
    """The ``addr_ht0`` / ``addr_ht1`` keys for one record.

    Each key pairs the house number with a ``address_prefix_length``-character
    prefix of one of the longest useful address words, so two records must
    agree on the building *and* roughly on the street before they are compared.

    A house number is required. Without one there is nothing to anchor the
    street word to, and a key made of a street word alone would group every
    business on that street; that is precisely the flooding the group cap
    exists to limit, so such records simply produce no address key. They remain
    reachable through the name key, or not at all, which is a real and intended
    recall cost rather than a crash.

    The layout is not parsed by country: the words come from the generic token
    list, so "12 MG Road" and "12 rue de la Paix" are treated the same way.
    """
    config = config or BlockingConfig()
    house_number = record.house_number
    if not house_number:
        return ()

    keys: list[BlockingKey] = []
    for position, word in enumerate(_longest_address_words(record, config.address_word_count)):
        keys.append((
            f"addr_ht{position}",
            record.country,
            house_number,
            word[: config.address_prefix_length],
        ))

    # Two long words can share a 3-character prefix. Deduplicating here keeps a
    # record from being inserted twice into one bucket, which would otherwise
    # inflate the bucket against the cap.
    return tuple(dict.fromkeys(keys))


def blocking_keys(
    record: PreprocessedRecord,
    config: BlockingConfig | None = None,
) -> tuple[BlockingKey, ...]:
    """Every key this record can be found by, across all strategies.

    Returned as a tuple rather than a set so that iteration order is fixed,
    which keeps the index and its statistics reproducible.
    """
    config = config or BlockingConfig()
    keys: list[BlockingKey] = []
    name_key = name_blocking_key(record, config)
    if name_key is not None:
        keys.append(name_key)
    keys.extend(address_blocking_keys(record, config))
    return tuple(dict.fromkeys(keys))


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockingStats:
    """Counters describing what an index actually contains.

    Exposed mainly so that silent recall loss is visible. ``keyless_records``
    in particular counts rows that produced no key at all and can therefore
    never be retrieved by anyone; a non-zero value is a data problem worth
    looking at, not something to ignore.
    """

    indexed_records: int
    keyless_records: int
    distinct_keys: int
    largest_group: int
    capped_groups: int
    stored_references: int


@dataclass
class BlockingIndex:
    """Hash-map index from blocking key to the records that own it.

    One flat dict is used rather than three separate ones, because the strategy
    tag inside each key keeps the families separate. Lookups stay a single dict
    probe per key and :meth:`stats` can report across all strategies at once.
    """

    config: BlockingConfig = field(default_factory=BlockingConfig)
    _buckets: dict[BlockingKey, list[RecordRef]] = field(default_factory=dict, repr=False)
    _capped: set[BlockingKey] = field(default_factory=set, repr=False)
    _indexed_records: int = 0
    _keyless_records: int = 0

    # -- construction ------------------------------------------------------

    def add(self, record: PreprocessedRecord, source: str) -> int:
        """Index one preprocessed record under every key it produces.

        Returns the number of keys it was filed under, which is ``0`` for a
        record that has no usable key. Such a record is *not* stored anywhere,
        so it can never be retrieved as a candidate; that loss is counted in
        :meth:`stats` rather than hidden.

        The per-key cap is applied here, at insertion time, and the first
        ``max_group_size`` records to reach a key are the ones kept. Two
        consequences are worth stating plainly: the cap also bounds memory, and
        it makes a key's contents depend on input order, because "first" is
        defined by arrival. Records are added in file order, which is
        deterministic, so repeated runs agree. A key that overflows is a key
        too generic to be evidence of a match, so dropping the overflow is the
        intended behaviour rather than a silent truncation.
        """
        if not source:
            raise ValueError("source must be a non-empty string")
        ref = RecordRef(source, record.entity_id)
        keys = blocking_keys(record, self.config)
        if not keys:
            self._keyless_records += 1
            return 0

        cap = self.config.max_group_size
        for key in keys:
            bucket = self._buckets.get(key)
            if bucket is None:
                self._buckets[key] = [ref]
            elif len(bucket) < cap:
                bucket.append(ref)
            else:
                self._capped.add(key)

        self._indexed_records += 1
        return len(keys)

    def extend(self, records: Iterable[PreprocessedRecord], source: str) -> "BlockingIndex":
        """Index every record in ``records`` and return ``self`` for chaining."""
        for record in records:
            self.add(record, source)
        return self

    # -- lookup ------------------------------------------------------------

    def keys_for(self, record: PreprocessedRecord) -> tuple[BlockingKey, ...]:
        """The keys a query record would be looked up under."""
        return blocking_keys(record, self.config)

    def retrieve(self, record: PreprocessedRecord) -> Iterator[RecordRef]:
        """Yield every record sharing at least one key, possibly more than once.

        This is the raw union of the matching buckets. A record reachable
        through two different keys is yielded twice on purpose: the caller
        decides how to deduplicate, and
        :func:`generate_candidates` does it once at the end.
        """
        for key in blocking_keys(record, self.config):
            bucket = self._buckets.get(key)
            if bucket:
                yield from bucket

    def candidates_for(self, record: PreprocessedRecord) -> tuple[RecordRef, ...]:
        """Deduplicated candidates for one query record, in a stable order.

        The union is a set, and Python randomises string hashing per process,
        so a set's iteration order is not reproducible between runs. The result
        is therefore sorted by ``(source, entity_id)``: a stable, obvious order
        that keeps any later feature table or output file byte-identical across
        runs.
        """
        return tuple(sorted(set(self.retrieve(record))))

    # -- introspection -----------------------------------------------------

    def stats(self) -> BlockingStats:
        """Summary counters for the built index."""
        return BlockingStats(
            indexed_records=self._indexed_records,
            keyless_records=self._keyless_records,
            distinct_keys=len(self._buckets),
            largest_group=max((len(b) for b in self._buckets.values()), default=0),
            capped_groups=len(self._capped),
            stored_references=sum(len(b) for b in self._buckets.values()),
        )

    def __len__(self) -> int:
        return self._indexed_records


def build_index(
    records: Iterable[PreprocessedRecord],
    source: str,
    config: BlockingConfig | None = None,
) -> BlockingIndex:
    """Index all candidate records of one source.

    Call this once per candidate source (``S2``, then ``S3``) and merge the
    results, or extend a single index with both. Records are expected to be
    preprocessed already, which is what keeps this module free of any
    normalisation logic of its own.
    """
    return BlockingIndex(config=config or BlockingConfig()).extend(records, source)


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------

def generate_candidates(
    record: PreprocessedRecord,
    index: BlockingIndex,
    source: str,
) -> tuple[CandidatePair, ...]:
    """All candidate pairs for one reference row, deduplicated and stable.

    ``source`` labels the reference row, so the returned pairs always say which
    reference a candidate belongs to. Every pair here is only a *candidate*:
    nothing has been compared yet, and deciding which of these pairs are real
    matches is the matching stage's job.
    """
    if not source:
        raise ValueError("source must be a non-empty string")
    reference = RecordRef(source, record.entity_id)
    return tuple(
        CandidatePair(reference, candidate)
        for candidate in index.candidates_for(record)
        # Guards the case where a caller indexed the reference source too. S1
        # and S2 may legitimately share an entity_id, and only an exact
        # (source, id) match is a self-pair, so cross-source ids are untouched.
        if candidate != reference
    )
'''

_SRC_FEATURES = r'''
"""Pairwise similarity features for candidate pairs produced by blocking.

This module sits between :mod:`src.blocking` and the (not yet written) matching
stage. For one candidate pair it turns the two
:class:`~src.preprocessing.PreprocessedRecord` objects into a small, fixed,
numeric vector that a classifier can consume later.

Three rules shape every decision here.

**1. Missing data must never look like agreement.**
The failure this avoids: when both records have no address, a naive
``address_exact`` comparison returns ``1.0``, and the model learns "identical
address means match" from a pair that carries no address evidence whatsoever.
So every field-level similarity is *gated*. If either record is missing the
field, the similarity is forced to ``0.0`` and a separate ``*_present_*`` flag
carries the fact that the field was absent. ``0.0`` therefore means "no
evidence", and the companion flag is what lets the model tell "no evidence"
apart from "evidence of difference". This is why the quality flags at the end
of :data:`FEATURE_NAMES` exist as real columns instead of being thrown away.

**2. Character level and token level are both needed.**
Token overlap is blind to spelling errors; character overlap is sensitive to
word order. A business recorded as "Blue Sky Restaurant" in one source and
"Restaurant Sky Blue" in another scores badly on character similarity but
perfectly on sorted-token equality, and "Restaurante" versus "Restaurant"
scores well on character similarity while sharing no tokens at all. Keeping
only one of the two loses a whole failure mode.

**3. No learned representations yet.**
There are no embeddings and no external models, so every feature is cheap,
deterministic and explainable, and the whole vector can be recomputed from two
records without loading a model.

Unicode is preserved throughout. No feature ASCII-strips a name, so a
Devanagari or Tamil business is compared on its own script instead of silently
collapsing to an empty string and looking identical to every other such
business.

No file I/O, no dataset access, no third-party imports.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from difflib import SequenceMatcher
from functools import lru_cache

from .preprocessing import PreprocessedRecord

__all__ = [
    "FEATURE_NAMES",
    "PairFeatures",
    "extract_features",
    "to_vector",
    "featurize",
    "featurize_batch",
]


# --------------------------------------------------------------------------
# Similarity primitives
# --------------------------------------------------------------------------

def _ratio(a: str, b: str) -> float:
    """Character-level similarity in ``[0, 1]``.

    ``SequenceMatcher`` is used with ``autojunk=False``. The default
    ``autojunk=True`` silently discards characters that appear in more than 1%
    of a sequence of length 200 or more, which for a long business address
    would throw away exactly the repeated street and city words that carry the
    signal. Business names and addresses are short enough that the heuristic
    would rarely fire, but relying on that would make the feature depend on a
    length threshold nobody remembers, so it is switched off explicitly.
    """
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _count_ratio(a: int, b: int) -> float:
    """Shorter count divided by longer count, in ``(0, 1]``. ``0.0`` if both 0."""
    longest = max(a, b)
    return min(a, b) / longest if longest else 0.0


def _length_ratio(a: str, b: str) -> float:
    """Shorter length divided by longer length, in ``(0, 1]``."""
    return _count_ratio(len(a), len(b))


@lru_cache(maxsize=1 << 16)
def _bigrams(text: str) -> frozenset[str]:
    """Character bigrams of ``text``, cached.

    Cached because one S1 record is compared against many candidates, so its
    name and address are re-read for every single pair. The cache is keyed on
    the string and is bounded, so a full pass cannot grow it without limit.
    """
    if len(text) < 2:
        return frozenset((text,)) if text else frozenset()
    return frozenset(text[i:i + 2] for i in range(len(text) - 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """``|A & B| / |A | B|``. ``0.0`` when either side is empty.

    Only ``len`` and set algebra are used, never iteration order, so the result
    does not depend on set ordering and is identical across processes.
    """
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return intersection / (len(a) + len(b) - intersection)


@lru_cache(maxsize=1 << 16)
def _token_set(tokens: tuple[str, ...]) -> frozenset[str]:
    """Set view of a token tuple, cached for the same reuse reason as bigrams."""
    return frozenset(tokens)


@lru_cache(maxsize=1 << 16)
def _core_prefix_key(core: tuple[str, ...]) -> tuple[str, ...]:
    """The two alphabetically-first core tokens, cut to four characters.

    This mirrors the ``name_2tok`` blocking key on purpose. When it fires, the
    pair was retrieved by the name route, so it is close to a constant for that
    subset of candidates and mostly informative for the others; it is cheap and
    it lets the model recognise a first-4-character agreement explicitly.
    """
    return tuple(token[:4] for token in sorted(core)[:2])


# --------------------------------------------------------------------------
# Feature container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PairFeatures:
    """The numeric description of one candidate pair.

    Field order *is* the vector order; :data:`FEATURE_NAMES` mirrors it and the
    test suite asserts the two cannot drift apart. Every value is a ``float`` in
    ``[0, 1]`` except the two counts (``*_overlap``), which are small integers
    stored as floats. There is no ``NaN`` anywhere: a tree model can route
    around ``NaN`` but a linear model propagates it into every downstream
    product, so absence is encoded as ``0.0`` plus a presence flag instead.
    """

    # -- name ---------------------------------------------------------------
    #: Normalised names identical, both present.
    name_exact: float
    #: Character similarity of the full normalised name.
    name_char_ratio: float
    #: Order-insensitive character similarity of the name.
    name_bigram_jaccard: float
    #: Length agreement of the name.
    name_len_ratio: float
    #: Core token *sets* identical, i.e. same words in any order.
    name_core_sorted_exact: float
    #: Token overlap of the core name tokens.
    name_core_jaccard: float
    #: How many core name tokens the two records share.
    name_core_overlap: float
    #: Core token count agreement.
    name_core_token_ratio: float
    #: First four characters of the two leading core tokens agree.
    name_core_prefix4_equal: float

    # -- address ------------------------------------------------------------
    #: Normalised addresses identical, both present.
    address_exact: float
    #: Character similarity of the full normalised address.
    address_char_ratio: float
    #: Order-insensitive character similarity of the address.
    address_bigram_jaccard: float
    #: Token overlap including house numbers and short words.
    address_token_jaccard: float
    #: Token overlap of meaningful address words only.
    address_alpha_jaccard: float
    #: How many meaningful address words the two records share.
    address_alpha_overlap: float
    #: Address token count agreement.
    address_token_ratio: float

    # -- house number -------------------------------------------------------
    #: Both records carry the same house number.
    house_number_equal: float
    #: Both records carry *different* house numbers. The strongest single piece
    #: of negative evidence available without external data.
    house_number_conflict: float

    # -- country ------------------------------------------------------------
    #: Countries equal and both non-empty. Open set: any string is accepted and
    #: no country is ever mapped to a code or enumerated.
    country_equal: float
    #: At least one record states a country.
    country_present_either: float

    # -- quality / missingness ---------------------------------------------
    #: Neither record is missing its name.
    name_present_both: float
    #: Neither record is missing its address.
    address_present_both: float
    #: At least one record has usable core name tokens.
    name_core_present_either: float
    #: At least one record has meaningful address words.
    address_alpha_present_either: float


#: Fixed, explicit vector order. Mirrors :class:`PairFeatures` field order.
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in fields(PairFeatures))


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract_features(
    left: PreprocessedRecord,
    right: PreprocessedRecord,
) -> PairFeatures:
    """Compute every feature for one candidate pair.

    The pair is treated as unordered: swapping the two arguments gives the same
    values, so the features cannot encode a source ranking that the matching
    stage is not supposed to have.

    Every similarity is gated on both sides actually having the field, per the
    rule described in the module docstring.
    """
    # -- name ---------------------------------------------------------------
    name_l, name_r = left.name, right.name
    core_l, core_r = left.name_core, right.name_core
    name_present = 1.0 if (name_l and name_r) else 0.0
    core_l_set, core_r_set = _token_set(core_l), _token_set(core_r)
    core_present = 1.0 if (core_l or core_r) else 0.0
    core_sorted_l, core_sorted_r = sorted(core_l), sorted(core_r)

    if name_present:
        name_exact = 1.0 if name_l == name_r else 0.0
        name_char_ratio = _ratio(name_l, name_r)
        name_bigram_jaccard = _jaccard(_bigrams(name_l), _bigrams(name_r))
        name_len_ratio = _length_ratio(name_l, name_r)
    else:
        # No name evidence: report "no evidence", never "no similarity found".
        name_exact = name_char_ratio = name_bigram_jaccard = name_len_ratio = 0.0

    if core_present:
        intersection = len(core_l_set & core_r_set)
        name_core_sorted_exact = 1.0 if core_sorted_l == core_sorted_r else 0.0
        name_core_jaccard = intersection / (len(core_l_set) + len(core_r_set) - intersection)
        name_core_overlap = float(intersection)
        name_core_token_ratio = _count_ratio(len(core_l), len(core_r)) if core_l and core_r else 0.0
        name_core_prefix4_equal = 1.0 if (
            core_l and core_r and _core_prefix_key(core_l) == _core_prefix_key(core_r)
        ) else 0.0
    else:
        name_core_sorted_exact = name_core_jaccard = 0.0
        name_core_overlap = name_core_token_ratio = name_core_prefix4_equal = 0.0

    # -- address ------------------------------------------------------------
    addr_l, addr_r = left.address, right.address
    alpha_l, alpha_r = left.address_alpha_tokens, right.address_alpha_tokens
    address_present = 1.0 if (addr_l and addr_r) else 0.0
    alpha_present = 1.0 if (alpha_l or alpha_r) else 0.0
    tokens_l, tokens_r = left.address_tokens, right.address_tokens

    if address_present:
        address_exact = 1.0 if addr_l == addr_r else 0.0
        address_char_ratio = _ratio(addr_l, addr_r)
        address_bigram_jaccard = _jaccard(_bigrams(addr_l), _bigrams(addr_r))
    else:
        address_exact = address_char_ratio = address_bigram_jaccard = 0.0
        address_token_jaccard = address_token_ratio = 0.0

    if tokens_l and tokens_r:
        address_token_jaccard = _jaccard(_token_set(tokens_l), _token_set(tokens_r))
        address_token_ratio = _count_ratio(len(tokens_l), len(tokens_r))
    if alpha_l and alpha_r:
        alpha_l_set, alpha_r_set = _token_set(alpha_l), _token_set(alpha_r)
        address_alpha_jaccard = _jaccard(alpha_l_set, alpha_r_set)
        address_alpha_overlap = float(len(alpha_l_set & alpha_r_set))
    else:
        address_alpha_jaccard = address_alpha_overlap = 0.0

    # -- house number -------------------------------------------------------
    house_l, house_r = left.house_number, right.house_number
    both_have_house = house_l is not None and house_r is not None
    house_number_equal = 1.0 if (both_have_house and house_l == house_r) else 0.0
    house_number_conflict = 1.0 if (both_have_house and house_l != house_r) else 0.0

    # -- country ------------------------------------------------------------
    country_l, country_r = left.country, right.country
    country_present_either = 1.0 if (country_l or country_r) else 0.0
    country_equal = 1.0 if (country_l and country_r and country_l == country_r) else 0.0

    return PairFeatures(
        name_exact=name_exact,
        name_char_ratio=name_char_ratio,
        name_bigram_jaccard=name_bigram_jaccard,
        name_len_ratio=name_len_ratio,
        name_core_sorted_exact=name_core_sorted_exact,
        name_core_jaccard=name_core_jaccard,
        name_core_overlap=name_core_overlap,
        name_core_token_ratio=name_core_token_ratio,
        name_core_prefix4_equal=name_core_prefix4_equal,
        address_exact=address_exact,
        address_char_ratio=address_char_ratio,
        address_bigram_jaccard=address_bigram_jaccard,
        address_token_jaccard=address_token_jaccard,
        address_alpha_jaccard=address_alpha_jaccard,
        address_alpha_overlap=address_alpha_overlap,
        address_token_ratio=address_token_ratio,
        house_number_equal=house_number_equal,
        house_number_conflict=house_number_conflict,
        country_equal=country_equal,
        country_present_either=country_present_either,
        name_present_both=name_present,
        address_present_both=address_present,
        name_core_present_either=core_present,
        address_alpha_present_either=alpha_present,
    )


# --------------------------------------------------------------------------
# Vector conversion
# --------------------------------------------------------------------------

def to_vector(features: PairFeatures) -> tuple[float, ...]:
    """Flatten to a tuple of floats in :data:`FEATURE_NAMES` order.

    Built from the dataclass field order rather than by looking names up, so it
    is a straight read with no dictionary and no chance of a name typo silently
    producing a column of zeros.
    """
    return tuple(getattr(features, name) for name in FEATURE_NAMES)


def featurize(
    left: PreprocessedRecord,
    right: PreprocessedRecord,
) -> tuple[float, ...]:
    """Extract and flatten in one call. The hot path for a full pass."""
    return to_vector(extract_features(left, right))


def featurize_batch(
    pairs,
) -> list[tuple[float, ...]]:
    """Featurize an iterable of ``(left, right)`` pairs, preserving order.

    A plain list rather than a generator so a caller can measure length and
    index straight into it. Row *i* corresponds to input pair *i*, which is what
    keeps the feature matrix aligned with the candidate list and therefore with
    any labels attached to it.
    """
    return [featurize(left, right) for left, right in pairs]
'''

_SRC_MATCHING = r'''
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
#: reproducible; this is a reporting grid, not a tuned hyper-parameter set.
DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
)


# --------------------------------------------------------------------------
# Input loading
# --------------------------------------------------------------------------

def source_of(entity_id: str) -> str:
    """``"S2-681193310"`` -> ``"S2"``.

    The competition ids already carry the source prefix, so the source is
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

    Both records of every pair must be present in ``records``. A missing id is
    a hard error rather than a skipped row: skipping would quietly change the
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

    F_beta = ``(1 + b^2) * P * R / (b^2 * P + R)``. With ``beta=0.5`` that
    is ``1.25 * P * R / (0.25 * P + R)``, the competition's measure. An undefined
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
        with: a sign that contradicts the field it belongs to is a bug signal,
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
'''

EMBEDDED_SOURCES = {
    "preprocessing.py": _SRC_PREPROCESSING,
    "blocking.py": _SRC_BLOCKING,
    "features.py": _SRC_FEATURES,
    "matching.py": _SRC_MATCHING,
}

#: SHA-256 of each source file as committed. Checked on every run.
EMBEDDED_SHA256 = {
    "preprocessing.py":
        "03528872cbfbaa1c15fa289bcf14526c486986438f88cd55806a966cfd5d3f99",
    "blocking.py":
        "a6feaef3953795de81fd337b8d7570e1e2940e3301a2670e19f5094789d25338",
    "features.py":
        "5568d645321a12983b7dbfb86a23b2d7b155d24cd66d4700ec0b0026b5429c57",
    "matching.py":
        "5eb5cdab5aef1b6c40aa9c9a012e984d6406b16ce522e3ad2691f40aa3352ce1",
}

# ===========================================================================
# LOGGING AND SMALL UTILITIES
# ===========================================================================
def step(message: str) -> None:
    """A major heading: stage boundaries and configuration lines."""
    print(f"\n{message}", flush=True)


def sub(message: str) -> None:
    """An indented detail line under the current heading."""
    print(f"   {message}", flush=True)


def warn(message: str) -> None:
    """Something the user should notice but that does not stop the run."""
    print(f"   [warn] {message}", flush=True)


def human(seconds: float) -> str:
    """Seconds as ``1h 02m 03s`` / ``2m 03s`` / ``3.4s``."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    return f"{minutes}m {sec:02d}s"


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record a hard check in the report. A failed check does not stop the run."""
    CHECKS.append((name, bool(ok), detail))
    mark = "ok  " if ok else "FAIL"
    sub(f"[{mark}] {name}{(' - ' + detail) if detail else ''}")
    return bool(ok)


def expect(name: str, observed_value, expected_value) -> bool:
    """Record a soft, advisory check against a reference count.

    Non-fatal by design. These numbers were produced by the same code on the
    same data, so a mismatch means either the methodology changed or the run
    was interrupted - and neither is improved by aborting before the report is
    written. The report prints what was observed, never the expectation.
    """
    if observed_value is None:
        warn(f"{name} unavailable, expected {expected_value:,}")
        return False
    ok = int(observed_value) == int(expected_value)
    if ok:
        sub(f"[ok  ] {name} = {int(observed_value):,}")
    else:
        warn(f"{name} = {int(observed_value):,}, expected {int(expected_value):,}")
    return ok


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write a file so a reader never sees a half-written one.

    Every checkpoint goes through here. The alternative - writing in place - means
    a run interrupted mid-write leaves a corrupt checkpoint that the resume logic
    then cannot tell apart from a good one, and the whole run has to start over.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_pickle(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def atomic_write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, default=str)
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def read_json(path: Path, default: Any = None) -> Any:
    """Load JSON, returning ``default`` for a missing or corrupt file.

    A truncated checkpoint is treated as absent on purpose: the stage that needs
    it will simply rebuild, which costs time, whereas raising would cost the run.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def describe_threads() -> dict[str, str]:
    """Snapshot the thread-limit environment, for the report."""
    keys = (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    )
    return {key: os.environ.get(key, "unset") for key in keys}


def hashed(entity_id: str) -> bool:
    """The salted hash that decides whether an S1 entity is in the subset.

    Fixed salt, fixed modulus, no state: the subset is therefore identical on
    every machine and every re-run, which is what makes the expected counts in
    this file meaningful.
    """
    digest = hashlib.blake2b(
        S1_SALT + entity_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % S1_MOD == 0


def observed(meta: dict, key: str, _expected: int) -> int | None:
    """Read an observed count out of the metadata, or ``None`` if absent.

    The reference count is accepted only so the call sites read symmetrically
    with :func:`expect`; it is deliberately never returned. Substituting an
    expectation for a missing observation would let a report claim a recall it
    did not measure.
    """
    value = meta.get(key)
    return None if value is None else int(value)


# ===========================================================================
# DUTY CYCLE
# ===========================================================================
# Kaggle does not expose a CPU temperature sensor, so there is no thermal guard
# here. What there is instead is a wall-clock duty cycle: the process works for
# a fixed fraction of the time and sleeps the rest in short bursts. That keeps
# the notebook courteous to a shared machine, and it is also the direct answer to
# a process that otherwise sits at 100% CPU for hours.

class Pacer:
    """Hold the process to a wall-clock duty cycle with short sleeps.

    It measures how long the thread actually worked between ticks and sleeps the
    proportional remainder, so a cheap loop and an expensive loop get the same
    fraction of relief. A plain per-row sleep would give the cheap loops far more
    relief than the expensive ones, which is the opposite of what is wanted.
    """

    def __init__(self, fraction: float = 0.85, tick_every: int = 2000,
                 max_sleep: float = 2.0) -> None:
        self.fraction = min(0.99, max(0.5, float(fraction)))
        self.tick_every = max(1, int(tick_every))
        self.max_sleep = max(0.0, float(max_sleep))
        self.slept_seconds = 0.0
        self.ticks = 0
        self._n = 0
        self._last = time.perf_counter()

    def tick(self) -> None:
        """Count one unit of work; sleep if the elapsed allowance says so."""
        self._n += 1
        self.ticks += 1
        if self._n % self.tick_every:
            return
        now = time.perf_counter()
        busy = now - self._last
        self._last = now
        if busy <= 0:
            return
        allowance = busy * self.fraction / (1.0 - self.fraction)
        allowance = min(allowance, self.max_sleep)
        if allowance > 0.001:
            time.sleep(allowance)
            self.slept_seconds += allowance

    def report(self) -> dict:
        return {
            "duty_fraction": self.fraction,
            "tick_every": self.tick_every,
            "work_units": self.ticks,
            "slept_seconds": round(self.slept_seconds, 1),
        }


def pace() -> None:
    """One unit of work against the duty cycle, if it is installed."""
    if PACE is not None:
        PACE.tick()


# ===========================================================================
# CONFIGURATION RESOLUTION
# ===========================================================================
def resolve_paths() -> dict[str, Path]:
    """Kaggle-first paths, overridable by environment, no platform literals.

    Defaults are the Kaggle locations. Every one can be overridden with an
    environment variable so the same cell can be pointed at a different mount
    without editing the file.
    """
    global WORK, CKPT, DATA_ROOT, SRC_ROOT, OUT_DIR

    WORK = Path(os.environ.get("MATCH_A2_WORK", str(KAGGLE_WORKING))).expanduser()
    CKPT = Path(os.environ.get("MATCH_A2_CKPT", str(WORK / "checkpoints" / "match_a2"))).expanduser()
    DATA_ROOT = Path(os.environ.get("MATCH_A2_DATASET", str(WORK / "dataset"))).expanduser()
    SRC_ROOT = Path(os.environ.get("MATCH_A2_SRC", str(WORK / "src"))).expanduser()
    OUT_DIR = Path(os.environ.get("MATCH_A2_OUT", str(CKPT))).expanduser()
    return {"work": WORK, "checkpoints": CKPT, "dataset": DATA_ROOT,
            "sources": SRC_ROOT, "output": OUT_DIR}


def lower_process_priority() -> str:
    """Ask the kernel for a lower scheduling priority, best effort.

    Never fatal. Kaggle may refuse, and a run that is merely not-niced still
    produces correct results, so a refusal is reported and ignored.
    """
    try:
        os.nice(10)
        return "niced to +10"
    except (AttributeError, OSError) as exc:
        return f"unavailable ({exc.__class__.__name__})"


def pin_to_cpus(requested: int) -> str:
    """Restrict this process to ``requested`` CPUs via affinity.

    A *warning*, not an error, on this platform. Kaggle containers frequently do
    not grant ``sched_setaffinity`` to the notebook process, and failing the whole
    run over a best-effort optimisation would be a bad trade: the BLAS thread cap
    and the duty cycle are the load-bearing limits, affinity is a bonus.

    Pinning to at most two cores is what stops a multi-threaded native library
    from spreading work across every core the container appears to have.
    """
    if not hasattr(os, "sched_setaffinity"):
        return "unsupported on this platform, not fatal"
    available = sorted(os.sched_getaffinity(0))
    wanted = max(1, min(int(requested), len(available)))
    chosen = available[:wanted]
    try:
        os.sched_setaffinity(0, set(chosen))
    except OSError as exc:
        return (f"could not set affinity ({exc.__class__.__name__}); "
                f"continuing with the thread cap only")
    return f"{len(chosen)} of {len(available)} allowed: {chosen}"


def install_thread_limit(requested: int) -> str:
    """Clamp native thread pools at runtime, if threadpoolctl is available.

    The environment variables set at the top of this file are the primary
    defence, because they are read when the native library loads. This is a
    belt-and-braces second line for the case where something already imported
    numpy or sklearn before this cell ran. Absent or failing, it is reported and
    ignored rather than allowed to stop the experiment.
    """
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return "unavailable (threadpoolctl not installed)"
    try:
        threadpool_limits(limits=max(1, int(requested)))
        return f"clamped native pools to {max(1, int(requested))}"
    except Exception as exc:
        return f"failed ({exc.__class__.__name__}); env cap still applies"


# ===========================================================================
# SOURCE MATERIALISATION AND IMPORT
# ===========================================================================
#: Filled in by :func:`materialize_sources`. Records the outcome of the digest
#: comparison for the run report. A mismatch is fatal, so by the time anything
#: else runs this is always the "all N match" form.
SOURCE_DIGEST_NOTE = "not checked yet"


def materialize_sources() -> None:
    """Write the embedded project sources out and verify them against digests.

    Writing them at run time is what makes this a single self-contained cell
    rather than something that needs the repository checked out alongside it.

    The SHA-256 comparison is a **hard gate**. :data:`EMBEDDED_SHA256` was
    recorded from the committed ``src/`` files, and the whole point of embedding
    them is that the run measures *those* files. If the embedded bytes are not
    identical to the recorded digests, the experiment is no longer the recorded
    experiment, and a result produced from drifted sources would be
    indistinguishable from a result produced from the right ones. There is no
    override: a mismatch stops the run.
    """
    global SOURCE_DIGEST_NOTE

    SRC_ROOT.mkdir(parents=True, exist_ok=True)
    (SRC_ROOT / "__init__.py").write_text(
        '"""Embedded verbatim from the project src/ package."""\n',
        encoding="utf-8")
    problems: list[tuple[str, str, str]] = []
    for name, text in EMBEDDED_SOURCES.items():
        # The triple-quote delimiters leave one leading and one trailing newline
        # on each literal, and those are not part of the file. Only newlines are
        # stripped, so the indentation of the final line survives, and CRLF is
        # folded to LF so the bytes written are the committed bytes however the
        # host happens to check the file out.
        data = text.strip("\n").replace("\r\n", "\n").encode("utf-8") + b"\n"
        (SRC_ROOT / name).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        recorded = EMBEDDED_SHA256[name]
        if digest != recorded:
            problems.append((name, recorded, digest))

    if problems:
        SOURCE_DIGEST_NOTE = (
            f"{len(problems)} of {len(EMBEDDED_SOURCES)} embedded sources do "
            f"not match the recorded digests")
        for name, recorded, digest in problems:
            warn(f"digest differs: {name} recorded {recorded[:12]}..., "
                 f"embedded {digest[:12]}...")
        raise SystemExit(
            f"{SOURCE_DIGEST_NOTE}. The embedded copies are not byte-identical "
            f"to the committed src/ files, so this run would not be measuring "
            f"the recorded experiment. Re-copy the sources into the _SRC_* "
            f"literals and update EMBEDDED_SHA256 to match."
        )
    SOURCE_DIGEST_NOTE = (
        f"all {len(EMBEDDED_SOURCES)} embedded sources match the recorded "
        f"digests")
    check("embedded sources match the recorded digests", True,
          f"{len(EMBEDDED_SOURCES)} modules")

    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    # Re-running this cell in one notebook session would otherwise keep the
    # already-imported src modules, so corrected sources would silently not take
    # effect. Purge them before the import.
    for module in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[module]


M = None
B = None
FEATURE_NAMES: tuple[str, ...] = ()
N_FEATURES = 0
FINGERPRINT = ""


def load_project() -> None:
    """Import the freshly written sources and confirm their contract.

    These are **usability** checks, not the digest check: they fail only when the
    embedded modules cannot support the experiment as written - a missing
    function, a feature count other than 24, or a cap/prefix/beta that disagrees
    with the runner's own constants. Failing here is the cheapest possible place
    to find out, long before a subtly wrong feature matrix would exist.

    A digest *difference* is not one of these conditions and is not an error; see
    :func:`materialize_sources`.
    """
    global M, B, FEATURE_NAMES, N_FEATURES, BETA, FINGERPRINT, BLOCKING_CONFIG

    from src import blocking, features, matching, preprocessing

    M = matching
    B = blocking
    FEATURE_NAMES = features.FEATURE_NAMES
    N_FEATURES = len(FEATURE_NAMES)
    BETA = matching.BETA
    if N_FEATURES != 24:
        raise SystemExit(f"expected 24 features, embedded copy has {N_FEATURES}")
    if not preprocessing.LEGAL_SUFFIX_TOKENS or not preprocessing.MISSING_PLACEHOLDERS:
        raise SystemExit(
            "preprocessing loaded with an empty token set: "
            f"LEGAL_SUFFIX_TOKENS={len(preprocessing.LEGAL_SUFFIX_TOKENS)} "
            f"MISSING_PLACEHOLDERS={len(preprocessing.MISSING_PLACEHOLDERS)}")

    # The two shaping constants live in BlockingConfig upstream rather than as
    # module-level names, so they are read off a default instance.
    embedded_config = blocking.BlockingConfig()
    if (embedded_config.max_group_size != CAP
            or embedded_config.name_prefix_length != PREFIX4):
        raise SystemExit(
            f"embedded blocking uses cap={embedded_config.max_group_size} "
            f"prefix={embedded_config.name_prefix_length} but the runner is "
            f"configured for cap={CAP} prefix={PREFIX4}")
    # Built explicitly from the experiment's own two constants, so the config
    # that drives every key in both arms is stated here rather than inherited
    # from a default nobody in this file can see.
    BLOCKING_CONFIG = blocking.BlockingConfig(
        max_group_size=CAP, name_prefix_length=PREFIX4)
    if matching.BETA != BETA_EXPECTED:
        raise SystemExit(
            f"embedded matching uses beta={matching.BETA}, "
            f"the experiment specifies {BETA_EXPECTED}")

    # Each module is checked against its own __all__, which is the contract the
    # module declares for itself. A hand-written name list here drifts: it is how
    # this runner came to demand matching.extract_features, which lives in
    # features and is never re-exported by matching.
    required: dict[str, tuple[object, tuple[str, ...]]] = {
        "preprocessing": (preprocessing, (
            "MISSING_PLACEHOLDERS", "LEGAL_SUFFIX_TOKENS", "PreprocessedRecord",
            "is_missing", "normalize_text", "normalize_name",
            "normalize_address", "normalize_country", "core_tokens",
            "address_tokens", "address_alpha_tokens", "extract_house_number",
            "preprocess_record")),
        "blocking": (blocking, (
            "BlockingConfig", "RecordRef", "CandidatePair", "BlockingStats",
            "BlockingIndex", "name_blocking_key", "address_blocking_keys",
            "blocking_keys", "build_index", "generate_candidates")),
        "features": (features, (
            "FEATURE_NAMES", "PairFeatures", "extract_features", "to_vector",
            "featurize", "featurize_batch")),
        "matching": (matching, (
            "DEFAULT_THRESHOLDS", "BETA", "ThresholdReport", "LabelledPair",
            "FitSummary", "source_of", "load_ground_truth", "iter_records",
            "load_records", "label_candidates", "feature_matrix",
            "split_by_reference", "precision_recall_fbeta",
            "evaluate_thresholds", "best_validation_threshold",
            "LogisticMatcher")),
    }
    for module_name, (module, names) in required.items():
        for name in names:
            if not hasattr(module, name):
                raise SystemExit(
                    f"embedded {module_name} is missing {name}; its __all__ is "
                    f"{getattr(module, '__all__', None)!r}")

    # Not in __all__ but used directly by the S1-level split below.
    for name in ("_split_bucket", "featurize", "CandidatePair"):
        if not hasattr(matching, name):
            raise SystemExit(
                f"embedded matching is missing {name}; the runner uses it "
                f"directly, so it cannot go unexported")

    # The fingerprint ties every checkpoint to the exact configuration that
    # produced it, so a stale checkpoint from a different cap, prefix or feature
    # set is detected instead of being silently mixed into the results.
    FINGERPRINT = hashlib.sha256("|".join([
        f"n_s1={N_S1}", f"salt={S1_SALT!r}", f"mod={S1_MOD}",
        f"cap={CAP}", f"prefix={PREFIX4}",
        f"split={SPLIT_FRACTION}", f"split_salt={SPLIT_SALT!r}",
        f"beta={BETA}", f"features={N_FEATURES}",
        ",".join(FEATURE_NAMES),
    ]).encode("utf-8")).hexdigest()[:16]


# ===========================================================================
# THE A2 KEY - EXPERIMENT CODE, NOT PART OF src/blocking.py
# ===========================================================================
#: The production blocking config, built from the two constants this experiment
#: was specified with. The values are asserted against the embedded module's own
#: defaults in :func:`load_project`, so a drift upstream is caught before stage 3.
BLOCKING_CONFIG = None


def a2_blocking_key(record, config):
    """The A2 key: the first two core name tokens **in original order**.

    This is arm B's addition, and it lives here rather than in the embedded
    ``src/blocking.py`` on purpose: upstream exposes ``name_2tok`` only, and
    that key sorts its tokens alphabetically. A2 is the deliberate opposite, so
    it is experiment code that reads upstream's own
    :class:`~src.preprocessing.PreprocessedRecord` and
    :class:`~src.blocking.BlockingConfig` rather than a redefinition of them.

    Three properties define it, and all three are what distinguish it from
    ``name_2tok``:

    1. **Original order, not sorted.** ``"sky blue"`` and ``"blue sky"`` land in
       the same ``name_2tok`` bucket, which is right for recall but merges
       genuinely different names whose first two words are anagrams. Keeping
       order trades a little recall for precision.
    2. **Truncated to four characters**, taken from the config rather than
       hard-coded, so ``restaurants`` and ``restaurant`` still meet.
    3. **Country in the key.** Two same-named businesses in different countries
       are almost never the same business, and this is the cheapest place to
       say so.

    The single-token case follows ``name_2tok``: one core token produces a
    one-token key rather than being dropped, so a bare brand name stays
    reachable. A record whose core tokens are all legal suffixes has no key at
    all and returns ``None``; padding it would group every business called
    "Private Ltd" into one bucket.
    """
    tokens = record.name_core
    if not tokens:
        return None
    prefix = config.name_prefix_length
    return ("name_2tok_a2", record.country,
            *(token[:prefix] for token in tokens[: config.name_token_count]))


# ===========================================================================
# STAGE 1 - LOCATE AND EXTRACT THE DATASET
# ===========================================================================
def find_dataset_zip(explicit: str | None = None) -> Path:
    """Find the uploaded archive, preferring an explicit path.

    Search order: the explicit path, the environment variable, then a recursive
    scan of ``/kaggle/input``. The name is matched loosely because Kaggle renames
    uploads, but the ``.zip`` suffix is required so a stray file is not opened.
    """
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file() or candidate.is_dir():
            return candidate
        raise SystemExit(
            f"--dataset-zip does not point at a file or directory: {candidate}")

    from_env = os.environ.get("MATCH_A2_ZIP")
    if from_env:
        candidate = Path(from_env).expanduser()
        if candidate.is_file() or candidate.is_dir():
            return candidate
        raise SystemExit(
            f"MATCH_A2_ZIP does not point at a file or directory: {candidate}")

    for root in (KAGGLE_INPUT, WORK, Path.cwd()):
        if not root.is_dir():
            continue
        direct = sorted(p for p in root.glob("*.zip") if ZIP_NAME_HINT.search(p.name))
        if direct:
            return direct[0]
    matches = []
    for root in (KAGGLE_INPUT, WORK):
        if root.is_dir():
            matches.extend(
                sorted(p for p in root.rglob("*.zip")
                       if ZIP_NAME_HINT.search(p.name)
                       and (S3_HINT.search(p.name) or S3_HINT.search(str(p.parent))))
            )
    if not matches:
        # Fallback, only when no archive exists: Kaggle may mount the four
        # train files loose instead of zipped. A real archive always wins.
        for root in (KAGGLE_INPUT, WORK):
            if root.is_dir():
                train_dir = find_train_dir(root)
                if train_dir is not None:
                    return train_dir
        raise SystemExit(
            f"no dataset zip found under {KAGGLE_INPUT}. Attach the "
            f"student_resource archive to the notebook, or set MATCH_A2_ZIP.")
    if len(matches) > 1:
        raise SystemExit(
            "several candidate zips found; pass --dataset-zip to choose one:\n  "
            + "\n  ".join(str(m) for m in matches))
    return matches[0]


def find_train_dir(root: Path) -> Path | None:
    """The directory that actually contains ``train_source1.tsv``."""
    if not root.is_dir():
        return None
    if all((root / name).is_file() for name in TRAIN_FILES):
        return root
    for path in sorted(root.rglob("train_source1.tsv")):
        parent = path.parent
        if all((parent / name).is_file() for name in TRAIN_FILES):
            return parent
    return None


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract an archive, refusing any member that escapes the destination.

    Zip entries carry their own paths, and ``..`` in one of them would otherwise
    write outside the working directory. This is a guard against a malformed or
    hostile archive, not against the competition data, which is well behaved.
    """
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = member.filename
            if name.endswith("/"):
                continue
            if SKIP_NAMES & set(Path(name).parts):
                continue
            target = (destination / name).resolve()
            if not str(target).startswith(str(resolved)):
                raise SystemExit(f"archive member escapes the destination: {name}")
        zf.extractall(destination)


def stage1_dataset(args) -> None:
    """Locate, extract and validate the dataset. Idempotent via a marker."""
    global TRAIN_DIR
    started = time.perf_counter()
    step("STAGE 1/6  DATASET")

    marker_path = CKPT / DATASET_MARKER
    archive = find_dataset_zip(args.dataset_zip)
    marker = read_json(marker_path, default=None)
    reuse = (not args.force and isinstance(marker, dict)
             and marker.get("fingerprint") == FINGERPRINT
             and Path(str(marker.get("train_dir", ""))).is_dir())
    if reuse:
        TRAIN_DIR = Path(marker["train_dir"])
        if find_train_dir(TRAIN_DIR) is not None:
            sub(f"already extracted: {TRAIN_DIR}")
        else:
            reuse = False

    if not reuse:
        if archive.is_dir():
            # Loose dataset already present as train/ directory
            sub(f"archive: {archive} (loose directory)")
            TRAIN_DIR = find_train_dir(archive) or archive
            for name in TRAIN_FILES:
                if not (TRAIN_DIR / name).is_file():
                    raise SystemExit(f"Missing train file: {name}")
            sub(f"train dir: {TRAIN_DIR}")
            for name in TRAIN_FILES:
                size = (TRAIN_DIR / name).stat().st_size
                sub(f"  {name:<24} {size:>15,} bytes")
            atomic_write_json(marker_path, {
                "archive": str(archive), "train_dir": str(TRAIN_DIR),
                "fingerprint": FINGERPRINT,
            })
        else:
            sub(f"archive: {archive}")
            safe_extract(archive, DATA_ROOT)
            found = find_train_dir(DATA_ROOT)
            if found is None:
                raise SystemExit(
                    f"extracted {archive} to {DATA_ROOT} but none of the required "
                    f"train files are present there")
            TRAIN_DIR = found
            atomic_write_json(marker_path, {
                "archive": str(archive), "train_dir": str(TRAIN_DIR),
                "fingerprint": FINGERPRINT,
            })
            sub(f"train dir: {TRAIN_DIR}")
            for name in TRAIN_FILES:
                size = (TRAIN_DIR / name).stat().st_size
                sub(f"  {name:<24} {size:>15,} bytes")
        check("test split never opened", True,
              "only train_* files are read; test_source* is not touched")
        step(f"  STAGE 1 done in {human(time.perf_counter() - started)}")


def train_path(name: str) -> Path:
    """Absolute path of one train file, checked so a typo fails loudly."""
    if name not in TRAIN_FILES:
        raise SystemExit(f"refusing to open non-train file: {name}")
    if TRAIN_DIR is None:
        raise SystemExit("dataset stage has not run yet")
    return TRAIN_DIR / name


# ===========================================================================
# STAGE 2 - THE 20,000-ENTITY S1 SUBSET
# ===========================================================================
def stage2_subset(args) -> tuple[dict, dict]:
    """Select the 20,000 S1 records and their ground truth. Resumable.

    S1 is streamed, not loaded: the full file is far larger than the subset, and
    only the selected ids are kept. The selection is a salted hash, so it is
    the same 20,000 entities on every machine and every re-run.
    """
    started = time.perf_counter()
    step("STAGE 2/6  S1 SUBSET (20,000 entities)")

    cache = CKPT / S1_RECORDS
    cached = pickle.loads(cache.read_bytes()) if (not args.force and cache.is_file()) else None
    if isinstance(cached, dict) and len(cached) == N_S1:
        s1_records = cached
        sub(f"loaded {len(s1_records):,} S1 records from the checkpoint")
    else:
        sub(f"streaming S1, selecting blake2b(key={S1_SALT!r}) % {S1_MOD} == 0")
        s1_records = {}
        scanned = 0
        for rec in M.iter_records(train_path("train_source1.tsv")):
            pace()
            scanned += 1
            if len(s1_records) < N_S1 and hashed(rec.entity_id):
                s1_records[rec.entity_id] = rec
            if scanned % 200_000 == 0:
                sub(f"  scanned {scanned:,}, selected {len(s1_records):,}")
            if len(s1_records) >= N_S1:
                break
        if len(s1_records) != N_S1:
            raise SystemExit(
                f"selected {len(s1_records):,} S1 records, expected {N_S1:,}")
        atomic_write_pickle(cache, s1_records)
    check("S1 subset size", len(s1_records) == N_S1, f"{len(s1_records):,}")

    truth_cache = CKPT / TRUTH_CACHE
    cached_truth = (pickle.loads(truth_cache.read_bytes())
                    if (not args.force and truth_cache.is_file()) else None)
    if isinstance(cached_truth, dict) and set(cached_truth) == set(s1_records):
        truth = cached_truth
    else:
        sub("streaming train_ground_truth.tsv, keeping only the subset's S1 ids")
        wanted = set(s1_records)
        truth = {}
        with train_path(TRUTH_FILE).open(
                "r", encoding="utf-8", errors="replace", newline="") as fh:
            fh.readline()
            for line in fh:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 2:
                    continue
                entity = parts[0].strip()
                if entity not in wanted:
                    continue
                truth[entity] = frozenset(
                    token for token in
                    (t.strip() for t in parts[1].split(",")) if token)
        missing = set(s1_records) - set(truth)
        check("every S1 entity has a ground-truth row", not missing,
              f"{len(missing):,} missing" if missing else "")
        atomic_write_pickle(truth_cache, truth)

    total_true = sum(len(v) for v in truth.values())
    expect("total true pairs for the subset", total_true, EXPECT_TOTAL_TRUE)
    step(f"  STAGE 2 done in {human(time.perf_counter() - started)}")
    return s1_records, truth


# ===========================================================================
# STAGE 3 - BLOCKING
# ===========================================================================
#: A candidate pair on disk: ``a`` is the S1 position, ``b`` the S2/S3 handle,
#: ``y`` the ground-truth label. Nine bytes per pair, so the union arm's 7.58M
#: pairs are 68 MB rather than the ~180 MB a pair of strings costs.
PAIR_DTYPE = np.dtype([("a", "<i4"), ("b", "<i4"), ("y", "i1")])

#: Rows buffered before a block is flushed to a numpy array. One flat Python
#: list of 15.2M ints converted at the end would peak around 600 MB for the list
#: alone; flushing keeps the peak at one block plus the finished array.
PAIR_BLOCK = 500_000


def write_npy(path: Path, array: np.ndarray) -> None:
    """Write a real ``.npy`` file atomically, so ``np.load`` can memory-map it."""
    import io

    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    atomic_write_bytes(path, buffer.getvalue())


class PairAccumulator:
    """Collect candidate pairs into int32 blocks, then one sorted array.

    Rows end up sorted by candidate handle. That ordering is what lets Stage 4
    serve *both* arms from a single sequential pass over S2 and S3 while holding
    exactly one candidate record in memory at a time - the alternative, indexing
    6.4M preprocessed records by id to allow random access, costs well over a
    gigabyte for the dictionary alone.
    """

    def __init__(self, block: int = PAIR_BLOCK) -> None:
        self.block = max(1, int(block))
        self._parts: list[np.ndarray] = []
        self._a: list[int] = []
        self._b: list[int] = []
        self._y: list[int] = []
        self.count = 0
        self.n_true = 0

    def add(self, a: int, b: int, label: int) -> None:
        self._a.append(a)
        self._b.append(b)
        self._y.append(label)
        self.count += 1
        if label:
            self.n_true += 1
        if len(self._a) >= self.block:
            self._flush()

    def _flush(self) -> None:
        if not self._a:
            return
        part = np.empty(len(self._a), dtype=PAIR_DTYPE)
        part["a"] = self._a
        part["b"] = self._b
        part["y"] = self._y
        self._parts.append(part)
        self._a, self._b, self._y = [], [], []

    def finish(self) -> np.ndarray:
        """The accumulated pairs as one array sorted by ``(b, a)``."""
        self._flush()
        if not self._parts:
            return np.zeros(0, dtype=PAIR_DTYPE)
        pairs = (self._parts[0] if len(self._parts) == 1
                 else np.concatenate(self._parts))
        # lexsort's last key is the primary one: group by candidate handle
        # first, then by S1 position, so each candidate's rows are contiguous.
        return pairs[np.lexsort((pairs["a"], pairs["b"]))]


def stage3_candidates(args, s1_records: dict) -> dict:
    """Build both candidate arms and write them as packed arrays.

    Deliberately atomic rather than checkpointed mid-build: a half-built index
    cannot be validated or resumed, and the honest cost of losing this stage is
    re-running it, which is minutes. Stages 1, 2, 4 and 5 all resume normally.
    """
    import gc

    started = time.perf_counter()
    step("STAGE 3/6  BLOCKING (arm A production, arm B production + A2)")

    truth = pickle.loads((CKPT / TRUTH_CACHE).read_bytes())
    s1_ids = list(s1_records)
    s1_pos = {entity_id: i for i, entity_id in enumerate(s1_ids)}

    s1_a2_keys = set()
    for rec in s1_records.values():
        key = a2_blocking_key(rec, BLOCKING_CONFIG)
        if key is not None:
            s1_a2_keys.add(key)
    sub(f"{len(s1_a2_keys):,} distinct A2 keys across the {len(s1_ids):,} S1 records")

    # Arm A uses upstream's own BlockingIndex, so it is exactly the committed
    # blocking behaviour, cap included. The A2 buckets are the experiment's own
    # dict and are capped the same way upstream caps at insertion (the first
    # CAP records to reach a key are the ones kept), so arm B differs from arm A
    # by the extra key family and by nothing else.
    index = B.BlockingIndex(config=BLOCKING_CONFIG)
    a2_buckets: dict[tuple, list[str]] = {}
    a2_capped: set[tuple] = set()
    handles: dict[str, int] = {}
    handle = 0
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        sub(f"indexing {fname}")
        n = 0
        for rec in M.iter_records(train_path(fname)):
            pace()
            n += 1
            handles[rec.entity_id] = handle
            handle += 1
            index.add(rec, M.source_of(rec.entity_id))
            key = a2_blocking_key(rec, BLOCKING_CONFIG)
            if key is not None and key in s1_a2_keys:
                bucket = a2_buckets.get(key)
                if bucket is None:
                    a2_buckets[key] = [rec.entity_id]
                elif len(bucket) < CAP:
                    bucket.append(rec.entity_id)
                else:
                    a2_capped.add(key)
            if n % 500_000 == 0:
                sub(f"    {n:,} rows indexed")
        sub(f"    {n:,} rows indexed")
    n_handles = handle
    index_stats = index.stats()
    sub(f"{index_stats.distinct_keys:,} production keys, "
        f"{index_stats.capped_groups:,} capped at {CAP:,}")
    sub(f"{len(a2_buckets):,} A2 buckets, "
        f"{len(a2_capped):,} capped at {CAP:,}")

    # The true-pair set is tiny (69k) next to the candidate pool (7.6M), so it
    # is built once by walking the ground truth rather than by testing 7.6M
    # pairs against it one at a time.
    true_pairs: set[tuple[int, int]] = set()
    for entity_id, matched in truth.items():
        a = s1_pos.get(entity_id)
        if a is None:
            continue
        for candidate_id in matched:
            b = handles.get(candidate_id)
            if b is not None:
                true_pairs.add((a, b))
    sub(f"{len(true_pairs):,} true pairs fall inside the blocked candidate space")

    prod = PairAccumulator()
    union = PairAccumulator()
    a2_only = PairAccumulator()

    for entity_id in s1_ids:
        pace()
        rec = s1_records[entity_id]
        a = s1_pos[entity_id]

        prod_hits: set[int] = set()
        for ref in index.candidates_for(rec):
            b = handles.get(ref.entity_id)
            if b is not None:
                prod_hits.add(b)

        a2_hits: set[int] = set()
        key = a2_blocking_key(rec, BLOCKING_CONFIG)
        if key is not None:
            for candidate_id in a2_buckets.get(key, ()):
                b = handles.get(candidate_id)
                if b is not None:
                    a2_hits.add(b)

        for b in prod_hits:
            label = 1 if (a, b) in true_pairs else 0
            prod.add(a, b, label)
            union.add(a, b, label)
        for b in a2_hits - prod_hits:
            label = 1 if (a, b) in true_pairs else 0
            union.add(a, b, label)
            a2_only.add(a, b, label)

    prod_pairs = prod.finish()
    union_pairs = union.finish()
    n_a2_only_true = a2_only.n_true
    n_a2_only = a2_only.count

    write_npy(CKPT / PROD_PAIRS, prod_pairs)
    write_npy(CKPT / UNION_PAIRS, union_pairs)

    meta = {
        "fingerprint": FINGERPRINT,
        "cap": CAP,
        "prefix4": PREFIX4,
        "n_s1": len(s1_ids),
        "n_handles": n_handles,
        "production_candidates": int(prod.count),
        "union_candidates": int(union.count),
        "a2_only_candidates": int(n_a2_only),
        "n_prod_true": int(prod.n_true),
        "n_union_true": int(union.n_true),
        "n_a2_only_true": int(n_a2_only_true),
        "production_keys": int(index_stats.distinct_keys),
        "capped_keys": int(index_stats.capped_groups),
        "a2_capped_keys": len(a2_capped),
        "true_pairs_in_candidate_space": len(true_pairs),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    atomic_write_json(CKPT / STAGE3_META, meta)

    check("production rows are a subset of the union",
          union.count == prod.count + n_a2_only,
          f"{prod.count:,} + {n_a2_only:,} = {prod.count + n_a2_only:,} "
          f"vs {union.count:,}")
    check("no duplicate (S1, candidate) pair in either arm",
          _unique_pairs(prod_pairs) and _unique_pairs(union_pairs))
    check("every stored handle is inside the indexed range",
          n_handles == 0 or int(union_pairs["b"].max(initial=-1)) < n_handles)

    # Release the inverted index, the A2 buckets and the 6.4M-entry id
    # dictionary *before* Stage 4 allocates its feature matrices. Peak memory
    # is the larger of the two, not their sum.
    del index, a2_buckets, a2_capped, handles, true_pairs, prod_pairs, union_pairs
    gc.collect()

    report_candidate_counts(meta)
    step(f"  STAGE 3 done in {human(time.perf_counter() - started)}")
    return meta


def _unique_pairs(pairs: np.ndarray) -> bool:
    """True when no ``(a, b)`` pair appears twice in a sorted pair array.

    A view is used rather than a set of 7.6M tuples, which would cost more
    memory than the array being checked.
    """
    if pairs.size < 2:
        return True
    same = (pairs["a"][1:] == pairs["a"][:-1]) & (pairs["b"][1:] == pairs["b"][:-1])
    return not bool(same.any())


def report_candidate_counts(meta: dict) -> None:
    """Compare the observed candidate counts with the reference counts."""
    expect("production candidates",
           meta.get("production_candidates"), EXPECT_PROD_CANDIDATES)
    expect("union candidates",
           meta.get("union_candidates"), EXPECT_UNION_CANDIDATES)
    expect("A2-only candidates",
           meta.get("a2_only_candidates"), EXPECT_A2_ONLY_CANDIDATES)
    expect("production true pairs",
           meta.get("n_prod_true"), EXPECT_PROD_TRUE)
    expect("A2-only true pairs",
           meta.get("n_a2_only_true"), EXPECT_A2_ONLY_TRUE)


def read_candidate_meta() -> dict | None:
    """Load Stage 3 metadata, but only if it still matches what is on disk.

    Every field that could have been written by a different configuration, a
    different cap, or an interrupted write is re-checked against the arrays
    themselves. A checkpoint that passes this has the counts it claims.
    """
    meta = read_json(CKPT / STAGE3_META, default=None)
    if not isinstance(meta, dict):
        return None
    for key in ("fingerprint", "production_candidates", "union_candidates",
                "n_prod_true", "n_union_true", "a2_only_true", "n_handles"):
        if meta.get(key) is None:
            return None
    if meta["fingerprint"] != FINGERPRINT:
        warn("candidate metadata is from a different configuration; rebuilding")
        return None
    if not (CKPT / PROD_PAIRS).is_file() or not (CKPT / UNION_PAIRS).is_file():
        return None
    try:
        prod = np.load(CKPT / PROD_PAIRS, mmap_mode="r", allow_pickle=False)
        union = np.load(CKPT / UNION_PAIRS, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return None
    if prod.dtype != PAIR_DTYPE or union.dtype != PAIR_DTYPE:
        warn("candidate arrays have an unexpected layout; rebuilding")
        return None
    if int(prod.size) != int(meta["production_candidates"]):
        return None
    if int(union.size) != int(meta["union_candidates"]):
        return None
    if int(np.count_nonzero(prod["y"])) != int(meta["n_prod_true"]):
        return None
    if int(np.count_nonzero(union["y"])) != int(meta["n_union_true"]):
        return None
    # A truncated array keeps the right shape and label count, so the tail
    # handle is checked too: a real run never points past the source rows.
    n_handles = int(meta["n_handles"])
    if n_handles > 0:
        if int(union["b"][-1]) >= n_handles or int(prod["b"][-1]) >= n_handles:
            return None
        if int(union["a"].max(initial=-1)) >= int(meta["n_s1"]):
            return None
    return meta


# ===========================================================================
# STAGE 4 - FEATURES
# ===========================================================================
def _open_features(path: Path, n_rows: int) -> np.ndarray:
    """Open the feature matrix for memory-mapped writing, recreating if stale.

    The matrix is its own checkpoint: writes land in the file, so re-running the
    stage reopens it and continues rather than recomputing what is already
    there. A shape or dtype that does not match the current run is discarded,
    because a half-sized matrix is worse than no matrix.
    """
    if path.is_file():
        try:
            existing = np.lib.format.open_memmap(path, mode="r+")
            if (existing.shape == (n_rows, N_FEATURES)
                    and existing.dtype == np.float32):
                return existing
            del existing
        except (OSError, ValueError):
            pass
        path.unlink(missing_ok=True)
    matrix = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(n_rows, N_FEATURES))
    return matrix


def _fill_run(matrix: np.ndarray, a_col: np.ndarray, lo: int, hi: int,
              record, s1_by_pos: list) -> None:
    """Featurize rows ``[lo, hi)`` of ``matrix`` against one candidate record.

    Filled through a small buffer rather than one ``matrix[i] = ...`` per row,
    because a per-row assignment into a memory-mapped file is a separate page
    fault each time.
    """
    block = np.empty((hi - lo, N_FEATURES), dtype=np.float32)
    for i in range(lo, hi):
        block[i - lo] = M.featurize(s1_by_pos[a_col[i]], record)
    matrix[lo:hi] = block


def _matrix_health(matrix: np.ndarray, chunk: int = 500_000) -> tuple[int, int]:
    """Count non-finite values and all-zero rows, reading in chunks.

    An all-zero row would mean a pair was never featurized - a freshly created
    memmap is zero-filled, so this is the check that catches an interrupted
    pass that was then marked complete.
    """
    nonfinite = 0
    empty = 0
    total = int(matrix.shape[0])
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        block = np.asarray(matrix[start:stop])
        nonfinite += int(np.count_nonzero(~np.isfinite(block)))
        empty += int(np.count_nonzero(~np.any(block, axis=1)))
    return nonfinite, empty


def stage4_features(args, meta: dict, s1_records: dict) -> None:
    """Compute the 24 features for every candidate row, in both arms.

    One pass over S2 and S3 fills both matrices at once. Each candidate row set
    is contiguous because Stage 3 sorted by handle, so the pass never needs more
    than the current record in memory and never seeks.

    Resumable at a record boundary: the state file records how many candidate
    handles have been consumed, and re-running re-parses the skipped rows
    without re-featurizing them.
    """
    started = time.perf_counter()
    step("STAGE 4/6  FEATURES (24 per pair, both arms in one pass)")

    s1_ids = list(s1_records)
    s1_by_pos = [s1_records[entity_id] for entity_id in s1_ids]

    prod = np.load(CKPT / PROD_PAIRS, mmap_mode="r", allow_pickle=False)
    union = np.load(CKPT / UNION_PAIRS, mmap_mode="r", allow_pickle=False)
    prod_b, prod_a = np.asarray(prod["b"]), np.asarray(prod["a"])
    union_b, union_a = np.asarray(union["b"]), np.asarray(union["a"])
    n_prod, n_union = int(prod_b.size), int(union_b.size)

    matrix_prod = _open_features(CKPT / FEATURES_PROD, n_prod)
    matrix_union = _open_features(CKPT / FEATURES_UNION, n_union)

    state_path = CKPT / FEATURE_STATE
    state = read_json(state_path, default=None)
    resume_at = 0
    if (not args.force and isinstance(state, dict)
            and state.get("fingerprint") == FINGERPRINT
            and state.get("n_prod") == n_prod
            and state.get("n_union") == n_union
            and state.get("n_features") == N_FEATURES):
        resume_at = int(state.get("handle", 0))
    if resume_at:
        sub(f"resuming after candidate handle {resume_at:,}")

    lo_prod = int(np.searchsorted(prod_b, resume_at, side="left"))
    lo_union = int(np.searchsorted(union_b, resume_at, side="left"))
    if resume_at and (lo_prod or lo_union):
        sub(f"rows already done: production {lo_prod:,}, union {lo_union:,}")

    handle = 0
    filled = 0
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        if lo_prod >= n_prod and lo_union >= n_union:
            break
        for record in M.iter_records(train_path(fname)):
            pace()
            if handle < resume_at:
                handle += 1
                continue
            hi_prod = lo_prod
            while hi_prod < n_prod and prod_b[hi_prod] == handle:
                hi_prod += 1
            hi_union = lo_union
            while hi_union < n_union and union_b[hi_union] == handle:
                hi_union += 1
            if hi_prod > lo_prod:
                _fill_run(matrix_prod, prod_a, lo_prod, hi_prod, record, s1_by_pos)
                filled += hi_prod - lo_prod
                lo_prod = hi_prod
            if hi_union > lo_union:
                _fill_run(matrix_union, union_a, lo_union, hi_union, record, s1_by_pos)
                filled += hi_union - lo_union
                lo_union = hi_union
            handle += 1
            if handle % 500_000 == 0:
                sub(f"  handle {handle:,}, rows featurized this pass {filled:,}")
            if lo_prod >= n_prod and lo_union >= n_union:
                break

    if lo_prod != n_prod or lo_union != n_union:
        raise SystemExit(
            f"candidate handles outran the source files: production "
            f"{lo_prod:,}/{n_prod:,}, union {lo_union:,}/{n_union:,}. "
            f"Re-run without --force once the data is complete.")

    matrix_prod.flush()
    matrix_union.flush()
    atomic_write_json(state_path, {
        "fingerprint": FINGERPRINT,
        "handle": handle,
        "n_prod": n_prod,
        "n_union": n_union,
        "n_features": N_FEATURES,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    })

    check("every candidate row was featurized",
          lo_prod == n_prod and lo_union == n_union,
          f"production {n_prod:,}, union {n_union:,}")
    for name, matrix in (("production", matrix_prod), ("union", matrix_union)):
        bad, empty = _matrix_health(matrix)
        check(f"{name} features are all finite", bad == 0, f"{bad} non-finite")
        check(f"{name} features have no empty row", empty == 0, f"{empty} all-zero")
    step(f"  STAGE 4 done in {human(time.perf_counter() - started)}")


# ===========================================================================
# STAGE 5 - MODEL
# ===========================================================================
def _validation_mask(a_col: np.ndarray, s1_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The S1-level train/validation split, as a boolean mask.

    ``matching.split_by_reference`` is the public API for this, but it takes
    ``LabelledPair`` objects: materialising 7.6M of them would cost more memory
    than the feature matrix itself. The hash bucket test inside it is applied
    directly to the 20,000 S1 positions instead, which is the same computation
    with the same salt and the same result - and
    :func:`check_split_equivalence` proves that on real rows at run time.
    """
    cut = SPLIT_FRACTION * 10_000
    is_val_s1 = np.array(
        [M._split_bucket(entity_id, SPLIT_SALT) < cut for entity_id in s1_ids],
        dtype=bool)
    return is_val_s1, is_val_s1[a_col]


def check_split_equivalence(a_col: np.ndarray, s1_ids: list[str],
                            sample: int = 5_000) -> bool:
    """Confirm the fast split matches ``matching.split_by_reference`` exactly.

    The candidate id is irrelevant to the split - it only looks at the
    reference id - so synthetic pairs with a placeholder candidate are enough to
    compare the two implementations on the same rows.
    """
    total = int(a_col.size)
    if total == 0:
        return False
    idx = np.unique(np.linspace(0, total - 1, min(sample, total)).astype(np.int64))
    labelled = [
        M.LabelledPair(
            M.CandidatePair(
                B.RecordRef("S1", s1_ids[int(a_col[i])]),
                B.RecordRef("S2", "not-used-by-the-split")),
            0)
        for i in idx
    ]
    _, validation = M.split_by_reference(
        labelled, validation_fraction=SPLIT_FRACTION, salt=SPLIT_SALT)
    in_validation: set[str] = {item.pair.reference.entity_id for item in validation}
    cut = SPLIT_FRACTION * 10_000
    for i in idx:
        entity_id = s1_ids[int(a_col[i])]
        if (entity_id in in_validation) != (M._split_bucket(entity_id, SPLIT_SALT) < cut):
            return False
    return True


def _run_arm(name: str, matrix: np.ndarray, y: np.ndarray, a_col: np.ndarray,
             s1_ids: list[str]) -> dict:
    """Fit one arm on its own candidate pool and sweep thresholds."""
    import gc

    is_val_s1, is_val = _validation_mask(a_col, s1_ids)
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    y_train = y[train_idx]
    y_val = y[val_idx]
    sub(f"{name}: {train_idx.size:,} train / {val_idx.size:,} validation rows, "
        f"{int(y_train.sum()):,} / {int(y_val.sum()):,} positive")
    if y_train.sum() == 0 or y_train.sum() == y_train.size:
        raise SystemExit(f"{name}: training split has a single class; cannot fit")

    model = M.LogisticMatcher(C=1.0, class_weight=None, max_iter=1000,
                              solver="lbfgs", random_state=0)
    # A single-threaded environment cap plus the runtime clamp keep this from
    # fanning out across every core the container appears to have.
    x_train = np.asarray(matrix[train_idx])
    x_val = np.asarray(matrix[val_idx])
    model.fit(x_train, y_train)
    reports = model.report(x_val, y_val)
    best = M.best_validation_threshold(reports)
    summary = model.summary
    del x_train, x_val
    gc.collect()

    return {
        "name": name,
        "candidates": int(y.size),
        "train_rows": int(train_idx.size),
        "train_positive": int(y_train.sum()),
        "validation_rows": int(val_idx.size),
        "validation_positive": int(y_val.sum()),
        "validation_s1": int(is_val_s1.sum()),
        "train_s1": int((~is_val_s1).sum()),
        "best_threshold": float(best.threshold),
        "best_f_beta": float(best.f_beta),
        "precision": float(best.precision),
        "recall": float(best.recall),
        "true_positives": int(best.true_positives),
        "false_positives": int(best.false_positives),
        "false_negatives": int(best.false_negatives),
        "n_iter": int(summary.n_iter) if summary else None,
        "converged": bool(summary.converged) if summary else None,
        "coefficients": model.coefficients(),
        "sweep": [
            {"threshold": float(r.threshold), "precision": float(r.precision),
             "recall": float(r.recall), "f_beta": float(r.f_beta),
             "predicted_positives": int(r.predicted_positives)}
            for r in reports
        ],
    }


def load_model_results(n_prod: int, n_union: int) -> dict | None:
    """Load cached model results only when they match the current inputs."""
    results = read_json(CKPT / MODEL_RESULTS, default=None)
    if not isinstance(results, dict):
        return None
    if results.get("fingerprint") != FINGERPRINT:
        return None
    if results.get("n_prod") != n_prod or results.get("n_union") != n_union:
        return None
    arms = results.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        return None
    for arm in arms:
        best = arm.get("best")
        if not isinstance(best, dict) or "f_beta" not in best:
            return None
    return results


def stage5_model(args, meta: dict) -> dict:
    """Fit both arms and cache the results."""
    started = time.perf_counter()
    step("STAGE 5/6  MODEL (LogisticRegression, F0.5 threshold sweep)")

    n_prod = int(meta["production_candidates"])
    n_union = int(meta["union_candidates"])
    if not args.force:
        cached = load_model_results(n_prod, n_union)
        if cached is not None:
            sub("model results match this configuration; reusing them")
            step(f"  STAGE 5 done in {human(time.perf_counter() - started)} (cached)")
            return cached

    s1_records = pickle.loads((CKPT / S1_RECORDS).read_bytes())
    s1_ids = list(s1_records)
    prod = np.load(CKPT / PROD_PAIRS, mmap_mode="r", allow_pickle=False)
    union = np.load(CKPT / UNION_PAIRS, mmap_mode="r", allow_pickle=False)
    matrix_prod = np.load(CKPT / FEATURES_PROD, mmap_mode="r", allow_pickle=False)
    matrix_union = np.load(CKPT / FEATURES_UNION, mmap_mode="r", allow_pickle=False)

    prod_a = np.asarray(prod["a"])
    union_a = np.asarray(union["a"])
    check("fast split agrees with matching.split_by_reference",
          check_split_equivalence(prod_a, s1_ids))

    arms = [
        _run_arm("A: production", matrix_prod,
                 np.asarray(prod["y"]).astype(np.int8), prod_a, s1_ids),
        _run_arm("B: production + A2", matrix_union,
                 np.asarray(union["y"]).astype(np.int8), union_a, s1_ids),
    ]
    arm_a, arm_b = arms
    results = {
        "fingerprint": FINGERPRINT,
        "n_prod": n_prod,
        "n_union": n_union,
        "beta": BETA,
        "split": {
            "level": "S1 entity",
            "validation_fraction": SPLIT_FRACTION,
            "salt": SPLIT_SALT,
            "train_s1": arm_a["train_s1"],
            "validation_s1": arm_a["validation_s1"],
        },
        "arms": [
            {
                "name": arm["name"],
                "candidates": arm["candidates"],
                "train_rows": arm["train_rows"],
                "train_positive": arm["train_positive"],
                "validation_rows": arm["validation_rows"],
                "validation_positive": arm["validation_positive"],
                "n_iter": arm["n_iter"],
                "converged": arm["converged"],
                "best": {
                    "threshold": arm["best_threshold"],
                    "f_beta": arm["best_f_beta"],
                    "precision": arm["precision"],
                    "recall": arm["recall"],
                    "true_positives": arm["true_positives"],
                    "false_positives": arm["false_positives"],
                    "false_negatives": arm["false_negatives"],
                },
                "coefficients": arm["coefficients"],
                "sweep": arm["sweep"],
            }
            for arm in arms
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    atomic_write_json(CKPT / MODEL_RESULTS, results)
    step(f"  STAGE 5 done in {human(time.perf_counter() - started)}")
    return results


# ===========================================================================
# STAGE 6 - REPORT
# ===========================================================================
def _fmt(value, suffix: str = "") -> str:
    return "unavailable" if value is None else f"{int(value):,}{suffix}"


def stage6_report(args, meta: dict, results: dict, s1_records: dict) -> None:
    """Write the human-readable report and the machine-readable summary."""
    started = time.perf_counter()
    step("STAGE 6/6  REPORT")

    txt_path = OUT_DIR / REPORT_TXT
    json_path = OUT_DIR / REPORT_JSON
    if (not args.force and txt_path.is_file() and json_path.is_file()
            and not args.rerun_report):
        cached = read_json(json_path, default=None)
        if isinstance(cached, dict) and cached.get("fingerprint") == FINGERPRINT:
            sub("report is current; re-run with --rerun-report to rebuild")
            step(f"  STAGE 6 done in {human(time.perf_counter() - started)} (cached)")
            print()
            print(txt_path.read_text(encoding="utf-8"))
            return
        sub("cached report is from a different configuration; rebuilding")

    truth = pickle.loads((CKPT / TRUTH_CACHE).read_bytes())
    total_true = sum(len(matched) for matched in truth.values())
    n_s1 = len(s1_records)
    prod_cand = int(meta["production_candidates"])
    union_cand = int(meta["union_candidates"])
    prod_true = observed(meta, "n_prod_true", EXPECT_PROD_TRUE)
    a2_true = observed(meta, "n_a2_only_true", EXPECT_A2_ONLY_TRUE)
    arm_a, arm_b = results["arms"]

    prod_recall = (None if prod_true is None
                   else prod_true / max(1, total_true))
    union_recall = (None if (prod_true is None or a2_true is None)
                    else (prod_true + a2_true) / max(1, total_true))

    blocking = {
        "production": {
            "candidates": prod_cand,
            "candidates_per_s1": prod_cand / max(1, n_s1),
            "true_pairs": prod_true,
            "recall": prod_recall,
        },
        "production+A2": {
            "candidates": union_cand,
            "candidates_per_s1": union_cand / max(1, n_s1),
            "true_pairs": (None if prod_true is None or a2_true is None
                           else prod_true + a2_true),
            "recall": union_recall,
        },
        "a2_only_additions": union_cand - prod_cand,
    }

    lines: list[str] = []
    add = lines.append
    rule = "=" * 78
    add(rule)
    add("BUSINESS ENTITY RESOLUTION - 20k S1 MATCHING A/B")
    add("Arm A = production blocking    Arm B = production blocking + A2")
    add(rule)
    add("")
    add("1. CONTROLS")
    add(f"   S1 records in the subset            {n_s1:,}")
    add(f"   features per pair                   {N_FEATURES}")
    add(f"   production blocking                 name_2tok + addr_ht0 + "
        f"addr_ht1, cap {CAP}")
    add(f"   A2 blocking                         original-order first 2 core "
        f"name tokens, prefix {PREFIX4}, country, cap {CAP}")
    add(f"   model                               LogisticRegression (C=1.0, "
        f"class_weight=None, max_iter=1000, solver=lbfgs, random_state=0, "
        f"features unscaled)")
    add(f"   split                               S1-level, "
        f"validation_fraction={SPLIT_FRACTION}, salt {SPLIT_SALT!r} "
        f"({results['split']['train_s1']:,} train / "
        f"{results['split']['validation_s1']:,} validation S1)")
    add(f"   metric                              F{BETA}")
    add(f"   total true pairs for these S1      {total_true:,}")
    add(f"   source fingerprint                  {FINGERPRINT}")
    add(f"   embedded source digests             {SOURCE_DIGEST_NOTE}")
    add("")
    add("2. BLOCKING")
    add(f"   {'arm':<18}{'candidates':>14}{'per S1':>10}{'true pairs':>13}"
        f"{'recall':>12}")
    for name, stats in blocking.items():
        if not isinstance(stats, dict):
            add(f"   {name:<18}{stats:>14,}")
            continue
        recall = stats["recall"]
        add(f"   {name:<18}{stats['candidates']:>14,}"
            f"{stats['candidates_per_s1']:>10.1f}"
            f"{_fmt(stats['true_pairs']):>13}"
            f"{'unavailable' if recall is None else format(recall, '.4%'):>12}")
    add(f"   A2-only additions                  {union_cand - prod_cand:,}")
    add(f"   A2-only true pairs recovered       {_fmt(a2_true)}")
    add(f"   A2 share of the candidate pool     "
        f"{(union_cand - prod_cand) / max(1, union_cand):.2%} of union rows "
        f"are A2-only")
    add(f"   production keys                    {meta.get('production_keys', 0):,}")
    add(f"   production keys truncated by cap   {meta.get('capped_keys', 0):,}")
    add(f"   A2 keys truncated by cap           {meta.get('a2_capped_keys', 0):,}")
    add("")
    add("3. MODEL (validation split, threshold swept)")
    add(f"   {'arm':<20}{'rows':>12}{'positive':>10}{'F0.5':>9}"
        f"{'P':>9}{'R':>9}{'thr':>7}")
    for arm in (arm_a, arm_b):
        best = arm["best"]
        add(f"   {arm['name']:<20}{arm['validation_rows']:>12,}"
            f"{arm['validation_positive']:>10,}"
            f"{best['f_beta']:>9.4f}{best['precision']:>9.4f}"
            f"{best['recall']:>9.4f}{best['threshold']:>7.1f}")
    add("")
    add("   threshold sweep (validation):")
    add(f"   {'arm':<20}{'thr':>6}{'F0.5':>10}{'P':>9}{'R':>9}{'predicted':>12}")
    for arm in (arm_a, arm_b):
        for row in arm["sweep"]:
            add(f"   {arm['name']:<20}{row['threshold']:>6.1f}"
                f"{row['f_beta']:>10.4f}{row['precision']:>9.4f}"
                f"{row['recall']:>9.4f}{row['predicted_positives']:>12,}")
    add("")
    add("4. TOP COEFFICIENTS (standardised? no - features are already 0..1)")
    for arm in (arm_a, arm_b):
        top = sorted(arm["coefficients"].items(), key=lambda kv: -abs(kv[1]))[:8]
        add(f"   {arm['name']}")
        for name, value in top:
            add(f"      {name:<30} {value:>10.4f}")
        add("")
    add("5. CHECKS")
    for name, ok, detail in CHECKS:
        add(f"   [{'ok  ' if ok else 'FAIL'}] {name}"
            f"{(' - ' + detail) if detail else ''}")
    add("")
    add("6. RESOURCES")
    add(f"   blas threads (env, pre-import)  {BLAS_THREADS}")
    for key, value in sorted(describe_threads().items()):
        add(f"     {key:<26} {value}")
    add(f"   runtime thread clamp            {THREAD_LIMIT}")
    add(f"   cpu affinity                     {CPUS_ALLOWED}")
    add(f"   duty cycle                       "
        f"{PACE.report() if PACE is not None else 'off'}")
    add(f"   total elapsed                    {human(time.perf_counter() - T0)}")
    add(rule)

    text = "\n".join(lines) + "\n"
    atomic_write_text(txt_path, text)
    atomic_write_json(json_path, {
        "fingerprint": FINGERPRINT,
        "n_s1": n_s1,
        "total_true_pairs": total_true,
        "blocking": blocking,
        "candidates": meta,
        "arms": results["arms"],
        "split": results["split"],
        "beta": BETA,
        "source_digest_check": SOURCE_DIGEST_NOTE,
        "resources": {
            "blas_threads": BLAS_THREADS,
            "blas_env": describe_threads(),
            "runtime_thread_limit": THREAD_LIMIT,
            "cpus_allowed": CPUS_ALLOWED,
            "priority": PRIORITY,
            "duty_cycle": PACE.report() if PACE is not None else None,
            "total_elapsed_seconds": round(time.perf_counter() - T0, 1),
        },
        "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in CHECKS],
    })
    step(f"  STAGE 6 done in {human(time.perf_counter() - started)}")
    print()
    print(text)


def print_finisher(meta: dict, results: dict) -> None:
    """The one-paragraph answer to the question the A/B was run to ask."""
    arm_a, arm_b = results["arms"]
    total_true = sum(len(matched) for matched in pickle.loads(
        (CKPT / TRUTH_CACHE).read_bytes()).values())
    a2_true = observed(meta, "n_a2_only_true", EXPECT_A2_ONLY_TRUE)
    prod_true = observed(meta, "n_prod_true", EXPECT_PROD_TRUE)
    print()
    print("=" * 78)
    print("WHAT THE A/B SHOWS")
    print("=" * 78)
    print(f"  A2 added {int(meta['union_candidates']) - int(meta['production_candidates']):,} "
          f"candidates "
          f"({(int(meta['union_candidates']) - int(meta['production_candidates'])) / max(1, int(meta['union_candidates'])):.2%} "
          f"of the union pool).")
    if a2_true is None:
        print("  A2-only true pairs: unavailable (Stage 3 metadata incomplete).")
    else:
        before = (prod_true / max(1, total_true)) if prod_true is not None else None
        after = ((prod_true + a2_true) / max(1, total_true)) if prod_true is not None else None
        print(f"  A2 recovered {_fmt(a2_true)} true pairs that production "
              f"blocking never retrieved.")
        if before is not None and after is not None:
            print(f"  blocking recall {before:.4%} -> {after:.4%} "
                  f"(+{(after - before) * 100:.2f} points).")
    print(f"  best validation F{BETA}: arm A {arm_a['best']['f_beta']:.4f} "
          f"at threshold {arm_a['best']['threshold']:.1f}, arm B "
          f"{arm_b['best']['f_beta']:.4f} at {arm_b['best']['threshold']:.1f}.")
    print(f"  report: {OUT_DIR / REPORT_TXT}")
    print("=" * 78)
    print()


# ===========================================================================
# ARGUMENTS AND ENTRY POINT
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="20k S1 matching A/B on Kaggle, resumable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=STAGES, default="all",
                        help="run one stage, or all of them in order")
    parser.add_argument("--dataset-zip", default=None,
                        help="explicit path to the uploaded archive")
    parser.add_argument("--work-dir", default=None, help="override the work root")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="override the checkpoint directory")
    parser.add_argument("--dataset-root", default=None,
                        help="override the extraction directory")
    parser.add_argument("--output-dir", default=None,
                        help="override where the report is written")
    parser.add_argument("--force", action="store_true",
                        help="ignore existing checkpoints and recompute")
    parser.add_argument("--rerun-report", action="store_true",
                        help="rebuild the report even if it looks current")
    parser.add_argument("--cpus", type=int, default=2,
                        help="CPUs to pin to; best effort, never fatal")
    parser.add_argument("--threads", type=int, default=1,
                        help="native thread cap (only effective before numpy loads)")
    parser.add_argument("--duty-fraction", type=float, default=0.85,
                        help="fraction of wall-clock time spent working")
    parser.add_argument("--duty-every", type=int, default=2000,
                        help="work units between duty-cycle measurements")
    return parser


def apply_path_overrides(args) -> None:
    """Let a command-line flag win over the environment variable."""
    for flag, variable in (
        ("work_dir", "MATCH_A2_WORK"),
        ("checkpoint_dir", "MATCH_A2_CKPT"),
        ("dataset_root", "MATCH_A2_DATASET"),
        ("output_dir", "MATCH_A2_OUT"),
    ):
        value = getattr(args, flag, None)
        if value:
            os.environ[variable] = value


def main(argv=None) -> int:
    global PACE, PRIORITY, CPUS_ALLOWED, THREAD_LIMIT, TRAIN_DIR, B

    parser = build_parser()
    # parse_known_args, not parse_args: inside a notebook sys.argv holds the
    # kernel's own flags, and a strict parser would abort on them.
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        warn(f"ignoring unrecognised arguments: {unknown}")
    if args.threads != BLAS_THREADS:
        warn(f"--threads {args.threads} could not take effect because numpy was "
             f"already imported; {BLAS_THREADS} is in force for this interpreter")

    apply_path_overrides(args)
    paths = resolve_paths()
    step("BUSINESS ENTITY RESOLUTION - 20k matching A/B")
    step(f"python {sys.version.split()[0]}   numpy {np.__version__}")
    try:
        import sklearn
        step(f"scikit-learn {sklearn.__version__}")
    except Exception:
        step("scikit-learn not importable yet")
    for key, value in paths.items():
        step(f"{key:<11} {value}")

    materialize_sources()
    load_project()
    step(f"embedded sources verified: {len(EMBEDDED_SOURCES)} modules, "
         f"{N_FEATURES} features, cap {CAP}, prefix {PREFIX4}")

    for directory in (CKPT, DATA_ROOT, OUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    PRIORITY = lower_process_priority()
    CPUS_ALLOWED = pin_to_cpus(args.cpus)
    THREAD_LIMIT = install_thread_limit(args.threads)
    step(f"priority: {PRIORITY}")
    step(f"cpus: {CPUS_ALLOWED} of {os.cpu_count()} logical")
    step(f"threads: env capped to {BLAS_THREADS} before numpy; "
         f"runtime clamp {THREAD_LIMIT}")
    if "unavailable" in THREAD_LIMIT or "failed" in THREAD_LIMIT:
        warn("no runtime thread clamp available; only the environment cap "
             "applies, and a pre-loaded BLAS would ignore it")

    PACE = Pacer(fraction=args.duty_fraction, tick_every=args.duty_every)

    order = list(STAGES[:-1]) if args.stage == "all" else [args.stage]
    try:
        s1_records = None
        meta = None
        results = None
        for stage in order:
            if stage == "dataset":
                if TRAIN_DIR is None:
                    stage1_dataset(args)
                continue
            if TRAIN_DIR is None:
                stage1_dataset(args)
            if s1_records is None:
                s1_records, _ = stage2_subset(args)
            if stage == "subset":
                continue
            if meta is None:
                meta = read_candidate_meta()
                if meta is None:
                    meta = stage3_candidates(args, s1_records)
            if stage == "candidates":
                continue
            # Stages 4-6 each validate their own checkpoint and return at once
            # when it is current, so running them in sequence is free. Keeping
            # the calls unconditional is what makes --stage model and --stage
            # report correct: both depend on Stage 4 having run.
            stage4_features(args, meta, s1_records)
            if stage == "features":
                continue
            if results is None:
                results = stage5_model(args, meta)
            if stage == "model":
                continue
            stage6_report(args, meta, results, s1_records)

        if args.stage == "all" and meta is not None and results is not None:
            print_finisher(meta, results)
        return 0
    except KeyboardInterrupt:
        step("interrupted; checkpointed work is kept, re-run this cell to resume")
        return 130
    finally:
        if PACE is not None:
            sub(f"duty cycle slept {PACE.report()['slept_seconds']}s "
                f"over {PACE.ticks:,} work units")


# ---------------------------------------------------------------------------
# Entry point. Runs when pasted into a notebook cell, where __name__ is
# "__main__", and when executed as a plain script.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _rc = main()
    if _rc not in (0, None):
        raise SystemExit(_rc)
