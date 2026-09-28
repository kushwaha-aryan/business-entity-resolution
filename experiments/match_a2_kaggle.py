"""Business Entity Resolution - 20k S1 matching A/B (production vs production+A2).

SELF-CONTAINED KAGGLE RUNNER. Paste this whole file into ONE Kaggle notebook
cell and run it. No arguments, no editing, no second file.

    /kaggle/input/**/..._student_resource.zip   (auto-discovered)
            |
            v
    /kaggle/working/dataset/                    (extraction target)
    /kaggle/working/checkpoints/match_a2/       (atomic, resumable)
    /kaggle/working/src/                        (embedded sources, written out)

Re-running the cell RESUMES. Every stage is skipped only when all of its files
are present AND pass validation AND carry a matching run fingerprint. If a
Kaggle session dies, paste the cell again.

Methodology
-----------
Imported verbatim from the project ``src`` package, which is embedded below as
exact bytes and written to ``/kaggle/working/src/`` on startup. Nothing in the
methodology is reimplemented, and no project file is modified:

    20,000 S1 subset     blake2b(entity_id, key=b'exp20k-s1', 8) % 110 == 0
    production blocking  name_2tok + addr_ht0 + addr_ht1, max_group_size 1000
    A2 blocking          ("A2_name_2tok_order", country,
                          name_core[0][:4], name_core[1][:4]), cap 1000
    features             the existing 24 pairwise features, float32
    model                LogisticMatcher: lbfgs, C=1.0, class_weight=None,
                         max_iter=1000, random_state=0, unscaled features
    split                S1-level, validation_fraction=0.2,
                         salt 'exp20k-split-v1'
    metric               F0.5, threshold chosen on validation

No external data, no lookup APIs, no geocoding, no embeddings, no LLM calls.
The competition test split is never read.

CPU / memory controls
---------------------
Kaggle has no thermal sensor and a fixed CPU quota, so the controls here are
about staying inside the quota and not being OOM-killed:

  1. BLAS/OpenMP capped to 1 thread via the environment, set BEFORE numpy is
     imported, AND clamped again at runtime through ``threadpoolctl`` if it is
     installed. The env vars alone are not a guarantee: they are read when a
     BLAS library loads, and if numpy or a backend is already loaded in this
     process the cap silently does nothing.
  2. ``os.nice(10)`` so the platform scheduler deprioritises this kernel.
  3. ``os.sched_setaffinity`` where the sandbox permits it. This NEVER aborts:
     Kaggle's cgroup may forbid CPU pinning, and refusing to run would be worse
     than running unpinned. A warning is printed instead.
  4. A wall-clock duty cycle in the CPU-bound loops.

Honest limit: a single long C call cannot be interrupted. During the
logistic-regression fit the only real control is the 1-thread clamp. The final
report says so rather than claiming protection that does not exist.

Memory notes
------------
Peak memory is dominated by two structures: the blocking index over all 10.3M
S2+S3 records, and the candidate-id lookup rebuilt on disk. Three changes
reduce the peak without changing a single computed value:

  * the blocking index and the A2 bucket table are released BEFORE the
    candidate arrays are materialised, because they are dead by then;
  * Stage 4 builds the candidate lookup one source at a time instead of both;
  * Stage 5 scans the feature matrix for NaN/inf in chunks rather than
    materialising the whole thing at once.
"""
from __future__ import annotations

import os
import sys


# ---------------------------------------------------------------------------
# BLAS thread cap. This MUST happen before numpy is imported anywhere, or the
# logistic-regression fit fans out across every available core.
# ---------------------------------------------------------------------------
def preparse_int_flag(argv, name, default):
    """Read ``--name N`` / ``--name=N`` out of argv before numpy exists."""
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return default
        if arg.startswith(name + "="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return default
    return default


def cap_blas_threads(n):
    value = str(max(1, int(n)))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = value


BLAS_THREADS = preparse_int_flag(sys.argv[1:], "--threads", 1)
cap_blas_threads(BLAS_THREADS)

import argparse                                            # noqa: E402
import dataclasses                                         # noqa: E402
import gc                                                  # noqa: E402
import hashlib                                             # noqa: E402
import json                                                # noqa: E402
import pickle                                              # noqa: E402
import shutil                                              # noqa: E402
import threading                                           # noqa: E402
import time                                                # noqa: E402
import zipfile                                             # noqa: E402
from collections import deque                             # noqa: E402
from pathlib import Path                                  # noqa: E402

import numpy as np                                         # noqa: E402

IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")

# ===========================================================================
# EMBEDDED PROJECT SOURCES - byte-exact copies of the project's src package.
# Raw triple-single-quoted strings: the project sources contain no ''' and do
# not end in a backslash, so r'''...''' reproduces them exactly. Each block is
# checked against its SHA-256 at startup.
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

@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    """``name_2tok`` key, or ``None`` when the name has no core token.

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

    ``F_beta`` is ``(1 + b^2) * P * R / (b^2 * P + R)``. With ``beta=0.5`` that
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
        return dict(zip(FEATURE_NAMES, (float(v) for v in self._model.coef_[0])))

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

EMBEDDED_SHA256 = {
    "preprocessing.py":
        "03528872cbfbaa1c15fa289bcf14526c486986438f88cd55806a966cfd5d3f99",
    "blocking.py":
        "a6feaef3953795de81fd337b8d7570e1e2940e3301a2670e19f5094789d25338",
    "features.py":
        "5568d645321a12983b7dbfb86a23b2d7b155d24cd66d4700ec0b0026b5429c57",
    "matching.py":
        "cc00141145e89c9ab9e11507d18dc14570492875b8bc96b7644084225743b6f2",
}

# ===========================================================================
# PATHS
# ===========================================================================
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")
ZIP_NAME_HINT = "student_resource"

WORK = KAGGLE_WORKING
CKPT = WORK / "checkpoints" / "match_a2"
DATA_ROOT = WORK / "dataset"
SRC_ROOT = WORK / "src"
OUT_DIR = CKPT
ZIP_ENV = "MATCH_A2_ZIP"
WORK_ENV = "MATCH_A2_WORK"
CKPT_ENV = "MATCH_A2_CKPT"
DATASET_ENV = "MATCH_A2_DATASET"
OUT_ENV = "MATCH_A2_OUT"

T0 = time.perf_counter()
CHECKS = []
GUARD = None
PACE = None
PRIORITY = "not set"
CPUS_ALLOWED = "all"
THREAD_LIMIT = "not applied"
DATASET_FP = None
FINGERPRINT = "unknown"
TRAIN_DIR = None


def step(msg):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


def sub(msg):
    print(f"           {msg}", flush=True)


def warn(msg):
    print(f"           [WARN] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"           [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def expect(name, got, want):
    """Sanity control against the completed local blocking A/B. Never fatal."""
    if got == want:
        check(name, True, f"{got:,} == {want:,}")
    else:
        CHECKS.append((name, False, f"{got:,} != {want:,}"))
        print(f"           [DIFF] {name}: got {got:,}, expected {want:,} "
              f"(delta {got - want:+,}). Logged, not fatal.", flush=True)


def human(seconds):
    seconds = int(seconds)
    return (f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m"
            f"{seconds % 60:02d}s")


def guard_tick():
    if PACE is not None:
        PACE.tick()


def guard_check(force=True):
    if PACE is not None:
        PACE.check(force=force)


# ===========================================================================
# CONFIGURATION RESOLUTION
# ===========================================================================
def resolve_paths():
    """Kaggle-first, environment-overridable, no platform-specific literals."""
    global WORK, CKPT, DATA_ROOT, SRC_ROOT, OUT_DIR
    work_env = os.environ.get(WORK_ENV, "").strip()
    if work_env:
        WORK = Path(work_env).expanduser()
    elif KAGGLE_WORKING.is_dir():
        WORK = KAGGLE_WORKING
    else:
        WORK = Path.cwd() / "_match_a2_work"
    WORK = WORK.resolve()

    ckpt_env = os.environ.get(CKPT_ENV, "").strip()
    CKPT = Path(ckpt_env).expanduser() if ckpt_env else WORK / "checkpoints" / "match_a2"
    CKPT = CKPT.resolve()

    ds_env = os.environ.get(DATASET_ENV, "").strip()
    DATA_ROOT = Path(ds_env).expanduser() if ds_env else WORK / "dataset"
    DATA_ROOT = DATA_ROOT.resolve()

    out_env = os.environ.get(OUT_ENV, "").strip()
    OUT_DIR = Path(out_env).expanduser() if out_env else CKPT
    OUT_DIR = OUT_DIR.resolve()

    SRC_ROOT = WORK / "src"
    return {"work": WORK, "checkpoints": CKPT, "dataset": DATA_ROOT,
            "output": OUT_DIR, "sources": SRC_ROOT}


# ===========================================================================
# CPU CONTROLS
# ===========================================================================
def lower_process_priority():
    """Best effort, never fatal."""
    if IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel32.SetPriorityClass.restype = ctypes.c_int
            if kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000):
                return "below-normal (Windows)"
            return "unchanged (SetPriorityClass returned 0)"
        except Exception as exc:
            return f"unchanged ({exc.__class__.__name__})"
    try:
        os.nice(10)
        return f"nice={os.nice(0)} (POSIX)"
    except Exception as exc:
        return f"unchanged ({exc.__class__.__name__}: {exc})"


def pin_to_cpus(n_cpus):
    """Restrict the process to at most ``n_cpus`` logical processors.

    Deliberately WARN-AND-CONTINUE rather than abort. A Kaggle sandbox may
    forbid CPU pinning through its cgroup, and refusing to start in that case
    would be a worse outcome than running unpinned. The return value is a
    human-readable status for the report.
    """
    if n_cpus <= 0:
        return "unlimited (--cpus 0)"
    if IS_LINUX and hasattr(os, "sched_setaffinity"):
        try:
            allowed = sorted(os.sched_getaffinity(0))
            chosen = allowed[:max(1, min(n_cpus, len(allowed)))]
            os.sched_setaffinity(0, set(chosen))
            return f"sched_setaffinity={chosen} (of allowed {allowed})"
        except (AttributeError, OSError) as exc:
            return f"unavailable ({exc.__class__.__name__}: {exc}); continuing unpinned"
    if IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,
                                                        ctypes.c_size_t]
            kernel32.SetProcessAffinityMask.restype = ctypes.c_int
            mask = (1 << max(1, min(n_cpus, os.cpu_count() or 1))) - 1
            if kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(),
                                               ctypes.c_size_t(mask)):
                return f"SetProcessAffinityMask={bin(mask).count('1')} cpus"
            return "unavailable (SetProcessAffinityMask returned 0); continuing unpinned"
        except Exception as exc:
            return f"unavailable ({exc.__class__.__name__}); continuing unpinned"
    return "unsupported on this platform; continuing unpinned"


def install_thread_limit(n):
    """Clamp every already-loaded native thread pool to ``n`` threads.

    This is the control that actually binds. The environment variables set
    before the numpy import are only read when a BLAS library initialises, so
    they do nothing if a backend was already loaded. ``threadpoolctl`` reaches
    into the loaded libraries and forces the count down at runtime.
    """
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except Exception as exc:
        return (f"unavailable ({exc.__class__.__name__}: {exc}); "
                f"relying on environment variables only")
    try:
        limiter = threadpool_limits(limits=max(1, int(n)))
        globals()["_THREADPOOL_LIMITER"] = limiter      # keep it applied
        try:
            pools = ", ".join(
                f"{info.get('internal_api', '?')}={info.get('num_threads', '?')}"
                for info in threadpool_info())
        except Exception:
            pools = "pools not enumerable"
        return f"active ({pools})"
    except Exception as exc:
        return f"failed ({exc.__class__.__name__}: {exc})"


def describe_threads():
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
            "NUMBA_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys if k in os.environ}


# ===========================================================================
# THERMAL MONITORING - Linux sysfs only, no subprocess, no PowerShell
# ===========================================================================
def read_sysfs_temps():
    """Highest CPU-ish temperature in degrees C, or None if the sandbox has none."""
    root = Path("/sys/class/thermal")
    if not root.is_dir():
        return None
    best = None
    fallback = None
    try:
        zones = sorted(root.glob("thermal_zone*"))
    except OSError:
        return None
    for zone in zones:
        try:
            raw = (zone / "temp").read_text().strip()
        except OSError:
            continue
        try:
            celsius = int(raw) / 1000.0
        except ValueError:
            continue
        if not (-40.0 < celsius < 150.0):
            continue
        if fallback is None or celsius > fallback:
            fallback = celsius
        try:
            kind = (zone / "type").read_text().strip().lower()
        except OSError:
            kind = ""
        if any(token in kind for token in ("x86", "cpu", "k10temp", "coretemp",
                                           "tdie", "tdp", "package")):
            if best is None or celsius > best:
                best = celsius
    if best is not None:
        return best
    return fallback


class _SysfsSampler(threading.Thread):
    """Polls sysfs on a timer. Uses no subprocess and no new process at all."""

    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval = max(1, int(interval))
        self.readings = deque(maxlen=4096)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            value = read_sysfs_temps()
            if value is not None:
                self.readings.append(value)
            self._stop.wait(self.interval)

    def latest(self):
        return self.readings[-1] if self.readings else None

    def stop(self):
        self._stop.set()


class ThermalGuard:
    def __init__(self, pause_above, resume_below, abort_above,
                 sample_seconds=20, enabled=True):
        if resume_below >= pause_above:
            raise ValueError("resume_below must be below pause_above")
        if abort_above <= pause_above:
            raise ValueError("abort_above must be above pause_above")
        self.pause_above = float(pause_above)
        self.resume_below = float(resume_below)
        self.abort_above = float(abort_above)
        self.max_temp = None
        self.paused_seconds = 0.0
        self.pause_count = 0
        self.enabled = bool(enabled)
        self._sampler = None
        self.available = False
        if self.enabled:
            try:
                self._sampler = _SysfsSampler(sample_seconds)
                self._sampler.start()
                for _ in range(6):
                    time.sleep(0.5)
                    if self._sampler.latest() is not None:
                        break
                self.available = self._sampler.latest() is not None
            except Exception:
                self.available = False

    def temp(self):
        if self._sampler is None:
            return None
        value = self._sampler.latest()
        if value is not None:
            self.available = True
            if self.max_temp is None or value > self.max_temp:
                self.max_temp = value
        return value

    def check(self, force=False):
        if not self.enabled:
            return
        value = self.temp()
        if value is None:
            return
        if value < self.pause_above and not force:
            return
        if value >= self.abort_above:
            raise ThermalAbort(
                f"CPU {value:.1f} C reached the abort ceiling "
                f"{self.abort_above:.1f} C. Stopping. Checkpointed work is kept; "
                f"re-run this cell to resume.")
        self.pause_count += 1
        print(f"           [thermal] CPU {value:.1f} C >= "
              f"{self.pause_above:.1f} C; pausing until <= "
              f"{self.resume_below:.1f} C", flush=True)
        started = time.perf_counter()
        while True:
            if time.perf_counter() - started > 20 * 60:
                raise ThermalAbort(
                    f"still at/above {self.pause_above:.1f} C after 20 min of "
                    f"cooling. Stopping. Checkpointed work is kept.")
            time.sleep(5.0)
            value = self.temp()
            if value is None or value < self.resume_below:
                break
        waited = time.perf_counter() - started
        self.paused_seconds += waited
        print(f"           [thermal] resumed after {human(waited)} cooling",
              flush=True)

    def stop(self):
        if self._sampler is not None:
            self._sampler.stop()

    def report(self):
        return {
            "temperature_monitoring": self.available,
            "max_cpu_temp_c": self.max_temp,
            "pause_above_c": self.pause_above,
            "resume_below_c": self.resume_below,
            "abort_above_c": self.abort_above,
            "pause_events": self.pause_count,
            "total_paused_seconds": round(self.paused_seconds, 1),
        }


class ThermalAbort(RuntimeError):
    """Raised when the CPU is too hot to continue safely."""


# ===========================================================================
# DUTY CYCLE
# ===========================================================================
class DutyCycle:
    """Hold the process to a wall-clock duty cycle with short sleeps.

    Measures how long the thread actually worked between ticks and sleeps the
    proportional remainder, so a cheap loop and an expensive one get the same
    fraction of relief. A per-row counter would give the cheap loops far more
    relief than the expensive ones, which is the opposite of what is wanted.
    """

    def __init__(self, fraction=0.85, tick_every=2000, max_sleep=2.0):
        self.fraction = min(0.99, max(0.5, float(fraction)))
        self.tick_every = max(1, int(tick_every))
        self.max_sleep = max(0.0, float(max_sleep))
        self.slept_seconds = 0.0
        self._n = 0
        self._last = time.perf_counter()
        self._sample_every = 20

    def tick(self):
        self._n += 1
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
        if self._n % (self.tick_every * self._sample_every) == 0:
            guard_check(force=False)

    def report(self):
        return {"duty_fraction": self.fraction, "tick_every_rows": self.tick_every,
                "slept_seconds": round(self.slept_seconds, 1)}


# ===========================================================================
# SOURCE MATERIALISATION AND IMPORT
# ===========================================================================
def materialize_sources(allow_drift=False):
    """Write the embedded project sources to disk and verify them byte-exactly."""
    SRC_ROOT.mkdir(parents=True, exist_ok=True)
    (SRC_ROOT / "__init__.py").write_text(
        '"""Embedded verbatim from the project src/ package."""\n',
        encoding="utf-8")
    problems = []
    for name, text in EMBEDDED_SOURCES.items():
        data = text.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        want = EMBEDDED_SHA256[name]
        if digest != want and not allow_drift:
            problems.append((name, want, digest))
        (SRC_ROOT / name).write_bytes(data)
    if problems:
        print()
        print("=" * 78)
        print("EMBEDDED SOURCE MISMATCH - refusing to run")
        print("=" * 78)
        for name, want, got in problems:
            print(f"  {name}")
            print(f"    expected sha256 {want}")
            print(f"    actual   sha256 {got}")
        print()
        print("The copied source does not match the project file byte for byte,")
        print("so the methodology this run would use is not provably the same.")
        print("Re-copy the file from the project, or, only if you have confirmed")
        print("the difference is comments/docstrings, pass")
        print("    --allow-embedded-drift")
        print("=" * 78)
        raise SystemExit(3)
    if problems:
        warn("embedded source digest mismatch tolerated by "
             "--allow-embedded-drift; methodology is NOT provably identical")
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    if str(SRC_ROOT) in sys.path:
        sys.path.remove(str(SRC_ROOT))
    # Re-running this cell in one notebook session would otherwise keep the
    # already-imported src modules, so a re-extracted or corrected source
    # would silently not take effect. Purge them before the import below.
    for name in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[name]


# ===========================================================================
# EXPERIMENT CONSTANTS - these define the experiment. Do not tune them.
# ===========================================================================
N_S1 = 20_000
S1_SALT, S1_MOD = b"exp20k-s1", 110
SPLIT_FRACTION = 0.2
SPLIT_SALT = "exp20k-split-v1"
BETA = 0.5

EXPECT_PROD_CANDIDATES = 6_373_030
EXPECT_UNION_CANDIDATES = 7_580_413
EXPECT_A2_ONLY_CANDIDATES = 1_207_383
EXPECT_PROD_TRUE = 55_325
EXPECT_A2_ONLY_TRUE = 2_995
EXPECT_TOTAL_TRUE = 69_301

SOURCES = (("train_source2.tsv", "S2"), ("train_source3.tsv", "S3"))
REQUIRED_TRAIN_FILES = ("train_source1.tsv", "train_source2.tsv",
                        "train_source3.tsv", "train_ground_truth.tsv")

# Populated immediately after the src import below.
CFG = None
CAP = 0
PREFIX4 = 0
N_FEATURES = 0
FEATURE_NAMES = ()
M = None
BlockingIndex = None
CandidatePair = None
RecordRef = None
generate_candidates = None
featurize = None


def load_project():
    """Import the freshly written src package and bind what the stages need."""
    global M, CFG, CAP, PREFIX4, N_FEATURES, FEATURE_NAMES
    global BlockingIndex, CandidatePair, RecordRef, generate_candidates
    global featurize
    from src import matching as matching_module
    from src.blocking import (BlockingConfig, BlockingIndex as _Index,
                              CandidatePair as _Pair, RecordRef as _Ref,
                              generate_candidates as _generate)
    from src.features import FEATURE_NAMES as _names, featurize as _featurize

    M = matching_module
    CFG = BlockingConfig()
    CAP = CFG.max_group_size
    PREFIX4 = CFG.name_prefix_length
    FEATURE_NAMES = _names
    N_FEATURES = len(_names)
    BlockingIndex = _Index
    CandidatePair = _Pair
    RecordRef = _Ref
    generate_candidates = _generate
    featurize = _featurize


# ===========================================================================
# RUN FINGERPRINT - a checkpoint is reusable only if it was built by exactly
# this methodology against exactly this dataset.
# ===========================================================================
def dataset_fingerprint(archive):
    """Cheap identity for the archive: name, size, mtime.

    Deliberately NOT a content hash. The archive is ~1.1 GB and hashing it
    would burn more CPU than the run it is protecting.
    """
    try:
        stat = archive.stat()
    except OSError:
        return {"name": archive.name, "size": None, "mtime": None}
    return {"name": archive.name, "size": int(stat.st_size),
            "mtime": int(stat.st_mtime)}


def compute_fingerprint(dataset_fp):
    payload = {
        "n_s1": N_S1,
        "s1_salt": S1_SALT.decode(),
        "s1_mod": S1_MOD,
        "cap": CAP,
        "prefix4": PREFIX4,
        "split_fraction": SPLIT_FRACTION,
        "split_salt": SPLIT_SALT,
        "beta": BETA,
        "n_features": N_FEATURES,
        "feature_names": list(FEATURE_NAMES),
        "sources": [src for _, src in SOURCES],
        "embedded_sha256": EMBEDDED_SHA256,
        "dataset": dataset_fp,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


# ===========================================================================
# ATOMIC CHECKPOINT IO
# ===========================================================================
def _fsync_dir(directory):
    if os.name == "nt":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def _json_default(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, indent=2, default=_json_default)
                      + "\n")


def atomic_write_pickle(path, obj):
    atomic_write_bytes(path, pickle.dumps(obj, protocol=4))


def atomic_save_npy(path, array, allow_pickle=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        np.save(fh, array, allow_pickle=allow_pickle)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def discard(path, why):
    for candidate in (path, path.with_name(path.name + ".part")):
        try:
            if candidate.is_file():
                candidate.unlink()
                print(f"           [discard] {candidate.name} ({why})", flush=True)
        except OSError:
            pass


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ===========================================================================
# STAGE 1 - dataset
# ===========================================================================
def find_dataset_zip(explicit=""):
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_dir():
            zips = sorted(p for p in candidate.glob("*.zip") if p.is_file())
            if len(zips) == 1:
                return zips[0]
            found = sorted(p for p in candidate.rglob(f"*{ZIP_NAME_HINT}*.zip")
                           if p.is_file())
            if len(found) == 1:
                return found[0]
            raise FileNotFoundError(
                f"--dataset-zip {candidate} is a directory holding "
                f"{len(zips)} zip file(s); expected exactly one")
        if not candidate.is_file():
            raise FileNotFoundError(f"--dataset-zip {candidate} is not a file")
        return candidate

    env = os.environ.get(ZIP_ENV, "").strip()
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"{ZIP_ENV}={env} is not a file")

    search_roots = []
    if KAGGLE_INPUT.is_dir():
        search_roots.append(KAGGLE_INPUT)
    if DATA_ROOT.parent.is_dir():
        search_roots.append(DATA_ROOT.parent)
    search_roots.append(Path.cwd())

    listing = {}
    for root in search_roots:
        try:
            zips = sorted(p for p in root.rglob("*.zip") if p.is_file())
        except OSError:
            continue
        if not zips:
            continue
        listing[str(root)] = [p.name for p in zips]
        named = [p for p in zips if ZIP_NAME_HINT in p.name.lower()]
        if len(named) == 1:
            return named[0]
        if len(zips) == 1:
            return zips[0]

    raise FileNotFoundError(
        f"could not find a dataset zip containing '{ZIP_NAME_HINT}'.\n"
        f"Attach the competition dataset to this notebook so it appears under "
        f"{KAGGLE_INPUT}, or set {ZIP_ENV} to its full path, or pass "
        f"--dataset-zip.\nZips seen: {listing}")


def safe_extract(archive, dest):
    """Extract, refusing any member that resolves outside ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    written = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(
                    f"refusing zip member {member.filename!r}: it resolves to "
                    f"{target}, outside the extraction root {root}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
            written += 1
    return written


def find_train_dir(root):
    for candidate in (root, root / "train"):
        if all((candidate / n).is_file() for n in REQUIRED_TRAIN_FILES):
            return candidate
    frontier = [root]
    for _ in range(5):
        nxt = []
        for parent in frontier:
            try:
                children = sorted(p for p in parent.iterdir() if p.is_dir())
            except OSError:
                continue
            for child in children:
                if all((child / n).is_file() for n in REQUIRED_TRAIN_FILES):
                    return child
                nxt.append(child)
        frontier = nxt
    return None


def train_path(name):
    if "test" in name.lower():
        raise AssertionError(f"refusing non-train file: {name}")
    if TRAIN_DIR is None:
        raise RuntimeError("TRAIN_DIR is not set; run stage 1 first")
    path = (TRAIN_DIR / name).resolve()
    if TRAIN_DIR.resolve() not in path.parents:
        raise AssertionError(f"refusing path outside train/: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset file: {path}")
    return path


DATASET_MARKER = "dataset_ready.json"


def stage1_dataset(args):
    global TRAIN_DIR, DATASET_FP, FINGERPRINT
    started = time.perf_counter()
    step("STAGE 1/6  DATASET")
    CKPT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    archive = find_dataset_zip(args.dataset_zip)
    DATASET_FP = dataset_fingerprint(archive)
    FINGERPRINT = compute_fingerprint(DATASET_FP)
    try:
        sub(f"zip: {archive} ({archive.stat().st_size / 1e9:.2f} GB)")
    except OSError:
        sub(f"zip: {archive}")
    sub(f"extraction root: {DATA_ROOT}")
    sub(f"checkpoints: {CKPT}")

    marker = CKPT / DATASET_MARKER
    previous = read_json(marker)
    reuse = False
    if previous is None:
        if marker.is_file():
            discard(marker, "unreadable")
    elif previous.get("fingerprint") != FINGERPRINT:
        sub("checkpoint fingerprint does not match this run; re-extracting")
    elif args.force:
        sub("--force given; re-extracting")
    else:
        reuse = True

    if reuse and find_train_dir(DATA_ROOT) is not None:
        sub(f"already extracted for this exact configuration, reusing: "
            f"{TRAIN_DIR}")
    else:
        sub(f"extracting to {DATA_ROOT} - a few minutes")
        t = time.perf_counter()
        count = safe_extract(archive, DATA_ROOT)
        sub(f"extracted {count:,} entries in {human(time.perf_counter() - t)}")

    found = find_train_dir(DATA_ROOT)
    if found is None:
        listing = sorted(p.name for p in DATA_ROOT.glob("*"))[:20] \
            if DATA_ROOT.is_dir() else []
        raise FileNotFoundError(
            f"no folder under {DATA_ROOT} holds all of "
            f"{list(REQUIRED_TRAIN_FILES)}. Expected something like "
            f"{DATA_ROOT}/student_resource/dataset/train/. Found: {listing}")
    TRAIN_DIR = found
    for name in REQUIRED_TRAIN_FILES:
        path = train_path(name)
        size = path.stat().st_size
        if size <= 0:
            raise FileNotFoundError(f"dataset file is empty: {path}")
        sub(f"  {name:<26} {size / 1e6:>10.1f} MB")

    # The marker is written only after every required file validated, so an
    # interrupted extraction can never be mistaken for a finished one.
    atomic_write_json(marker, {
        "fingerprint": FINGERPRINT,
        "dataset": DATASET_FP,
        "train_dir": str(TRAIN_DIR),
        "files": {name: (TRAIN_DIR / name).stat().st_size
                  for name in REQUIRED_TRAIN_FILES},
    })
    check("all four training files present and non-empty", True, str(TRAIN_DIR))
    step(f"  STAGE 1 done in {human(time.perf_counter() - started)}")


# ===========================================================================
# STAGE 2 - S1 subset and ground truth
# ===========================================================================
def hashed(entity_id, salt=S1_SALT, mod=S1_MOD):
    digest = hashlib.blake2b(entity_id.encode("utf-8"), key=salt,
                             digest_size=8).digest()
    return int.from_bytes(digest, "big") % mod == 0


def a2_key(record):
    if len(record.name_core) < 2:
        return None
    return ("A2_name_2tok_order", record.country,
            record.name_core[0][:PREFIX4], record.name_core[1][:PREFIX4])


SUBSET_META = "stage2_meta.json"


def load_subset():
    subset_path = CKPT / "s1_records.pkl"
    truth_path = CKPT / "truth.pkl"
    meta = read_json(CKPT / SUBSET_META)
    if meta is None or not (subset_path.is_file() and truth_path.is_file()):
        return None
    if meta.get("fingerprint") != FINGERPRINT:
        sub("stage 2 checkpoint was built for a different configuration; "
            "rebuilding the subset")
        return None
    try:
        s1_records = pickle.loads(subset_path.read_bytes())
        truth = pickle.loads(truth_path.read_bytes())
    except Exception as exc:
        discard(subset_path, f"unpicklable: {exc}")
        discard(truth_path, f"unpicklable: {exc}")
        return None
    if len(s1_records) != N_S1:
        discard(subset_path, f"{len(s1_records):,} records, need {N_S1:,}")
        return None
    if not truth:
        discard(truth_path, "ground truth is empty")
        return None
    if not set(truth).issubset(set(s1_records)):
        sub("ground truth holds S1 ids outside the subset; rebuilding")
        return None
    return s1_records, truth


def stage2_subset(args):
    started = time.perf_counter()
    step("STAGE 2/6  S1 SUBSET + GROUND TRUTH")
    if not args.force:
        loaded = load_subset()
        if loaded is not None:
            s1_records, truth = loaded
            total = sum(len(v) for v in truth.values())
            sub(f"reusing validated subset: {len(s1_records):,} S1 records, "
                f"{len(truth):,} GT rows")
            check("S1 subset size", len(s1_records) == N_S1,
                  f"{len(s1_records):,}")
            expect("total true pairs", total, EXPECT_TOTAL_TRUE)
            step(f"  STAGE 2 done in {human(time.perf_counter() - started)} "
                 f"(cached)")
            return s1_records, truth

    sub(f"streaming S1, selecting blake2b(key={S1_SALT!r}) % {S1_MOD} == 0")
    s1_records = {}
    scanned = 0
    for rec in M.iter_records(train_path("train_source1.tsv")):
        scanned += 1
        if len(s1_records) < N_S1 and hashed(rec.entity_id):
            s1_records[rec.entity_id] = rec
        if scanned % 200_000 == 0:
            guard_tick()
            sub(f"  scanned {scanned:,}, selected {len(s1_records):,}")
        if len(s1_records) >= N_S1:
            break
    if len(s1_records) != N_S1:
        raise AssertionError(
            f"selected {len(s1_records):,} S1 records, expected {N_S1:,}")
    sub(f"  S1 scanned {scanned:,}, selected {len(s1_records):,}")
    check("S1 subset size", len(s1_records) == N_S1, f"{len(s1_records):,}")
    atomic_write_pickle(CKPT / "s1_records.pkl", s1_records)

    sub("streaming train_ground_truth.tsv, keeping only the subset's S1 ids")
    wanted = set(s1_records)
    truth = {}
    with train_path("train_ground_truth.tsv").open(
            "r", encoding="utf-8", errors="replace", newline="") as fh:
        fh.readline()
        rows = 0
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            entity = parts[0].strip()
            if entity not in wanted:
                continue
            truth[entity] = frozenset(p.strip() for p in parts[1].split(",")
                                      if p.strip())
            rows += 1
            if rows % 20_000 == 0:
                guard_tick()
    total_true = sum(len(v) for v in truth.values())
    sub(f"  {len(truth):,} GT rows, {total_true:,} true pairs")
    expect("total true pairs", total_true, EXPECT_TOTAL_TRUE)
    atomic_write_pickle(CKPT / "truth.pkl", truth)
    atomic_write_json(CKPT / SUBSET_META, {
        "fingerprint": FINGERPRINT, "n_s1": len(s1_records),
        "gt_rows": len(truth), "true_pairs": total_true,
    })
    step(f"  STAGE 2 done in {human(time.perf_counter() - started)}")
    return s1_records, truth


# ===========================================================================
# STAGE 3 - candidate generation
# ===========================================================================
def cand_paths(src):
    return CKPT / f"cand_{src}_ids.npy", CKPT / f"cand_{src}_packed.npy"


STAGE3_META = "stage3_meta.json"


def read_candidate_meta():
    """Validated stage-3 metadata, rebuilt from the arrays if it went missing.

    Returns None whenever the arrays are not complete, so the caller
    regenerates. Never raises for a missing file.
    """
    meta_path = CKPT / STAGE3_META
    rows_per_src = {}
    for _, src in SOURCES:
        ids_path, packed_path = cand_paths(src)
        if not (ids_path.is_file() and packed_path.is_file()):
            return None
        try:
            ids = np.load(ids_path, allow_pickle=True)
            packed = np.load(packed_path)
        except Exception as exc:
            discard(ids_path, f"unreadable: {exc}")
            discard(packed_path, f"unreadable: {exc}")
            return None
        if ids.shape[0] != packed.shape[0] or ids.ndim != 1 or packed.ndim != 1:
            discard(packed_path, f"shape {packed.shape} vs ids {ids.shape}")
            return None
        rows_per_src[src] = int(ids.shape[0])
        del ids, packed
        guard_tick()

    needed = ("production_candidates", "union_candidates", "n_s1", "true_pairs",
              "n_prod_true", "n_a2_only_true", "rows_per_src", "fingerprint")
    meta = read_json(meta_path)
    if meta is not None and all(k in meta for k in needed):
        if meta.get("fingerprint") == FINGERPRINT \
                and meta.get("rows_per_src") == rows_per_src \
                and sum(rows_per_src.values()) == meta.get("union_candidates"):
            return meta
        sub("stage 3 metadata is for a different configuration; rebuilding")

    # Rebuild the counts from the arrays themselves. Bit 0 of a packed row is
    # the production flag by construction, so the totals are derivable. The
    # per-arm true-pair counts are NOT derivable from the arrays, so they stay
    # None and are recomputed in stage 4 rather than guessed here.
    n_union = 0
    n_prod = 0
    for _, src in SOURCES:
        packed = np.load(cand_paths(src)[1], mmap_mode="r")
        n_union += int(packed.shape[0])
        for start in range(0, packed.shape[0], 2_000_000):
            block = np.asarray(packed[start:start + 2_000_000])
            n_prod += int(np.count_nonzero(block & 1))
            guard_tick()
        del packed
    meta = {
        "n_s1": N_S1,
        "true_pairs": None,
        "production_candidates": n_prod,
        "union_candidates": n_union,
        "a2_only_candidates": n_union - n_prod,
        "n_prod_true": None,
        "n_a2_only_true": None,
        "rows_per_src": rows_per_src,
        "fingerprint": FINGERPRINT,
        "reconstructed": True,
    }
    atomic_write_json(meta_path, meta)
    sub("rebuilt stage 3 metadata from the candidate arrays "
        "(per-arm true-pair counts will come from stage 4)")
    return meta


def report_candidate_counts(meta):
    prod = int(meta.get("production_candidates") or 0)
    union = int(meta.get("union_candidates") or 0)
    n_s1 = int(meta.get("n_s1") or N_S1)
    expect("production candidate count", prod, EXPECT_PROD_CANDIDATES)
    expect("union candidate count", union, EXPECT_UNION_CANDIDATES)
    expect("A2-only additions", union - prod, EXPECT_A2_ONLY_CANDIDATES)
    sub(f"  average candidates per S1: production "
        f"{prod / max(1, n_s1):.1f}, production+A2 "
        f"{union / max(1, n_s1):.1f}")
    if meta.get("n_prod_true") is not None:
        expect("production true pairs", int(meta["n_prod_true"]),
               EXPECT_PROD_TRUE)
        expect("A2-only true pairs", int(meta["n_a2_only_true"]),
               EXPECT_A2_ONLY_TRUE)


def stage3_candidates(args, s1_records):
    started = time.perf_counter()
    step("STAGE 3/6  CANDIDATE GENERATION")
    if not args.force:
        meta = read_candidate_meta()
        if meta is not None:
            sub("candidate arrays + metadata validated; skipping the full "
                "S2/S3 index rebuild")
            report_candidate_counts(meta)
            step(f"  STAGE 3 done in {human(time.perf_counter() - started)} "
                 f"(cached)")
            return meta

    s1_order = sorted(s1_records)
    s1_a2_keys = {a2_key(r) for r in s1_records.values()}
    s1_a2_keys.discard(None)
    sub(f"  {len(s1_a2_keys):,} distinct A2 keys across the subset")

    sub("streaming the FULL S2 + S3 into the production index and A2 buckets")
    index = BlockingIndex(CFG)
    a2_buckets = {}
    for fname, src in SOURCES:
        n = 0
        for rec in M.iter_records(train_path(fname)):
            n += 1
            index.add(rec, src)
            key = a2_key(rec)
            if key is not None and key in s1_a2_keys:
                bucket = a2_buckets.get(key)
                if bucket is None:
                    a2_buckets[key] = [RecordRef(src, rec.entity_id)]
                elif len(bucket) < CAP:
                    bucket.append(RecordRef(src, rec.entity_id))
            if n % 200_000 == 0:
                guard_tick()
                sub(f"    {src}: {n:,} rows indexed")
        sub(f"    {src}: {n:,} rows indexed")
    n_buckets = len(a2_buckets)
    sub(f"  {n_buckets:,} A2 buckets within cap {CAP}")
    guard_check(force=True)

    sub("querying both arms for all 20,000 S1 records")
    truth = pickle.loads((CKPT / "truth.pkl").read_bytes())
    want = {"S2": {}, "S3": {}}
    n_prod_seen = 0
    n_prod_true = 0
    n_a2_true = 0

    def _add(store, key, packed):
        current = store.get(key)
        if current is None:
            store[key] = packed
        elif type(current) is int:
            store[key] = [current, packed]
        else:
            current.append(packed)

    s1_index = {eid: i for i, eid in enumerate(s1_order)}
    for done, eid in enumerate(s1_order, 1):
        rec = s1_records[eid]
        base = s1_index[eid] * 2
        truth_set = truth.get(eid, frozenset())
        prod = {(c.candidate.source, c.candidate.entity_id)
                for c in generate_candidates(rec, index, "S1")}
        n_prod_seen += len(prod)
        for src, cid in prod:
            _add(want[src], cid, base | 1)
            if cid in truth_set:
                n_prod_true += 1
        key = a2_key(rec)
        if key is not None:
            for ref in a2_buckets.get(key, ()):
                if (ref.source, ref.entity_id) not in prod:
                    _add(want[ref.source], ref.entity_id, base)
                    if ref.entity_id in truth_set:
                        n_a2_true += 1
        guard_tick()
        if done % 2_000 == 0:
            sub(f"    queried {done:,}/{N_S1:,} S1")

    # ---- MEMORY: the index and the bucket table are dead from here on.
    # They are by far the largest structures in the run (a bucket per blocking
    # key over 10.3M records). Releasing them BEFORE materialising the
    # candidate arrays lowers peak memory substantially and changes no value.
    del index, a2_buckets
    gc.collect()
    sub("released the blocking index and A2 bucket table before writing arrays")
    guard_check(force=True)

    n_union = 0
    rows_per_src = {}
    for _, src in SOURCES:
        store = want[src]
        total = sum(1 if type(v) is int else len(v) for v in store.values())
        ids = np.empty(total, dtype=object)
        packed = np.empty(total, dtype=np.int64)
        i = 0
        for cid, value in store.items():
            if type(value) is int:
                ids[i] = cid
                packed[i] = value
                i += 1
            else:
                for pk in value:
                    ids[i] = cid
                    packed[i] = pk
                    i += 1
        ids_path, packed_path = cand_paths(src)
        atomic_save_npy(ids_path, ids, allow_pickle=True)
        atomic_save_npy(packed_path, packed)
        rows_per_src[src] = int(ids.shape[0])
        n_union += ids.shape[0]
        sub(f"    {src}: {ids.shape[0]:,} candidate rows over "
            f"{len(store):,} distinct ids")
        del ids, packed, store
        want[src] = None            # release each source as it is written
        gc.collect()
        guard_check(force=True)

    del want
    gc.collect()

    truth_pairs_total = sum(len(v) for v in truth.values())
    meta = {
        "n_s1": len(s1_records),
        "true_pairs": truth_pairs_total,
        "production_candidates": n_prod_seen,
        "union_candidates": n_union,
        "a2_only_candidates": n_union - n_prod_seen,
        "n_prod_true": n_prod_true,
        "n_a2_only_true": n_a2_true,
        "a2_buckets": n_buckets,
        "rows_per_src": rows_per_src,
        "max_group_size": CAP,
        "name_prefix_length": PREFIX4,
        "blas_threads": BLAS_THREADS,
        "thread_limit": THREAD_LIMIT,
        "duty_cycle": PACE.report() if PACE is not None else None,
        "fingerprint": FINGERPRINT,
        "reconstructed": False,
    }
    atomic_write_json(CKPT / STAGE3_META, meta)
    report_candidate_counts(meta)
    step(f"  STAGE 3 done in {human(time.perf_counter() - started)}")
    return meta


# ===========================================================================
# STAGE 4 - features
# ===========================================================================
def feature_paths():
    return (CKPT / "stage4_state.json", CKPT / "stage4_rows.npz",
            CKPT / "X.npy", CKPT / "X.partial.npy")


def load_feature_state(n_union):
    state_path, rows_path, _, _ = feature_paths()
    if not (state_path.is_file() and rows_path.is_file()):
        for path in (state_path, rows_path):
            if path.is_file():
                discard(path, "the other half of this checkpoint is missing")
        return None
    state = read_json(state_path)
    if state is None:
        discard(state_path, "unparseable")
        return None
    needed = ("pos", "pos_in_file", "n_prod_true", "done_src", "n_union",
              "n_features", "fingerprint")
    if not all(k in state for k in needed):
        discard(state_path, f"missing {[k for k in needed if k not in state]}")
        return None
    if state.get("fingerprint") != FINGERPRINT:
        sub("stage 4 matrix was built for a different configuration; "
            "recomputing every feature")
        discard(state_path, "fingerprint mismatch")
        discard(rows_path, "fingerprint mismatch")
        return None
    if int(state["n_features"]) != N_FEATURES:
        discard(state_path, f"{state['n_features']} features, need {N_FEATURES}")
        return None
    if int(state["n_union"]) != n_union:
        discard(state_path,
                f"built for {int(state['n_union']):,} rows, stage 3 says "
                f"{n_union:,}")
        return None
    if not 0 <= int(state["pos"]) <= n_union:
        discard(state_path, f"pos {state['pos']} outside 0..{n_union}")
        return None
    if 0 <= int(state["pos"]) < n_union and int(state["pos_in_file"]) == 0 \
            and not state["done_src"]:
        discard(state_path, "resume point is inconsistent")
        return None
    try:
        with np.load(rows_path) as saved:
            keys = ("y", "is_prod", "is_a2only_true", "s1_of_row")
            if any(k not in saved for k in keys):
                discard(rows_path, "missing label arrays")
                return None
            if any(saved[k].shape[0] != n_union for k in keys):
                discard(rows_path, "array length != n_union")
                return None
    except Exception as exc:
        discard(rows_path, f"unreadable: {exc}")
        return None
    return state


def open_feature_matrix(n_union, resuming):
    state_path, rows_path, x_final, x_partial = feature_paths()
    want = (n_union, N_FEATURES)
    if resuming:
        for path in (x_final, x_partial):
            if not path.is_file():
                continue
            try:
                array = np.load(path, mmap_mode="r+")
            except Exception as exc:
                discard(path, f"unreadable memmap: {exc}")
                continue
            if array.shape != want:
                discard(path, f"shape {array.shape} != {want}")
                continue
            return array, True
    for path in (x_final, x_partial):
        if path.is_file():
            discard(path, "no validated state; rebuilding the matrix from zero")
    return np.lib.format.open_memmap(x_partial, mode="w+", dtype=np.float32,
                                     shape=want), False


def load_want_source(src):
    """Candidate-id lookup for ONE source, straight off disk.

    Built per source rather than for both at once. The stage loop only ever
    touches one source at a time, so holding both roughly doubles peak memory
    for no benefit.
    """
    ids = np.load(cand_paths(src)[0], allow_pickle=True)
    packed = np.load(cand_paths(src)[1])
    store = {}
    for cid, pk in zip(ids.tolist(), packed.tolist()):
        current = store.get(cid)
        if current is None:
            store[cid] = pk
        elif type(current) is int:
            store[cid] = [current, pk]
        else:
            current.append(pk)
    del ids, packed
    return store


def stage4_features(args, meta, s1_records):
    started = time.perf_counter()
    step("STAGE 4/6  FEATURE GENERATION")
    n_union = int(meta["union_candidates"])
    if n_union <= 0:
        raise AssertionError(f"stage 3 reported {n_union} candidate rows")
    state_path, rows_path, x_final, x_partial = feature_paths()

    if args.force:
        # Without this, --force rebuilds the matrix but leaves the old state
        # and label arrays on disk. A crash before the first new checkpoint
        # would then leave a state file that points at rows which no longer
        # exist, and the next run would trust it.
        for path in (state_path, rows_path, x_final, x_partial):
            if path.is_file():
                discard(path, "--force: rebuilding from zero")

    state = load_feature_state(n_union)
    if state is None and not args.force:
        for path in (x_final, x_partial):
            if path.is_file():
                discard(path, "no validated state accompanies this matrix")

    resuming = state is not None
    X, matrix_ok = open_feature_matrix(n_union, resuming)

    if not matrix_ok or not resuming:
        pos = 0
        pos_in_file = 0
        n_prod_true = 0
        done_src = []
        y = np.zeros(n_union, dtype=np.int8)
        is_prod = np.zeros(n_union, dtype=bool)
        is_a2only_true = np.zeros(n_union, dtype=bool)
        s1_of_row = np.zeros(n_union, dtype=np.int32)
        sub(f"  starting fresh: {n_union:,} rows x {N_FEATURES} float32 "
            f"({n_union * N_FEATURES * 4 / 1e6:.0f} MB memmap)")
    else:
        pos = int(state["pos"])
        pos_in_file = int(state["pos_in_file"])
        n_prod_true = int(state["n_prod_true"])
        done_src = list(state["done_src"])
        with np.load(rows_path) as saved:
            y = saved["y"].copy()
            is_prod = saved["is_prod"].copy()
            is_a2only_true = saved["is_a2only_true"].copy()
            s1_of_row = saved["s1_of_row"].copy()
        sub(f"  resuming at {pos:,} of {n_union:,} rows "
            f"(pos_in_file={pos_in_file:,}, completed={done_src})")
    guard_check(force=True)

    if pos < n_union:
        s1_order = sorted(s1_records)
        truth = pickle.loads((CKPT / "truth.pkl").read_bytes())
        last_checkpoint = pos

        def write_checkpoint(src):
            """Flush, then labels, then the state file.

            Labels are written BEFORE the state, so a crash in between leaves
            the state pointing at older but consistent work rather than at rows
            whose labels were never saved.
            """
            X.flush()
            atomic_save_npz(rows_path, y=y, is_prod=is_prod,
                            is_a2only_true=is_a2only_true, s1_of_row=s1_of_row)
            atomic_write_json(state_path, {
                "pos": int(pos), "pos_in_file": int(pos_in_file),
                "n_prod_true": int(n_prod_true), "done_src": list(done_src),
                "n_union": int(n_union), "n_features": int(N_FEATURES),
                "fingerprint": FINGERPRINT,
            })
            sub(f"  checkpoint: {pos:,} rows after {src} (done={done_src})")

        for fname, src in SOURCES:
            if pos >= n_union:
                break
            if src in done_src:
                sub(f"  {src} already complete, skipping")
                continue
            sub(f"  featurising {src}")
            store = load_want_source(src)
            guard_check(force=True)
            skip = pos_in_file
            seen = 0
            joined = 0
            file_started = pos
            for rec in M.iter_records(train_path(fname)):
                value = store.get(rec.entity_id)
                if value is None:
                    guard_tick()
                    continue
                seen += 1
                pks = (value,) if type(value) is int else value
                if skip > 0:
                    skip -= len(pks)
                    guard_tick()
                    continue
                for pk in pks:
                    si = pk >> 1
                    prod_row = bool(pk & 1)
                    eid = s1_order[si]
                    X[pos] = featurize(s1_records[eid], rec)
                    s1_of_row[pos] = si
                    is_prod[pos] = prod_row
                    hit = rec.entity_id in truth.get(eid, frozenset())
                    y[pos] = 1 if hit else 0
                    is_a2only_true[pos] = (not prod_row) and hit
                    if hit and prod_row:
                        n_prod_true += 1
                    pos += 1
                    pos_in_file += 1
                    joined += 1
                    guard_tick()
                if pos - last_checkpoint >= args.feature_checkpoint_every:
                    write_checkpoint(f"{src} (partial)")
                    last_checkpoint = pos
            sub(f"    {src}: joined {seen:,}/{len(store):,} ids, "
                f"{pos - file_started:,} rows this pass, {pos:,} total")
            if skip != 0:
                raise AssertionError(
                    f"{src}: resume skipped {skip:,} rows but the file ran out")
            if seen != len(store):
                raise AssertionError(
                    f"{src}: {len(store) - seen:,} wanted candidate ids were "
                    f"never found in the file")
            del store
            gc.collect()
            pos_in_file = 0
            if src not in done_src:
                done_src.append(src)
            write_checkpoint(src)
            last_checkpoint = pos
            guard_check(force=True)

    if pos != n_union:
        raise AssertionError(f"featurised {pos:,} of {n_union:,} expected rows")

    X.flush()
    del X
    gc.collect()
    if x_partial.is_file():
        os.replace(x_partial, x_final)
        _fsync_dir(CKPT)
    check("all rows featurised", True, f"{pos:,}")
    check("production-arm true pairs", n_prod_true == EXPECT_PROD_TRUE,
          f"{n_prod_true:,} == {EXPECT_PROD_TRUE:,}")
    a2_true = int(is_a2only_true.sum())
    check("A2-only true pairs", a2_true == EXPECT_A2_ONLY_TRUE,
          f"{a2_true:,} == {EXPECT_A2_ONLY_TRUE:,}")
    check("total true pairs", int(y.sum()) == EXPECT_PROD_TRUE + EXPECT_A2_ONLY_TRUE,
          f"{int(y.sum()):,}")

    # Persist the observed per-arm true-pair counts so a later stage-3 metadata
    # rebuild never has to guess them and the report never falls back to the
    # expected constants.
    meta["n_prod_true"] = n_prod_true
    meta["n_a2_only_true"] = a2_true
    if meta.get("reconstructed"):
        atomic_write_json(CKPT / STAGE3_META, meta)
    step(f"  STAGE 4 done in {human(time.perf_counter() - started)}")


# ===========================================================================
# STAGE 5 - model
# ===========================================================================
STAGE5_RESULTS = "stage5_results.json"


def load_model_results(n_union, n_s1):
    path = CKPT / STAGE5_RESULTS
    payload = read_json(path)
    if payload is None:
        if path.is_file():
            discard(path, "unparseable")
        return None
    if payload.get("fingerprint") != FINGERPRINT:
        sub("stage 5 results were produced by a different configuration; "
            "refitting")
        return None
    if int(payload.get("n_union") or 0) != n_union \
            or int(payload.get("n_s1") or 0) != n_s1 \
            or int(payload.get("n_features") or 0) != N_FEATURES:
        sub("stage 5 results do not match the current feature matrix; refitting")
        return None
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        discard(path, "expected exactly two arms")
        return None
    for arm in arms:
        if not isinstance(arm, dict):
            discard(path, "malformed arm")
            return None
        best = arm.get("best")
        if not isinstance(best, dict) or "reports" not in arm:
            discard(path, f"arm {arm.get('name')!r} is incomplete")
            return None
        for key in ("threshold", "precision", "recall", "f_beta",
                    "true_positives", "false_positives", "false_negatives",
                    "predicted_positives"):
            if key not in best:
                discard(path, f"arm {arm.get('name')!r} best missing {key}")
                return None
    return payload


def matrix_is_finite(X, chunk=500_000):
    """Chunked so a 700 MB scan cannot hold the whole matrix at once."""
    for start in range(0, X.shape[0], chunk):
        if not bool(np.isfinite(np.asarray(X[start:start + chunk])).all()):
            return False
        guard_tick()
    return True


def stage5_model(args, meta):
    started = time.perf_counter()
    step("STAGE 5/6  LOGISTIC REGRESSION")
    _, rows_path, x_final, _ = feature_paths()
    n_union = int(meta["union_candidates"])

    if not args.force:
        cached = load_model_results(n_union, N_S1)
        if cached is not None:
            sub("reusing validated stage 5 results")
            for arm in cached["arms"]:
                sub(f"  {arm['name']}: F0.5={arm['best']['f_beta']:.4f} "
                    f"@ threshold {arm['best']['threshold']}")
            step(f"  STAGE 5 done in {human(time.perf_counter() - started)} "
                 f"(cached)")
            return cached

    for needed in (x_final, rows_path, CKPT / "s1_records.pkl"):
        if not needed.is_file():
            raise FileNotFoundError(
                f"stage 5 needs {needed.name}, which stage 4 did not produce")

    s1_order = sorted(pickle.loads((CKPT / "s1_records.pkl").read_bytes()))
    X = np.load(x_final, mmap_mode="r")
    with np.load(rows_path) as saved:
        y = saved["y"].copy()
        is_prod = saved["is_prod"].copy()
        is_a2only_true = saved["is_a2only_true"].copy()
        s1_of_row = saved["s1_of_row"].copy()
    sub(f"  loaded {X.shape[0]:,} x {X.shape[1]} features "
        f"(positives {int(y.sum()):,})")
    guard_check(force=True)

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
    sub(f"  split: {len(train_s1):,} train S1, {len(val_s1):,} validation S1 "
        f"(salt {SPLIT_SALT!r}, fraction {SPLIT_FRACTION})")
    check("every S1 entity on exactly one side",
          len(train_s1) + len(val_s1) == len(s1_order),
          f"{len(train_s1):,} + {len(val_s1):,} = {len(s1_order):,}")
    check("no S1 entity on both sides", not (train_s1 & val_s1))

    def run_arm(name, mask):
        if mask is None:
            ya, a2t, row_val = y, is_a2only_true, row_is_val
        else:
            idx = np.flatnonzero(mask)
            ya, a2t, row_val = y[idx], is_a2only_true[idx], row_is_val[idx]
            del idx
        val_idx = np.flatnonzero(row_val)
        trn_idx = np.flatnonzero(~row_val)
        # numpy arrays, not .tolist() lists: a Python list of several million
        # ints costs far more memory to hold and to walk than the array.
        ytr = np.ascontiguousarray(ya[trn_idx])
        yva = np.ascontiguousarray(ya[val_idx])
        Xtr = X if trn_idx.size == ya.size else X[trn_idx]
        Xva = np.asarray(X[val_idx])
        sub(f"  [{name}] {ya.size:,} rows (train {trn_idx.size:,}, "
            f"valid {val_idx.size:,}), positives {int(ya.sum()):,} "
            f"({int(ya.sum()) / max(1, ya.size):.4%})")
        guard_check(force=True)
        t = time.perf_counter()
        model = M.LogisticMatcher()
        # A single long C call. Nothing can interrupt it; the 1-thread clamp is
        # the only real protection here, which is why it is installed above.
        model.fit(Xtr, ytr)
        summary = model.summary
        coefficients = model.coefficients()
        sub(f"  [{name}] fit in {human(time.perf_counter() - t)}: "
            f"n_iter={summary.n_iter}, converged={summary.converged}, "
            f"class_weight={summary.class_weight}")
        guard_check(force=True)
        scores = model.predict_proba(Xva)
        reports = M.evaluate_thresholds(yva, scores, beta=BETA)
        best = M.best_validation_threshold(reports)
        predicted = np.fromiter((s >= best.threshold for s in scores),
                                dtype=bool, count=len(scores))
        a2_val = a2t[val_idx]
        out = {
            "name": name,
            "rows": int(ya.size),
            "positives": int(ya.sum()),
            "pos_rate": float(ya.sum()) / max(1, int(ya.size)),
            "train_rows": int(trn_idx.size),
            "valid_rows": int(val_idx.size),
            "valid_positives": int(ya[val_idx].sum()),
            "candidates_per_s1": float(ya.size) / max(1, len(s1_order)),
            "n_iter": summary.n_iter,
            "converged": summary.converged,
            "class_weight": summary.class_weight,
            "C": model.C, "solver": model.solver,
            "max_iter": model.max_iter, "random_state": model.random_state,
            "coefficients": coefficients,
            "reports": [dataclasses.asdict(r) for r in reports],
            "best": dataclasses.asdict(best),
            "tp": best.true_positives, "fp": best.false_positives,
            "fn": best.false_negatives, "precision": best.precision,
            "recall": best.recall, "f_beta": best.f_beta,
            "pred_pos": best.predicted_positives,
            "a2_only_true_total": int(a2t.sum()),
            "a2_only_true_valid": int(a2_val.sum()),
            "a2_only_true_caught": int(((a2_val) & (predicted)).sum()),
        }
        del Xtr, Xva, scores, predicted, ya, a2t, row_val, val_idx, trn_idx
        ytr = yva = None
        gc.collect()
        return out

    arms = [run_arm("production", is_prod), run_arm("production+A2", None)]
    guard_check(force=True)

    check("no NaN/inf anywhere in the feature matrix", matrix_is_finite(X),
          f"all {X.shape[0]:,} rows, scanned in chunks")
    payload = {
        "n_union": n_union,
        "n_features": int(X.shape[1]),
        "n_s1": len(s1_order),
        "split": {"validation_fraction": SPLIT_FRACTION, "salt": SPLIT_SALT,
                  "train_s1": len(train_s1), "validation_s1": len(val_s1)},
        "beta": BETA,
        "arms": arms,
        "fingerprint": FINGERPRINT,
        "thread_limit": THREAD_LIMIT,
    }
    atomic_write_json(CKPT / STAGE5_RESULTS, payload)
    for arm in arms:
        sub(f"  {arm['name']}: P={arm['precision']:.4f} R={arm['recall']:.4f} "
            f"F0.5={arm['f_beta']:.4f} @ {arm['best']['threshold']}")
    step(f"  STAGE 5 done in {human(time.perf_counter() - started)}")
    return payload


# ===========================================================================
# STAGE 6 - report
# ===========================================================================
REPORT_TXT = "match_a2_report.txt"
REPORT_JSON = "match_a2_summary.json"


def observed(meta, key, expected):
    """Observed value, or None. Never substitutes an expected constant.

    The local runner fell back to EXPECT_* here, which meant a reconstructed
    metadata file could report a remembered number as if it had been measured.
    """
    value = meta.get(key)
    return None if value is None else int(value)


def stage6_report(args, meta, model, s1_records):
    started = time.perf_counter()
    step("STAGE 6/6  REPORT")
    txt_path = OUT_DIR / REPORT_TXT
    json_path = OUT_DIR / REPORT_JSON
    if not args.force and txt_path.is_file() and json_path.is_file() \
            and not args.rerun_report:
        sub("report already written; re-run with --rerun-report to rebuild")
        step(f"  STAGE 6 done in {human(time.perf_counter() - started)} (cached)")
        print()
        print(txt_path.read_text(encoding="utf-8"))
        return

    truth = pickle.loads((CKPT / "truth.pkl").read_bytes())
    total_true = sum(len(v) for v in truth.values())
    n_s1 = len(s1_records)
    prod_cand = int(meta["production_candidates"])
    union_cand = int(meta["union_candidates"])
    prod_true = observed(meta, "n_prod_true", EXPECT_PROD_TRUE)
    a2_true = observed(meta, "n_a2_only_true", EXPECT_A2_ONLY_TRUE)
    arm1, arm2 = model["arms"]

    fmt = lambda v: "unavailable" if v is None else f"{v:,}"
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
    }
    summary = {
        "n_s1": n_s1,
        "n_features": N_FEATURES,
        "total_true_pairs": total_true,
        "candidate_counts": {
            "production": prod_cand, "production+A2": union_cand,
            "a2_only_additions": union_cand - prod_cand},
        "blocking": blocking,
        "arms": model["arms"],
        "split": model["split"],
        "beta": BETA,
        "fingerprint": FINGERPRINT,
        "resources": {
            "blas_threads": BLAS_THREADS,
            "blas_env": describe_threads(),
            "runtime_thread_limit": THREAD_LIMIT,
            "priority": PRIORITY,
            "cpus_allowed": CPUS_ALLOWED,
            "duty_cycle": PACE.report() if PACE is not None else None,
            "thermal": GUARD.report() if GUARD is not None else None,
            "total_elapsed_seconds": round(time.perf_counter() - T0, 1),
        },
        "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in CHECKS],
    }
    atomic_write_json(json_path, summary)

    lines = []
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
    add(f"   model                               LogisticRegression "
        f"(solver={arm1['solver']}, C={arm1['C']}, "
        f"class_weight={arm1['class_weight']}, max_iter={arm1['max_iter']}, "
        f"random_state={arm1['random_state']}, features unscaled)")
    add(f"   split                               S1-level, "
        f"validation_fraction={SPLIT_FRACTION}, salt {SPLIT_SALT!r} "
        f"({model['split']['train_s1']:,} train / "
        f"{model['split']['validation_s1']:,} validation S1)")
    add(f"   metric                              F{BETA}")
    add(f"   total true pairs for these S1      {total_true:,}")
    add("")
    add("2. BLOCKING")
    add(f"   {'arm':<18}{'candidates':>14}{'per S1':>10}{'true pairs':>13}"
        f"{'recall':>11}")
    for name, stats in blocking.items():
        recall = stats["recall"]
        add(f"   {name:<18}{stats['candidates']:>14,}"
            f"{stats['candidates_per_s1']:>10.1f}"
            f"{fmt(stats['true_pairs']):>13}"
            f"{'unavailable' if recall is None else format(recall, '.4%'):>11}")
    add(f"   A2-only additions                  {union_cand - prod_cand:,}")
    add(f"   A2-only true pairs recovered       {fmt(a2_true)}")
    add(f"   A2 share of the candidate pool     "
        f"{(union_cand - prod_cand) / max(1, union_cand):.2%} of union rows "
        f"are A2-only")
    add("")
    add("3. VALIDATION RESULT AT EACH ARM'S BEST F0.5 THRESHOLD")
    add(f"   {'arm':<18}{'rows':>12}{'positives':>12}{'prec':>9}{'recall':>9}"
        f"{'F0.5':>9}{'thr':>6}{'TP':>9}{'FP':>11}{'FN':>10}")
    for arm in (arm1, arm2):
        best = arm["best"]
        add(f"   {arm['name']:<18}{arm['rows']:>12,}{arm['positives']:>12,}"
            f"{best['precision']:>9.4f}{best['recall']:>9.4f}"
            f"{best['f_beta']:>9.4f}{best['threshold']:>6.1f}"
            f"{best['true_positives']:>9,}{best['false_positives']:>11,}"
            f"{best['false_negatives']:>10,}")
    add("")
    add(f"   F0.5   (+A2 - production)          "
        f"{arm2['f_beta'] - arm1['f_beta']:+.4f}")
    add(f"   precision change                    "
        f"{arm2['precision'] - arm1['precision']:+.4f}")
    add(f"   recall change                       "
        f"{arm2['recall'] - arm1['recall']:+.4f}")
    add(f"   true positives                      {arm2['tp'] - arm1['tp']:+,}")
    add(f"   false positives                     {arm2['fp'] - arm1['fp']:+,}")
    add("")
    add("4. THE A2-RECOVERED TRUE PAIRS")
    add(f"   in the candidate set                {arm2['a2_only_true_total']:,}")
    add(f"   in the validation split             {arm2['a2_only_true_valid']:,}")
    add(f"   predicted positive at the +A2 best  "
        f"{arm2['a2_only_true_caught']:,} "
        f"(threshold {arm2['best']['threshold']})")
    add(f"   recall on that subset               "
        f"{arm2['a2_only_true_caught'] / max(1, arm2['a2_only_true_valid']):.4%}")
    add("")
    add("   These pairs do not exist in the production arm at all: production")
    add("   blocking never retrieved them, so the production model never had the")
    add("   chance to score them. The comparison above is therefore the whole")
    add("   effect of A2: the extra candidates it brings, and what the matcher")
    add("   does with them.")
    add("")
    add("5. THRESHOLD SWEEPS")
    for arm in (arm1, arm2):
        add("")
        add(f"   {arm['name']}")
        add(f"     {'thr':>5}{'prec':>9}{'rec':>9}{'F0.5':>9}{'TP':>9}"
            f"{'FP':>10}{'FN':>9}{'pred+':>10}")
        for r in arm["reports"]:
            add(f"     {r['threshold']:>5.1f}{r['precision']:>9.4f}"
                f"{r['recall']:>9.4f}{r['f_beta']:>9.4f}"
                f"{r['true_positives']:>9,}{r['false_positives']:>10,}"
                f"{r['false_negatives']:>9,}{r['predicted_positives']:>10,}")
    add("")
    add("6. FITTED COEFFICIENTS (unscaled features)")
    add(f"   {'feature':<28}{'production':>13}{'production+A2':>15}")
    for feature in FEATURE_NAMES:
        add(f"   {feature:<28}{arm1['coefficients'][feature]:>13.4f}"
            f"{arm2['coefficients'][feature]:>15.4f}")
    add("")
    add("7. RESOURCES AND CHECKS")
    res = summary["resources"]
    add(f"   BLAS/OpenMP threads                 {BLAS_THREADS} "
        f"(env {res['blas_env']})")
    add(f"   runtime thread clamp                {res['runtime_thread_limit']}")
    add(f"   process priority                    {res['priority']}")
    add(f"   CPU restriction                     {res['cpus_allowed']}")
    duty = res["duty_cycle"]
    if duty:
        add(f"   duty cycle                          target fraction "
            f"{duty['duty_fraction']:.2f}, slept "
            f"{human(duty['slept_seconds'])}")
    thermal = res["thermal"]
    if thermal:
        hottest = thermal["max_cpu_temp_c"]
        add(f"   temperature monitoring              "
            f"{'active' if thermal['temperature_monitoring'] else 'unavailable'}")
        add(f"   max CPU temp observed               "
            f"{'unavailable' if hottest is None else format(hottest, '.1f') + ' C'}")
    add(f"   total elapsed                       "
        f"{human(res['total_elapsed_seconds'])}")
    add("")
    add("   Note: a single long C call cannot be interrupted by the guard. The")
    add("   logistic-regression fit is protected only by the thread clamp, not")
    add("   by a temperature check.")
    add("")
    for name, ok, detail in CHECKS:
        add(f"   [{'PASS' if ok else 'FAIL'}] {name}"
            + (f" -- {detail}" if detail else ""))
    add("")
    add(rule)
    add("Validation F0.5 on a bounded S1 subset against the full S2/S3 pool.")
    add("This is NOT competition performance and NOT a submitted result. The 20k")
    add("S1 subset is a sample, and the competition metric is macro-averaged over")
    add("S1 entities, whereas the numbers above are pair-level on pooled")
    add("candidates.")
    add(rule)

    text = "\n".join(lines) + "\n"
    atomic_write_text(txt_path, text)
    step(f"  wrote {txt_path}")
    step(f"  wrote {json_path}")
    step(f"  STAGE 6 done in {human(time.perf_counter() - started)}")
    print()
    print(text)


def print_finisher(meta, model):
    arm1, arm2 = model["arms"]
    prod = int(meta["production_candidates"])
    union = int(meta["union_candidates"])
    n_s1 = int(meta.get("n_s1") or N_S1)
    total_true = sum(len(v) for v in
                     pickle.loads((CKPT / "truth.pkl").read_bytes()).values())
    prod_true = observed(meta, "n_prod_true", EXPECT_PROD_TRUE)
    a2_true = observed(meta, "n_a2_only_true", EXPECT_A2_ONLY_TRUE)
    union_true = (None if prod_true is None or a2_true is None
                  else prod_true + a2_true)
    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Production candidates:  {prod:,} ({prod / max(1, n_s1):.1f} per S1)")
    print(f"Union candidates:       {union:,} ({union / max(1, n_s1):.1f} per S1)")
    print(f"A2-only candidates:     {union - prod:,}")
    print(f"Production recall:      "
          + ("unavailable" if prod_true is None
             else f"{prod_true / max(1, total_true):.4%} "
                  f"({prod_true:,}/{total_true:,})"))
    print(f"Production+A2 recall:   "
          + ("unavailable" if union_true is None
             else f"{union_true / max(1, total_true):.4%} "
                  f"({union_true:,}/{total_true:,})"))
    print(f"A2-only true caught:    {arm2['a2_only_true_caught']:,} of "
          f"{arm2['a2_only_true_valid']:,} in validation")
    print(f"Production F0.5:        {arm1['f_beta']:.4f} "
          f"(P {arm1['precision']:.4f} / R {arm1['recall']:.4f} "
          f"@ {arm1['best']['threshold']})")
    print(f"Production+A2 F0.5:     {arm2['f_beta']:.4f} "
          f"(P {arm2['precision']:.4f} / R {arm2['recall']:.4f} "
          f"@ {arm2['best']['threshold']})")
    print(f"F0.5 delta (+A2):       {arm2['f_beta'] - arm1['f_beta']:+.4f}")
    print("=" * 60)
    print(f"Report     : {OUT_DIR / REPORT_TXT}")
    print(f"Metrics    : {OUT_DIR / REPORT_JSON}")
    print(f"Checkpoints: {CKPT}")
    print("=" * 60)


# ===========================================================================
# DRIVER
# ===========================================================================
STAGES = ("dataset", "subset", "candidates", "features", "model", "report",
          "all")


def build_parser():
    parser = argparse.ArgumentParser(
        description="20k S1 matching A/B, staged and resumable. "
                    "Run with no arguments for everything.")
    parser.add_argument("--stage", default="all", choices=STAGES,
                        help="one stage, or 'all' (default)")
    parser.add_argument("--dataset-zip", default="",
                        help="path to the dataset zip (default: auto-discover "
                             "under /kaggle/input)")
    parser.add_argument("--dataset-root", default="",
                        help="extraction directory (default: <work>/dataset)")
    parser.add_argument("--work-dir", default="", help="scratch/working root")
    parser.add_argument("--checkpoint-dir", default="",
                        help="where stage checkpoints are written")
    parser.add_argument("--output-dir", default="",
                        help="where the final report is written")
    parser.add_argument("--threads", type=int, default=1,
                        help="native thread cap; must precede the numpy import")
    parser.add_argument("--cpus", type=int, default=2,
                        help="maximum logical CPUs (0 = no restriction). "
                             "Warn-and-continue if the sandbox forbids pinning.")
    parser.add_argument("--duty-fraction", type=float, default=0.85,
                        help="target fraction of wall time spent computing")
    parser.add_argument("--duty-every", type=int, default=2000,
                        help="rows between duty-cycle ticks")
    parser.add_argument("--feature-checkpoint-every", type=int, default=400_000,
                        help="rows between mid-file feature checkpoints")
    parser.add_argument("--thermal", default="auto", choices=("auto", "on", "off"),
                        help="temperature guard (default auto)")
    parser.add_argument("--pause-above", type=float, default=84.0)
    parser.add_argument("--resume-below", type=float, default=79.0)
    parser.add_argument("--abort-above", type=float, default=90.0)
    parser.add_argument("--force", action="store_true",
                        help="ignore validated checkpoints and recompute")
    parser.add_argument("--rerun-report", action="store_true")
    parser.add_argument("--allow-embedded-drift", action="store_true",
                        help="continue even if an embedded source does not "
                             "match its recorded SHA-256")
    return parser


def apply_path_overrides(args):
    """CLI path flags win over the environment, which wins over the defaults."""
    if args.work_dir:
        os.environ[WORK_ENV] = args.work_dir
    if args.checkpoint_dir:
        os.environ[CKPT_ENV] = args.checkpoint_dir
    if args.dataset_root:
        os.environ[DATASET_ENV] = args.dataset_root
    if args.output_dir:
        os.environ[OUT_ENV] = args.output_dir


def main(argv=None):
    global GUARD, PACE, PRIORITY, CPUS_ALLOWED, THREAD_LIMIT, TRAIN_DIR

    parser = build_parser()
    # parse_known_args, not parse_args: inside a notebook sys.argv holds the
    # kernel's own flags (-f /root/...ipykernel_launcher.py) and a strict
    # parser would abort on them instead of running the experiment.
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        warn(f"ignoring unrecognised arguments: {unknown}")
    if args.threads != BLAS_THREADS:
        warn(f"--threads {args.threads} did not take effect before the numpy "
             f"import; {BLAS_THREADS} is in force for this interpreter")

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

    materialize_sources(allow_drift=args.allow_embedded_drift)
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
    step(f"threads: env capped to {BLAS_THREADS} before numpy; runtime clamp "
         f"{THREAD_LIMIT}")
    if "unavailable" in THREAD_LIMIT or "failed" in THREAD_LIMIT:
        warn("no runtime thread clamp available; only the environment cap "
             "applies, and a pre-loaded BLAS would ignore it")

    PACE = DutyCycle(fraction=args.duty_fraction, tick_every=args.duty_every)

    thermal_on = args.thermal == "on" or (args.thermal == "auto"
                                          and read_sysfs_temps() is not None)
    GUARD = ThermalGuard(pause_above=args.pause_above,
                         resume_below=args.resume_below,
                         abort_above=args.abort_above, enabled=thermal_on)
    if not thermal_on:
        sub(f"thermal guard off (--thermal {args.thermal}); this sandbox "
            f"usually exposes no CPU temperature sensor")
    else:
        now = GUARD.temp()
        sub(f"thermal guard on, CPU {'unavailable' if now is None else format(now, '.1f') + ' C'}"
            f"; pause >= {args.pause_above:.0f} C, resume <= "
            f"{args.resume_below:.0f} C, abort >= {args.abort_above:.0f} C")

    order = list(STAGES[:-1]) if args.stage == "all" else [args.stage]
    try:
        s1_records = None
        meta = None
        model = None
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
            if stage == "features":
                stage4_features(args, meta, s1_records)
                continue
            if model is None:
                model = stage5_model(args, meta)
            if stage == "model":
                continue
            stage6_report(args, meta, model, s1_records)

        if args.stage == "all" and meta is not None and model is not None:
            print_finisher(meta, model)
        return 0
    except ThermalAbort as exc:
        step(f"THERMAL STOP: {exc}")
        step(f"completed work is checkpointed in {CKPT}; re-run this cell to "
             f"resume")
        return 2
    except KeyboardInterrupt:
        step("interrupted; checkpointed work is kept, re-run this cell to "
             "resume")
        return 130
    finally:
        if GUARD is not None:
            GUARD.stop()


# ---------------------------------------------------------------------------
# Entry point. Runs when pasted into a notebook cell, where __name__ is
# "__main__", and when executed as a plain script.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _rc = main()
    if _rc not in (0, None):
        raise SystemExit(_rc)
