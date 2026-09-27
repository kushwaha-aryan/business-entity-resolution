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
