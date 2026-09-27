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
