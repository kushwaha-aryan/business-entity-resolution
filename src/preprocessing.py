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
