"""Name/address normalization: legal-suffix stripping, punctuation, and
script-general transliteration for non-Latin business names.

Country-aware suffix lists cover US, India, and France (test-only country --
per problem_statement.md, country must stay an open set, so `LEGAL_SUFFIXES`
is used as a normalization aid, never as a filter/blocking key restricted to
these three).

Real data (train_source2.tsv, 2026-09-25) shows India business names use
multiple Indic scripts, not just Devanagari -- Tamil and Gujarati samples were
observed too. `romanize()` uses `indic_transliteration` (MIT license, covers
Devanagari/Tamil/Gujarati/Bengali/Kannada/Malayalam/Telugu/Gurumukhi/Oriya) to
get a Latin-script phonetic approximation, so downstream string-similarity
features (rapidfuzz/TF-IDF) have something to compare against the English
name. This is script conversion, not a business-identity lookup -- it does
not touch the "no external data lookup" rule.
"""
from __future__ import annotations

import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Legal-entity suffixes to strip from the end of a normalized name, longest
# first so e.g. "private limited" matches before "limited" alone.
LEGAL_SUFFIXES = sorted(
    [
        # US
        "incorporated", "inc", "corporation", "corp", "company", "co",
        "limited liability company", "llc", "l l c", "limited", "ltd",
        "l p", "lp", "l l p", "llp", "p c", "pc", "pllc",
        # India
        "private limited", "pvt ltd", "pvt", "private", "proprietorship",
        "and sons", "sons",
        # France
        "societe a responsabilite limitee", "sarl", "societe par actions simplifiee",
        "sas", "sasu", "societe anonyme", "sa", "societe en nom collectif", "snc",
        "societe civile immobiliere", "sci", "entreprise unipersonnelle a responsabilite limitee",
        "eurl", "societe", "ste",
    ],
    key=len,
    reverse=True,
)

_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES) + r")\b\.?\s*$"
)
_PUNCT_PATTERN = re.compile(r"[^\w\s]")
_WS_PATTERN = re.compile(r"\s+")

# Unicode block ranges -> indic_transliteration scheme. Checked against the
# first non-space character of the name.
_SCRIPT_RANGES = [
    ((0x0900, 0x097F), sanscript.DEVANAGARI),
    ((0x0B80, 0x0BFF), sanscript.TAMIL),
    ((0x0A80, 0x0AFF), sanscript.GUJARATI),
    ((0x0980, 0x09FF), sanscript.BENGALI),
    ((0x0C80, 0x0CFF), sanscript.KANNADA),
    ((0x0D00, 0x0D7F), sanscript.MALAYALAM),
    ((0x0C00, 0x0C7F), sanscript.TELUGU),
    ((0x0A00, 0x0A7F), sanscript.GURMUKHI),
    ((0x0B00, 0x0B7F), sanscript.ORIYA),
]


def detect_script(text: str) -> str | None:
    """Return the indic_transliteration scheme name for text's script, or
    None if the text is already Latin/ASCII (no transliteration needed)."""
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            continue
        cp = ord(ch)
        for (lo, hi), scheme in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                return scheme
        return None  # first alphabetic char is Latin or another non-Indic script
    return None


def romanize(text: str) -> str:
    """Convert a non-Latin-script business name to a Latin phonetic
    approximation. Returns the input unchanged if it's already Latin script
    or its script isn't one of the supported Indic schemes."""
    scheme = detect_script(text)
    if scheme is None:
        return text
    try:
        return transliterate(text, scheme, sanscript.HK)
    except Exception:
        return text


def strip_diacritics(text: str) -> str:
    """Fold Latin diacritics (e.g. 'Sólar' -> 'Solar') via NFKD decomposition.
    Stdlib-only, no dependency/license question."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_name(name: str) -> str:
    """Full name normalization pipeline: romanize -> fold diacritics ->
    lowercase -> strip punctuation -> strip legal suffix -> collapse
    whitespace. Returns "" for empty/missing input."""
    if not name:
        return ""
    text = romanize(name)
    text = strip_diacritics(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = _PUNCT_PATTERN.sub(" ", text)
    text = _WS_PATTERN.sub(" ", text).strip()
    # Strip one trailing legal suffix (suffixes can stack, e.g. "x pvt ltd" ->
    # "ltd" then "pvt"); loop until no more match, bounded to avoid over-stripping.
    for _ in range(3):
        new_text = _SUFFIX_PATTERN.sub("", text).strip()
        if new_text == text:
            break
        text = new_text
    return text


def name_tokens(name: str) -> set[str]:
    """Normalized, whitespace-split token set for blocking/features."""
    normalized = normalize_name(name)
    return set(normalized.split()) if normalized else set()


def normalize_address(address: str) -> str:
    """Lighter normalization for addresses: romanize, fold diacritics,
    lowercase, strip punctuation, collapse whitespace. No suffix stripping
    (address abbreviations like Rd/Road are handled as features, not folded
    away, since they can carry disambiguating signal)."""
    if not address:
        return ""
    text = romanize(address)
    text = strip_diacritics(text)
    text = text.lower()
    text = _PUNCT_PATTERN.sub(" ", text)
    text = _WS_PATTERN.sub(" ", text).strip()
    return text
