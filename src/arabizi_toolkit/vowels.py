"""Short-vowel explicitness: parsing, alignment and the per-writer rate (H3).

The protocol computes, for each short vowel in the fully vocalised Arabic
counterpart (layer 2), whether a Latin vowel letter appears in the matching
position of the Arabizi form. The original protocol delegated this to two
undefined helpers; this module defines them.

Conventions (documented so that annotators and code agree):
- A short vowel followed by its own long-vowel letter (fatha + alif, damma + waw,
  kasra + ya) is one long vowel, not a short-vowel slot. Long vowels are written
  in Arabic script, so their appearance in Latin is not "explicitness".
- Taa marbuta is not a slot; the vowel on the consonant before it is.
- Shadda duplicates the consonant; the Latin form may write it once or twice.
- An initial alif carrying a vowel mark (e.g. "اِلْ") contributes that vowel.
- Every automatic alignment is a proposal: the annotator confirms or corrects it
  (layer 3b), and agreement on layer 3b is measured separately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FATHA, DAMMA, KASRA, SUKUN, SHADDA = "َ", "ُ", "ِ", "ْ", "ّ"
TANWEEN = {"ً", "ٌ", "ٍ"}
HARAKA_TO_VOWEL = {FATHA: "a", DAMMA: "u", KASRA: "i"}
MADD_FOR = {FATHA: "ا", DAMMA: "و", KASRA: "ي"}
LONG_LETTER_VOWEL = {"ا": "a", "ى": "a", "و": "u", "ي": "i"}
HAMZA_FORMS = set("ءأإؤئ")
WEAK = set("عءأإؤئه")  # often written as a vowel letter or dropped

# Broad candidate sets: which Latin units may represent each Arabic consonant.
CANDIDATES = {
    "ب": {"b", "p"}, "ت": {"t"}, "ث": {"th", "s", "t"}, "ج": {"g", "j", "dj"},
    "ح": {"7", "h"}, "خ": {"kh", "5", "7'", "x"}, "د": {"d"}, "ذ": {"th", "dh", "z", "d"},
    "ر": {"r"}, "ز": {"z"}, "س": {"s"}, "ش": {"sh", "ch"}, "ص": {"s", "9"},
    "ض": {"d", "9'"}, "ط": {"t", "6"}, "ظ": {"z", "6'", "th", "dh"}, "ع": {"3"},
    "غ": {"gh", "3'"}, "ف": {"f"}, "ق": {"2", "q", "k", "'"}, "ك": {"k"}, "ل": {"l"},
    "م": {"m"}, "ن": {"n"}, "ه": {"h"}, "و": {"w"}, "ي": {"y"},
}
for _h in HAMZA_FORMS:
    CANDIDATES[_h] = {"2", "'"}

LATIN_VOWELS = set("aeiou")
DIGRAPHS = ("sh", "kh", "gh", "th", "dh", "ch", "dj", "3'", "7'", "9'", "6'")


@dataclass(frozen=True)
class Seg:
    kind: str       # "C" consonant, "SV" short vowel, "LV" long vowel
    value: str      # Arabic letter for C, vowel quality for SV/LV
    geminate: bool = False


def parse_vocalised(word: str) -> list[Seg]:
    """Turn a fully vocalised Arabic word into consonant / short / long vowel segments."""
    segs: list[Seg] = []
    chars = [ch for ch in word if ch not in TANWEEN and ch != "ـ"]
    i = 0
    while i < len(chars):
        ch = chars[i]
        marks = []
        j = i + 1
        while j < len(chars) and chars[j] in (FATHA, DAMMA, KASRA, SUKUN, SHADDA):
            marks.append(chars[j])
            j += 1
        haraka = next((m for m in marks if m in HARAKA_TO_VOWEL), None)

        if ch == "ة":
            i = j
            continue
        if ch == "آ":
            segs += [Seg("C", "ء"), Seg("LV", "a")]
            i = j
            continue
        if ch == "ا" and i == 0:
            # hamzat al-wasl: contributes a vowel only if one is written on it
            if haraka:
                segs.append(Seg("SV", HARAKA_TO_VOWEL[haraka]))
            i = j
            continue
        if ch in LONG_LETTER_VOWEL and not marks and segs and segs[-1].kind == "C" \
                and ch in ("ا", "ى"):
            segs.append(Seg("LV", LONG_LETTER_VOWEL[ch]))
            i = j
            continue

        segs.append(Seg("C", ch))
        if SHADDA in marks:
            segs.append(Seg("C", ch, geminate=True))
        if haraka:
            nxt = chars[j] if j < len(chars) else ""
            nxt_marks = chars[j + 1] if j + 1 < len(chars) else ""
            absorbs = nxt == MADD_FOR[haraka] and nxt_marks not in (
                FATHA, DAMMA, KASRA, SUKUN, SHADDA)
            if absorbs:
                segs.append(Seg("LV", HARAKA_TO_VOWEL[haraka]))
                j += 1  # consume the madd letter
            else:
                segs.append(Seg("SV", HARAKA_TO_VOWEL[haraka]))
        i = j
    return segs


def normalize_latin(form: str) -> str:
    """Lower-case and reduce expressive lengthening (3+ repeats) to one letter.

    Mid-word capitals are dropped only for alignment; the case rule of the
    protocol is applied to the consonant layer, not here.
    """
    s = form.lower()
    s = re.sub(r"(.)\1{2,}", r"\1", s)
    return re.sub(r"[^a-z0-9']", "", s)


def latin_units(form: str) -> list[str]:
    s = normalize_latin(form)
    units, i = [], 0
    while i < len(s):
        two = s[i:i + 2]
        if two in DIGRAPHS:
            units.append(two)
            i += 2
        else:
            units.append(s[i])
            i += 1
    return units


def _sub_cost(seg: Seg, unit: str) -> float:
    is_vowel = unit in LATIN_VOWELS
    if seg.kind == "C":
        if not is_vowel:
            return 0.0 if unit in CANDIDATES.get(seg.value, set()) else 1.0
        return 0.8 if seg.value in WEAK else 1.5
    # vowels
    if is_vowel:
        return 0.0
    if seg.kind == "LV" and ((seg.value == "i" and unit == "y") or (seg.value == "u" and unit == "w")):
        return 0.2
    return 1.5


def _del_cost(seg: Seg) -> float:
    if seg.kind == "C":
        if seg.geminate:
            return 0.3
        return 0.6 if seg.value in WEAK else 1.0
    return 0.9 if seg.kind == "SV" else 0.5


def _ins_cost(unit: str) -> float:
    return 0.7 if unit in LATIN_VOWELS else 1.0


def align(segs: list[Seg], units: list[str]) -> list[tuple[Seg | None, str | None]]:
    """Minimum-cost alignment (Needleman-Wunsch) between Arabic segments and Latin units."""
    n, m = len(segs), len(units)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + _del_cost(segs[i - 1])
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + _ins_cost(units[j - 1])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + _sub_cost(segs[i - 1], units[j - 1]),
                          D[i - 1][j] + _del_cost(segs[i - 1]),
                          D[i][j - 1] + _ins_cost(units[j - 1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i][j] - (D[i - 1][j - 1] + _sub_cost(segs[i - 1], units[j - 1]))) < 1e-9:
            out.append((segs[i - 1], units[j - 1])); i -= 1; j -= 1
        elif i > 0 and abs(D[i][j] - (D[i - 1][j] + _del_cost(segs[i - 1]))) < 1e-9:
            out.append((segs[i - 1], None)); i -= 1
        else:
            out.append((None, units[j - 1])); j -= 1
    return out[::-1]


def short_vowel_slots(arabic_vocalised: str, latin: str) -> list[bool]:
    """For each short-vowel slot, whether a Latin vowel occupies it."""
    pairs = align(parse_vocalised(arabic_vocalised), latin_units(latin))
    return [u is not None and u in LATIN_VOWELS for s, u in pairs if s is not None and s.kind == "SV"]


def vowel_rate(word_pairs) -> float:
    """Explicitness rate over (vocalised Arabic, Latin form) pairs. Long vowels excluded."""
    slots = shown = 0
    for arabic, latin in word_pairs:
        flags = short_vowel_slots(arabic, latin)
        slots += len(flags)
        shown += sum(flags)
    return shown / slots if slots else float("nan")
