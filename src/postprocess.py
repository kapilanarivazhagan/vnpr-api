# src/postprocess.py

from rapidfuzz import fuzz

from src.config import (
    CONFUSION_MAP,
    DIGIT_TO_LETTER,
    LETTER_TO_DIGIT,
    VALID_STATES,
)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_plate(text: str) -> str:
    return "".join(CONFUSION_MAP.get(c, c) for c in text)

# ============================================================
# PLATE GRAMMAR ENFORCEMENT (INDIA)
# ============================================================

def apply_plate_grammar(clean_lines):
    """
    Input: list of character lists (lines)
    Output: corrected plate string
    """
    plate = "".join("".join(line) for line in clean_lines)

    if len(plate) < 8:
        return plate

    chars = list(plate)

    # STATE (0–1): letters only
    for i in (0, 1):
        if chars[i].isdigit():
            chars[i] = DIGIT_TO_LETTER.get(chars[i], chars[i])

    state = "".join(chars[0:2])
    if state not in VALID_STATES:
        for i in (0, 1):
            if chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]
        for i in (0, 1):
            if chars[i].isdigit():
                chars[i] = DIGIT_TO_LETTER.get(chars[i], chars[i])

    # DISTRICT (2–3): digits only
    for i in (2, 3):
        if chars[i].isalpha():
            chars[i] = LETTER_TO_DIGIT.get(chars[i], "0")

    # SERIES (4–5): letters only
    for i in (4, 5):
        if chars[i].isdigit():
            chars[i] = DIGIT_TO_LETTER.get(chars[i], chars[i])

    # NUMBER (last 4): digits only
    for i in range(len(chars) - 4, len(chars)):
        if chars[i].isalpha():
            chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])

    return "".join(chars)

# ============================================================
# VERIFICATION / MATCHING
# ============================================================

def verify_plate(recognized: str, assigned: str | None = None):
    """
    Compare recognized plate with assigned vehicle number
    """
    if not assigned:
        return {
            "recognized": recognized,
            "verdict": "NO_REFERENCE"
        }

    norm_rec = normalize_plate(recognized)
    norm_ass = normalize_plate(assigned)

    similarity = fuzz.ratio(norm_rec, norm_ass)

    if similarity >= 92:
        verdict = "MATCH"
    elif similarity >= 80:
        verdict = "POSSIBLE_MATCH"
    else:
        verdict = "NOT_MATCH"

    return {
        "recognized": recognized,
        "normalized": norm_rec,
        "assigned": assigned,
        "similarity": similarity,
        "verdict": verdict
    }
