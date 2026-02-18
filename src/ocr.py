# src/ocr.py

import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.models import ocr_model, device
from src.config import CLASSES, TOP_K

# ============================================================
# OCR TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# ============================================================
# OCR INFERENCE (CHAR LEVEL)
# ============================================================

def ocr_crop_with_conf(crop):
    """
    Input:
        crop (np.ndarray): BGR image of a single character
    Output:
        List of (char, probability), top-K
    """
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inp = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(ocr_model(inp), dim=1)[0]

    topk = torch.topk(probs, TOP_K)

    return [
        (CLASSES[i], float(p))
        for i, p in zip(topk.indices, topk.values)
    ]

# ============================================================
# CONTEXT-AWARE CHARACTER FIXING
# ============================================================

def snap_char_with_context(topk, expected, allow_7_to_1=False):
    """
    Resolve OCR ambiguity using context (LETTER or DIGIT expected)
    """
    ch0, p0 = topk[0]

    # ---------------- LETTER EXPECTED ----------------
    if expected == "L":

        if p0 >= 0.90:
            return ch0

        if ch0 == "N":
            for ch, p in topk[1:]:
                if ch == "W" and p >= p0 * 0.90:
                    return "W"

        if ch0 == "H":
            for ch, p in topk[1:]:
                if ch == "M" and p >= p0 * 0.90:
                    return "M"

        if ch0 == "M":
            return "M"

        letter_fallback = {
            "0": {"O": 0.40, "D": 0.35, "Q": 0.25},
            "1": {"I": 0.60, "L": 0.40},
            "2": {"Z": 1.00},
            "5": {"S": 1.00},
            "6": {"G": 1.00},
            "7": {"Z": 1.00},
            "8": {"B": 1.00},
            "V": {"W": 1.00},
            "U": {"V": 1.00},
            "K": {"X": 0.50, "R": 0.50},
            "P": {"R": 1.00},
            "C": {"G": 1.00},
        }

        scores = {}
        if ch0 in letter_fallback:
            for tgt, w in letter_fallback[ch0].items():
                for ch, p in topk:
                    if ch == tgt and p > 0.20:
                        scores[tgt] = max(scores.get(tgt, 0), p * w)

        return max(scores, key=scores.get) if scores else ch0

    # ---------------- DIGIT EXPECTED ----------------
    else:

        if p0 >= 0.80:
            return ch0

        if ch0 == "7" and allow_7_to_1:
            for ch, p in topk[1:]:
                if ch in ("1", "I") and p > 0.20:
                    return "1"

        digit_fallback = {
            "O": {"0": 1.00},
            "Q": {"0": 1.00},
            "D": {"0": 1.00},
            "I": {"1": 1.00},
            "L": {"1": 1.00},
            "S": {"5": 1.00},
            "Z": {"7": 0.70, "2": 0.30},
            "A": {"4": 0.80, "7": 0.20},
            "B": {"8": 1.00},
            "G": {"6": 0.60, "0": 0.40},
            "T": {"1": 0.60, "7": 0.40},
            "Y": {"7": 1.00},
            "X": {"7": 1.00},
            "R": {"7": 1.00},
        }

        scores = {}
        if ch0 in digit_fallback:
            for tgt, w in digit_fallback[ch0].items():
                for ch, p in topk:
                    if ch == tgt and p > 0.20:
                        scores[tgt] = max(scores.get(tgt, 0), p * w)

        return max(scores, key=scores.get) if scores else ch0
