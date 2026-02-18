# src/config.py

# ============================================================
# MODEL PATHS
# ============================================================

PLATE_YOLO_MODEL = "models/best_plate_finetuned_final.pt"
CHAR_YOLO_MODEL  = "models/best_char_yolo_finetuned.pt"
OCR_MODEL_PATH   = "models/mobilenet_char_classifier_final.pth"

# ============================================================
# OCR CONFIG
# ============================================================

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
TOP_K = 3

# ============================================================
# OCR CONFUSION MAP
# ============================================================

CONFUSION_MAP = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2", "S": "5",
    "B": "8", "G": "6",
    "M": "N", "H": "N",
}

# ============================================================
# PLATE GRAMMAR
# ============================================================

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}

LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
    "A": "4",
}

VALID_STATES = {
    "KA","TN","DL","MH","AP","TS","KL","WB","RJ","UP","MP","GJ",
    "HR","PB","CH","OD","BR","CG","JK","HP","UK","GA","AS",
    "MN","ML","MZ","NL","TR","SK","AR","AN","DN","DD","LD","PY"
}
