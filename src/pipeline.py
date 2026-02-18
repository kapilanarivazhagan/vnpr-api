# src/pipeline.py

import cv2

from src.models import plate_yolo, char_yolo
from src.ocr import ocr_crop_with_conf
from src.utils import (
    pad_plate,
    remove_duplicate_boxes,
    group_boxes_into_lines,
)
from src.postprocess import apply_plate_grammar, verify_plate


# ============================================================
# PLATE DETECTION
# ============================================================

def detect_plates(image):
    """
    Detect number plates in an image.
    Returns list of cropped plate images.
    """
    result = plate_yolo.predict(
        image,
        imgsz=640,
        conf=0.25,
        verbose=False
    )[0]

    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        boxes.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        })

    boxes = remove_duplicate_boxes(boxes)

    plates = [
        image[b["y1"]:b["y2"], b["x1"]:b["x2"]]
        for b in boxes
        if image[b["y1"]:b["y2"], b["x1"]:b["x2"]].size
    ]

    return plates


# ============================================================
# CHARACTER RECOGNITION ON PLATE
# ============================================================

def recognize_plate_text(plate_img):
    """
    Plate image -> recognized plate string
    """
    padded, pad = pad_plate(plate_img)

    result = char_yolo.predict(
        padded,
        imgsz=512,
        conf=0.2,
        verbose=False
    )[0]

    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        boxes.append({
            "x1": max(0, x1 - pad),
            "y1": y1,
            "x2": max(0, x2 - pad),
            "y2": y2
        })

    lines = group_boxes_into_lines(boxes)
    clean_lines = []

    for line in lines:
        chars = []
        for b in line:
            crop = plate_img[b["y1"]:b["y2"], b["x1"]:b["x2"]]
            if crop.size:
                topk = ocr_crop_with_conf(crop)
                chars.append(topk[0][0])
        clean_lines.append(chars)

    return apply_plate_grammar(clean_lines)


# ============================================================
# FULL PIPELINE (PUBLIC API)
# ============================================================

def run_anpr(image, assigned_vehicle_number=None):
    """
    Full ANPR pipeline.
    """
    plates = detect_plates(image)
    results = []

    for plate in plates:
        plate_text = recognize_plate_text(plate)
        verdict = verify_plate(
            plate_text,
            assigned_vehicle_number
        )
        results.append(verdict)

    return results
