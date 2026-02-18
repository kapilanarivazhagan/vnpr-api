# src/models.py

import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import mobilenet_v2

from src.config import (
    CLASSES,
    PLATE_YOLO_MODEL,
    CHAR_YOLO_MODEL,
    OCR_MODEL_PATH,
)

# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# YOLO MODELS
# ============================================================

plate_yolo = YOLO(PLATE_YOLO_MODEL)
char_yolo  = YOLO(CHAR_YOLO_MODEL)

# ============================================================
# OCR MODEL (MobileNetV2)
# ============================================================

def build_ocr_model():
    model = mobilenet_v2(weights=None)

    # Change first conv to accept 1-channel (grayscale)
    model.features[0][0] = nn.Conv2d(
        1,
        model.features[0][0].out_channels,
        kernel_size=3,
        stride=2,
        padding=1
    )

    # Change classifier head
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASSES)
    )

    return model


ocr_model = build_ocr_model().to(device)
ocr_model.load_state_dict(
    torch.load(OCR_MODEL_PATH, map_location=device)
)
ocr_model.eval()
