# VNPR – Deployment Notes

## Overview
- Vehicle Number Plate Recognition (ANPR) service
- End-to-end pipeline: detection → OCR → grammar → verification
- Exposed via FastAPI

## Tech Stack
- Python 3.9+
- YOLO (Ultralytics) for plate & character detection
- MobileNetV2 for character OCR
- OpenCV for image handling
- RapidFuzz for plate similarity matching
- FastAPI + Uvicorn for API serving

## Architecture
- src/config.py       → constants & grammar rules
- src/models.py       → model loading (YOLO, OCR)
- src/ocr.py          → character-level OCR inference
- src/utils.py        → geometric & structural helpers
- src/postprocess.py → normalization, grammar, verification
- src/pipeline.py    → full ANPR orchestration
- vnpr.py             → local test runner
- api.py              → FastAPI wrapper

## Security
- API key required via HTTP header
- Header name: X-API-Key
- API key loaded from environment variable:
  FACE_API_KEY

## Local Run (API)
```bash
$env:FACE_API_KEY="my-test-key"
uvicorn api:app --reload
