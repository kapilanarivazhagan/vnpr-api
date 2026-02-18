import os
import cv2
import numpy as np
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends, Form
from fastapi.responses import JSONResponse

from src.pipeline import run_anpr

# ============================================================
# API KEY CONFIG
# ============================================================

API_KEY = os.getenv("FACE_API_KEY")

def verify_api_key(x_api_key: str = Header(...)):
    if API_KEY is None:
        raise HTTPException(
            status_code=500,
            detail="API key not configured on server"
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="VNPR / ANPR API",
    description="Vehicle Number Plate Recognition Service",
    version="v2"
)

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "vnpr"
    }

# ============================================================
# CONFIDENCE LABEL LOGIC
# ============================================================

def get_confidence_level(similarity: float):
    if similarity >= 95:
        return "HIGH"
    elif similarity >= 80:
        return "MEDIUM"
    else:
        return "LOW"

# ============================================================
# MAIN ANPR ENDPOINT
# ============================================================

@app.post("/anpr")
async def anpr_api(
    image: UploadFile = File(...),
    assigned_vehicle_number: Optional[str] = Form(None),  # <-- IMPORTANT FIX
    _: None = Depends(verify_api_key)
):
    try:
        # Read uploaded image
        contents = await image.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image file")

        # Run pipeline
        results = run_anpr(
            image=img,
            assigned_vehicle_number=assigned_vehicle_number
        )

        # If nothing detected
        if not results or len(results) == 0:
            return JSONResponse(content={
                "matched": False,
                "assigned_vehicle_number": assigned_vehicle_number,
                "recognized_vehicle_number": None,
                "similarity": 0.0,
                "verdict": "NO_PLATE_DETECTED",
                "confidence_level": "LOW"
            })

        # Take best result
        best = results[0]

        # Extract detected plate text safely
        recognized = (
            best.get("final_vehicle_number")
            or best.get("recognized")
            or best.get("plate")
            or best.get("recognized_vehicle_number")
        )

        similarity = float(best.get("similarity", 0))
        verdict = best.get("verdict", "UNKNOWN")

        matched = verdict == "MATCH"

        return JSONResponse(content={
            "matched": matched,
            "assigned_vehicle_number": assigned_vehicle_number,
            "recognized_vehicle_number": recognized,
            "similarity": similarity,
            "verdict": verdict,
            "confidence_level": get_confidence_level(similarity)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "matched": False,
                "error": str(e)
            }
        )
