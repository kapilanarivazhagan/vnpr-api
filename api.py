# api.py

import os
import cv2
import numpy as np
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import JSONResponse

from src.pipeline import run_anpr

# ============================================================
# API KEY CONFIG (ENV VAR)
# ============================================================

API_KEY = os.getenv("FACE_API_KEY")
API_KEY_HEADER = "X-API-Key"

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
    version="v1"
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
# ANPR ENDPOINT
# ============================================================

@app.post("/anpr")
async def anpr_api(
    image: UploadFile = File(...),
    assigned_vehicle_number: Optional[str] = None,
    _: None = Depends(verify_api_key)
):
    try:
        contents = await image.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image file")

        results = run_anpr(
            image=img,
            assigned_vehicle_number=assigned_vehicle_number
        )

        return JSONResponse(content={
            "plates_detected": len(results),
            "results": results
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )
