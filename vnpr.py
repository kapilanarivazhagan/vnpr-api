# vnpr.py

import cv2
from src.pipeline import run_anpr

# ============================================================
# CONFIG (LOCAL TEST ONLY)
# ============================================================

IMAGE_PATH = r"C:\Users\SPURGE\Downloads\WhatsApp Image 2026-01-15 at 9.40.54 AM.jpeg"
ASSIGNED_VEHICLE_NUMBER = "KA03AN6757"  # optional

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    img = cv2.imread(IMAGE_PATH)

    if img is None:
        raise ValueError("Image not found or unreadable")

    results = run_anpr(
        image=img,
        assigned_vehicle_number=ASSIGNED_VEHICLE_NUMBER
    )

    print("\n================ ANPR RESULT ================\n")

    if not results:
        print("No plates detected")
    else:
        for i, r in enumerate(results, 1):
            print(f"[Plate {i}]")
            for k, v in r.items():
                print(f"{k:12}: {v}")
            print()

    print("============================================\n")
