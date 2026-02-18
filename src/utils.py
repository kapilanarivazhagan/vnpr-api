# src/utils.py

import cv2
import numpy as np

# ============================================================
# PLATE PADDING
# ============================================================

def pad_plate(img, ratio=0.18):
    pad = int(img.shape[1] * ratio)
    padded = cv2.copyMakeBorder(
        img, 0, 0, pad, pad,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )
    return padded, pad

# ============================================================
# IOU + BOX CLEANUP
# ============================================================

def compute_iou(b1, b2):
    x1 = max(b1["x1"], b2["x1"])
    y1 = max(b1["y1"], b2["y1"])
    x2 = min(b1["x2"], b2["x2"])
    y2 = min(b1["y2"], b2["y2"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    area2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
    union = area1 + area2 - inter

    return 0 if union <= 0 else inter / union


def remove_duplicate_boxes(boxes, thr=0.42):
    out = []
    for b in boxes:
        keep = True
        for o in out:
            if compute_iou(b, o) > thr:
                keep = False
                break
        if keep:
            out.append(b)
    return out

# ============================================================
# CHARACTER STRUCTURE RECOVERY
# ============================================================

def group_boxes_into_lines(boxes):
    if not boxes:
        return []

    for b in boxes:
        b["cy"] = (b["y1"] + b["y2"]) / 2
        b["h"] = b["y2"] - b["y1"]

    avg_h = np.mean([b["h"] for b in boxes])
    thr = avg_h * 0.6

    boxes = sorted(boxes, key=lambda x: x["cy"])
    lines = [[boxes[0]]]

    for b in boxes[1:]:
        if abs(b["cy"] - np.mean([x["cy"] for x in lines[-1]])) <= thr:
            lines[-1].append(b)
        else:
            lines.append([b])

    return [sorted(line, key=lambda x: x["x1"]) for line in lines]


def remove_duplicate_chars(dets, iou_thresh=0.6):
    if not dets:
        return []

    out = [dets[0]]

    for c in dets[1:]:
        p = out[-1]

        xa = max(c["x1"], p["x1"])
        ya = max(c["y1"], p["y1"])
        xb = min(c["x2"], p["x2"])
        yb = min(c["y2"], p["y2"])

        inter = max(0, xb - xa) * max(0, yb - ya)
        a1 = (c["x2"] - c["x1"]) * (c["y2"] - c["y1"])
        a2 = (p["x2"] - p["x1"]) * (p["y2"] - p["y1"])
        iou = inter / (a1 + a2 - inter + 1e-6)

        if c["topk"][0][0] == p["topk"][0][0] and iou > iou_thresh:
            continue

        out.append(c)

    return out


def recover_line_to_length(dets, plate_img, target_len=5):
    recovered = 0
    if not dets:
        return dets, recovered

    widths = [d["x2"] - d["x1"] for d in dets]
    avg_w = np.mean(widths)

    if len(dets) < target_len:
        first = dets[0]
        if first["x1"] > avg_w * 1.2:
            center = first["x1"] - avg_w * 0.5
            half = avg_w * 0.8
            x1 = int(max(0, center - half))
            x2 = int(min(plate_img.shape[1], center + half))
            crop = plate_img[first["y1"]:first["y2"], x1:x2]
            if crop.size:
                dets.insert(0, {
                    "x1": x1, "x2": x2,
                    "y1": first["y1"], "y2": first["y2"],
                    "topk": None,
                    "synthetic": True
                })
                recovered += 1

    if len(dets) < target_len:
        last = dets[-1]
        right_gap = plate_img.shape[1] - last["x2"]
        if right_gap > avg_w * 1.2:
            center = last["x2"] + avg_w * 0.5
            half = avg_w * 0.8
            x1 = int(max(0, center - half))
            x2 = int(min(plate_img.shape[1], center + half))
            crop = plate_img[last["y1"]:last["y2"], x1:x2]
            if crop.size:
                dets.append({
                    "x1": x1, "x2": x2,
                    "y1": last["y1"], "y2": last["y2"],
                    "topk": None,
                    "synthetic": True
                })
                recovered += 1

    return dets, recovered
