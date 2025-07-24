import cv2
import numpy as np
import os
from math import floor

# ---------------- CONFIG ----------------
VIDEO_PATH   = "videos/P303772MWP_10fps_Test1.avi"
STRIP_H      = 8           # rows copied each time we have >=1 px net motion
MOTION_THR   = 2.0         # mean abs-diff threshold → “motion started”
OUT_DIR      = "saved"
OUT_FILE     = "full_panorama_corrected.png"
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"❌ cannot open {VIDEO_PATH}")

ret, prev = cap.read()
if not ret:
    raise SystemExit("❌ empty video")

H, W = prev.shape[:2]
gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# choose a wide horizontal ROI in the *middle* of the frame for correlation
roi_y1, roi_y2 = int(H*0.25), int(H*0.75)
roi_prev = gray_prev[roi_y1:roi_y2, :]

cum_shift = 0.0          # cumulative sub-pixel motion
rows      = []           # list of STRIP_H-row bands

frame_idx = 1
started   = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------- wait until the tube actually starts moving --------
    if not started:
        diff = cv2.absdiff(gray_prev, gray)
        if diff.mean() > MOTION_THR:
            started = True
            print(f"🚀 motion starts at frame {frame_idx}")
        else:
            gray_prev = gray
            roi_prev  = gray[roi_y1:roi_y2, :]
            continue  # still idle; skip
    # ------------------------------------------------------------

    # ---- sub-pixel vertical shift between consecutive frames ----
    roi     = gray[roi_y1:roi_y2, :]
    (shift_y, _), _ = cv2.phaseCorrelate(np.float32(roi_prev), np.float32(roi))
    cum_shift += shift_y          # positive if surface moves up

    # When 1 px (or more) of *new* surface enters, copy that band
    while cum_shift >= 1.0:
        y_src = int(H * 0.5 + STRIP_H/2)  # centre slice
        band  = frame[y_src : y_src+STRIP_H, :, :]
        rows.append(band)
        cum_shift -= 1.0
    while cum_shift <= -1.0:              # opposite direction just in case
        y_src = int(H * 0.5 - STRIP_H/2)
        band  = frame[y_src : y_src+STRIP_H, :, :]
        rows.append(band)
        cum_shift += 1.0

    # roll state
    gray_prev = gray
    roi_prev  = roi

    if frame_idx % 500 == 0:
        print(f"🔄 processed {frame_idx}")

cap.release()
if not rows:
    raise SystemExit("❌ no motion detected / nothing stitched")

panorama = np.vstack(rows)
cv2.imwrite(os.path.join(OUT_DIR, OUT_FILE), panorama)
print(f"💾 saved {OUT_DIR}/{OUT_FILE}  {panorama.shape[1]}×{panorama.shape[0]} px")
