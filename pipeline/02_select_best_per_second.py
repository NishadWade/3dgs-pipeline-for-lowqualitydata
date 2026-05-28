"""
From every 6-frame window (1 second of 30fps video, sampled every 5 frames),
pick the single best frame by: sharpness x exposure_quality.

Sharpness  = Laplacian variance (higher = sharper)
Exposure   = penalise very dark (<30 mean) and blown-out (>220 mean) frames
             score = 1 - abs(mean_brightness - 128) / 128  (peaks at mid-grey)

Output: da3_output_iphone_v8_best  -- one frame per second per video
"""
import cv2
import numpy as np
import os
import shutil
from pathlib import Path

SRC_DIR = r"D:\3DGS_Project\da3_output_iphone_v8_frames"
DST_DIR = r"D:\3DGS_Project\da3_output_iphone_v8_best"
FRAMES_PER_SEC = 6   # every 5th frame from 30fps = 6 per second

os.makedirs(DST_DIR, exist_ok=True)

def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def exposure_score(img):
    mean = img.mean()
    if mean < 15:       # black frame
        return 0.0
    if mean > 235:      # completely blown out
        return 0.0
    return 1.0 - abs(mean - 128) / 128   # peaks at mid-grey

src = Path(SRC_DIR)
total_saved = 0

for prefix in ["v1", "v2", "v3", "v4"]:
    files = sorted(src.glob(f"{prefix}_*.jpg"))
    if not files:
        continue

    n = len(files)
    n_windows = (n + FRAMES_PER_SEC - 1) // FRAMES_PER_SEC
    saved = 0
    sharpness_scores = []

    for w in range(n_windows):
        window = files[w * FRAMES_PER_SEC : (w + 1) * FRAMES_PER_SEC]
        best_score = -1
        best_file  = None
        best_sharp = 0

        for f in window:
            img = cv2.imread(str(f))
            if img is None:
                continue
            sharp = sharpness(img)
            exp   = exposure_score(img)
            score = sharp * (exp + 0.1)   # exposure as weight; +0.1 so sharp dark frames aren't zeroed
            if score > best_score:
                best_score = score
                best_file  = f
                best_sharp = sharp

        if best_file is None:
            continue

        saved += 1
        total_saved += 1
        dst_name = f"{prefix}_{saved:04d}.jpg"
        shutil.copy2(str(best_file), os.path.join(DST_DIR, dst_name))
        sharpness_scores.append(best_sharp)

    print(f"[{prefix}] {n} frames -> {saved} best-per-second  "
          f"(sharpness: mean={np.mean(sharpness_scores):.0f}  min={np.min(sharpness_scores):.0f})")

print(f"\nTotal selected: {total_saved} -> {DST_DIR}")
