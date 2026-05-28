"""
From every N-frame window (1 second of 30fps video sampled every 5 frames = 6 frames/sec),
pick the single best frame by: sharpness x exposure_quality.

Sharpness  = Laplacian variance (higher = sharper)
Exposure   = penalise very dark (<15 mean) and blown-out (>235 mean) frames
             score = 1 - abs(mean_brightness - 128) / 128  (peaks at mid-grey)
Score      = sharpness * (exposure_score + 0.1)

Usage:
    python 02_select_best_per_second.py --src-dir frames/ --dst-dir best_frames/
    python 02_select_best_per_second.py --src-dir frames/ --dst-dir best/ --frames-per-sec 6
"""
import cv2
import numpy as np
import os
import shutil
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Select best frame per second window")
parser.add_argument("--src-dir", required=True, help="Directory of extracted frames")
parser.add_argument("--dst-dir", required=True, help="Output directory for best frames")
parser.add_argument("--frames-per-sec", type=int, default=6,
                    help="Frames per 1-second window (default: 6, for every-5th-frame from 30fps)")
parser.add_argument("--prefixes", nargs="+", default=["v1", "v2", "v3", "v4"],
                    help="Video prefixes to process (default: v1 v2 v3 v4)")
args = parser.parse_args()

os.makedirs(args.dst_dir, exist_ok=True)

def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def exposure_score(img):
    mean = img.mean()
    if mean < 15:
        return 0.0
    if mean > 235:
        return 0.0
    return 1.0 - abs(mean - 128) / 128

src = Path(args.src_dir)
total_saved = 0

for prefix in args.prefixes:
    files = sorted(src.glob(f"{prefix}_*.jpg"))
    if not files:
        continue

    n = len(files)
    n_windows = (n + args.frames_per_sec - 1) // args.frames_per_sec
    saved = 0
    sharpness_scores = []

    for w in range(n_windows):
        window = files[w * args.frames_per_sec : (w + 1) * args.frames_per_sec]
        best_score = -1
        best_file  = None
        best_sharp = 0

        for f in window:
            img = cv2.imread(str(f))
            if img is None:
                continue
            sharp = sharpness(img)
            exp   = exposure_score(img)
            score = sharp * (exp + 0.1)
            if score > best_score:
                best_score = score
                best_file  = f
                best_sharp = sharp

        if best_file is None:
            continue

        saved += 1
        total_saved += 1
        dst_name = f"{prefix}_{saved:04d}.jpg"
        shutil.copy2(str(best_file), os.path.join(args.dst_dir, dst_name))
        sharpness_scores.append(best_sharp)

    if sharpness_scores:
        print(f"[{prefix}] {n} frames -> {saved} best-per-second  "
              f"(sharpness: mean={np.mean(sharpness_scores):.0f}  min={np.min(sharpness_scores):.0f})")

print(f"\nTotal selected: {total_saved} -> {args.dst_dir}")
