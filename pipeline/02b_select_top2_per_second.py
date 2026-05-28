"""
From every N-frame window (1 second of 30fps video sampled every 5 frames = 6 frames/sec),
pick the TOP 2 frames by: sharpness x exposure_quality.

Produces ~2x the frames of 02_select_best_per_second.py.
More frames = denser 3DGS reconstruction, at the cost of longer DA3 inference.

Usage:
    python 02b_select_top2_per_second.py --src-dir frames/ --dst-dir best2_frames/
    python 02b_select_top2_per_second.py --src-dir frames/ --dst-dir best/ --top-n 3
"""
import cv2
import numpy as np
import os
import shutil
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Select top N frames per second window")
parser.add_argument("--src-dir", required=True, help="Directory of extracted frames")
parser.add_argument("--dst-dir", required=True, help="Output directory for selected frames")
parser.add_argument("--frames-per-sec", type=int, default=6,
                    help="Frames per 1-second window (default: 6, for every-5th-frame from 30fps)")
parser.add_argument("--top-n", type=int, default=2,
                    help="Number of best frames to keep per window (default: 2)")
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

        scored = []
        for f in window:
            img = cv2.imread(str(f))
            if img is None:
                continue
            sharp = sharpness(img)
            exp   = exposure_score(img)
            score = sharp * (exp + 0.1)
            scored.append((score, sharp, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        picks = scored[:args.top_n]

        # Restore temporal order within the window
        picks.sort(key=lambda x: x[2].name)

        for score, sharp, f in picks:
            saved += 1
            total_saved += 1
            dst_name = f"{prefix}_{saved:04d}.jpg"
            shutil.copy2(str(f), os.path.join(args.dst_dir, dst_name))
            sharpness_scores.append(sharp)

    if sharpness_scores:
        print(f"[{prefix}] {n} frames -> {saved} top-{args.top_n}-per-second  "
              f"(sharpness: mean={np.mean(sharpness_scores):.0f}  min={np.min(sharpness_scores):.0f})")

print(f"\nTotal selected: {total_saved} -> {args.dst_dir}")
