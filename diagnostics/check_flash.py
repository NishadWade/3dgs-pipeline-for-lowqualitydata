"""
Quick check: count flash frames in one or more DA3 gs_video files.
Flags frames where the GS render (left half) is dominantly red or blue.

Usage:
    python check_flash.py --videos output_v1/gs_video/0000.mp4
    python check_flash.py --videos v1.mp4 v2.mp4 --threshold 2.0
"""
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Count flash frames in DA3 gs_video files")
parser.add_argument("--videos", nargs="+", required=True, help="One or more MP4 video paths")
parser.add_argument("--threshold", type=float, default=2.0,
                    help="R/B ratio threshold to flag as flash (default: 2.0)")
args = parser.parse_args()

def analyze_video(path, threshold):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open: {path}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[{path}] {total} frames @ {fps:.0f}fps ({total/fps:.1f}s)")

    flash_frames = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        gs = frame[:, :w//2, :]   # left half = GS render
        r = gs[:, :, 2].mean()
        g = gs[:, :, 1].mean()
        b = gs[:, :, 0].mean()
        if r > 80 and r > threshold * g and r > threshold * b:
            flash_frames.append(("RED",   fi, r, g, b))
        elif b > 80 and b > threshold * r and b > threshold * g:
            flash_frames.append(("BLUE",  fi, r, g, b))
        elif (r + g + b) / 3 > 200:
            flash_frames.append(("WHITE", fi, r, g, b))
        fi += 1
    cap.release()

    print(f"Flash frames: {len(flash_frames)} / {fi}  (threshold={threshold})")
    for ftype, fvi, r, g, b in flash_frames[:30]:
        print(f"  [{ftype}] frame={fvi:4d}  t={fvi/fps:.1f}s  R={r:.0f} G={g:.0f} B={b:.0f}")

for video_path in args.videos:
    analyze_video(video_path, args.threshold)
