"""
Extract every Nth frame from one or more videos.
Default: every 5th frame from 30fps video = 6 candidates per second,
which feeds into 02_select_best_per_second.py.

Usage:
    python 01_extract_frames.py --video-dir /path/to/videos --out-dir /path/to/frames
    python 01_extract_frames.py --videos vid1.mov vid2.mov --out-dir /path/to/frames
    python 01_extract_frames.py --videos vid1.mov --out-dir frames --interval 10
"""
import cv2
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Extract every Nth frame from videos")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--video-dir", help="Directory containing video files (.mov, .mp4)")
group.add_argument("--videos", nargs="+", help="One or more video file paths")
parser.add_argument("--out-dir", required=True, help="Output directory for extracted frames")
parser.add_argument("--interval", type=int, default=5,
                    help="Save every Nth frame (default: 5 → 6 frames/sec from 30fps)")
parser.add_argument("--quality", type=int, default=95, help="JPEG quality 0-100 (default: 95)")
parser.add_argument("--prefix", nargs="+",
                    help="Prefix labels for each video (e.g. v1 v2). Defaults to v1, v2, ...")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

if args.video_dir:
    exts = (".mov", ".mp4", ".avi", ".mkv")
    video_paths = sorted(p for p in Path(args.video_dir).iterdir() if p.suffix.lower() in exts)
else:
    video_paths = [Path(v) for v in args.videos]

if args.prefix:
    if len(args.prefix) != len(video_paths):
        parser.error(f"--prefix count ({len(args.prefix)}) must match video count ({len(video_paths)})")
    prefixes = args.prefix
else:
    prefixes = [f"v{i+1}" for i in range(len(video_paths))]

total_saved = 0
for video_path, prefix in zip(video_paths, prefixes):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[{prefix}] WARNING: could not open {video_path}")
        continue
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[{prefix}] {total_frames} frames @ {fps:.0f}fps ({total_frames/fps:.0f}s) -- {video_path.name}")

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.interval == 0:
            saved += 1
            total_saved += 1
            fname = f"{prefix}_{saved:04d}.jpg"
            cv2.imwrite(os.path.join(args.out_dir, fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        frame_idx += 1

    cap.release()
    print(f"  saved={saved}  (every {args.interval} frames)")

print(f"\nTotal frames saved: {total_saved} -> {args.out_dir}")
