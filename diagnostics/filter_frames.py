"""
Filter frames to remove pose discontinuity regions.

Loads extrinsics from a DA3 mini_npz, finds frames where the camera jumps
more than JUMP_THRESH world units in one step, then removes MARGIN frames
on either side of each jump. The clean subset is copied to a new directory.

Use this before re-running DA3 if your scene has pose discontinuities
(multiple separate video clips, tracking failures, etc.)

Usage:
    python filter_frames.py \\
        --npz       da3_output/exports/mini_npz/results.npz \\
        --src-dir   input_frames/ \\
        --dst-dir   filtered_frames/ \\
        --threshold 2.0 \\
        --margin    5
"""
import sys
import os
import shutil
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Remove frames near pose discontinuities")
parser.add_argument("--npz",       required=True, help="DA3 mini_npz results.npz")
parser.add_argument("--src-dir",   required=True, help="Source frame directory")
parser.add_argument("--dst-dir",   required=True, help="Output directory for filtered frames")
parser.add_argument("--threshold", type=float, default=2.0,
                    help="Jump threshold in world units (default: 2.0)")
parser.add_argument("--margin",    type=int, default=5,
                    help="Frames to remove on each side of a jump (default: 5)")
args = parser.parse_args()

os.makedirs(args.dst_dir, exist_ok=True)

npz  = np.load(args.npz)
extr = npz["extrinsics"].astype(np.float32)
R    = extr[:, :3, :3]
t    = extr[:, :3, 3]
cam_pos = np.einsum("vij,vj->vi", -R.transpose(0, 2, 1), t)
V = len(cam_pos)

diffs  = np.linalg.norm(np.diff(cam_pos, axis=0), axis=1)
print(f"Frames: {V}  |  mean disp: {diffs.mean():.4f}  max: {diffs.max():.4f}")

jump_at = np.where(diffs > args.threshold)[0]
print(f"\nJumps > {args.threshold} units ({len(jump_at)} total):")
for j in jump_at:
    print(f"  frame {j:3d}->{j+1:3d}: {diffs[j]:.3f} units")

bad = np.zeros(V, dtype=bool)
for j in jump_at:
    lo = max(0, j - args.margin + 1)
    hi = min(V - 1, j + args.margin)
    bad[lo:hi+1] = True

n_bad  = bad.sum()
n_keep = (~bad).sum()
print(f"\nBad frames (within {args.margin} of any jump): {n_bad} / {V}")
print(f"Frames to keep: {n_keep}")

frame_files = sorted(Path(args.src_dir).glob("*.jpg")) + sorted(Path(args.src_dir).glob("*.png"))
frame_files = sorted(frame_files)

n_use = min(V, len(frame_files))
if len(frame_files) != V:
    print(f"\nWARNING: {len(frame_files)} image files vs {V} NPZ poses — using first {n_use}")

saved = 0
for i in range(n_use):
    if bad[i]:
        continue
    src_f = frame_files[i]
    dst_f = os.path.join(args.dst_dir, f"frame_{saved:04d}{src_f.suffix}")
    shutil.copy2(str(src_f), dst_f)
    saved += 1

print(f"\nSaved {saved} frames -> {args.dst_dir}")
