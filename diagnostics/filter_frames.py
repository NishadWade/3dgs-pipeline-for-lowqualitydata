"""
Filter iPhone v4 frames to remove discontinuous pose regions, then prep v7.

How it works:
1. Load extrinsics from v4 NPZ
2. Compute frame-to-frame camera displacement
3. Mark any frame within MARGIN frames of a "jump" (>JUMP_THRESH) as bad
4. Copy only good frames to da3_output_iphone_v7_frames
5. Report coverage stats
"""
import sys, os, shutil
import numpy as np
from pathlib import Path

NPZ_PATH   = r"D:\3DGS_Project\da3_output_iphone_v4\exports\mini_npz\results.npz"
SRC_FRAMES = r"D:\3DGS_Project\da3_output_iphone_v4_frames"
DST_FRAMES = r"D:\3DGS_Project\da3_output_iphone_v7_frames"
JUMP_THRESH = 2.0   # world-units: jumps above this are discontinuities
MARGIN      = 5     # remove MARGIN frames on each side of a jump

# Load extrinsics
npz  = np.load(NPZ_PATH)
extr = npz['extrinsics'].astype(np.float32)  # (V, 3, 4) w2c
R    = extr[:, :3, :3]
t    = extr[:, :3, 3]
cam_pos = np.einsum('vij,vj->vi', -R.transpose(0, 2, 1), t)  # (V, 3) world positions
V = len(cam_pos)

# Frame-to-frame displacement
diffs = np.linalg.norm(np.diff(cam_pos, axis=0), axis=1)
print(f"Frames: {V}  |  mean disp: {diffs.mean():.4f}  max: {diffs.max():.4f}")

# Find jump indices
jump_at = np.where(diffs > JUMP_THRESH)[0]  # frame i where i->(i+1) is a jump
print(f"\nJumps > {JUMP_THRESH} units ({len(jump_at)} total):")
for j in jump_at:
    print(f"  frame {j:3d}->{j+1:3d}: {diffs[j]:.3f} units")

# Build bad frame mask
bad = np.zeros(V, dtype=bool)
for j in jump_at:
    lo = max(0, j - MARGIN + 1)
    hi = min(V - 1, j + MARGIN)
    bad[lo:hi+1] = True

n_bad  = bad.sum()
n_keep = (~bad).sum()
print(f"\nBad frames (within {MARGIN} of any jump): {n_bad} / {V}")
print(f"Frames to keep: {n_keep}")

# Find the frame files — sorted alphabetically (same order DA3 used)
src_dir = Path(SRC_FRAMES)
frame_files = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
frame_files = sorted(frame_files)
print(f"Frame files in source dir: {len(frame_files)}")
if len(frame_files) != V:
    print(f"WARNING: frame count mismatch ({len(frame_files)} files vs {V} NPZ poses)")
    n_use = min(len(frame_files), V)
    frame_files = frame_files[:n_use]
    bad = bad[:n_use]
    n_keep = (~bad).sum()

# Clear and create destination
dst_dir = Path(DST_FRAMES)
if dst_dir.exists():
    shutil.rmtree(dst_dir)
dst_dir.mkdir(parents=True)

# Copy good frames with new sequential naming (v7_XXXX.jpg)
kept_indices = []
copy_count   = 0
for i, fpath in enumerate(frame_files):
    if not bad[i]:
        copy_count += 1
        dst_name = f"v7_{copy_count:04d}.jpg"
        shutil.copy2(str(fpath), str(dst_dir / dst_name))
        kept_indices.append(i)

print(f"\nCopied {copy_count} frames to {DST_FRAMES}")
print(f"Input frame indices kept: {kept_indices[:5]} ... {kept_indices[-5:]}")

# Show continuous segments
segs = []
seg_start = kept_indices[0] if kept_indices else 0
prev = seg_start
for idx in kept_indices[1:]:
    if idx != prev + 1:
        segs.append((seg_start, prev))
        seg_start = idx
    prev = idx
segs.append((seg_start, prev))
print(f"\nContinuous segments ({len(segs)}):")
for s, e in segs:
    print(f"  frames {s:3d}-{e:3d}  ({e-s+1} frames)")
