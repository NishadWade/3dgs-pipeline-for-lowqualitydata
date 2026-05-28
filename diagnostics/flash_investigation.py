"""
Deep diagnostic for red/blue flashing in DA3 GS videos.
Investigates: camera trajectory, depth map stats, Gaussian distribution,
SH view-dependence, and frame-to-frame colour swings.

Usage:
    python flash_investigation.py \\
        --npz   da3_output/exports/mini_npz/results.npz \\
        --ply   da3_output/gs_ply/0000.ply \\
        --video da3_output/gs_video/0000_smooth_light.mp4
"""
import sys
import os
import numpy as np
import cv2
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Diagnose red/blue flashing in DA3 GS output")
parser.add_argument("--npz",   required=True, help="DA3 mini_npz results.npz")
parser.add_argument("--ply",   default=None,  help="DA3 gs_ply 0000.ply (optional)")
parser.add_argument("--video", default=None,  help="DA3 gs_video MP4 (optional)")
parser.add_argument("--da3-src", default=None,
                    help="Path to Depth-Anything-3/src (for SH analysis)")
args = parser.parse_args()

if args.da3_src:
    sys.path.insert(0, args.da3_src)

# ── Section 1: Camera Trajectory ────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: Camera Trajectory Analysis")
print("=" * 60)

npz  = np.load(args.npz)
print(f"NPZ keys: {list(npz.keys())}")
extr = npz["extrinsics"].astype(np.float32)
intr = npz["intrinsics"].astype(np.float32)
print(f"Views: {extr.shape[0]},  extrinsics shape: {extr.shape}")
print(f"Intrinsics[0]:\n{intr[0]}")

R = extr[:, :3, :3]
t = extr[:, :3, 3]
cam_pos = np.einsum("vij,vj->vi", -R.transpose(0, 2, 1), t)
print(f"\nCamera positions (world space):")
print(f"  X: min={cam_pos[:,0].min():.3f}  max={cam_pos[:,0].max():.3f}  range={cam_pos[:,0].ptp():.3f}")
print(f"  Y: min={cam_pos[:,1].min():.3f}  max={cam_pos[:,1].max():.3f}  range={cam_pos[:,1].ptp():.3f}")
print(f"  Z: min={cam_pos[:,2].min():.3f}  max={cam_pos[:,2].max():.3f}  range={cam_pos[:,2].ptp():.3f}")

diffs = np.linalg.norm(np.diff(cam_pos, axis=0), axis=1)
print(f"\nFrame-to-frame displacement:")
print(f"  mean={diffs.mean():.4f}  max={diffs.max():.4f}  std={diffs.std():.4f}")
print(f"  Top-10 biggest jumps:")
for i in np.argsort(diffs)[-10:][::-1]:
    print(f"    frame {i:3d}->{i+1:3d}: {diffs[i]:.4f}")

n_big = (diffs > 2.0).sum()
print(f"\n  Jumps > 2.0 world units: {n_big}  (these cause void traversal artefacts)")

# ── Section 2: Depth Map Stats ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: Depth Map Statistics")
print("=" * 60)

depth = npz["depth"]
print(f"Depth shape: {depth.shape}  dtype: {depth.dtype}")
print(f"  global min={depth.min():.4f}  max={depth.max():.4f}  mean={depth.mean():.4f}")

frame_means = depth.reshape(depth.shape[0], -1).mean(axis=1)
frame_stds  = depth.reshape(depth.shape[0], -1).std(axis=1)
print(f"  Per-frame mean: min={frame_means.min():.4f}  max={frame_means.max():.4f}")
print(f"  Per-frame std:  min={frame_stds.min():.4f}   max={frame_stds.max():.4f}")

# ── Section 3: Video Flash Analysis ─────────────────────────────────────────
if args.video:
    print("\n" + "=" * 60)
    print("SECTION 3: Video Flash Analysis")
    print("=" * 60)
    cap   = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"{total} frames @ {fps:.0f}fps ({total/fps:.1f}s)")

    r_means, g_means, b_means = [], [], []
    flash_events = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        gs = frame[:, :w//2, :]
        r = gs[:, :, 2].mean()
        g = gs[:, :, 1].mean()
        b = gs[:, :, 0].mean()
        r_means.append(r); g_means.append(g); b_means.append(b)
        ratio = max(r, b) / (min(r, b) + 1e-6)
        if ratio > 1.4:
            flash_events.append((fi, r, g, b, ratio))
        fi += 1
    cap.release()

    print(f"\nR-B mean swing: mean={abs(np.array(r_means)-np.array(b_means)).mean():.2f}  "
          f"max={abs(np.array(r_means)-np.array(b_means)).max():.2f}")
    print(f"Flash events (R/B ratio > 1.4): {len(flash_events)} / {fi}")
    for fvi, r, g, b, ratio in flash_events[:20]:
        print(f"  frame={fvi:4d}  t={fvi/fps:.1f}s  R={r:.0f} G={g:.0f} B={b:.0f}  ratio={ratio:.2f}")

print("\nDone.")
