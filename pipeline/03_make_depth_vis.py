"""
Generate side-by-side depth visualisation images from a DA3 mini_npz export.
Applies jet colormap with red=close, blue=far (standard disparity convention).

Run DA3 with --export-format mini_npz first to get the results.npz file.

Usage:
    python 03_make_depth_vis.py --npz results.npz --img-dir best_frames/ --out-dir depth_vis/
"""
import numpy as np
import cv2
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate depth visualisations from DA3 mini_npz export")
parser.add_argument("--npz",     required=True, help="Path to DA3 results.npz (mini_npz export)")
parser.add_argument("--img-dir", required=True, help="Directory of input images (same order DA3 processed them)")
parser.add_argument("--out-dir", required=True, help="Output directory for depth vis images")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

npz   = np.load(args.npz)
depth = npz["depth"]   # (N, H, W) metric depth

img_files = sorted(Path(args.img_dir).glob("*.jpg")) + sorted(Path(args.img_dir).glob("*.png"))
img_files = sorted(img_files)
print(f"Depth frames: {depth.shape[0]}  |  Image files: {len(img_files)}")

for i in range(depth.shape[0]):
    d = depth[i]

    d_min = np.percentile(d, 2)
    d_max = np.percentile(d, 98)
    d_norm = np.clip((d - d_min) / (d_max - d_min + 1e-8), 0, 1)
    d_uint8 = ((1.0 - d_norm) * 255).astype(np.uint8)  # inverted: close=red

    d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)

    if i < len(img_files):
        src = cv2.imread(str(img_files[i]))
        if src is not None:
            h, w = d_color.shape[:2]
            src_resized = cv2.resize(src, (w, h))
            out_img = np.hstack([src_resized, d_color])
        else:
            out_img = d_color
    else:
        out_img = d_color

    fname = img_files[i].stem if i < len(img_files) else f"frame_{i:04d}"
    cv2.imwrite(os.path.join(args.out_dir, f"{fname}_depth.jpg"), out_img,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{depth.shape[0]}")

print(f"\nSaved {depth.shape[0]} depth vis images -> {args.out_dir}")
