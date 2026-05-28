"""
Generate depth_vis images from an existing DA3 mini_npz export.
Applies the same jet colormap DA3 uses internally.
"""
import numpy as np
import cv2
import os
from pathlib import Path

NPZ_PATH  = r"D:\3DGS_Project\da3_output_iphone_v8\exports\mini_npz\results.npz"
IMG_DIR   = r"D:\3DGS_Project\da3_output_iphone_v8_best"
OUT_DIR   = r"D:\3DGS_Project\da3_output_iphone_v8\depth_vis"

os.makedirs(OUT_DIR, exist_ok=True)

npz   = np.load(NPZ_PATH)
depth = npz['depth']          # (N, H, W)  -- metric depth in world units
conf  = npz['conf']           # (N, H, W)  -- confidence map

# Sort input images to match the order DA3 processed them
img_files = sorted(Path(IMG_DIR).glob("*.jpg")) + sorted(Path(IMG_DIR).glob("*.png"))
img_files = sorted(img_files)
print(f"Depth frames: {depth.shape[0]}  |  Image files: {len(img_files)}")

for i in range(depth.shape[0]):
    d = depth[i]              # (H, W)

    # Normalise to 0-255 using per-frame percentile
    # Invert so close=red (warm), far=blue (cold) -- standard disparity convention
    d_min = np.percentile(d, 2)
    d_max = np.percentile(d, 98)
    d_norm = np.clip((d - d_min) / (d_max - d_min + 1e-8), 0, 1)
    d_uint8 = ((1.0 - d_norm) * 255).astype(np.uint8)  # inverted: close=255=red

    # Apply jet colormap (close to DA3's default)
    d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)

    # Stack input image (resized to match depth) + depth side by side
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
    cv2.imwrite(os.path.join(OUT_DIR, f"{fname}_depth.jpg"), out_img,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{depth.shape[0]}")

print(f"\nSaved {depth.shape[0]} depth vis images -> {OUT_DIR}")
