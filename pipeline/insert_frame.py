"""
Insert a frame from the original v1 video between v1_0046 and v1_0047
in da3_output_iphone_v8_best, then generate a depth vis for it.

v1_0046 in best = best of source frames 1350-1375 (seconds 45-46 of video)
v1_0047 in best = best of source frames 1380-1405 (seconds 46-47 of video)
Midpoint source frame = ~1377 (t=45.9s)
"""
import cv2
import numpy as np
import os, shutil
from pathlib import Path

VIDEO_PATH = r"C:\Users\aistudio\Downloads\3DGS\Video May 18 2026, 11 48 56 AM.mov"
BEST_DIR   = r"D:\3DGS_Project\da3_output_iphone_v8_best"
NPZ_PATH   = r"D:\3DGS_Project\da3_output_iphone_v8\exports\mini_npz\results.npz"
DEPTH_DIR  = r"D:\3DGS_Project\da3_output_iphone_v8\depth_vis"

# The new frame will sort between v1_0046 and v1_0047 alphabetically
# v1_0046_5.jpg works: '5' < '.' is false... use v1_0046b.jpg
# Confirmed: v1_0046.jpg < v1_0046b.jpg < v1_0047.jpg
NEW_FRAME_NAME  = "v1_0046b.jpg"
NEW_DEPTH_NAME  = "v1_0046b_depth.jpg"
SOURCE_FRAME_IDX = 1377   # midpoint between the two 1-second windows

# --- Extract the source frame ---
print(f"Extracting source frame {SOURCE_FRAME_IDX} from v1 video...")
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, SOURCE_FRAME_IDX)
ret, frame = cap.read()
cap.release()
if not ret:
    print("ERROR: could not read frame")
    exit(1)

out_img_path = os.path.join(BEST_DIR, NEW_FRAME_NAME)
cv2.imwrite(out_img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"Saved: {out_img_path}  ({frame.shape[1]}x{frame.shape[0]})")

# --- Generate depth vis by interpolating existing depth frames 45 and 46 (0-indexed) ---
print("Generating depth vis by interpolating depth frames 46 and 47...")
npz = np.load(NPZ_PATH)
depth = npz['depth']       # (N, H, W)
# frames 45 and 46 in 0-indexed = v1_0046 and v1_0047 (1-indexed)
d_prev = depth[45]         # v1_0046
d_next = depth[46]         # v1_0047
d_interp = 0.5 * d_prev + 0.5 * d_next   # simple average (midpoint)

# Normalise and colorise (red=close, blue=far)
d_min  = np.percentile(d_interp, 2)
d_max  = np.percentile(d_interp, 98)
d_norm = np.clip((d_interp - d_min) / (d_max - d_min + 1e-8), 0, 1)
d_uint8 = ((1.0 - d_norm) * 255).astype(np.uint8)
d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)

# Resize source frame to match depth and stack side-by-side
h, w = d_color.shape[:2]
src_resized = cv2.resize(frame, (w, h))
out_vis = np.hstack([src_resized, d_color])

vis_path = os.path.join(DEPTH_DIR, NEW_DEPTH_NAME)
cv2.imwrite(vis_path, out_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"Saved depth vis: {vis_path}")

# Verify sort order
files = sorted(Path(BEST_DIR).glob("v1_004*.jpg"))
print(f"\nSort order around insertion point:")
for f in files:
    marker = " <-- NEW" if f.name == NEW_FRAME_NAME else ""
    print(f"  {f.name}{marker}")
