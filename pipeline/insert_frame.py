"""
Insert a single frame from a source video into the best-frames folder,
then generate an interpolated depth visualisation for it.

The new frame name must sort alphabetically between the two neighbouring
frames. Convention: use a letter suffix, e.g. v1_0046b.jpg sorts between
v1_0046.jpg and v1_0047.jpg.

Usage:
    python insert_frame.py \\
        --video      path/to/source.mov \\
        --best-dir   path/to/best_frames/ \\
        --npz        path/to/results.npz \\
        --depth-dir  path/to/depth_vis/ \\
        --frame-name v1_0046b.jpg \\
        --src-idx    1377 \\
        --npz-prev   45 \\
        --npz-next   46
"""
import cv2
import numpy as np
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Insert a frame + interpolated depth vis")
parser.add_argument("--video",      required=True, help="Source video file")
parser.add_argument("--best-dir",   required=True, help="Best-frames directory to insert into")
parser.add_argument("--npz",        required=True, help="DA3 mini_npz results.npz")
parser.add_argument("--depth-dir",  required=True, help="Depth-vis directory to insert into")
parser.add_argument("--frame-name", required=True,
                    help="Output filename, e.g. v1_0046b.jpg (must sort between neighbours)")
parser.add_argument("--src-idx",    type=int, required=True,
                    help="0-based frame index to extract from the source video")
parser.add_argument("--npz-prev",   type=int, required=True,
                    help="0-based NPZ index of the frame before the insertion point")
parser.add_argument("--npz-next",   type=int, required=True,
                    help="0-based NPZ index of the frame after the insertion point")
args = parser.parse_args()

# Extract frame from video
print(f"Extracting frame {args.src_idx} from {args.video}...")
cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.src_idx)
ret, frame = cap.read()
cap.release()
if not ret:
    print("ERROR: could not read frame")
    exit(1)

out_img_path = os.path.join(args.best_dir, args.frame_name)
cv2.imwrite(out_img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"Saved: {out_img_path}  ({frame.shape[1]}x{frame.shape[0]})")

# Interpolate depth and generate visualisation
print("Generating interpolated depth vis...")
npz   = np.load(args.npz)
depth = npz["depth"]
d_interp = 0.5 * depth[args.npz_prev] + 0.5 * depth[args.npz_next]

d_min  = np.percentile(d_interp, 2)
d_max  = np.percentile(d_interp, 98)
d_norm = np.clip((d_interp - d_min) / (d_max - d_min + 1e-8), 0, 1)
d_uint8 = ((1.0 - d_norm) * 255).astype(np.uint8)
d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)

h, w = d_color.shape[:2]
src_resized = cv2.resize(frame, (w, h))
out_vis = np.hstack([src_resized, d_color])

depth_name = Path(args.frame_name).stem + "_depth.jpg"
vis_path = os.path.join(args.depth_dir, depth_name)
cv2.imwrite(vis_path, out_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"Saved depth vis: {vis_path}")

# Show sort order around insertion point
prefix = args.frame_name[:6]
files = sorted(Path(args.best_dir).glob(f"{prefix}*.jpg"))
print(f"\nSort order around {args.frame_name}:")
for f in files:
    marker = " <-- NEW" if f.name == args.frame_name else ""
    print(f"  {f.name}{marker}")
