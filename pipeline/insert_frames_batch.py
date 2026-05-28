"""
Insert multiple clear frames into the best-frames folder by scanning the source
video for the sharpest frame in a given time range, then generating interpolated
depth visualisations.

Define insertions as a list of tuples in the INSERTIONS variable below, then run:
    python insert_frames_batch.py \\
        --video     path/to/source.mov \\
        --best-dir  path/to/best_frames/ \\
        --npz       path/to/results.npz \\
        --depth-dir path/to/depth_vis/

Each insertion tuple:
    (new_frame_name, search_start_sec, search_end_sec, npz_prev_idx, npz_next_idx)

    new_frame_name   : output filename; must sort alphabetically between neighbours
                       e.g. "v1_0039b.jpg" sorts between v1_0039.jpg and v1_0040.jpg
    search_start_sec : scan source video from this time (seconds)
    search_end_sec   : to this time (seconds) — picks the sharpest frame in range
    npz_prev_idx     : 0-based NPZ depth index of the frame before insertion
    npz_next_idx     : 0-based NPZ depth index of the frame after insertion
"""
import cv2
import numpy as np
import os
import argparse
from pathlib import Path

# ── Configure your insertions here ──────────────────────────────────────────
INSERTIONS = [
    # (frame_name, start_sec, end_sec, npz_prev, npz_next)
    ("v1_0039b.jpg", 39, 42, 38, 39),
    ("v1_0063b.jpg", 62, 64, 62, 63),
]
# ────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Batch-insert clear frames + depth vis")
parser.add_argument("--video",      required=True, help="Source video file")
parser.add_argument("--best-dir",   required=True, help="Best-frames directory to insert into")
parser.add_argument("--npz",        required=True, help="DA3 mini_npz results.npz")
parser.add_argument("--depth-dir",  required=True, help="Depth-vis directory to insert into")
args = parser.parse_args()

npz   = np.load(args.npz)
depth = npz["depth"]

def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def find_best_frame(cap, start_sec, end_sec, fps):
    start_idx = int(start_sec * fps)
    end_idx   = int(end_sec   * fps)
    best_score, best_frame, best_idx = -1, None, -1
    for fi in range(start_idx, end_idx, 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        s = sharpness(frame)
        if s > best_score:
            best_score = s
            best_frame = frame.copy()
            best_idx   = fi
    return best_frame, best_idx, best_score

def make_depth_vis(frame, d_prev, d_next):
    d_interp = 0.5 * d_prev + 0.5 * d_next
    d_min  = np.percentile(d_interp, 2)
    d_max  = np.percentile(d_interp, 98)
    d_norm = np.clip((d_interp - d_min) / (d_max - d_min + 1e-8), 0, 1)
    d_uint8 = ((1.0 - d_norm) * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)
    h, w = d_color.shape[:2]
    return np.hstack([cv2.resize(frame, (w, h)), d_color])

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps:.1f}")

for frame_name, t0, t1, npz_prev, npz_next in INSERTIONS:
    print(f"\n--- {frame_name}: scanning seconds {t0}-{t1} ---")
    frame, src_idx, score = find_best_frame(cap, t0, t1, fps)
    if frame is None:
        print("  ERROR: no frame read")
        continue
    print(f"  Best source frame: {src_idx}  (t={src_idx/fps:.2f}s)  sharpness={score:.0f}")

    img_path = os.path.join(args.best_dir, frame_name)
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved image: {img_path}")

    depth_name = Path(frame_name).stem + "_depth.jpg"
    vis = make_depth_vis(frame, depth[npz_prev], depth[npz_next])
    vis_path = os.path.join(args.depth_dir, depth_name)
    cv2.imwrite(vis_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Saved depth vis: {vis_path}")

cap.release()

print("\n--- Sort order verification ---")
for frame_name, *_ in INSERTIONS:
    prefix = frame_name[:6]
    nearby = sorted(Path(args.best_dir).glob(f"{prefix}*.jpg"))
    print(f"\nAround {frame_name}:")
    for f in nearby:
        marker = " <-- NEW" if f.name == frame_name else ""
        print(f"  {f.name}{marker}")
