"""
Insert additional clear frames at specified positions.
Scans source video range for sharpest frame, adds to best folder + depth_vis.

Insertions:
  - v1_0039b: best clear frame from v1 seconds 39-42 (sorts after v1_0039, before v1_0040)
               NPZ interp: depth[38] and depth[39]
  - v1_0063b: best clear frame from v1 seconds 62-64 (sorts after v1_0063, before v1_0064)
               63 is blurry; this provides a clear alternative
               NPZ interp: depth[62] and depth[63]
"""
import cv2
import numpy as np
import os
from pathlib import Path

VIDEO_PATH = r"C:\Users\aistudio\Downloads\3DGS\Video May 18 2026, 11 48 56 AM.mov"
BEST_DIR   = r"D:\3DGS_Project\da3_output_iphone_v8_best"
NPZ_PATH   = r"D:\3DGS_Project\da3_output_iphone_v8\exports\mini_npz\results.npz"
DEPTH_DIR  = r"D:\3DGS_Project\da3_output_iphone_v8\depth_vis"

npz   = np.load(NPZ_PATH)
depth = npz['depth']  # (256, H, W)

def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def find_best_frame(cap, start_sec, end_sec, fps=30.0):
    """Scan every 2nd frame in [start_sec, end_sec) and return the sharpest."""
    start_idx = int(start_sec * fps)
    end_idx   = int(end_sec   * fps)
    best_score = -1
    best_frame = None
    best_idx   = -1
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
    src_resized = cv2.resize(frame, (w, h))
    return np.hstack([src_resized, d_color])

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps:.1f}")

INSERTIONS = [
    # (new_frame_name, new_depth_name, search_start_sec, search_end_sec, npz_prev_idx, npz_next_idx)
    ("v1_0039b.jpg", "v1_0039b_depth.jpg", 39, 42, 38, 39),
    ("v1_0063b.jpg", "v1_0063b_depth.jpg", 62, 64, 62, 63),
]

for frame_name, depth_name, t0, t1, npz_prev, npz_next in INSERTIONS:
    print(f"\n--- {frame_name}: scanning v1 seconds {t0}-{t1} ---")
    frame, src_idx, score = find_best_frame(cap, t0, t1, fps)
    if frame is None:
        print(f"  ERROR: no frame read")
        continue
    print(f"  Best source frame: {src_idx}  (t={src_idx/fps:.2f}s)  sharpness={score:.0f}")

    img_path = os.path.join(BEST_DIR, frame_name)
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved image: {img_path}")

    vis = make_depth_vis(frame, depth[npz_prev], depth[npz_next])
    vis_path = os.path.join(DEPTH_DIR, depth_name)
    cv2.imwrite(vis_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Saved depth vis: {vis_path}")

cap.release()

# Show sort order around each insertion
print("\n--- Sort order verification ---")
for frame_name, _, _, _, _, _ in INSERTIONS:
    stem = frame_name.replace(".jpg", "")
    prefix = stem[:6]  # e.g. v1_003 or v1_006
    nearby = sorted(Path(BEST_DIR).glob(f"{prefix}*.jpg"))
    print(f"\nAround {frame_name}:")
    for f in nearby:
        marker = " <-- NEW" if f.name == frame_name else ""
        print(f"  {f.name}{marker}")
