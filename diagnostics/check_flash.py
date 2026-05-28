"""Check v7 video for flash frames and compare against v4."""
import cv2
import numpy as np

def analyze_video(path, label):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[{label}] Cannot open: {path}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[{label}] {total} frames @ {fps:.0f}fps ({total/fps:.1f}s)")

    flash_frames = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        gs = frame[:, :w//2, :]   # left half = GS render (right = depth)
        r = gs[:,:,2].mean()
        g = gs[:,:,1].mean()
        b = gs[:,:,0].mean()
        if r > 80 and r > 2.0*g and r > 2.0*b:
            flash_frames.append(('RED',  fi, r, g, b))
        elif b > 80 and b > 2.0*r and b > 2.0*g:
            flash_frames.append(('BLUE', fi, r, g, b))
        elif (r+g+b)/3 > 200:
            flash_frames.append(('WHITE',fi, r, g, b))
        fi += 1
    cap.release()

    print(f"Flash frames: {len(flash_frames)} / {fi}")
    for ftype, fvi, r, g, b in flash_frames[:30]:
        print(f"  [{ftype}] frame={fvi:4d}  t={fvi/fps:.1f}s  R={r:.0f} G={g:.0f} B={b:.0f}")

# Check new v7 run
analyze_video(
    r"D:\3DGS_Project\da3_output_iphone_v7\gs_video\0000_extend.mp4",
    "v7 (pose-filtered 339 frames)"
)

# Compare against v4 original
analyze_video(
    r"D:\3DGS_Project\da3_output_iphone_v4\gs_video\0000_extend.mp4",
    "v4 (original 412 frames)"
)

# Also print new NPZ pose continuity
import sys
sys.path.insert(0, r"D:\3DGS_Project\Depth-Anything-3\src")
import numpy as np
for label, npz_path in [
    ("v7", r"D:\3DGS_Project\da3_output_iphone_v7\exports\mini_npz\results.npz"),
    ("v4", r"D:\3DGS_Project\da3_output_iphone_v4\exports\mini_npz\results.npz"),
]:
    try:
        npz  = np.load(npz_path)
        extr = npz['extrinsics'].astype(np.float32)
        R = extr[:,:3,:3]; t = extr[:,:3,3]
        cam_pos = np.einsum('vij,vj->vi', -R.transpose(0,2,1), t)
        diffs = np.linalg.norm(np.diff(cam_pos, axis=0), axis=1)
        n_big = (diffs > 2.0).sum()
        print(f"\n[{label}] {len(cam_pos)} views | mean_disp={diffs.mean():.3f} max_disp={diffs.max():.3f} | jumps>2.0: {n_big}")
    except Exception as e:
        print(f"\n[{label}] NPZ error: {e}")
