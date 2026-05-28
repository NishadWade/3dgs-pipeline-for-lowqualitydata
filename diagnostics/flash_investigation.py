"""
Deep diagnostic for red/blue flashing in iPhone GS videos.
Investigates: camera trajectory, depth map stats, Gaussian distribution,
SH view-dependence, and wander section timing.
"""
import sys, os
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, r"D:\3DGS_Project\Depth-Anything-3\src")

NPZ_PATH   = r"D:\3DGS_Project\da3_output_iphone_v4\exports\mini_npz\results.npz"
PLY_PATH   = r"D:\3DGS_Project\da3_output_iphone_v4\gs_ply\0000.ply"
DEPTHVIS   = r"D:\3DGS_Project\da3_output_iphone_v4\depth_vis"
VIDEO_PATH = r"D:\3DGS_Project\da3_output_iphone_v4\gs_video\0000_extend.mp4"

print("=" * 60)
print("SECTION 1: Camera Trajectory Analysis")
print("=" * 60)

npz = np.load(NPZ_PATH)
print(f"NPZ keys: {list(npz.keys())}")
extr = npz['extrinsics'].astype(np.float32)  # (V, 3, 4) w2c
intr = npz['intrinsics'].astype(np.float32)  # (V, 3, 3)
print(f"Views: {extr.shape[0]},  extrinsics shape: {extr.shape}")
print(f"Intrinsics[0]:\n{intr[0]}")

# Camera positions in world space: c = -R^T t
R = extr[:, :3, :3]  # (V, 3, 3)
t = extr[:, :3, 3]   # (V, 3)
cam_pos = np.einsum('vij,vj->vi', -R.transpose(0, 2, 1), t)  # (V, 3)
print(f"\nCamera positions (world space):")
print(f"  X: min={cam_pos[:,0].min():.3f}  max={cam_pos[:,0].max():.3f}  range={cam_pos[:,0].ptp():.3f}")
print(f"  Y: min={cam_pos[:,1].min():.3f}  max={cam_pos[:,1].max():.3f}  range={cam_pos[:,1].ptp():.3f}")
print(f"  Z: min={cam_pos[:,2].min():.3f}  max={cam_pos[:,2].max():.3f}  range={cam_pos[:,2].ptp():.3f}")

# Frame-to-frame displacement (motion between consecutive views)
diffs = np.linalg.norm(np.diff(cam_pos, axis=0), axis=1)
print(f"\nFrame-to-frame displacement:")
print(f"  mean={diffs.mean():.4f}  max={diffs.max():.4f}  std={diffs.std():.4f}")
print(f"  Top-10 biggest jumps (frame_idx, displacement):")
top10 = np.argsort(diffs)[-10:][::-1]
for i in top10:
    print(f"    frame {i:3d}->{i+1:3d}: {diffs[i]:.4f}")

# Check if the extend trajectory wander orbit is present
# With 412 views, inter_len=1 (no interpolation), wander at midpoint
# mid_idx = 412//2 = 206; wander_frames = max(36, min(60, 206//2)) = max(36, min(60, 103)) = 60
V = extr.shape[0]
mid_idx = V // 2
wander_frames = max(36, min(60, mid_idx // 2))
wander_r_est = 24.0 / intr[0, 0, 0]  # 24 / fx in pixel units (approx)
print(f"\nExtend trajectory wander estimate (from DA3 source):")
print(f"  V={V}, mid_idx={mid_idx}, wander_frames={wander_frames}")
print(f"  fx={intr[0,0,0]:.1f}, wander_r (world units) = 24.0/fx = {wander_r_est:.4f}")
print(f"  Camera translation scale: mean displacement per frame = {diffs.mean():.4f}")
print(f"  Wander radius vs scene scale: {wander_r_est:.4f} vs range_X={cam_pos[:,0].ptp():.3f}")

print("\n" + "=" * 60)
print("SECTION 2: Depth Map Statistics (from depth_vis images)")
print("=" * 60)

depth_dir = Path(DEPTHVIS)
if depth_dir.exists():
    frames = sorted(depth_dir.glob("*.png")) + sorted(depth_dir.glob("*.jpg"))
    print(f"Found {len(frames)} depth_vis frames")

    # Sample every 20th frame for speed, plus check a dense window around mid_idx
    sample_indices = list(range(0, len(frames), 20))
    # Add frames around mid_idx (wander region)
    wander_start = mid_idx - 10
    wander_end   = mid_idx + wander_frames + 10
    for fi in range(max(0, wander_start), min(len(frames), wander_end)):
        if fi not in sample_indices:
            sample_indices.append(fi)
    sample_indices = sorted(set(sample_indices))

    bright_frames = []
    for fi in sample_indices:
        img = cv2.imread(str(frames[fi]))
        if img is None:
            continue
        r = img[:, :, 2].astype(float)
        g = img[:, :, 1].astype(float)
        b = img[:, :, 0].astype(float)
        mean_r = r.mean()
        mean_g = g.mean()
        mean_b = b.mean()
        gray   = img.mean(axis=2)
        variance = gray.var()
        # Check for near-uniform depth (variance < threshold = bad depth estimation)
        if variance < 100 or mean_r > 200 or mean_b > 200:
            bright_frames.append((fi, mean_r, mean_g, mean_b, variance))

    if bright_frames:
        print(f"\nSuspicious depth frames (low variance OR high R/B channel):")
        for fi, mr, mg, mb, var in bright_frames:
            print(f"  frame {fi:3d}: R={mr:.0f} G={mg:.0f} B={mb:.0f} variance={var:.1f}")
    else:
        print("\nNo obviously bad depth frames detected in sampled set.")

    # Sample 5 frames and show their stats
    check_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
    print(f"\nDepth stat samples at key frames:")
    for fi in check_indices:
        if fi >= len(frames):
            fi = len(frames) - 1
        img = cv2.imread(str(frames[fi]))
        if img is None:
            continue
        gray = img.mean(axis=2)
        print(f"  frame {fi:3d}: mean_brightness={gray.mean():.1f}  variance={gray.var():.1f}  "
              f"R={img[:,:,2].mean():.0f} G={img[:,:,1].mean():.0f} B={img[:,:,0].mean():.0f}")
else:
    print(f"depth_vis dir not found: {DEPTHVIS}")

print("\n" + "=" * 60)
print("SECTION 3: Gaussian PLY Analysis")
print("=" * 60)

try:
    from plyfile import PlyData
    ply = PlyData.read(PLY_PATH)
    data = ply.elements[0].data
    n = len(data)
    print(f"Total Gaussians: {n:,}")

    log_scales = np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1)
    max_log_scale = log_scales.max(axis=1)
    opacity_raw   = data['opacity'].astype(np.float32)
    opacity_prob  = 1.0 / (1.0 + np.exp(-opacity_raw))

    means = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)

    print(f"\nScale distribution (log space):")
    for thresh in [0.0, 0.5, 1.0, 2.0, 5.0]:
        pct = (max_log_scale > thresh).mean() * 100
        print(f"  log_scale > {thresh}: {pct:.1f}%  ({(max_log_scale > thresh).sum():,} Gaussians)")

    print(f"\nActual scale distribution (exp of log_scale):")
    actual_max_scale = np.exp(max_log_scale)
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"  p{p}: {np.percentile(actual_max_scale, p):.4f}")

    print(f"\nOpacity distribution:")
    for thresh in [0.001, 0.005, 0.01, 0.05, 0.1]:
        pct = (opacity_prob < thresh).mean() * 100
        print(f"  opacity < {thresh}: {pct:.1f}%")

    print(f"\nGaussian positions in world space:")
    print(f"  X: {means[:,0].min():.3f} to {means[:,0].max():.3f}  range={means[:,0].ptp():.3f}")
    print(f"  Y: {means[:,1].min():.3f} to {means[:,1].max():.3f}  range={means[:,1].ptp():.3f}")
    print(f"  Z: {means[:,2].min():.3f} to {means[:,2].max():.3f}  range={means[:,2].ptp():.3f}")

    # Check for floaters far from camera path bounding box
    path_min = cam_pos.min(axis=0)
    path_max = cam_pos.max(axis=0)
    path_center = cam_pos.mean(axis=0)
    path_std    = cam_pos.std(axis=0)

    # Gaussians within 2x scene range of camera path
    margin = np.maximum(cam_pos.ptp(axis=0), 0.5)  # at least 0.5 units margin
    in_bounds_x = (means[:,0] >= path_min[0] - 3*margin[0]) & (means[:,0] <= path_max[0] + 3*margin[0])
    in_bounds_y = (means[:,1] >= path_min[1] - 3*margin[1]) & (means[:,1] <= path_max[1] + 3*margin[1])
    in_bounds_z = (means[:,2] >= path_min[2] - 3*margin[2]) & (means[:,2] <= path_max[2] + 3*margin[2])
    in_bounds = in_bounds_x & in_bounds_y & in_bounds_z
    print(f"\nCamera path bounding box (with 3x margin):")
    print(f"  X: {path_min[0]-3*margin[0]:.3f} to {path_max[0]+3*margin[0]:.3f}")
    print(f"  Y: {path_min[1]-3*margin[1]:.3f} to {path_max[1]+3*margin[1]:.3f}")
    print(f"  Z: {path_min[2]-3*margin[2]:.3f} to {path_max[2]+3*margin[2]:.3f}")
    print(f"  Gaussians OUTSIDE 3x path bounds: {(~in_bounds).sum():,} / {n:,} ({100*(~in_bounds).mean():.1f}%)")

    # Check Gaussians very close to each camera position
    floater_scale_keep = max_log_scale <= 0.5
    print(f"\nAfter log_scale<=0.5 prune: {floater_scale_keep.sum():,} remain ({100*floater_scale_keep.mean():.1f}%)")

    # Floaters specifically: large scale AND positioned within camera bounds (in-scene floaters)
    bad_scale = max_log_scale > 0.5
    bad_in_scene = bad_scale & in_bounds
    print(f"Large floaters (log_scale>0.5) that ARE within scene bounds: {bad_in_scene.sum():,}")
    if bad_in_scene.sum() > 0:
        bs_means = means[bad_in_scene]
        bs_scales = actual_max_scale[bad_in_scene]
        print(f"  Their scale range: {bs_scales.min():.3f} to {bs_scales.max():.3f}  mean={bs_scales.mean():.3f}")
        print(f"  Their positions: X={bs_means[:,0].mean():.3f}  Y={bs_means[:,1].mean():.3f}  Z={bs_means[:,2].mean():.3f}")

except Exception as e:
    print(f"PLY analysis error: {e}")

print("\n" + "=" * 60)
print("SECTION 4: SH Color Analysis")
print("=" * 60)

try:
    from plyfile import PlyData
    ply = PlyData.read(PLY_PATH)
    data = ply.elements[0].data

    # DC component of SH — this is the view-independent base color
    # SH DC: color = sigmoid(f_dc * 0.2820947918 + 0.5)  (C0 constant)
    C0 = 0.2820947918
    f_dc_0 = data['f_dc_0'].astype(np.float32)
    f_dc_1 = data['f_dc_1'].astype(np.float32)
    f_dc_2 = data['f_dc_2'].astype(np.float32)

    # Check for full SH (higher-order coefficients)
    all_fields = [d.name for d in data.dtype.descr]
    sh_fields = [f for f in all_fields if f.startswith('f_rest_')]
    print(f"SH DC fields: f_dc_0/1/2 (degree 0)")
    print(f"SH higher-order fields: {len(sh_fields)} (e.g. {sh_fields[:3] if sh_fields else 'NONE'})")

    if sh_fields:
        print(f"  PLY has FULL SH coefficients saved (not DC only)")
        # Check the magnitude of higher-order SH vs DC
        f_rest_vals = np.stack([data[f].astype(np.float32) for f in sh_fields[:9]], axis=1)
        dc_mag  = np.sqrt(f_dc_0**2 + f_dc_1**2 + f_dc_2**2)
        rest_mag = np.linalg.norm(f_rest_vals, axis=1)
        ratio    = rest_mag / (dc_mag + 1e-6)
        print(f"  Higher-order SH magnitude vs DC: mean ratio = {ratio.mean():.3f}  p95 = {np.percentile(ratio, 95):.3f}")
        print(f"  If ratio >> 1, view-dependent color swing will be large (causing flashing)")
    else:
        print(f"  PLY has DC-ONLY SH (save_sh_dc_only=True was used)")

    # Base color from DC
    r_base = 1.0 / (1.0 + np.exp(-(f_dc_0 * C0 + 0.5)))
    g_base = 1.0 / (1.0 + np.exp(-(f_dc_1 * C0 + 0.5)))
    b_base = 1.0 / (1.0 + np.exp(-(f_dc_2 * C0 + 0.5)))

    print(f"\nBase color (DC-only) distribution:")
    print(f"  R: mean={r_base.mean():.3f}  std={r_base.std():.3f}")
    print(f"  G: mean={g_base.mean():.3f}  std={g_base.std():.3f}")
    print(f"  B: mean={b_base.mean():.3f}  std={b_base.std():.3f}")

    # High-saturation red Gaussians (R >> G, B)
    is_red_gs = (r_base > 0.7) & (r_base > 2*g_base) & (r_base > 2*b_base)
    is_blue_gs = (b_base > 0.7) & (b_base > 2*r_base) & (b_base > 2*g_base)
    print(f"\nHigh-saturation RED Gaussians (R>0.7 and R>2G and R>2B): {is_red_gs.sum():,} ({100*is_red_gs.mean():.2f}%)")
    print(f"High-saturation BLUE Gaussians (B>0.7 and B>2R and B>2G): {is_blue_gs.sum():,} ({100*is_blue_gs.mean():.2f}%)")

    if is_red_gs.sum() > 0:
        log_scales = np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1)
        red_log_scale = log_scales[is_red_gs].max(axis=1)
        print(f"\nRed Gaussian log_scale distribution:")
        for p in [50, 75, 90, 95, 99]:
            print(f"  p{p}: {np.percentile(red_log_scale, p):.3f}")
        red_means = np.stack([data['x'], data['y'], data['z']], axis=1)[is_red_gs].astype(np.float32)
        print(f"  Red G positions: X={red_means[:,0].mean():.3f}  Y={red_means[:,1].mean():.3f}  Z={red_means[:,2].mean():.3f}")

except Exception as e:
    print(f"SH analysis error: {e}")

print("\n" + "=" * 60)
print("SECTION 5: Video Frame Flash Analysis")
print("=" * 60)

cap = cv2.VideoCapture(VIDEO_PATH)
if cap.isOpened():
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_vid_frames} frames @ {fps_vid:.0f}fps ({total_vid_frames/fps_vid:.1f}s)")

    flash_frames = []
    frame_vi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Only left half (GS render, not depth vis side-by-side)
        h, w = frame.shape[:2]
        gs_half = frame[:, :w//2, :]

        mean_r = gs_half[:, :, 2].mean()
        mean_g = gs_half[:, :, 1].mean()
        mean_b = gs_half[:, :, 0].mean()

        # Red flash: R dominates strongly
        if mean_r > 80 and mean_r > 2.0 * mean_g and mean_r > 2.0 * mean_b:
            flash_frames.append(('RED',   frame_vi, mean_r, mean_g, mean_b))
        # Blue flash: B dominates strongly
        elif mean_b > 80 and mean_b > 2.0 * mean_r and mean_b > 2.0 * mean_g:
            flash_frames.append(('BLUE',  frame_vi, mean_r, mean_g, mean_b))
        # Overall very bright (whiteout)
        elif (mean_r + mean_g + mean_b) / 3 > 200:
            flash_frames.append(('WHITE', frame_vi, mean_r, mean_g, mean_b))

        frame_vi += 1
    cap.release()

    print(f"Analyzed {frame_vi} video frames")
    print(f"Flash frames detected: {len(flash_frames)}")
    if flash_frames:
        print(f"\nFlash frame list (type, vid_frame, R, G, B):")
        for ftype, fvi, mr, mg, mb in flash_frames[:50]:
            t_sec = fvi / fps_vid
            print(f"  [{ftype}] vid_frame={fvi:4d}  t={t_sec:.1f}s  R={mr:.0f} G={mg:.0f} B={mb:.0f}")
        if len(flash_frames) > 50:
            print(f"  ... and {len(flash_frames)-50} more")
else:
    print(f"Cannot open video: {VIDEO_PATH}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
