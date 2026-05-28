import cv2
import numpy as np
import open3d as o3d
import json
import os
import argparse
from collections import deque

parser = argparse.ArgumentParser(description='ToF depth + IR video to 3D point cloud')
parser.add_argument('--recording-dir', type=str, required=True)
parser.add_argument('--max-depth', type=float, default=4000.0)
parser.add_argument('--min-depth', type=float, default=100.0)
parser.add_argument('--temporal-frames', type=int, default=3)
parser.add_argument('--bilateral-d', type=int, default=5)
parser.add_argument('--point-size', type=float, default=1.0)
parser.add_argument('--cleanup-every', type=int, default=10,
                    help='Run outlier removal every N frames (default 10)')
args = parser.parse_args()

npz_path  = os.path.join(args.recording_dir, 'depth_frames.npz')
npy_path  = os.path.join(args.recording_dir, 'depth_frames_recovered.npy')
ir_path   = os.path.join(args.recording_dir, 'ir_video.avi')

print(f"[INFO] Loading depth data ...")
if os.path.exists(npy_path):
    depth_data = np.load(npy_path)
else:
    depth_data = np.load(npz_path)['depth']
num_frames, tof_h, tof_w = depth_data.shape
print(f"[INFO] {num_frames} frames at {tof_w}x{tof_h}")

cap = cv2.VideoCapture(ir_path)
ir_w = int(cap.get(3))
ir_h = int(cap.get(4))

fx = 110.0
fy = 110.0
cx = tof_w / 2.0
cy = tof_h / 2.0
pinhole = o3d.camera.PinholeCameraIntrinsic(tof_w, tof_h, fx, fy, cx, cy)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='ToF 3D Point Cloud v2', width=1280, height=720)
opt = vis.get_render_option()
opt.point_size = args.point_size
opt.background_color = np.array([0.05, 0.05, 0.05])

temporal_buffer = deque(maxlen=args.temporal_frames)

first_time = True
last_pcd = None
frame_idx = 0

print(f"[INFO] Press Q to quit.")

while cap.isOpened() and frame_idx < num_frames:
    ret, ir_frame = cap.read()
    if not ret:
        break

    depth_mm = depth_data[frame_idx].copy()
    frame_idx += 1

    if ir_w != tof_w or ir_h != tof_h:
        ir_frame = cv2.resize(ir_frame, (tof_w, tof_h))

    depth_mm[depth_mm < args.min_depth] = 0
    depth_mm[depth_mm > args.max_depth] = 0

    depth_filtered = cv2.bilateralFilter(
        depth_mm.astype(np.float32),
        d=args.bilateral_d,
        sigmaColor=75,
        sigmaSpace=75
    )

    temporal_buffer.append(depth_filtered)
    depth_smooth = np.mean(np.stack(temporal_buffer, axis=0), axis=0).astype(np.float32)

    ir_bright = cv2.convertScaleAbs(ir_frame, alpha=3.0, beta=30)
    ir_rgb = cv2.cvtColor(ir_bright, cv2.COLOR_BGR2RGB)

    depth_m = (depth_smooth / 1000.0).astype(np.float32)

    o3d_color = o3d.geometry.Image(ir_rgb.astype(np.uint8))
    o3d_depth = o3d.geometry.Image(depth_m)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        depth_scale=1.0,
        depth_trunc=args.max_depth / 1000.0,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0,  1]])

    if frame_idx % args.cleanup_every == 0:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.05)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    if first_time:
        vis.add_geometry(pcd)
        last_pcd = pcd
        first_time = False
    else:
        vis.add_geometry(pcd, reset_bounding_box=False)
        vis.remove_geometry(last_pcd, reset_bounding_box=False)
        last_pcd = pcd
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    depth_clipped = np.clip(depth_smooth, 0, args.max_depth)
    depth_norm = (depth_clipped / args.max_depth * 255.0).astype(np.uint8)
    depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_RAINBOW)
    combined = cv2.hconcat([ir_bright, depth_colour])
    cv2.imshow('IR  |  Depth', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vis.destroy_window()
cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
