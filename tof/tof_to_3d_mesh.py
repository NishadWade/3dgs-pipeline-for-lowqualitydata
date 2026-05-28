import cv2
import numpy as np
import open3d as o3d
import json
import os
import argparse

parser = argparse.ArgumentParser(description='ToF depth + IR video to 3D point cloud')
parser.add_argument('--recording-dir', type=str, required=True)
parser.add_argument('--max-depth', type=float, default=4000.0)
parser.add_argument('--fps', type=float, default=30.0,
                    help='Playback speed in frames per second (default 30)')
parser.add_argument('--min-depth', type=float, default=100.0,
                    help='Minimum depth in mm to include (default 100)')
parser.add_argument('--confidence-threshold', type=int, default=30,
                    help='Minimum confidence to include a pixel (default 30)')
args = parser.parse_args()

npz_path  = os.path.join(args.recording_dir, 'depth_frames.npz')
npy_path  = os.path.join(args.recording_dir, 'depth_frames_recovered.npy')
ir_path   = os.path.join(args.recording_dir, 'ir_video.avi')
meta_path = os.path.join(args.recording_dir, 'metadata.json')

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
vis.create_window(window_name='ToF 3D Point Cloud', width=1280, height=720)

opt = vis.get_render_option()
opt.point_size = 1.0
opt.background_color = np.array([0.1, 0.1, 0.1])

first_time = True
last_pcd = None
frame_idx = 0

print("[INFO] Running. Press Q in the CV2 window to quit.")

while cap.isOpened() and frame_idx < num_frames:
    ret, ir_frame = cap.read()
    if not ret:
        break

    depth_mm = depth_data[frame_idx]
    frame_idx += 1

    if ir_w != tof_w or ir_h != tof_h:
        ir_frame = cv2.resize(ir_frame, (tof_w, tof_h))

    ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)

    valid_mask = (depth_mm > args.min_depth) & (depth_mm < args.max_depth) & (ir_gray > args.confidence_threshold)

    depth_m = (depth_mm / 1000.0).astype(np.float32)
    depth_m[~valid_mask] = 0.0

    ir_rgb = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2RGB)

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

    depth_clipped = np.clip(depth_mm, 0, args.max_depth)
    depth_norm = (depth_clipped / args.max_depth * 255.0).astype(np.uint8)
    depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_RAINBOW)
    combined = cv2.hconcat([ir_frame, depth_colour])
    cv2.imshow('IR  |  Depth', combined)

    delay_ms = max(1, int(1000.0 / args.fps))
    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
        break

vis.destroy_window()
cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
