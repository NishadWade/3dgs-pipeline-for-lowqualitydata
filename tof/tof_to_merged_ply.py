import cv2
import numpy as np
import open3d as o3d
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Merge all ToF frames into one point cloud')
parser.add_argument('--recording-dir', type=str, required=True)
parser.add_argument('--max-depth', type=float, default=4000.0)
parser.add_argument('--voxel-size', type=float, default=0.005)
parser.add_argument('--skip-frames', type=int, default=1)
args = parser.parse_args()

npz_path  = os.path.join(args.recording_dir, 'depth_frames.npz')
ir_path   = os.path.join(args.recording_dir, 'ir_video.avi')
meta_path = os.path.join(args.recording_dir, 'metadata.json')
out_path  = os.path.join(args.recording_dir, 'merged_pointcloud.ply')

print(f"[INFO] Loading depth data ...")
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

all_points = []
all_colors = []
frame_idx = 0
processed = 0

print(f"[INFO] Processing all frames (skip={args.skip_frames}) ...")

while cap.isOpened() and frame_idx < num_frames:
    ret, ir_frame = cap.read()
    if not ret:
        break

    depth_mm = depth_data[frame_idx]
    frame_idx += 1

    if frame_idx % args.skip_frames != 0:
        continue

    if ir_w != tof_w or ir_h != tof_h:
        ir_frame = cv2.resize(ir_frame, (tof_w, tof_h))

    ir_rgb = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2RGB)

    depth_m = (depth_mm / 1000.0).astype(np.float32)
    depth_m = np.clip(depth_m, 0, args.max_depth / 1000.0)

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

    all_points.append(np.asarray(pcd.points))
    all_colors.append(np.asarray(pcd.colors))
    processed += 1

    if processed % 50 == 0:
        print(f"  {frame_idx}/{num_frames} frames processed ...", end='\r')

cap.release()

print(f"\n[INFO] Combining {processed} frames into one point cloud ...")
all_points = np.concatenate(all_points, axis=0)
all_colors = np.concatenate(all_colors, axis=0)

merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(all_points)
merged_pcd.colors = o3d.utility.Vector3dVector(all_colors)

print(f"[INFO] Total points before cleanup: {len(merged_pcd.points)}")

merged_pcd = merged_pcd.voxel_down_sample(args.voxel_size)
print(f"[INFO] After voxel downsampling: {len(merged_pcd.points)}")

merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"[INFO] After outlier removal: {len(merged_pcd.points)}")

print(f"[INFO] Saving to {out_path} ...")
o3d.io.write_point_cloud(out_path, merged_pcd)
print(f"[OK] Saved: {out_path}")
print(f"[OK] Final point cloud has {len(merged_pcd.points)} points")
