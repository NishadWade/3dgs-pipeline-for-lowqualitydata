"""
Extract every 15th frame from all 4 iPhone videos. No filtering of any kind.
"""
import cv2
import os

VIDEOS = [
    (r"C:\Users\aistudio\Downloads\3DGS\Video May 18 2026, 11 48 56 AM.mov", "v1"),
    (r"C:\Users\aistudio\Downloads\3DGS\Video May 18 2026, 11 51 05 AM.mov", "v2"),
    (r"C:\Users\aistudio\Downloads\3DGS\Video May 18 2026, 11 52 04 AM.mov", "v3"),
    (r"C:\Users\aistudio\Downloads\3DGS\Video May 18 2026, 11 53 33 AM.mov", "v4"),
]
OUT_DIR        = r"D:\3DGS_Project\da3_output_iphone_v8_frames"
FRAME_INTERVAL = 5
JPEG_QUALITY   = 95

os.makedirs(OUT_DIR, exist_ok=True)

total_saved = 0
for video_path, prefix in VIDEOS:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[{prefix}] {total_frames} frames @ {fps:.0f}fps  ({total_frames/fps:.0f}s) -- {video_path.split(chr(92))[-1]}")

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            saved += 1
            total_saved += 1
            fname = f"{prefix}_{saved:04d}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_idx += 1

    cap.release()
    print(f"  saved={saved}  (every {FRAME_INTERVAL} frames, no filtering)")

print(f"\nTotal frames saved: {total_saved} -> {OUT_DIR}")
