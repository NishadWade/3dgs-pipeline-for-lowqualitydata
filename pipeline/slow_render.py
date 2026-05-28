"""
Re-encode a DA3 gs_video at a lower FPS so the camera flythrough plays slower.
No re-rendering needed — just re-encodes the existing frames at a different speed.

Usage:
    python slow_render.py --input 0000_smooth_light.mp4 --output slow.mp4
    python slow_render.py --input 0000_smooth_light.mp4 --output slow.mp4 --fps 10
"""
import moviepy.editor as mpy
import argparse

parser = argparse.ArgumentParser(description="Re-encode video at slower FPS")
parser.add_argument("--input",  required=True, help="Input MP4 file")
parser.add_argument("--output", required=True, help="Output MP4 file")
parser.add_argument("--fps", type=float, default=8,
                    help="Target FPS (default: 8, which is 3x slower than DA3's default 24fps)")
args = parser.parse_args()

clip = mpy.VideoFileClip(args.input)
print(f"Original: {clip.fps:.1f}fps  {clip.duration:.1f}s  ({int(clip.fps * clip.duration)} frames)")

frames = list(clip.iter_frames())
print(f"Frames extracted: {len(frames)}")

slow_clip = mpy.ImageSequenceClip(frames, fps=args.fps)
print(f"Output:   {args.fps}fps  {len(frames)/args.fps:.1f}s")

slow_clip.write_videofile(
    args.output,
    codec="libx264",
    audio=False,
    fps=args.fps,
    ffmpeg_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"],
)
print(f"\nSaved: {args.output}")
