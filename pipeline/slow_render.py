"""
Re-encode an existing gs_video at a slower FPS so the camera moves slower.
No re-rendering needed — just changes playback speed.

Current: 154 frames @ 24fps = ~6.4s
Target:  154 frames @ 8fps  = ~19s  (3x slower)
"""
import moviepy.editor as mpy
import sys

INPUT  = r"D:\3DGS_Project\da3_output_iphone_v10\gs_video\Garage_Final3DGS.mp4"
OUTPUT = r"D:\3DGS_Project\da3_output_iphone_v10\gs_video\Garage_Final3DGS_slow.mp4"
TARGET_FPS = 8   # change this to taste (lower = slower)

clip = mpy.VideoFileClip(INPUT)
print(f"Original: {clip.fps:.1f}fps  {clip.duration:.1f}s  ({int(clip.fps * clip.duration)} frames)")

# Re-encode at lower FPS — same frames, just played back slower
frames = list(clip.iter_frames())
print(f"Frames extracted: {len(frames)}")

slow_clip = mpy.ImageSequenceClip(frames, fps=TARGET_FPS)
print(f"Output:   {TARGET_FPS}fps  {len(frames)/TARGET_FPS:.1f}s")

slow_clip.write_videofile(
    OUTPUT,
    codec="libx264",
    audio=False,
    fps=TARGET_FPS,
    ffmpeg_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"],
)
print(f"\nSaved: {OUTPUT}")
