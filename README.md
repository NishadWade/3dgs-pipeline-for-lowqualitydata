# 3DGS Pipeline for Low-Quality / Challenging Video Data

A complete pipeline for turning low-quality iPhone and ToF camera footage into clean,
flash-free 3D Gaussian Splat (3DGS) reconstructions using [Depth Anything 3 (DA3)](https://github.com/DepthAnything/Depth-Anything-V3).

Built through iterative debugging of a real parking garage scan — the code addresses
problems specific to challenging footage: motion blur, exposure variation, pose
discontinuities, and SH-degree flashing artifacts.

---

## The Problem This Solves

Standard DA3 on raw video produces:
- **Red/blue flashing** — caused by degree-3 spherical harmonics (SH) amplifying
  view-dependent color at novel viewpoints
- **Void traversal artifacts** — camera pose discontinuities smoothed into sweeping
  voids by DA3's trajectory smoother
- **Blurry reconstructions** — too many low-quality frames fed into the model

This repo patches DA3 and provides a frame selection pipeline to fix all three.

---

## Pipeline Overview

```
iPhone Videos (.mov)
        │
        ▼
01_extract_frames.py         ← extract every 5th frame (6 candidates/second)
        │
        ▼
02_select_best_per_second.py ← pick sharpest + best-exposed frame per second
  OR
02b_select_top2_per_second.py← pick top 2 frames per second (denser, more detail)
        │
        ▼
[Manual review + insert_frames_batch.py for gap-filling]
        │
        ▼
da3 images <best_frames_dir> --infer-gs --export-format gs_ply --export-format gs_video
  (with DA3 patches applied — see da3_patches/)
        │
        ▼
slow_render.py               ← optional: re-encode flythrough at slower FPS
```

---

## DA3 Patches (Critical — Apply Before Running)

The patches in `da3_patches/` fix the red/blue flashing issue that affects DA3's
GS rendering on most real-world footage. Copy them over your DA3 installation:

```
da3_patches/gs_renderer.py  →  <da3_env>/Lib/site-packages/depth_anything_3/model/utils/gs_renderer.py
da3_patches/gs.py           →  <da3_env>/Lib/site-packages/depth_anything_3/utils/export/gs.py
```

**What the patches do:**
- **`gs_renderer.py`** — adds `smooth_light` trajectory mode (Gaussian k=11 stabilization,
  no wander/dolly zoom) and a `max_sh_degree` parameter to cap SH at degree 1
- **`gs.py`** — sets `smooth_light` as the default trajectory, passes `max_sh_degree=1`
  to the renderer (this is the key fix for red/blue flashing)

See `da3_patches/README.md` for full details.

---

## Frame Selection — How It Works

**Sharpness score:** Laplacian variance of the grayscale image (higher = sharper)

**Exposure score:** `1 - abs(mean_brightness - 128) / 128`
- Peaks at 128 (mid-grey), zero for fully black (<15) or blown-out (>235)

**Combined score:** `sharpness × (exposure_score + 0.1)`
- The `+0.1` prevents sharp but slightly dark frames from being zeroed out

Both scripts evaluate every frame in each 1-second window and rank by this score.
`02_select_best_per_second.py` keeps the top 1; `02b` keeps the top 2.

---

## ToF Camera

`tof/` contains scripts for Sony IMX316 ToF sensor recordings saved as
`depth_frames.npz` + `ir_video.avi`:

| Script | Purpose |
|--------|---------|
| `tof_to_3d_mesh.py` | Live playback: IR video + depth colormap + Open3D point cloud |
| `tof_to_3d_mesh_v2.py` | Same with additional filtering options |
| `tof_to_merged_ply.py` | Export all frames as a single merged PLY |

```bash
python tof/tof_to_3d_mesh.py --recording-dir /path/to/tof_recording/
```

---

## Diagnostics

`diagnostics/` contains investigation tools used to root-cause the flashing:

| Script | Purpose |
|--------|---------|
| `flash_investigation.py` | Full forensic: pose jumps, depth stats, SH analysis, frame-to-frame color swings |
| `filter_frames.py` | Remove ±5 frames around pose discontinuities >2.0 world units |
| `check_flash.py` | Quick check: counts flash events in a rendered video |

---

## Requirements

```
depth-anything-3   # DA3 with the patches applied
opencv-python
numpy
open3d
moviepy
```

DA3 install: follow the [official DA3 instructions](https://github.com/DepthAnything/Depth-Anything-V3).
After installing, apply the patches from `da3_patches/`.

---

## Hardware Used

- RTX PRO 6000 (95 GB VRAM) — process-res 504 runs fine
- On smaller GPUs, reduce `--process-res` to 336 or 252
