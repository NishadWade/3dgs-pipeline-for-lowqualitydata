# DA3 Patches — Flash Fix

These two files fix the **red/blue colour flashing** that occurs in DA3's Gaussian
Splatting renderer on most real-world footage, and add a new `smooth_light`
trajectory mode for stable flythroughs.

## How to Apply

Find your DA3 Python environment (the one you use to run `da3`):

```bash
# Windows example — adjust path to match your install
copy gs_renderer.py  D:\path\to\da3_env\Lib\site-packages\depth_anything_3\model\utils\gs_renderer.py
copy gs.py           D:\path\to\da3_env\Lib\site-packages\depth_anything_3\utils\export\gs.py
```

```bash
# Linux/Mac
cp gs_renderer.py  /path/to/da3_env/lib/python3.x/site-packages/depth_anything_3/model/utils/gs_renderer.py
cp gs.py           /path/to/da3_env/lib/python3.x/site-packages/depth_anything_3/utils/export/gs.py
```

## What Changed

### `gs_renderer.py`

1. **`smooth_light` trajectory mode** — Gaussian smoothing (k=11) over the input
   camera poses. No wander orbit, no dolly zoom. Keeps the camera on a stable path
   derived directly from your input frames.

2. **`max_sh_degree` parameter** — added to `render_3dgs()`. Lets callers cap the
   spherical harmonic degree used during rendering without modifying the stored PLY.

### `gs.py`

1. **Default `trj_mode` changed** from `"extend"` to `"smooth_light"`.

2. **`max_sh_degree=1`** passed to the renderer — this is the key fix. DA3 stores
   degree-3 SH coefficients (16 per channel). At novel viewpoints far from training
   cameras, degree 2 and 3 terms blow up, producing the red/blue flashes. Capping
   at degree 1 (4 coefficients, view-independent colour) eliminates the flashing
   while keeping the reconstruction faithful.

## Root Cause

Flash = **degree-3 SH evaluated at extrapolated viewpoints** + **pose discontinuities**
in the input that the trajectory smoother turns into smooth void traversals.

The SH cap (degree 1) fixes flashing. Frame filtering (`diagnostics/filter_frames.py`)
fixes void traversal. Both together give a clean result.
