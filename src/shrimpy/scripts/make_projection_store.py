"""
Z-projection of channels into a single new OME-Zarr store.

HEAVY pass. Reads the full 5D stacks of every position and writes a compact new
store containing ONLY the projected channels as true 2D images (Z=1). Each output
channel is pulled from a named input store, a source channel, and a method (max
or sum) -- so you can mix max/sum projections AND draw channels from more than one
input store (e.g. virtual-staining predictions + raw fluorescence) into a single
output. Values are stored raw float32 -- no clipping, no normalization -- so all
quantitative info is preserved.

max vs sum: max takes the brightest Z value per pixel (good for sparse,
high-contrast structures); sum adds all Z values (preserves integrated intensity
but also accumulates background/noise across slices).

OME-Zarr refresher:
  store.zarr (Plate) -> <row>/<col> (Well) -> <fov> (Position) -> "0" (5D array)
  A Position array is (T, C, Z, Y, X). Zarr arrays are lazy: slicing reads only
  the chunks you touch, so we never load a whole position into RAM. Input stores
  must share the same position names, (T, Z, Y, X) shape, and pixel scale.

What this writes per position: an array of shape (T, n_channels, 1, Y, X).
"""

from __future__ import annotations

import os

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta

# =============================================================================
# CONFIG -- edit these
# =============================================================================
# _BASE = Path(
#     "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/input_data/"
#     "2026_03_26_A549_CAAX_H2B_DENV_ZIKV.zarr/"
# )

# Named input stores. A single entry == single-input behaviour.
INPUTS = {
    "all": "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/input_data/2026_05_27_A549_SEC61B_TOMM20_G3BP1_ZIKV.zarr/",
}

OUTPUT_ZARR = Path(
    "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/fov_selection_output/"
    "2026_05_27_A549_SEC61B_TOMM20_G3BP1_ZIKV_proj.zarr"
)

# Output channel name -> (input store key, source channel name, method).
# method is "max" or "sum". Order MUST match the existing store so RESUME appends
# the missing positions instead of erroring.
# This dataset's only nuclei/membrane channels are the virtual-staining predictions
# (raw fluorescence here is organelle markers: SEC61B/TOMM20/G3BP1, not nuclei).
CHANNELS_TO_PROJECT = {
    "nuclei_prediction_maxproj": ("all", "nuclei_prediction", "max"),
    "nuclei_prediction_sumproj": ("all", "nuclei_prediction", "sum"),
    "membrane_prediction_maxproj": ("all", "membrane_prediction", "max"),
    "membrane_prediction_sumproj": ("all", "membrane_prediction", "sum"),
}

# Only process positions whose row (first path component) is in this set.
# None or empty == all rows.
ROWS: set[str] | None = None  # {"H2BC21"}

# Process only the first N positions (handy for a smoke test). None = all.
MAX_POSITIONS: int | None = None

# Resume: skip positions already present in the output store. Lets you re-run
# after an interruption without redoing finished FOVs.
RESUME = True

# Parallel reads across CPU cores (this step is disk/decompress bound, not GPU).
# None = all cores minus 2 (polite on a shared login node); set an int to override.
# Capped here because this dataset has a large T (67): each worker builds a full
# (T, C, 1, Y, X) float32 array (~2 GB) and ships it back to the main process
# (pickle transiently doubles it), so too many concurrent workers OOM the node.
NUM_WORKERS: int | None = 12
# =============================================================================

# Per-worker globals (set once per process by the initializer).
_W: dict = {}


def default_workers() -> int:
    """All available cores minus 2 (leave headroom on a shared login node)."""
    n = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 2)
    )
    return max(1, n - 2)


def _init_worker(inputs: dict[str, str], specs: list[tuple]) -> None:
    """specs: list of (input_key, channel_index, reduce_name), output-channel order."""
    _W["plates"] = {}
    for key, path in inputs.items():
        plate = open_ome_zarr(path, mode="r")
        _W["plates"][key] = {name: pos for name, pos in plate.positions()}
    _W["specs"] = specs


def _project_position(name: str):
    """Read one position and project over Z. Returns (name, scale, (T,C,1,Y,X))."""
    plates = _W["plates"]
    specs = _W["specs"]
    # Reference geometry/scale from the first spec's store (all inputs must match).
    ref_pos = plates[specs[0][0]][name]
    T, _, _, Y, X = ref_pos["0"].shape
    scale = list(ref_pos.scale)
    out = np.zeros((T, len(specs), 1, Y, X), np.float32)
    for out_c, (key, ch_idx, reduce_name) in enumerate(specs):
        reduce = np.max if reduce_name == "max" else np.sum
        arr = plates[key][name]["0"]  # lazy (T, C, Z, Y, X)
        for t in range(T):
            zyx = np.asarray(arr[t, ch_idx])  # disk read + decompress (the bottleneck)
            out[t, out_c, 0] = reduce(zyx, axis=0)
    return name, scale, out


def main() -> None:
    num_workers = NUM_WORKERS or default_workers()

    # Resolve each input store's channel names; build per-output (key, idx, method).
    in_channels = {}
    for key, path in INPUTS.items():
        z = open_ome_zarr(path, mode="r")
        in_channels[key] = list(z.channel_names)
        if key == next(iter(INPUTS)):
            all_names = [name for name, _ in z.positions()]
        z.close()

    out_names = list(CHANNELS_TO_PROJECT)
    specs = []
    for out_name, (key, src_chan, method) in CHANNELS_TO_PROJECT.items():
        assert key in INPUTS, f"unknown input key {key!r} for {out_name!r}"
        assert method in ("max", "sum"), f"method must be 'max' or 'sum', got {method!r}"
        assert src_chan in in_channels[key], f"{src_chan!r} not in input {key!r}"
        specs.append((key, in_channels[key].index(src_chan), method))

    if ROWS:
        all_names = [n for n in all_names if n.split("/")[0] in ROWS]
    if MAX_POSITIONS is not None:
        all_names = all_names[:MAX_POSITIONS]

    OUTPUT_ZARR.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (RESUME and OUTPUT_ZARR.exists()) else "w"
    out_plate = open_ome_zarr(OUTPUT_ZARR, layout="hcs", mode=mode, channel_names=out_names)
    done = {name for name, _ in out_plate.positions()} if mode == "a" else set()
    todo = [n for n in all_names if n not in done]

    print("Inputs:")
    for key, path in INPUTS.items():
        print(f"  [{key}] {path}")
    print(f"Output: {OUTPUT_ZARR}  (mode={mode})")
    print(f"Rows: {sorted(ROWS) if ROWS else 'all'}")
    for out_name, (key, src_chan, method) in CHANNELS_TO_PROJECT.items():
        print(f"  [{key}] {src_chan} -> {out_name}  ({method})")
    print(f"Positions: {len(todo)} to do, {len(done)} already done | workers={num_workers}\n")

    inputs_str = {k: str(v) for k, v in INPUTS.items()}
    n = 0
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(inputs_str, specs),
    ) as ex:
        futures = {ex.submit(_project_position, name): name for name in todo}
        for fut in as_completed(futures):
            name, scale, arr = fut.result()  # workers read in parallel
            row, col, fov = name.split("/")
            out_pos = out_plate.create_position(row, col, fov)  # main-only writes
            out_arr = out_pos.create_zeros(
                name="0",
                shape=arr.shape,
                dtype=np.float32,
                chunks=(1, 1, 1, arr.shape[-2], arr.shape[-1]),
                transform=[TransformationMeta(type="scale", scale=scale)],
            )
            out_arr[:] = arr
            n += 1
            print(f"[{n}/{len(todo)}] {name}: wrote {tuple(arr.shape)}")

    out_plate.close()
    print(f"\nDone. Projection store at:\n  {OUTPUT_ZARR}")


if __name__ == "__main__":
    print(f"(parallel I/O: {NUM_WORKERS or default_workers()} workers)")
    main()
