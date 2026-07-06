"""
Batch Cellpose segmentation of projection stores: segment EVERY channel of each
input store and write the projections + one label channel per input channel into a
SINGLE OME-Zarr per store (no separate merge step).

You no longer list channels by hand. For each store the script auto-builds one
segmentation task per input channel (suffix = the model name, e.g. "_cpdino"):

    <channel>  ->  <channel>_cpdino     (label channel, integer instance ids)

The Cellpose diameter is inferred from the channel name: channels containing
"membrane" use MEMBRANE_DIAMETER (whole-cell needs an explicit diameter -- every
backbone badly under-covers the membrane on auto-scale, see
compare_cellpose_models.py), everything else (nuclei) uses NUCLEI_DIAMETER (None =
Cellpose auto-scale, correct for filled nuclear blobs).

Output store per input: <input_stem>_cpdino.zarr, containing
    [all input channels copied through...] + [<channel>_cpdino for every channel].
The model-name suffix is one of archive/view_segmentation.py's LABEL_CHANNEL_HINTS, so
these channels load as napari *Labels* layers and downstream tools treat them as
instance masks. Masks are stored as float32 (integer label ids, exact up to
2**24). Resumable via a per-position "seg_done" flag.

    python segment_cellpose.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta

# =============================================================================
# CONFIG -- edit these
# =============================================================================
_OUT_DIR = Path(
    "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/fov_selection_output"
)

# Every projection store to segment. Each is processed independently into its own
# <input_stem>_seg.zarr; ALL of its channels are segmented automatically.
INPUT_ZARRS = [
    _OUT_DIR / "2026_03_25_A549_strong_organelles_DENV_ZIKV_time_course_H2BC21_proj.zarr",
    _OUT_DIR / "2026_03_26_A549_CAAX_H2B_DENV_ZIKV_proj.zarr",
    _OUT_DIR / "2026_06_24_A549_H2BC21_FOV_selection_HB_20x_proj.zarr",
    _OUT_DIR / "2026_06_24_A549_H2BC21_FOV_selection_proj.zarr",
]

# Cellpose diameter (px) by channel role, inferred from the channel name.
NUCLEI_DIAMETER: float | None = None    # None = auto-scale (correct for nuclei)
MEMBRANE_DIAMETER: float | None = 120.0  # whole-cell needs an explicit diameter
MEMBRANE_HINT = "membrane"               # channels containing this use MEMBRANE_DIAMETER

# ---- cellpose model + shared eval parameters ------------------------------
MODEL_NAME = "cpdino"            # Cellpose-DINO (fastest; quality ties cyto3/cpsam)
# Label channel/store suffix = model name (a LABEL_CHANNEL_HINTS entry), e.g. "_cpdino".
OUTPUT_SUFFIX = f"_{MODEL_NAME}"
USE_GPU = True
CELLPROB_THRESHOLD = 0.0         # lower=more/larger masks, higher=stricter
FLOW_THRESHOLD = 0.4             # mask shape QC
MIN_SIZE = 15                    # drop masks smaller than this many px
NORMALIZE_PERCENTILES: tuple[float, float] | None = None  # (low, high) or None=default(1,99)

# ---- throughput / scope ---------------------------------------------------
BATCH_SIZE = 64                  # cellpose tiles per forward pass
CHUNK_SIZE = 16                  # images per GPU batch (across positions+timepoints)
POSITIONS: list[str] | None = None  # specific positions, e.g. ["B/3/000000"]; None=all
MAX_POSITIONS: int | None = None    # None = entire dataset; int for a quick look
RESUME = True
# =============================================================================


def _diameter_for(channel: str) -> float | None:
    """Membrane channels need an explicit diameter; nuclei use auto-scale."""
    return MEMBRANE_DIAMETER if MEMBRANE_HINT in channel.lower() else NUCLEI_DIAMETER


def _base_kwargs() -> dict:
    normalize = (
        True if NORMALIZE_PERCENTILES is None else {"percentile": list(NORMALIZE_PERCENTILES)}
    )
    return dict(
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        batch_size=BATCH_SIZE,
        min_size=MIN_SIZE,
        normalize=normalize,
    )


def _auto_tasks(input_channels: list[str]) -> list[dict]:
    """One segmentation task per input channel: <channel> -> <channel>_seg."""
    tasks, seen = [], set()
    for inp in input_channels:
        out = f"{inp}{OUTPUT_SUFFIX}"
        if out in input_channels:
            raise SystemExit(f"output {out!r} collides with an input channel name")
        if out in seen:
            raise SystemExit(f"duplicate output channel {out!r}")
        seen.add(out)
        tasks.append({"input": inp, "output": out, "diameter": _diameter_for(inp)})
    return tasks


def segment_store(model, input_zarr: Path) -> None:
    output_zarr = input_zarr.with_name(input_zarr.stem + OUTPUT_SUFFIX + ".zarr")

    in_plate = open_ome_zarr(input_zarr, mode="r")
    input_channels = list(in_plate.channel_names)
    tasks = _auto_tasks(input_channels)
    out_channels = input_channels + [t["output"] for t in tasks]
    out_index = {name: i for i, name in enumerate(out_channels)}

    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (RESUME and output_zarr.exists()) else "w"
    out_plate = open_ome_zarr(output_zarr, layout="hcs", mode=mode, channel_names=out_channels)
    existing = {name: pos for name, pos in out_plate.positions()} if mode == "a" else {}

    positions = list(in_plate.positions())
    if POSITIONS is not None:
        wanted = set(POSITIONS)
        positions = [(n, p) for n, p in positions if n in wanted]
    elif MAX_POSITIONS is not None:
        positions = positions[:MAX_POSITIONS]

    # Build work list + (re)create output arrays for positions still to do.
    out_arr, remaining, work, skipped = {}, {}, [], 0
    for name, pos in positions:
        if name in existing and existing[name].zattrs.get("seg_done"):
            skipped += 1
            continue
        T, _, _, Y, X = pos["0"].shape
        if name in existing:
            op, oa = existing[name], existing[name]["0"]
        else:
            row, col, fov = name.split("/")
            op = out_plate.create_position(row, col, fov)
            oa = op.create_zeros(
                name="0",
                shape=(T, len(out_channels), 1, Y, X),
                dtype=np.float32,
                chunks=(1, 1, 1, Y, X),
                transform=[TransformationMeta(type="scale", scale=list(pos.scale))],
            )
        out_arr[name] = (op, oa)
        remaining[name] = T
        work.extend((name, t) for t in range(T))

    print(f"\n=== {input_zarr.name} ===")
    print(f"Input : channels={input_channels}")
    print(f"Output: {output_zarr.name}  (mode={mode})")
    print("Segmentations (label channels):")
    for t in tasks:
        print(f"  {t['input']:32s} -> {t['output']:36s} (diameter={t['diameter']})")
    print(f"Positions: {len(remaining)} to do, {skipped} done | images={len(work)} "
          f"chunk={CHUNK_SIZE} tile_batch={BATCH_SIZE}")

    in_arrs = {name: pos["0"] for name, pos in positions}
    base_kw = _base_kwargs()
    n_done = 0
    for start in range(0, len(work), CHUNK_SIZE):
        chunk = work[start:start + CHUNK_SIZE]
        # Read every input channel once for the chunk (used for copy + segmentation).
        chan_imgs = {
            ci: [np.asarray(in_arrs[name][t, ci, 0]) for name, t in chunk]
            for ci in range(len(input_channels))
        }
        # Copy projection channels through (same indices in the output).
        for ci in range(len(input_channels)):
            for (name, t), img in zip(chunk, chan_imgs[ci]):
                out_arr[name][1][t, ci, 0] = img
        # Segment each task (own diameter), write to its label channel.
        for task in tasks:
            in_ci = input_channels.index(task["input"])
            kw = dict(base_kw)
            if task["diameter"] is not None:
                kw["diameter"] = task["diameter"]
            masks = model.eval([x.copy() for x in chan_imgs[in_ci]], **kw)[0]
            oci = out_index[task["output"]]
            for (name, t), m in zip(chunk, masks):
                out_arr[name][1][t, oci, 0] = np.asarray(m, np.float32)
        # Mark positions complete once all their timepoints are written.
        for name, t in chunk:
            remaining[name] -= 1
            if remaining[name] == 0:
                out_arr[name][0].zattrs["seg_done"] = True
        n_done += len(chunk)
        print(f"  [{n_done}/{len(work)}] images done")

    out_plate.close()
    in_plate.close()
    print(f"Done. Combined store at: {output_zarr}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Segment a single store (default: every store in INPUT_ZARRS). "
             "Used to fan out one SLURM job per dataset (see submit_cellpose.sh).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the configured INPUT_ZARRS (one per line) and exit. "
             "Lets submit_cellpose.sh read the store list from this one source of truth.",
    )
    cli = parser.parse_args()

    if cli.list:
        for z in INPUT_ZARRS:
            print(z)
        return

    stores = [cli.input] if cli.input is not None else [Path(z) for z in INPUT_ZARRS]

    from cellpose import models

    print(f"Loading {MODEL_NAME} (gpu={USE_GPU}) ...")
    model = models.CellposeModel(gpu=USE_GPU, pretrained_model=MODEL_NAME)

    for input_zarr in stores:
        segment_store(model, Path(input_zarr))

    print(f"\nAll done. Segmented {len(stores)} store(s).")


if __name__ == "__main__":
    main()
