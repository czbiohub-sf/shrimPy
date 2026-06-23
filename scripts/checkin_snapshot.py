"""Check-in snapshot for a *live* OME-Zarr acquisition.

A long (multi-day) acquisition written by ``acquire-zarr`` keeps every
position's array-level ``0/zarr.json`` exclusively locked for the whole run,
so the store cannot be opened by ``zarr`` / ``iohub`` / napari while it is
being written.  This utility produces a separate, valid OME-Zarr containing the
**last fully-written timepoint of every position**, so progress can be
inspected without touching the live store.

How it stays safe and non-disturbing:
  * It only *reads* small metadata files (group ``zarr.json``, ``OME``,
    ``summary_metadata.json``) and *finalized* chunk files.  The currently
    being-written timepoint is detected and skipped.
  * It never opens the live store for writing and never locks it.

How it works around the locked array metadata:
  * The array ``0/zarr.json`` is never read.  Compressed blosc chunks are
    self-describing on decode, so we copy the raw chunk *files* into a brand
    new array whose metadata we author here with zarr-python.  Array shape is
    derived from readable sources (``summary_metadata.json`` for Y/X) and from
    decoding the z-chunks of one position (ground truth for Z), then confirmed
    by a read-back verification pass.

Usage (defaults wired to the 2026_06_08 dynatrack debug experiment)::

    python scripts/checkin_snapshot.py \
        --source "E:/2026_06_08_dynatrack_debug_48hpf/dynatrack_gfp_pcc_5_days_1.ome.zarr" \
        --out-dir "E:/2026_06_08_dynatrack_debug_48hpf" \
        [--no-verify] [--every-hours 12]

Produces ``<out-dir>/checkin_t<maxT>_<YYYYmmddTHHMMSS>.ome.zarr``.  With
``--every-hours N`` it takes a snapshot immediately and then repeats every N
hours until interrupted (Ctrl-C); a failure in one iteration is logged and the
loop continues to the next interval.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import zarr

from numcodecs.blosc import decompress as blosc_decompress
from zarr.codecs import BloscCodec, BytesCodec

DEFAULT_SOURCE = r"E:/2026_06_08_dynatrack_debug_48hpf/dynatrack_gfp_pcc_5_days_1.ome.zarr"
DEFAULT_OUT_DIR = r"E:/2026_06_08_dynatrack_debug_48hpf"

# zarr v3 default chunk-key encoding (matches what acquire-zarr wrote).
CHUNK_KEY_ENCODING = {"name": "default", "configuration": {"separator": "/"}}
DIMENSION_NAMES = ["t", "c", "z", "y", "x"]


def _read_json(path: Path) -> dict:
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _is_int_name(name: str) -> bool:
    return name.isdigit()


def discover_positions(source: Path) -> list[str]:
    """Top-level group dirs that contain a multiscale-0 child."""
    positions = []
    for child in sorted(p.name for p in source.iterdir() if p.is_dir()):
        if child == "OME":
            continue
        if (source / child / "0" / "c").is_dir():
            positions.append(child)
    return positions


def _file_ready(path: Path) -> bool:
    """True if the chunk file exists, is non-empty, and is not currently locked."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "rb") as fh:  # acquire-zarr locks the active chunk
            fh.read(1)
        return True
    except OSError:
        return False


def expected_chunk_rel_paths(t: int, n_channels: int, n_zchunks: int) -> list[str]:
    """Chunk keys (relative to ``<pos>/0``) for a single timepoint ``t``."""
    return [
        f"c/{t}/{c}/{z}/0/0"
        for c in range(n_channels)
        for z in range(n_zchunks)
    ]


def last_complete_timepoint(
    pos_dir: Path, n_channels: int, n_zchunks: int
) -> int | None:
    """Highest ``t`` whose every expected chunk file is present, non-empty, unlocked."""
    chunk_root = pos_dir / "0" / "c"
    t_indices = sorted(
        (int(p.name) for p in chunk_root.iterdir() if _is_int_name(p.name)),
        reverse=True,
    )
    for t in t_indices:
        rels = expected_chunk_rel_paths(t, n_channels, n_zchunks)
        if all(_file_ready(pos_dir / "0" / rel) for rel in rels):
            return t
    return None


def _zplan_count(z_plan: dict) -> int:
    """Number of z-planes for the acquisition's z-plan.

    Uses useq (the same library that drove the acquisition, so the count matches
    the array's stored Z exactly).  Falls back to the top/bottom/step formula.
    """
    try:
        import useq

        return int(useq.MDASequence(z_plan=z_plan).sizes["z"])
    except Exception:  # noqa: BLE001 - fall back to arithmetic
        top, bottom, step = z_plan["top"], z_plan["bottom"], z_plan["step"]
        return int(round(abs(top - bottom) / step)) + 1


def _derive_z_chunk(source: Path, positions: list[str], y: int, x: int) -> int:
    """Z extent of one stored chunk, by decoding an early (finalized) chunk.

    Chunk buffers are always stored at full chunk shape, so decoding any chunk
    yields the chunk's z size directly (the array's true Z comes from the z-plan,
    not from summing chunks -- edge chunks are zero-padded to full size).
    """
    for pos in positions:
        chunk_root = source / pos / "0" / "c"
        for t in sorted(int(p.name) for p in chunk_root.iterdir() if _is_int_name(p.name)):
            chunk = chunk_root / str(t) / "0" / "0" / "0" / "0"  # c=0, z=0, y=0, x=0
            if _file_ready(chunk):
                n_elems = len(blosc_decompress(chunk.read_bytes())) // np.dtype("uint16").itemsize
                return n_elems // (y * x)
    raise RuntimeError("Could not find a readable chunk to size the z-chunk from.")


def derive_dims(source: Path, positions: list[str]) -> dict:
    """Derive (C, Z, Y, X) and the z-chunk size from readable sources only.

    Y/X come from ``summary_metadata.json`` (camera plane shape); C from the
    position's ``omero`` channels; Z from the z-plan via useq; the z-chunk size
    by decoding one finalized chunk.  None of this touches the locked array
    ``0/zarr.json``.
    """
    summary = _read_json(source / "summary_metadata.json")
    plane = summary["image_infos"][0]["plane_shape"]  # [Y, X]
    y, x = int(plane[0]), int(plane[1])

    z_total = _zplan_count(summary["mda_sequence"]["z_plan"])

    pos0_ome = _read_json(source / positions[0] / "zarr.json")["attributes"]["ome"]
    n_channels = len(pos0_ome["omero"]["channels"])

    z_chunk = _derive_z_chunk(source, positions, y, x)
    return {
        "n_channels": n_channels,
        "Z": z_total,
        "Y": y,
        "X": x,
        "z_chunk": z_chunk,
        "n_zchunks": math.ceil(z_total / z_chunk),
    }


def build_snapshot(source: Path, out_dir: Path, verify: bool = True) -> Path:
    positions = discover_positions(source)
    if not positions:
        raise RuntimeError(f"No position groups found under {source}")
    print(f"Found {len(positions)} positions: {', '.join(positions)}")

    dims = derive_dims(source, positions)
    n_channels = dims["n_channels"]
    Z, Y, X = dims["Z"], dims["Y"], dims["X"]
    z_chunk = dims["z_chunk"]
    n_zchunks = math.ceil(Z / z_chunk)
    shape = (1, n_channels, Z, Y, X)
    chunks = (1, 1, z_chunk, Y, X)
    print(f"Array shape (per position) = {shape}, chunks = {chunks}, dtype = uint16")

    # Resolve last complete timepoint per position.
    chosen: dict[str, int] = {}
    for pos in positions:
        t = last_complete_timepoint(source / pos, n_channels, n_zchunks)
        if t is None:
            print(f"  WARNING: no complete timepoint for {pos}; skipping")
            continue
        chosen[pos] = t
    if not chosen:
        raise RuntimeError("No position had a complete timepoint to snapshot.")
    max_t = max(chosen.values())

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = out_dir / f"checkin_t{max_t}_{ts}.ome.zarr"
    print(f"Writing snapshot -> {dest}")

    # --- create destination store + metadata with zarr-python -----------------
    root = zarr.open_group(store=str(dest), mode="w", zarr_format=3)
    src_root_ome = _read_json(source / "zarr.json")["attributes"]["ome"]
    root.attrs["ome"] = src_root_ome

    ome_grp = root.create_group("OME")
    ome_grp.attrs["ome"] = {"version": "0.5", "series": list(chosen.keys())}

    info = {
        "source_store": str(source),
        "created": datetime.now().isoformat(timespec="seconds"),
        "max_timepoint": max_t,
        "positions": {},
    }

    for pos, t in chosen.items():
        src_pos_dir = source / pos
        src_ome = _read_json(src_pos_dir / "zarr.json")["attributes"]["ome"]
        pos_grp = root.create_group(pos)
        # keep only the OME image metadata (drop bulky pymmcore_plus summary)
        pos_grp.attrs["ome"] = {
            "version": src_ome.get("version", "0.5"),
            "multiscales": src_ome["multiscales"],
            "omero": src_ome["omero"],
        }
        pos_grp.create_array(
            "0",
            shape=shape,
            chunks=chunks,
            dtype="uint16",
            compressors=[BloscCodec(cname="zstd")],
            serializer=BytesCodec(endian="little"),
            fill_value=0,
            chunk_key_encoding=CHUNK_KEY_ENCODING,
            dimension_names=DIMENSION_NAMES,
        )

        # copy the timepoint's chunk files, remapping the T index to 0
        n_copied = 0
        chunk_mtime = 0.0
        for c in range(n_channels):
            for z in range(n_zchunks):
                src_chunk = src_pos_dir / "0" / "c" / str(t) / str(c) / str(z) / "0" / "0"
                dst_chunk = dest / pos / "0" / "c" / "0" / str(c) / str(z) / "0" / "0"
                dst_chunk.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_chunk, dst_chunk)
                chunk_mtime = max(chunk_mtime, src_chunk.stat().st_mtime)
                n_copied += 1
        info["positions"][pos] = {
            "source_timepoint": t,
            "chunks_copied": n_copied,
            "source_chunk_mtime": datetime.fromtimestamp(chunk_mtime).isoformat(
                timespec="seconds"
            ),
        }
        print(f"  {pos}: copied t={t} ({n_copied} chunks)")

    (dest / "checkin_info.json").write_text(json.dumps(info, indent=2))

    if verify:
        verify_snapshot(dest, chosen, shape)

    print(f"\nDone. Snapshot: {dest}")
    return dest


def verify_snapshot(dest: Path, chosen: dict[str, int], shape: tuple) -> None:
    """Re-open read-only and force a blosc decode of both z-edges per position."""
    print("\nVerifying snapshot (read-back)...")
    grp = zarr.open_group(store=str(dest), mode="r", zarr_format=3)
    width = max(len(p) for p in chosen)
    all_ok = True
    for pos, t in chosen.items():
        try:
            arr = grp[pos]["0"]
            if tuple(arr.shape) != tuple(shape):
                raise ValueError(f"shape {tuple(arr.shape)} != expected {tuple(shape)}")
            first = arr[0, :, 0]      # decode first z-chunk, all channels
            last = arr[0, :, -1]      # decode edge z-chunk, all channels
            nonzero = bool(np.any(first)) or bool(np.any(last))
            status = "ok" if nonzero else "ok (all-zero!?)"
            print(f"  {pos:<{width}}  source_t={t:<4} {status}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            all_ok = False
            print(f"  {pos:<{width}}  source_t={t:<4} FAILED: {exc}")
    if not all_ok:
        raise SystemExit(
            "Verification failed for one or more positions. The copied chunks did "
            "not decode against the authored metadata -- re-run with corrected dims "
            "or use the decode-and-rewrite fallback."
        )
    print("Verification passed.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="Live .ome.zarr store")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    # line-buffer stdout so a redirected loop log updates live (not in 8KB blocks)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    ap.add_argument("--no-verify", action="store_true", help="Skip read-back verify")
    ap.add_argument(
        "--every-hours",
        type=float,
        default=0.0,
        help="If > 0, snapshot now then repeat every N hours until Ctrl-C",
    )
    args = ap.parse_args(argv)

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    if not source.is_dir():
        print(f"ERROR: source store not found: {source}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    verify = not args.no_verify
    if args.every_hours <= 0:
        build_snapshot(source, out_dir, verify=verify)
        return 0

    period = timedelta(hours=args.every_hours)
    print(f"Loop mode: snapshot now, then every {args.every_hours} h. Ctrl-C to stop.")
    while True:
        start = datetime.now()
        print(f"\n===== check-in @ {start.isoformat(timespec='seconds')} =====")
        try:
            build_snapshot(source, out_dir, verify=verify)
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001 - keep the loop alive across failures
            print("ERROR during snapshot (continuing to next interval):", file=sys.stderr)
            traceback.print_exc()
        nxt = start + period
        print(f"Next check-in @ {nxt.isoformat(timespec='seconds')}")
        try:
            time.sleep(max(0.0, (nxt - datetime.now()).total_seconds()))
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
