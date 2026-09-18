# ReplayCamera / acquisition write-path — known issues (to fix later)

Two issues found while validating a ReplayCamera acquisition of the A549 HCS
dataset (`0-flatfield`, OME-NGFF 0.5 / zarr v3, `float32`, 147 positions,
T=1 C=2 Z=1068). The acquisition itself runs and reproduces pixels byte-for-byte;
these are about the *written output* being trustworthy for downstream analysis.

Repro used:
- mm-config: `config/MMConfig_A549_fov_replay.cfg`
- mda-config: `config/mda/mantis/replay_A549_demo.yaml`
- output: `/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/replay_test/replay_A549_test_1.ome.zarr`

---

## Issue 1 — output dtype mislabeled `uint32`, bytes are actually `float32`

**Symptom.** Input arrays are `float32`; the written output arrays are labeled
`uint32`. The `uint32` values are the *bit reinterpretation* of the source
`float32` (verified: `output == input_float32.view(uint32)` is True; a numeric
cast is False). E.g. input pixel `27550.4` is stored as `1188510925`.

**Impact.** Byte-level replay fidelity is fine, but any reader that trusts the
dtype label (iohub, napari, the FOV-selection pipeline) reads garbage
(~1.19e9 instead of ~27550.4). Real values are only recoverable via
`arr.view(np.float32)`.

**Root cause (hypothesis).** MM's image model has only integer pixel types.
ReplayCamera reports `float32` (`bytesPerPixel=4`); the core / `ome_writers`
map "4 bytes" to a 32-bit *integer*, so float bytes get stored under a `uint32`
label. Largely **replay-specific**: real cameras emit `uint8/uint16` and would
be labeled correctly. Surfaces here because we replay `float32`
(flatfielded/reconstructed) data through an integer-pixel pipeline.

**Fix ideas.**
- Make the write path carry the source dtype (`float32`) through to the store,
  or
- Define an explicit pixel-type policy for replay (e.g. only replay integer
  stores, or intentionally cast with documented behavior).
- Check where the pixel type is resolved: `ReplayCamera.dtype()` vs
  `core.getBytesPerPixel()/getImageBitDepth()` vs the `ome_writers` handler in
  `shrimpy/mantis/mantis_engine.py` (`acquire`, ~line 487+).

---

## Issue 2 — output HCS plate metadata incomplete (well `images` lists empty)

**Symptom.** iohub `open_ome_zarr(...).positions()` returns 0 positions and logs
`Skipped item at B/3/000000: invalid <class 'type'>`. Channel names read fine.
The per-well group metadata is empty:
- Input  `B/3/zarr.json` attributes: `{"version":"0.5","images":[{"path":"000000"}, ...]}`
- Output `B/3/zarr.json` attributes: `{}`  (no `well` / `images` list)

**Impact.** Pixel data is physically present (openable by explicit path, e.g.
`B/3/000000/0`), but OME-Zarr HCS readers can't enumerate FOVs via the plate
API. The FOV-selection pipeline would need to open positions by explicit path
instead of iterating the plate.

**Root cause (hypothesis).** The `ome_writers` HCS write path isn't emitting the
NGFF well-level `images` metadata. Not obviously replay-specific — a real plate
acquisition through the same writer could have the same gap. May be a version /
config issue on the `ome-writers` git pin.

**Fix ideas.**
- Inspect what `ome_writers` writes at the well level during a plate
  acquisition; ensure `images` is populated.
- Check the `ome-writers` pin in `pyproject.toml`
  (`ome-writers = { git = ... }`) for a newer revision that writes well metadata.
