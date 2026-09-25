# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

shrimPy is a Python framework for high-throughput smart microscopy that synchronizes data collection using hardware triggering and performs intelligent acquisition tasks like autofocus and autoexposure. The framework is designed to support multiple microscope platforms (mantis, iSIM, Dragonfly) through a modular, extensible architecture built on pymmcore-plus.

Current status: Alpha version, actively restructuring from mantis-only to multi-microscope support (branch: `215-restructure-repository-for-multi-microscope-support`).

## Common Development Commands

### Setup
```bash
# Install in development mode with all dependencies (uses uv)
uv sync

# Or using make
make install

# Install pre-commit hooks (required for contributors)
pre-commit install
```

### Code Quality
```bash
# Format code (ruff format + ruff check --fix)
make format

# Check formatting and linting without modifying files
make check
```

### Testing
```bash
# Run all tests
make test

# Or directly with pytest
uv run pytest

# Run specific test file
uv run pytest tests/test_mantis_logger.py
```

### Running the GUI
```bash
# Launch pymmcore-gui through the shrimPy CLI
uv run shrimpy gui
```
The older Mantis acquisition widget is deprecated and archived in
`archive/` — do not update it.

### Demo Mode Acquisition
No microscope hardware is required -- point `--mm-config` at Micro-Manager's
`MMConfig_Demo.cfg`:
```bash
uv run shrimpy acquire mantis \
    --mm-config path/to/MMConfig_Demo.cfg \
    --mda-config config/mda/mantis/demo.yaml \
    --output-dir ./YYYY_MM_DD_experiment \
    --name acquisition_name
```
The archived pycromanager CLI used a different flag spelling
(`--config-filepath` / `--output-dirpath` / `--mm-config-filepath`); its example
settings are in `archive/pycromanager/acquisition_settings/`.

## Architecture

### Repository Structure
Only `src/shrimpy/` is packaged; everything else is repo-only and stays out of
the wheel. The `src/` layout means an un-installed checkout cannot shadow the
installed package: anything that imports `shrimpy` is exercising the real
distribution, so a file missing from the wheel fails loudly instead of silently
resolving against the working tree.
```
src/shrimpy/             # the installed package
├── engines/             # One module per microscope, all sharing BaseEngine
│   ├── base_engine.py        # BaseEngine: MDAEngine subclass shared by all microscopes
│   ├── mantis_engine.py      # Label-free + light-sheet microscope (implemented)
│   ├── isim_engine.py        # iSIM (placeholder)
│   └── dragonfly_engine.py   # Dragonfly (placeholder)
│
├── config.py            # pydantic validation of the shrimPy metadata sections
├── logging/             # Logging configuration
│   ├── __init__.py           # configure_logging() and DEFAULT_LOGGING_CONFIG
│   └── logging.ini           # package data: the INI the above loads by default
├── dynatrack/           # DynaTrack position tracking (any engine, via BaseEngine)
├── viewer/              # napari viewer, reading the acquisition's OME-Zarr
└── cli/                 # CLI (`shrimpy acquire`, `shrimpy view`, `shrimpy gui`)

tests/                   # Unit and integration tests
config/mda/              # Example acquisition configs to copy and edit
archive/                 # Historical implementations (pycromanager, old pymmcore-plus,
                         # deprecated Mantis Qt widget and its launcher)
packages/                # uv workspace members: separate distributions developed here
└── napari-deskew-preview/   # napari deskew widget, also installable on its own
```

### Workspace members

`packages/*` are uv workspace members: they live in this repo but build and
publish as their own distributions, so they are *not* part of the `shrimpy`
wheel. `[tool.uv.sources]` points at them with `{ workspace = true }`.

`napari-deskew-preview` is the deskew preview widget shrimPy's viewer uses. It
never imports napari (napari discovers it through a `napari.manifest` entry
point), so its dependencies are just numpy and qtpy and its numpy-only tests run
in the default environment — they are in the root `testpaths`. shrimPy depends
on it through the `viewer` extra; it is also in the `dev` group so its tests
always run.

Install it without shrimPy:
```bash
pip install "napari-deskew-preview @ git+https://github.com/czbiohub-sf/shrimPy.git#subdirectory=packages/napari-deskew-preview"
```

`DeskewControls` lives in the public `napari_deskew_preview.controls` module
rather than being re-exported from the package root: it needs qtpy, and the root
`__init__` must stay importable without Qt so shrimPy's acquisition process can
import the package without a GUI toolkit.

Nothing is re-exported from `src/shrimpy/engines/__init__.py`: importing a
microscope engine pulls in its heavy optional dependencies (torch, via
DynaTrack), and the CLI controls when that happens. Import the module directly,
e.g. `from shrimpy.engines.mantis_engine import MantisEngine`.

### Key Design Patterns

#### 1. Engine Abstraction Pattern
`src/shrimpy/engines/base_engine.py` holds `BaseEngine`, the `MDAEngine` subclass
shared by every microscope. It owns the behavior that does not vary by platform:

- hardware-sequencing defaults (`use_hardware_sequencing=True`,
  `force_set_xy_position=False`) and registration with `mmc.mda`
- debug logging of property changes, ROI changes, and XY stage moves
- autofocus handling: reads `metadata.autofocus`, dispatches to the simulated
  `demo-PFS` method or to the microscope's `engage_autofocus()`, and skips the
  event (`SkipEvent`) when autofocus is enabled but did not engage
- Z positions are not written to the autofocus stage while autofocus is engaged
  (`_set_event_properties`)
- resetting `metadata.reset_hardware_sequencing_settings` in `teardown_sequence`
- shared smart-microscopy features: DynaTrack position tracking, built from
  `metadata.dynatrack` in `_setup_dynatrack()`, applied in `event_iterator()`,
  and shut down in `teardown_sequence` — so any engine can use it
- `acquire()`: runs the sequence and writes OME-Zarr to `<name>_<idx>.ome.zarr`

Each microscope subclasses it and overrides only what differs:
```python
class MantisEngine(BaseEngine):
    def __init__(mmc, *args, **kwargs):
        # Microscope-specific defaults (e.g. acquisition timeouts), then super()

    def engage_autofocus(event: MDAEvent) -> bool:
        # Required hook — BaseEngine raises NotImplementedError.
        # Mantis: Nikon PFS with z-offset retries; returns False if it never locks

    def setup_event(event: MDAEvent):
        # XY stage speed modulation, then super().setup_event()
        # - Variable speed (2.0 mm/s short, 5.75 mm/s long distances)
```

To add a new microscope:
1. Subclass `BaseEngine` in `src/shrimpy/engines/<microscope_name>_engine.py`
   (`isim_engine.py` and `dragonfly_engine.py` are placeholders to fill in)
2. Implement `engage_autofocus()`; override `setup_sequence()`, `setup_event()`,
   and positioning methods as needed, always calling `super()`
3. Define microscope-specific metadata schema
4. Add a `shrimpy acquire <microscope_name>` command in `src/shrimpy/cli/acquire.py`
5. Add tests in `tests/test_<microscope_name>_engine.py`

#### 2. Metadata Propagation Pattern
An acquisition config file *is* an `MDASequence`; the microscope settings are
folded directly into its `metadata`, which is how they reach the engine (the MDA
runner passes only the sequence to `setup_sequence` / `teardown_sequence`, and
`metadata` is also captured in the acquisition's summary metadata):

```yaml
setup: ...            # ROI, imaging path, device properties applied once
channels: ...
metadata:
  autofocus: {enabled: true, method: PFS, stage: ZDrive}
  reset_hardware_sequencing_settings:
    - ['TS2_DAC03', 'Sequence', 'Off']
  dynatrack: {enabled: true, input_channel: BF, tracking_channel: BF}
```

`src/shrimpy/config.py` validates those sections with pydantic, so a mistyped
setting fails before any hardware is touched:

```python
from shrimpy.config import ShrimpyMetadata, load_config

sequence = load_config("config/mda/mantis/demo.yaml")  # validates on load
meta = ShrimpyMetadata.from_sequence(sequence)  # engines read this

meta.autofocus  # AutofocusSettings
meta.reset_hardware_sequencing_settings  # [(device, property, value), ...]
meta.dynatrack  # DynaTrackConfig | None
```

Validation is strict (`extra="forbid"`): an unknown metadata section, or an
unknown key within one, is an error. A present-but-disabled `dynatrack` section
is still fully validated — omit the section to disable tracking. Autofocus and
DynaTrack may not both be enabled: both correct Z, so a config that enables the
two together is rejected.

#### 3. Logging Pattern
Every module logs through the `shrimpy` logger hierarchy
(`logger = logging.getLogger(__name__)`); the CLI configures the handlers once,
from the packaged `shrimpy/logging/logging.ini` (`src/` in the repo):
```python
from shrimpy.logging import configure_logging

# During acquisition setup, in the CLI entry point. Uses the packaged
# DEFAULT_LOGGING_CONFIG unless a config_file= path is passed.
log_file = configure_logging(output_dir, name)
# Creates dual handlers on the "shrimpy" logger:
# - Console: INFO level
# - File: DEBUG level (saved to <output_dir>/logs/)

# Also attaches the file handler to the pymmcore-plus logger
```

Use `logger.debug()` for detailed diagnostics (file only) and `logger.info()` for user-facing messages (console + file).

#### 4. Viewer Pattern

The viewer reads the acquisition's OME-Zarr **from disk**, read-only, for both a
finished dataset and one still being written (acquire-zarr no longer holds
`zarr.json` locked — acquire-project/acquire-zarr#234). No image data crosses a
process boundary and nothing is kept in RAM on the acquisition's behalf, so the
whole acquisition is browsable while it runs and a viewer crash or hang cannot
disturb the run. `shrimpy view` and `shrimpy acquire --napari-viewer` are the
same window on the same data; the live case only adds a child process and two
lifecycle messages (`open <path>`, `finish`) over a `multiprocessing.Queue`.

```
shrimpy/viewer/
├── store.py             # AcquisitionStore: read a live or finished OME-Zarr
├── _napari_process.py   # the window; run_viewer() is also `shrimpy view`
└── live.py              # LiveViewer: spawn it beside an acquisition
```

`AcquisitionStore` uses `iohub.open_ome_zarr` for layout, ordering, channels,
axes and scales — it already handles every layout ome-writers writes (single
image, bioformats2raw series, HCS plate) and re-reads array metadata on each
`Position.data` access, so a growing store needs no special handling. What
shrimPy adds is what reading a *live* store needs:

- **A decompressed-volume cache.** A whole z-stack is one chunk (see
  `dimension_overrides` in `BaseEngine.acquire`), so reading one plane
  decompresses the entire stack — ~275 ms for a mantis-sized volume, *every
  time*. Caching `(z, y, x)` volumes under a byte budget is what makes scrubbing
  z, and deskew (where every displayed plane mixes all of them), interactive.
- **`volume_ready()`.** Chunks are flushed asynchronously once complete, so "the
  frame was acquired" is not "the frame is readable", and an unflushed chunk
  reads back as zeros. Checking the chunk files both blanks half-written stacks
  instead of deskewing garbage and keeps a blank volume out of the cache.
- Positions whose array does not exist yet (acquire-zarr creates it on first
  write), reads past the acquired extent, and the `latest_complete` frontier the
  sliders follow.

Deskew geometry: the **scan step** comes from the store's z scale, where the
acquisition recorded it. The **lateral pixel size does not** — MM resolves
`getPixelSizeUm()` from whichever pixel-size config currently matches the device
properties, which has silently changed the deskew ratio between otherwise
identical runs (see `BaseEngine._setup_dynatrack`). It uses the
`napari-deskew-preview` default, and the Deskew widget overrides both.

### Configuration Files

Acquisitions are configured using YAML `MDASequence` files, validated by
`src/shrimpy/config.py`. Examples live in `config/mda/` — `mantis/` (`demo.yaml`,
`mantis.yaml`, `dynatrack_demo.yaml`), `dragonfly/dragonfly.yaml`, and
`replay_demo.yaml`. These are samples to copy and edit; they are *not* installed
with the package. The only runtime package data is `src/shrimpy/logging/logging.ini`.

**Key Configuration Sections:**
- `setup`: ROI, imaging path, and device properties applied once before the run
- `time_plan`: Timepoint intervals and loops
- `channels`: Channel configurations
- `z_plan`: Z-stack range and step size
- `stage_positions`: XY positions or a well-plate plan (optional)
- `metadata.autofocus`: `enabled`, `method` (`PFS` / `demo-PFS`), `stage`,
  `home_focus_device` (default false; opt in only when the Core-Focus device's
  position changes the plane autofocus locks onto, as on the Dragonfly)
- `metadata.reset_hardware_sequencing_settings`: properties restored in teardown
- `metadata.dynatrack`: DynaTrack position tracking, available to every engine
  (see `src/shrimpy/dynatrack/README.md`)

Configs with the settings nested one level deeper under `metadata.mantis` (the
older layout) are rejected by `load_config` with a migration message.

### Qt GUI

`shrimpy gui` launches pymmcore-gui. The Mantis-specific acquisition widget
(`archive/mantis_acquisition_widget.py` and `archive/launch_mantis_gui.py`) **is
deprecated** — do not update it. It still writes/reads the older
`metadata['mantis']` nesting and has not been migrated to `src/shrimpy/config.py`,
so its save/load and run paths are out of sync with the engine. Use the CLI
(`shrimpy acquire mantis --mda-config <config.yaml>`) instead.

## Key Dependencies

Core (always installed):
- **pymmcore-plus**: Python bindings for Micro-Manager with MDA engine
- **useq-schema**: Multi-dimensional acquisition sequence specification
- **ome-writers** / **acquire-zarr**: OME-Zarr output
- **PyYAML**: Configuration parsing
- **numpy**: Numerical operations
- **click**: CLI
- **qtpy**: Qt abstraction layer (only used lazily by the viewer child process)

Nothing in the core set pulls Qt bindings, so `shrimpy acquire` runs headless.

### Extras vs dependency groups

`[project.optional-dependencies]` (extras) are optional **runtime features**: they
ship in the package metadata, so a user of the installed package can opt in with
`uv sync --extra <name>` or `pip install "shrimpy[<name>]"`.

- `gui` — **pymmcore-gui**, required only by `shrimpy gui`
- `viewer` — **napari** + **napari-deskew-preview** + **iohub**, for `shrimpy view`
  and `shrimpy acquire --napari-viewer`
- `dynatrack` — **biahub[stain]** (pulls cytoland/VisCy), **matplotlib**, **torch**

`[dependency-groups]` (PEP 735) are **development/CI only**: they are not in the
published metadata and cannot be installed by a consumer, only with
`uv sync --group <name>` from a checkout.

- `dev` — pre-commit, pytest, ruff, iohub, ipykernel (synced by default)
- `build` — build, twine (release tooling)
- `test-cpu` — CI-only CPU torch wheel; conflicts with the `dynatrack` extra

The rule: a feature a *user* of the package might want is an extra; tooling only a
*contributor* needs is a group.

Other optional analysis libraries, used through `dynatrack` rather than imported
directly: **recOrder** (phase/orientation), **VisCy** (virtual staining).

## Code Style

- **Formatter**: black with line length 95, Python 3.11, skip string normalization (`-S`)
- **Import sorting**: isort (black profile)
- **Linter**: flake8 (disabled: C, R, W, import-error, unsubscriptable-object)
- **Pre-commit hooks**: Automatically run style checks on commit

Run `make format` before committing. The pre-commit hooks will catch violations.

## Package Management

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and `hatchling` + `hatch-vcs` as the build backend (version derived from git tags).

- `pymmcore-plus`, `ome-writers`, and `useq-schema` are pinned to git branches and fetched automatically (see `[tool.uv.sources]` in `pyproject.toml`); no local clones are needed. `make install-dev` optionally overlays editable installs from sibling checkouts when co-developing them.
- Dependencies are locked in `uv.lock` for reproducibility

## Testing

- Framework: pytest
- Test location: `tests/` (repo root, outside the package)
- Ignore: `scripts/`, `**/archive/` (configured in pyproject.toml)
- Run with: `make test` or `pytest . --disable-pytest-warnings`

Shared engine behavior is tested in `tests/test_base_engine.py`; keep
microscope-specific tests in `tests/test_<microscope>_*.py` and add
tests for new microscope engines there.

Viewer tests (`tests/test_viewer_store.py`, `tests/test_viewer_layers.py`) build
real stores with ome-writers/acquire-zarr rather than hand-rolled zarr, because
the behavior being tested *is* that writer's: arrays appear only when a position
is first written, and chunks are flushed asynchronously. They need no napari — a
fake viewer stands in for it — but they do need `iohub` (in the `dev` group).

## Current Development Focus

**Active restructuring:**
- Transitioning from mantis-only to a multi-microscope framework: all engines
  now live in `src/shrimpy/engines/` and share `BaseEngine`
- Archiving legacy code (pycromanager/V1-V2 engines, the Mantis Qt widget) in
  `archive/`
- iSIM and Dragonfly engines are placeholders for future work

**What's stable:**
- Shared acquisition engine (`BaseEngine`) and the Mantis engine (`MantisEngine`)
- CLI-based acquisition workflow (`shrimpy acquire mantis`)
- Logging infrastructure
- Configuration via YAML + metadata

**What's in flux:**
- Cross-microscope abstractions
- iSIM and Dragonfly implementations

## Important Implementation Notes

### Mantis-Specific Behavior
- **Autofocus**: Engages Nikon PFS after XY stage movements with retry logic (up to 3 attempts, 0.5s wait between)
- **Stage speed**: Variable speed based on distance (2.0 mm/s for <2000 µm, 5.75 mm/s for longer moves)
- **Hardware sequencing**: TriggerScope DAC/TTL control for synchronized imaging
- **Dual-arm imaging**: Label-free and light-sheet acquired on separate Micro-Manager instances

### Extending to New Microscopes
When filling in the iSIM / Dragonfly placeholders or adding another microscope:
1. Study `src/shrimpy/engines/base_engine.py` for the shared behavior and
   `src/shrimpy/engines/mantis_engine.py` as the reference subclass
2. Override only the methods that differ from `BaseEngine` behavior
3. Document microscope-specific metadata schema in docstrings
4. Log through `logging.getLogger(__name__)` so messages land in the shared
   `shrimpy` log file
5. Keep archived code in `archive/` for reference

### Data Output
Raw data follows OME-Zarr or NDTiff format. Reconstruction workflows handled by separate biahub library. See `docs/data_structure.md` for details.
