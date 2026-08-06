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
uv run pytest shrimpy/tests/test_mantis_logger.py
```

### Running the Mantis GUI
```bash
# Launch the GUI-based acquisition interface
uv run python -m shrimpy.mantis.launch_mantis_gui
```

### Demo Mode Acquisition (Legacy)
The legacy CLI is archived but provides a pattern for programmatic acquisition:
```bash
shrimpy acquire mantis \
    --config-filepath examples/acquisition_settings/example_mda_sequence.yaml \
    --output-dirpath ./YYYY_MM_DD_experiment/acquisition_name \
    --mm-config-filepath path/to/MMConfig_Demo.cfg
```

## Architecture

### Microscope Module Structure
```
shrimpy/
├── base_engine.py       # BaseEngine: MDAEngine subclass shared by all microscopes
├── config.py            # pydantic validation of the shrimPy metadata sections
│
├── mantis/              # Label-free + Light-sheet microscope (fully implemented)
│   ├── mantis_engine.py              # BaseEngine subclass
│   ├── mantis_acquisition_widget.py  # Qt GUI (deprecated, ~815 lines)
│   ├── mantis_logger.py              # Logging configuration
│   ├── launch_mantis_gui.py          # GUI entry point
│   └── archive/                      # Historical implementations (pycromanager, old pymmcore-plus)
│
├── isim/                # iSIM microscope (placeholder for future implementation)
├── viewer/              # Data visualization (placeholder)
├── cli/                 # Command-line interface
└── tests/               # Unit tests
```

### Key Design Patterns

#### 1. Engine Abstraction Pattern
`shrimpy/base_engine.py` holds `BaseEngine`, the `MDAEngine` subclass shared by
every microscope. It owns the behavior that does not vary by platform:

- hardware-sequencing defaults (`use_hardware_sequencing=True`,
  `force_set_xy_position=False`) and registration with `mmc.mda`
- debug logging of property changes, ROI changes, and XY stage moves
- autofocus handling: reads `metadata.autofocus`, dispatches to the simulated
  `demo-PFS` method or to the microscope's `engage_autofocus()`, and skips the
  event (`SkipEvent`) when autofocus is enabled but did not engage
- Z positions are not written to the autofocus stage while autofocus is engaged
  (`_set_event_properties`)
- resetting `metadata.reset_hardware_sequencing_settings` in `teardown_sequence`
- `acquire()`: runs the sequence and writes OME-Zarr to `<name>_<idx>.ome.zarr`

Each microscope subclasses it and overrides only what differs:
```python
class MantisEngine(BaseEngine):
    def __init__(mmc, *args, **kwargs):
        # Microscope-specific defaults (e.g. acquisition timeouts), then super()

    def engage_autofocus(event: MDAEvent) -> bool:
        # Required hook — BaseEngine raises NotImplementedError.
        # Mantis: Nikon PFS with z-offset retries; returns False if it never locks

    def setup_sequence(sequence: MDASequence) -> SummaryMetaV1:
        # DynaTrack setup around super().setup_sequence()

    def setup_event(event: MDAEvent):
        # XY stage speed modulation, then super().setup_event()
        # - Variable speed (2.0 mm/s short, 5.75 mm/s long distances)
```

To add a new microscope:
1. Create `shrimpy/<microscope_name>/` directory
2. Subclass `BaseEngine` in `<microscope_name>_engine.py`
3. Implement `engage_autofocus()`; override `setup_sequence()`, `setup_event()`,
   and positioning methods as needed, always calling `super()`
4. Define microscope-specific metadata schema
5. Create Qt widget for GUI (optional)

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

`shrimpy/config.py` validates those sections with pydantic, so a mistyped
setting fails before any hardware is touched:

```python
from shrimpy.config import ShrimpyMetadata, load_config

sequence = load_config('config/mda/mantis/demo.yaml')  # validates on load
meta = ShrimpyMetadata.from_sequence(sequence)         # engines read this
meta.autofocus                                         # AutofocusSettings
meta.reset_hardware_sequencing_settings                # [(device, property, value), ...]
meta.dynatrack                                         # DynaTrackConfig | None
```

Validation is strict (`extra="forbid"`): an unknown metadata section, or an
unknown key within one, is an error. A present-but-disabled `dynatrack` section
is still fully validated — omit the section to disable tracking.

#### 3. Logging Pattern
Each microscope module uses a separate logger instance:
```python
from shrimpy.mantis.mantis_logger import configure_mantis_logger, get_mantis_logger

# During acquisition setup
logger = configure_mantis_logger(save_dir, 'acquisition_name')
# Creates dual handlers:
# - Console: INFO level
# - File: DEBUG level (saved to logs/ subdirectory)

# Also captures pymmcore-plus logger to same file
```

Use `logger.debug()` for detailed diagnostics (file only) and `logger.info()` for user-facing messages (console + file).

### Configuration Files

Acquisitions are configured using YAML `MDASequence` files, validated by
`shrimpy/config.py`. Examples in `config/mda/mantis/` (`demo.yaml`, `mantis.yaml`,
`dynatrack_demo.yaml`, `replay_demo.yaml`).

**Key Configuration Sections:**
- `setup`: ROI, imaging path, and device properties applied once before the run
- `time_plan`: Timepoint intervals and loops
- `channels`: Channel configurations
- `z_plan`: Z-stack range and step size
- `stage_positions`: XY positions or a well-plate plan (optional)
- `metadata.autofocus`: `enabled`, `method` (`PFS` / `demo-PFS`), `stage`
- `metadata.reset_hardware_sequencing_settings`: properties restored in teardown
- `metadata.dynatrack`: DynaTrack position tracking (see `shrimpy/dynatrack/README.md`)

Configs with the settings nested one level deeper under `metadata.mantis` (the
older layout) are rejected by `load_config` with a migration message.

### Widget Composition (Qt GUI)
```
MantisAcquisitionWidget (main container)
├── ImagePreview (from pymmcore-widgets)
├── CustomCameraRoiWidget (workaround for camera snap issues)
├── StageWidget (XY and Z stage control)
├── MDAWidget (standard multi-dimensional acquisition configuration)
└── MantisSettingsWidget
    ├── TriggerScopeSettingsWidget (hardware triggering)
    └── MicroscopeSettingsWidget (focus device, autofocus, hardware sequencing)
```

Widgets communicate via Qt signals/slots.

**`mantis_acquisition_widget.py` is deprecated** — do not update it. It still
writes/reads the older `metadata['mantis']` nesting and has not been migrated to
`shrimpy/config.py`, so its save/load and run paths are out of sync with the
engine. Use the CLI (`shrimpy acquire mantis --mda-config <config.yaml>`) instead.

## Key Dependencies

- **pymmcore-plus** (0.17.0): Python bindings for Micro-Manager with MDA engine
- **pymmcore-widgets**: Qt widgets for microscope control
- **useq-schema**: Multi-dimensional acquisition sequence specification
- **PyYAML**: Configuration parsing
- **numpy**: Numerical operations
- **qtpy**: Qt abstraction layer (PyQt5/6, PySide2/6)

Optional (for analysis, not in core package):
- **biahub**: Image analysis library (deskewing, reconstruction, registration)
- **iohub**: OME-Zarr conversion and metadata management
- **recOrder**: Phase and orientation reconstruction
- **VisCy**: Virtual staining

## Code Style

- **Formatter**: black with line length 95, Python 3.11, skip string normalization (`-S`)
- **Import sorting**: isort (black profile)
- **Linter**: flake8 (disabled: C, R, W, import-error, unsubscriptable-object)
- **Pre-commit hooks**: Automatically run style checks on commit

Run `make format` before committing. The pre-commit hooks will catch violations.

## Package Management

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and `hatchling` + `hatch-vcs` as the build backend (version derived from git tags).

- `pymmcore-plus` and `ome-writers` are installed as editable local sources (see `[tool.uv.sources]` in `pyproject.toml`)
- Dependencies are locked in `uv.lock` for reproducibility

## Testing

- Framework: pytest
- Test location: `shrimpy/tests/`
- Ignore: `scripts/`, `**/archive/` (configured in pyproject.toml)
- Run with: `make test` or `pytest . --disable-pytest-warnings`

Shared engine behavior is tested in `shrimpy/tests/test_base_engine.py`; keep
microscope-specific tests in `shrimpy/tests/test_<microscope>_*.py` and add
tests for new microscope engines there.

## Current Development Focus

**Active restructuring** (branch `215-restructure-repository-for-multi-microscope-support`):
- Transitioning from mantis-only to multi-microscope framework
- Archiving legacy CLI and V1/V2 acquisition engines
- Establishing iSIM placeholder for future work
- Maintaining GUI-first approach with programmatic API

**What's stable:**
- Shared acquisition engine (`BaseEngine`) and the Mantis engine (`MantisEngine`)
- GUI-based acquisition workflow
- Logging infrastructure
- Configuration via YAML + metadata

**What's in flux:**
- CLI interface (currently empty, being redesigned)
- Cross-microscope abstractions
- iSIM implementation

## Important Implementation Notes

### Mantis-Specific Behavior
- **Autofocus**: Engages Nikon PFS after XY stage movements with retry logic (up to 3 attempts, 0.5s wait between)
- **Stage speed**: Variable speed based on distance (2.0 mm/s for <2000 µm, 5.75 mm/s for longer moves)
- **Hardware sequencing**: TriggerScope DAC/TTL control for synchronized imaging
- **Dual-arm imaging**: Label-free and light-sheet acquired on separate Micro-Manager instances

### Extending to New Microscopes
When adding iSIM or other microscopes:
1. Study `shrimpy/base_engine.py` for the shared behavior and
   `shrimpy/mantis/mantis_engine.py` as the reference subclass
2. Override only the methods that differ from `BaseEngine` behavior
3. Document microscope-specific metadata schema in docstrings
4. Create separate logger instance following mantis_logger pattern
5. Keep archived code in `archive/` subdirectory for reference

### Data Output
Raw data follows OME-Zarr or NDTiff format. Reconstruction workflows handled by separate biahub library. See `docs/data_structure.md` for details.
