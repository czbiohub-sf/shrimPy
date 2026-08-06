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
├── mantis/              # Label-free + Light-sheet microscope (fully implemented)
│   ├── mantis_engine.py              # MDAEngine subclass (~455 lines)
│   ├── mantis_acquisition_widget.py  # Qt GUI (~815 lines)
│   ├── mantis_logger.py              # Logging configuration
│   ├── launch_mantis_gui.py          # GUI entry point
│   └── archive/                      # Historical implementations (pycromanager, old pymmcore-plus)
│
├── isim/                # iSIM microscope (placeholder for future implementation)
├── viewer/              # Data visualization (placeholder)
├── cli/                 # Command-line interface (in transition, currently empty)
└── tests/               # Unit tests
```

### Key Design Patterns

#### 1. Engine Abstraction Pattern
Each microscope implements a custom `MDAEngine` subclass:
```python
class MantisEngine(MDAEngine):
    def setup_sequence(sequence: MDASequence) -> SummaryMetaV1:
        # Configure hardware before acquisition starts
        # - Set ROI, focus device, initialization settings
        # - Configure hardware sequencing
        # - Setup autofocus parameters

    def setup_event(event: MDAEvent):
        # Prepare for each acquisition event
        # - Configure TriggerScope if using hardware sequencing

    def _set_event_xy_position(event: MDAEvent):
        # Custom XY positioning with intelligent stage movement
        # - Variable speed (2.0 mm/s short, 5.75 mm/s long distances)
        # - Post-movement autofocus engagement with retry logic
        # - Stage settlement waiting
```

To add a new microscope:
1. Create `shrimpy/<microscope_name>/` directory
2. Subclass `MDAEngine` in `<microscope_name>_engine.py`
3. Override `setup_sequence()`, `setup_event()`, and positioning methods as needed
4. Define microscope-specific metadata schema
5. Create Qt widget for GUI (optional)

#### 2. Config Validation and Propagation Pattern
Acquisitions are described by a `ShrimpyConfig` (`shrimpy/config.py`), a pydantic
model with four top-level sections and `extra="forbid"` everywhere, so a mistyped
setting fails before any hardware is touched:

```python
from shrimpy.config import ShrimpyConfig

config = ShrimpyConfig.from_file('config/mda/mantis/demo.yaml')
config.mda         # useq.MDASequence (setup, stage_positions, time_plan, z_plan, channels, ...)
config.autofocus   # AutofocusConfig: enabled, method, stage
config.mantis      # MicroscopeConfig: use_hardware_sequencing, reset_hardware_sequencing_settings
                   # (other scopes will get their own section, e.g. isim: MicroscopeConfig)
config.dynatrack   # DynaTrackConfig | None
```

The MDA runner only passes an `MDASequence` to the engine, so the non-`mda`
sections travel with it under `sequence.metadata['shrimpy']`:

```python
sequence = config.to_sequence()                  # embeds the sections in metadata
config = ShrimpyConfig.from_sequence(sequence)   # engines read them back
```

`MantisEngine.setup_sequence` / `teardown_sequence` call `from_sequence`; missing
sections fall back to defaults (autofocus and DynaTrack disabled). This also
persists the sections in the acquisition's summary metadata.

A present-but-disabled `dynatrack` section is still fully validated (so
`input_channel` / `tracking_channel` are required); omit the section to disable
tracking.

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

Acquisitions are configured using YAML files validated by `ShrimpyConfig`. Examples
in `config/mda/mantis/` (`demo.yaml`, `mantis.yaml`, `dynatrack_demo.yaml`,
`replay_demo.yaml`).

**Top-level sections:**
- `mda`: the `useq.MDASequence` — `setup` (ROI, imaging path, device properties
  applied once before the run), `stage_positions`, `time_plan`, `z_plan`,
  `channels`, `grid_plan`, `axis_order`
- `autofocus`: `enabled`, `method` (`PFS` / `demo-PFS`), `stage`
- `mantis`: `use_hardware_sequencing`, `reset_hardware_sequencing_settings`
- `dynatrack`: DynaTrack position tracking (see `shrimpy/dynatrack/README.md`)

Only `mda` is required. Legacy configs (MDA fields at the top level with settings
nested under `metadata.mantis`) are rejected with a migration message.

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
writes/reads the legacy `metadata['mantis']` layout and has not been migrated to
`ShrimpyConfig`, so its save/load and run paths are out of sync with the engine.
Use the CLI (`shrimpy acquire mantis --mda-config <config.yaml>`) instead.

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

Current tests focus on logging infrastructure. Add tests for new microscope engines in `shrimpy/tests/test_<microscope>_*.py`.

## Current Development Focus

**Active restructuring** (branch `215-restructure-repository-for-multi-microscope-support`):
- Transitioning from mantis-only to multi-microscope framework
- Archiving legacy CLI and V1/V2 acquisition engines
- Establishing iSIM placeholder for future work
- Maintaining GUI-first approach with programmatic API

**What's stable:**
- Mantis acquisition engine (MantisEngine)
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
1. Study `shrimpy/mantis/mantis_engine.py` as the reference implementation
2. Override only the methods that differ from default MDAEngine behavior
3. Document microscope-specific metadata schema in docstrings
4. Create separate logger instance following mantis_logger pattern
5. Keep archived code in `archive/` subdirectory for reference

### Data Output
Raw data follows OME-Zarr or NDTiff format. Reconstruction workflows handled by separate biahub library. See `docs/data_structure.md` for details.
