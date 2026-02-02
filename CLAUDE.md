# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

shrimPy is a Python framework for high-throughput smart microscopy that synchronizes data collection using hardware triggering and performs intelligent acquisition tasks like autofocus and autoexposure. The framework is designed to support multiple microscope platforms (mantis, iSIM, Dragonfly) through a modular, extensible architecture built on pymmcore-plus.

Current status: Alpha version, actively restructuring from mantis-only to multi-microscope support (branch: `215-restructure-repository-for-multi-microscope-support`).

## Common Development Commands

### Setup
```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Or using make
make setup-develop

# Install pre-commit hooks (required for contributors)
pre-commit install
```

### Code Quality
```bash
# Format code (black + isort)
make format

# Check formatting without modifying files
make check-format

# Lint with flake8
make lint

# Run all pre-commit hooks
make pre-commit
```

### Testing
```bash
# Run all tests
make test

# Or directly with pytest
python -m pytest . --disable-pytest-warnings

# Run specific test file
pytest shrimpy/tests/test_mantis_logger.py
```

### Running the Mantis GUI
```bash
# Launch the GUI-based acquisition interface
python -m shrimpy.mantis.launch_mantis_gui
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

#### 2. Metadata Propagation Pattern
Configuration is passed through MDASequence metadata:
```python
sequence = MDASequence.from_file('config.yaml')
sequence.metadata = {
    'mantis': {
        'roi': [x, y, width, height],
        'z_stage': 'AP Galvo',
        'initialization_settings': [[device, property, value], ...],
        'setup_hardware_sequencing_settings': [...],
        'autofocus': {
            'enabled': True,
            'stage': 'ZDrive',
            'method': 'PFS',  # Nikon Perfect Focus System
            'wait_after_correction': 0.5,
            'wait_before_acquire': 0.1,
        },
    }
}
```

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

Acquisitions are configured using YAML files that define MDASequence parameters plus microscope-specific metadata. Examples in `examples/acquisition_settings/`.

**Key Configuration Sections:**
- `time_plan`: Timepoint intervals and loops
- `channels`: Channel configurations
- `z_plan`: Z-stack range and step size
- `stage_positions`: XY positions (optional)
- `metadata.mantis`: Mantis-specific settings (ROI, autofocus, hardware sequencing)

See `examples/acquisition_settings/example_mda_sequence.yaml` for a minimal example.

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

Widgets communicate via Qt signals/slots. Settings are propagated to MDASequence metadata before acquisition starts.

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
