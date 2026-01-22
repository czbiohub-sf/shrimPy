# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

shrimPy (pronounced: ʃrɪm-pai) is a pythonic framework for high-throughput smart microscopy and high-performance analysis. The codebase uses the legacy name `mantis` internally, which overlaps with the name of the microscope hardware. The mantis microscope implements simultaneous label-free and light-sheet imaging with two independent arms running separate instances of Micro-Manager and pycromanager.

The acquisition engine synchronizes data collection using hardware triggering and performs smart microscopy tasks (autofocus, autoexposure). Raw multidimensional datasets are processed with the biahub library to generate registered multimodal data in OME-Zarr format for analysis.

## Development Commands

### Environment Setup
```sh
# Create and activate conda environment
conda create -y --name mantis python=3.10
conda activate mantis

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality
```sh
make check-format    # Check black and isort formatting
make format          # Apply black and isort formatting
make lint            # Run flake8 linting
make pre-commit      # Run pre-commit hooks on all files
```

### Testing
```sh
make test            # Run pytest on all tests
python -m pytest .   # Run pytest directly
```

### Acquisition Commands
```sh
# Show all mantis commands
mantis --help

# Show acquisition arguments
mantis run-acquisition --help

# Run acquisition with config file
mantis run-acquisition \
    --config-filepath path/to/config.yaml \
    --output-dirpath ./YYYY_MM_DD_experiment_name/acquisition_name

# Run in demo mode (no hardware required)
mantis run-acquisition \
    --config-filepath path/to/config.yaml \
    --output-dirpath ./YYYY_MM_DD_experiment_name/acquisition_name \
    --mm-config-filepath path/to/MMConfig_Demo.cfg
```

## Architecture

### Dual-Microscope System

The mantis microscope runs two independent imaging modalities simultaneously:
- **Label-free (LF)**: Quantitative phase and orientation imaging
- **Light-sheet (LS)**: Fluorescence imaging

Each arm runs a separate instance of Micro-Manager/pycromanager and generates independent PTCZYX datasets. The acquisition engine coordinates both arms and synchronizes hardware triggering.

### Acquisition Engine (`mantis/acquisition/acq_engine.py`)

The `MantisAcquisition` class orchestrates dual acquisitions:
- Manages two sub-acquisitions (LF and LS) as independent pycromanager `Acquisition` objects
- Coordinates hardware setup, autofocus, autoexposure, and data collection
- Uses hardware triggering via TriggerScope and NI-DAQ for synchronization
- Implements smart microscopy features (autofocus via PFS, autoexposure, O3 refocus)

**Note**: This version uses an older version of pycromanager. API documentation is available at https://github.com/micro-manager/pycro-manager.

Key workflow:
1. `setup()` - Initialize both MM instances, configure hardware settings, setup autofocus/autoexposure
2. `acquire()` - Execute time-lapse acquisition with position iteration
3. Hook functions modify behavior at different stages (pre-hardware, post-hardware, post-camera, image-saved)

### V2 Acquisition Engine (Branch: `212-mantis-v2-poc`)

**IMPORTANT: A new version of the acquisition engine is under active development in `mantis/acquisition/mantis_v2.py`.**

The V2 engine represents a significant architectural shift:
- **Framework**: Uses the latest `pymmcore-plus` instead of `pycromanager` for Micro-Manager interface
  - pymmcore-plus: https://github.com/pymmcore-plus/pymmcore-plus
- **Sequence Definition**: Uses `useq-schema` (MDASequence, MDAEvent) for acquisition sequences
- **Engine Architecture**: Implements a custom `MantisEngine` class that extends `MDAEngine`
- **Event-Driven**: Leverages pymmcore-plus event system for hardware callbacks
- **Data Writers**: Testing new OME writers from https://github.com/pymmcore-plus/ome-writers (OMETiffWriter, OMEZarrWriter)
- **Simplified**: Single-arm acquisition with cleaner separation of concerns

Key features of MantisEngine:
- `setup_sequence()` - Configure mantis-specific hardware (TriggerScope, ROI, focus device)
- `setup_event()` - Prepare hardware for each acquisition event
- `_set_event_xy_position()` - Custom XY stage control with speed adjustment and autofocus
- `_engage_autofocus()` - Robust Nikon PFS engagement with multiple Z offset attempts

The V2 engine reads mantis-specific settings from `sequence.metadata['mantis']` and supports:
- ROI configuration
- TriggerScope DAC sequencing and TTL blanking
- Focus device selection (AP Galvo for light-sheet)
- Adaptive XY stage speed (2.0 mm/s for short moves, 5.75 mm/s for long moves)
- Automatic autofocus engagement after position changes

MDA sequences are defined in YAML format (see `examples/acquisition_settings/mantis2_mda.yaml`) and loaded with `MDASequence.from_file()`.

**Current Status**: This is a proof-of-concept in active development. The V1 engine (`acq_engine.py`) remains the production version on the `main` branch.

### Configuration System (`mantis/acquisition/AcquisitionSettings.py`)

All acquisition parameters are defined in YAML configuration files with Pydantic validation. Settings classes include:
- `TimeSettings` - Time-lapse parameters (shared between LF and LS)
- `PositionSettings` - Multi-position acquisition configuration
- `ChannelSettings` - Channel definitions and exposure times (separate for LF/LS)
- `SliceSettings` - Z-stack parameters including sequencing (separate for LF/LS)
- `MicroscopeSettings` - Device properties, ROI, autofocus config (separate for LF/LS)
- `AutoexposureSettings` - Autoexposure algorithm configuration (LS only)

See `examples/acquisition_settings/example_acquisition_settings.yaml` for detailed configuration structure.

### Hook Functions (`mantis/acquisition/hook_functions/`)

The acquisition engine uses pycromanager hooks to customize behavior at different stages:
- `pre_hardware_hook_functions.py` - Called before hardware updates (e.g., log events, prepare DAQ sequences)
- `post_hardware_hook_functions.py` - Called after hardware updates (e.g., update laser power, start DAQ)
- `post_camera_hook_functions.py` - Called after camera triggering
- `image_saved_hook_functions.py` - Called when images are saved

Hook functions can access global state via `mantis.acquisition.hook_functions.globals`.

### Hardware Control

Hardware is controlled through multiple interfaces:
- **Micro-Manager** - Camera, stages, microscope body (via pycromanager)
- **coPylot** - Laser control (Vortran lasers) and stage control (Thorlabs PIA13)
- **NI-DAQ** (nidaqmx) - Analog output sequencing for liquid crystals and galvo mirrors
- **TriggerScope** - Digital triggering and synchronization signals
- **waveorder** - Focus analysis for autofocus algorithms

### Data Organization

Raw data follows pycromanager/NDTiff structure:
```
YYYY_MM_DD_<experiment_description>/
├── <acq-name>_<n>/
│   ├── positions.csv
│   ├── platemap.csv
│   ├── <acq-name>_labelfree_1/      # PTCZYX dataset
│   │   ├── NDTiff.index
│   │   └── <acq-name>_labelfree_NDTiffStack*.tif
│   ├── <acq-name>_lightsheet_1/     # PTCZYX dataset
│   │   ├── NDTiff.index
│   │   └── <acq-name>_lightsheet_NDTiffStack*.tif
│   └── logs/
│       ├── mantis_acquisition_log_*.txt
│       └── conda_environment_log_*.txt
```

Processed data is converted to OME-Zarr v0.4 format using iohub for downstream processing with biahub.

### CLI Structure (`mantis/cli/`)

The CLI uses Click with commands registered in `main.py`:
- `run_acquisition` - Primary command for data acquisition
- `stir_plate_cli` - Control integrated stir plate

The entry point is defined in `pyproject.toml` as `mantis = "mantis.cli.main:cli"`.

### Microscope Configuration

The recommended Micro-Manager version is defined in `mantis/__init__.py` as `__mm_version__`. The mantis microscope requires:
- Nikon Ti2 Control (microscope body)
- SpinView 2.3.0.77 (FLIR cameras)
- CellDrive (Meadowlark liquid crystals)
- TriggerScope firmware (synchronization)
- Vortran Stradus (laser control)
- Thorlabs Kinesis (objective positioning stages)

Two Micro-Manager instances are required: one for interactive control and one for headless LS acquisition.

## Code Style

- Line length: 95 characters
- Formatting: black with `skip-string-normalization = true`
- Import sorting: isort with black profile
- Target: Python 3.10
- Pre-commit hooks enforce style checks before commits
