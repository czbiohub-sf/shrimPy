# shrimPy: Smart High-throughput Robust Imaging & Measurement in Python
![acquisition and reconstruction schematic](docs/figure_3a.png)

shrimPy (pronounced: ʃrɪm-pai) is a pythonic framework for high-throughput smart microscopy and high-performance analysis. The current alpha version of the framework is specific to the mantis microscope, described in our [manuscript](https://doi.org/10.1093/pnasnexus/pgae323), but extensible to a high throughput microscope that is controlled with [Micro-Manager](https://micro-manager.org/).

The acquisition engine synchronizes data collection using hardware triggering and carries out smart microscopy tasks such as autofocus and autoexposure.

The acquired multidimensional raw datasets are processed with the [biahub](https://github.com/czbiohub-sf/biahub) library to generate registered multimodal data that can be used for analysis. Raw data are first converted to the [OME-Zarr](https://ngff.openmicroscopy.org/) format using [iohub](https://github.com/czbiohub-sf/iohub) to facilitate parallel processing and metadata management. Discrete data volumes then undergo deskewing of fluorescence channels, reconstruction of phase and orientation (using [recOrder](https://github.com/mehta-lab/recOrder)), registration and virtual staining (using [VisCy](https://github.com/mehta-lab/viscy)).

This version of the code contains an acquisition engine for the `mantis` microscope, including several archived versions. We intend to develop additional acquisition engines for the `iSIM` and `Dragonfly` microscopes within this framework. These acquisition engines are expected to have shared features but also to accommodate differences between the microscope hardware and the acquisition needs on each microscope.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for package management. uv handles Python installation, virtual environments, and dependency locking automatically.

### Prerequisites

Install uv (if not already installed):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone repositories

shrimPy depends on local editable installs of `pymmcore-plus` and `ome-writers`. Clone all three repositories into the same parent directory:

```sh
git clone https://github.com/czbiohub-sf/shrimPy.git
git clone https://github.com/ieivanov/pymmcore-plus.git
git clone https://github.com/ieivanov/ome-writers.git
```

Your directory structure should look like:

```
parent-directory/
  shrimPy/
  pymmcore-plus/
  ome-writers/
```

### Install

```sh
cd shrimPy
uv sync
```

This will automatically:
- Download and install Python 3.11 (if not already available)
- Create a `.venv` virtual environment
- Install shrimPy and all dependencies (including dev tools)
- Install `pymmcore-plus` and `ome-writers` as editable packages from the local clones

### Verify the installation

```sh
uv run python -c "import shrimpy; print('shrimPy installed successfully')"
```

## Usage

### Running the Mantis GUI

```sh
uv run python -m shrimpy.mantis.launch_mantis_gui
```

### CLI

A list of available commands can be displayed with:

```sh
uv run shrimpy --help
```

Data are acquired using `shrimpy acquire <microscope_name>`:

```sh
uv run shrimpy acquire mantis \
    --config-filepath path/to/config.yaml \
    --output-dirpath ./YYYY_MM_DD_experiment_name/acquisition_name
```

The acquisition may also be run in "demo" mode with the Micro-Manager `MMConfig_Demo.cfg` config. This does not require any microscope hardware:

```sh
uv run shrimpy acquire mantis \
    --config-filepath path/to/config.yaml \
    --output-dirpath ./YYYY_MM_DD_experiment_name/acquisition_name \
    --mm-config-filepath path/to/MMConfig_Demo.cfg
```

Acquisitions are configured using YAML files. See [examples/acquisition_settings/](examples/acquisition_settings/) for configuration examples.

## Setting up the mantis microscope

The mantis microscope implements simultaneous label-free and light-sheet imaging as described in [Ivanov et al.](https://doi.org/10.1093/pnasnexus/pgae323) The two imaging modalities are acquired on two independent arms of the microscope running separate instances of [Micro-Manager](https://micro-manager.org/) and [pycromanager](https://pycro-manager.readthedocs.io/). shrimPy and [biahub](https://github.com/czbiohub-sf/biahub) were developed to enable robust long-term imaging with mantis and efficient analysis of the resulting TB-scale datasets on a high-performance compute cluster.

The [Setup Guide](docs/setup_guide.md) outlines how the mantis microscope is configured.

## Data reconstruction

Data reconstruction is accomplished with the [biahub](https://github.com/czbiohub-sf/biahub) library. Visit the link for the latest information on our reconstruction workflows.

## Data and metadata format

The format of the raw and reconstructed data and associated metadata is documented [here](/docs/data_structure.md).

## Development

### Code quality

```sh
# Format code
make format

# Check formatting and linting without modifying files
make check

# Run tests
make test
```

### Pre-commit hooks

Install pre-commit hooks (required for contributors):

```sh
uv run pre-commit install
```

## Contributing

We are updating the code to enable smart high throughput microscopy on any Micro-Manager controlled microscope. The code will have rough edges for the next several months. We appreciate the bug reports and code contributions if you use this package. If you would like to contribute to this package, please read the [contributing guide](CONTRIBUTING.md).
