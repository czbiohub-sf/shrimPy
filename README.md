# SHRIMPy: Smart High-throughput Robust Imaging & Measurement in Python
![acquisition and reconstruction schematic](docs/figure_3a.png)

SHRIMPy (pronounced: ʃrɪm-pai) is a pythonic framework for high-throughput smart microscopy and high-throughput analysis. It was developed for acquisition and reconstruction of image data acquired with the mantis microscope.  The mantis microscope implements simultaneous label-free and light-sheet imaging on two independent arms running separate instances of [Micro-manager](https://micro-manager.org/) and [pycromanager](https://pycro-manager.readthedocs.io/). The acquisition engine synchronizes data collection using hardware triggering and carries out smart microscopy tasks such as autofocus and autoexposure.

The acquired multidimensional raw datasets are processed by the reconstruction engine to generate registered multimodal data that can be used for analysis. Raw data are first converted to the [OME-Zarr](https://ngff.openmicroscopy.org/) format using [iohub](https://github.com/czbiohub-sf/iohub) to facilitate parallel processing and metadata management. Discrete data volumes then undergo deskewing of fluorescence channels, reconstruction of phase and orientation (using [recOrder](https://github.com/mehta-lab/recOrder)), registration and virtual staining (using [VisCy](https://github.com/mehta-lab/viscy)).

## Mantis Microscope setup

Mantis Microscope's set up is outlined in the [Setup Guide](docs/setup_guide.md).

## Installation

`shrimpy` can be installed as follows:

1. Create a new Python 3.10 virtual environment using conda:

```sh
conda create -y --name shrimpy python=3.10
conda activate shrimpy
```

2. Clone the repo and install this package:

```sh
pip install .
```

## Data Acquisition

Mantis acquisitions and analyses use a command-line interface.

A list of `shrimpy` commands can be displayed with:
```sh
shrimpy --help
```

Data are acquired using `shrimpy run-acquisition`, and a list of arguments can be displayed with:

```sh
shrimpy run-acquisition --help
```

The mantis acquisition is configured using a YAML file. An example of a configuration file can be found [here](mantis/acquisition/settings/example_acquisition_settings.yaml).

This is an example of a command which will start an acquisition on the mantis microscope:

```pwsh
shrimpy run-acquisition `
    --config-filepath path/to/config.yaml `
    --output-dirpath ./YYYY_MM_DD_experiment_name/acquisition_name
```

The acquisition may also be run in "demo" mode with the Micro-manager `MMConfig_Demo.cfg` config. This does not require any microscope hardware. A demo run can be started with:

```pwsh
shrimpy run-acquisition `
    --config-filepath path/to/config.yaml `
    --output-dirpath ./YYYY_MM_DD_experiment_name/acquisition_name `
    --mm-config-filepath path/to/MMConfig_Demo.cfg
```

## Data reconstruction

Data reconstruction also uses a command line interface. All reconstruction calls take an input `-i` and an output `-o`, and most reconstruction calls use configuration files passed via a `-c` option.

A typical set of CLI calls to go from raw data to registered volumes looks like:

```sh
# CONVERT TO ZARR
iohub convert \
    -i ./acq_name/acq_name_labelfree_1 \
    -o ./acq_name_labelfree.zarr \
iohub convert \
    -i ./acq_name/acq_name_lightsheet_1 \
    -o ./acq_name_lightsheet.zarr

# DESKEW FLUORESCENCE
# estimate deskew parameters
shrimpy estimate-deskew \
    -i ./acq_name_lightsheet.zarr/0/0/0 \
    -o ./deskew.yml
# apply deskew parameters
shrimpy deskew \
    -i ./acq_name_lightsheet.zarr/*/*/* \
    -c ./deskew_params.yml \
    -o ./acq_name_lightsheet_deskewed.zarr

# RECONSTRUCT PHASE/BIREFRINGENCE
recorder reconstruct \
    -i ./acq_name_labelfree.zarr/*/*/* \
    -c ./recon.yml \
    -o ./acq_name_labelfree_reconstructed.zarr

# TODO: rename function calls as below
# REGISTER
# estimate registration parameters
shrimpy estimate-registration \
    --input-source ./acq_name_labelfree_reconstructed.zarr/0/0/0 \
    --input-target ./acq_name_lightsheet_deskewed.zarr/0/0/0 \
    -o ./register.yml
# optimize registration parameters
shrimpy optimize-registration \
    --input-source ./acq_name_labelfree_reconstructed.zarr/0/0/0 \
    --input-target ./acq_name_lightsheet_deskewed.zarr/0/0/0 \
    -c ./register.yml \
    -o ./register_optimized.yml
# register data
shrimpy register \
    --input-source ./acq_name_labelfree_reconstructed.zarr/*/*/* \
    --input-target ./acq_name_lightsheet_deskewed.zarr/*/*/* \
    -c ./register_optimized.yml \
    -o ./acq_name_registered.zarr
```

## Data and metadata format

The format of the raw and reconstructed data and associated metadata is documented [here](/docs/data_structure.md).

## Contributing
We are updating the code to enable smart high throughput microscopy on any Micro-Manager controlled microscope. The code will have rough edges for the next several months. We appreciate the bug reports and code contributions if you use this package. If you would like to contribute to this package, please read the [contributing guide](CONTRIBUTING.md).
