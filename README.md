# Acquisition and analysis platform for the mantis microscope

This package facilitates acquiring and analyzing data from the mantis microscope. The microscope simultaneously acquires label-free and fluorescence light-sheet data using two independent arms. The acquisition module coordinates data collection using [Micro-manager](https://micro-manager.org/) and [pycromanager](https://pycro-manager.readthedocs.io/). The analysis module is used to process raw data (e.g. reconstruct, deskew, deconvolve, register) to enable biological interpretation.

## Hardware setup

Set up the microscope hardware according to the [Setup Guide](docs/setup_guide.md)

## Installation

**mantis** can be installed as follows:

1. Create a new Python 3.10 virtual environment using conda:

```sh
conda create -y --name mantis python=3.10
conda activate mantis
```

2. Clone the repo and install this package:

```sh
pip install .
```

## Usage

Mantis acquisitions and analyses use a command-line interface.

A list of `mantis` commands can be displayed with:
```sh
mantis --help
```

Data are acquired using `mantis run-acquisition`, and a list of arguments can be displayed with:

```sh
mantis run-acquisition --help
```

The mantis acquisition is configured using a YAML settings file. An example of a settings file can be found [here](mantis/acquisition/settings/example_acquisition_settings.yaml).

This is an example of a command which will start an acquisition on the mantis microscope:

```pwsh
mantis run-acquisition `
    --config-filepath path/to/config.yaml `
    --output-dirpath ./YYYY_MM_DD_experiment_name/acquisition_name
```

The acquisition may also be run in "demo" mode with the Micro-manager `MMConfig_Demo.cfg` config. This does not require any microscope hardware. A demo run can be started with:

```pwsh
mantis run-acquisition `
    --config-filepath path/to/config.yaml `
    --output-dirpath ./YYYY_MM_DD_experiment_name/acquisition_name `
    --mm-config-filepath path/to/MMConfig_Demo.cfg
```

After data has been acquired, we can run analyses from the command line. All analysis calls take an input `-i` and an output `-o`, and the main analysis calls (`deskew`, `reconstruct`, `register`) use configuration files passed via a `-c` flag.

A typical set of CLI calls to go from raw data to registered volumes looks like:

```sh
# CONVERT TO ZARR
iohub convert `
    -i ./acq_name/acq_name_labelfree_1 `
    -o ./acq_name_labelfree.zarr `
iohub convert `
    -i ./acq_name/acq_name_lightsheet_1
    -o ./acq_name_lightsheet.zarr

# DESKEW
mantis estimate-deskew `
    -i ./acq_name_lightsheet.zarr/0/0/0 `
    -o ./deskew.yml
mantis deskew `
    -i ./acq_name_lightsheet.zarr/*/*/*
    -c ./deskew_params.yml `
    -o ./acq_name_lightsheet_deskewed.zarr

# UPCOMING CALLS AHEAD
# RECONSTRUCT
recorder reconstruct `
    -i ./acq_name_labelfree.zarr/*/*/* `
    -c ./recon.yml `
    -o ./acq_name_labelfree_reconstructed.zarr

# REGISTER
mantis estimate-phase-to-fluor-affine `
    -lf ./acq_name_labelfree_reconstructed.zarr/0/0/0 `
    -ls ./acq_name_lightsheet_deskewed.zarr/0/0/0 `
    -o ./register.yml
mantis apply-affine `
    -i ./acq_name_labelfree_deskewed.zarr/*/*/* `
    -c ./register.yml
    -o ./acq_name_registerred.zarr
```

## Contributing

If you would like to contribute to this package, please read the [contributing guide](CONTRIBUTING.md)
