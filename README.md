# shrimPy: Smart High-throughput Robust Imaging & Measurement in Python
![acquisition and reconstruction schematic](docs/figure_3a.png)

shrimPy (pronounced: ʃrɪm-pai) is a pythonic framework for high-throughput smart microscopy and high-performance analysis. The current alpha version of the framework is specific to the mantis microscope, described in our [manuscript](https://doi.org/10.1093/pnasnexus/pgae323), but extensible to a high throughput microscope that is controlled with [Micro-Manager](https://micro-manager.org/).

The acquisition engine synchronizes data collection using hardware triggering and carries out smart microscopy tasks such as autofocus and autoexposure.

The acquired multidimensional raw datasets are processed by the reconstruction engine to generate registered multimodal data that can be used for analysis. Raw data are first converted to the [OME-Zarr](https://ngff.openmicroscopy.org/) format using [iohub](https://github.com/czbiohub-sf/iohub) to facilitate parallel processing and metadata management. Discrete data volumes then undergo deskewing of fluorescence channels, reconstruction of phase and orientation (using [recOrder](https://github.com/mehta-lab/recOrder)), registration and virtual staining (using [VisCy](https://github.com/mehta-lab/viscy)).

This version of the code still uses the legacy name `mantis`, which overlaps with the name of the microscope which is used to acquire data. In a future release we will transition the codebase to the name `shrimPy`.

## Installation

`mantis` can be installed as follows:

1. Create a new Python 3.10 virtual environment using conda:

```sh
conda create -y --name mantis python=3.10
conda activate mantis
```

2. Clone the repo and install this package:

```sh
pip install .
```

## Setting up the mantis microscope
The mantis microscope implements simultaneous label-free and light-sheet imaging as described in [Ivanov et al.](https://www.biorxiv.org/content/10.1101/2023.12.19.572435v1) The two imaging modalities are acquired on two independent arms of the microscope running separate instances of [Micro-Manager](https://micro-manager.org/) and [pycromanager](https://pycro-manager.readthedocs.io/). shrimPy was developed to enable robust long-term imaging with mantis and efficient analysis of resulting TB-scale datasets.

The [Setup Guide](docs/setup_guide.md) outlines how the mantis microscope is configured.


## Data acquisition with mantis

Mantis acquisitions and analyses use a command-line interface.

A list of `mantis` commands can be displayed with:
```sh
mantis --help
```

Data are acquired using `mantis run-acquisition`, and a list of arguments can be displayed with:

```sh
mantis run-acquisition --help
```

The mantis acquisition is configured using a YAML file. An example of a configuration file can be found [here](mantis/acquisition/settings/example_acquisition_settings.yaml).

This is an example of a command which will start an acquisition on the mantis microscope:

```pwsh
mantis run-acquisition \
    --config-filepath path/to/config.yaml \
    --output-dirpath ./YYYY_MM_DD_experiment_name/acquisition_name
```

The acquisition may also be run in "demo" mode with the Micro-manager `MMConfig_Demo.cfg` config. This does not require any microscope hardware. A demo run can be started with:

```pwsh
mantis run-acquisition \
    --config-filepath path/to/config.yaml \
    --output-dirpath ./YYYY_MM_DD_experiment_name/acquisition_name \
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

# DECONVOLVE FLUORESCENCE
# estimate PSF parameters
mantis estimate-psf \
    -i ./beads.zarr \
    -c ./psf_params.yml \
    -o ./psf.zarr
# deconvolve data
mantis deconvolve \
    -i ./acq_name_lightsheet.zarr \
    -c ./deconvolve_params.yml \
    --psf-dirpath ./psf.zarr
    -o ./acq_name_lightsheet_deconvolved.zarr

# DESKEW FLUORESCENCE
# estimate deskew parameters
mantis estimate-deskew \
    -i ./acq_name_lightsheet.zarr/0/0/0 \
    -o ./deskew.yml
# apply deskew parameters
mantis deskew \
    -i ./acq_name_lightsheet.zarr/*/*/* \
    -c ./deskew_params.yml \
    -o ./acq_name_lightsheet_deskewed.zarr

# RECONSTRUCT PHASE/BIREFRINGENCE
recorder reconstruct \
    -i ./acq_name_labelfree.zarr/*/*/* \
    -c ./recon.yml \
    -o ./acq_name_labelfree_reconstructed.zarr

# STABILIZE
# estimate stabilization parameters
mantis estimate-stabilization \
    -i ./acq_name_labelfree.zarr/*/*/* \
    -o ./stabilization.yml \
    --stabilize-xy \
    --stabilize-z
# stabilize data
mantis stabilize \
    -i ./acq_name_labelfree.zarr/*/*/* \
    -c ./stabilization.yml \
    -o ./acq_name_labelfree_stabilized.zarr/*/*/*

# REGISTER
# estimate registration parameters
mantis estimate-registration \
    -s ./acq_name_labelfree_reconstructed.zarr/0/0/0 \
    -t ./acq_name_lightsheet_deskewed.zarr/0/0/0 \
    -o ./register.yml
# optimize registration parameters
mantis optimize-registration \
    -s ./acq_name_labelfree_reconstructed.zarr/0/0/0 \
    -t ./acq_name_lightsheet_deskewed.zarr/0/0/0 \
    -c ./register.yml \
    -o ./register_optimized.yml
# register data
mantis register \
    -s ./acq_name_labelfree_reconstructed.zarr/*/*/* \
    -t ./acq_name_lightsheet_deskewed.zarr/*/*/* \
    -c ./register_optimized.yml \
    -o ./acq_name_registered.zarr

# CONCATENATE CHANNELS
mantis concatenate \
    -c ./concatenate.yml \
    -o ./acq_name_concatenated.zarr

# STITCH
# estimate stitching parameters
mantis estimate-stitching \
    -i ./acq_name.zarr/*/*/* \
    -o ./stitching.yml \
    --channel DAPI
    --percent-overlap 0.05
# stitch fields of view
mantis stitch \
    -i ./acq_name.zarr/*/*/* \
    -c ./stitching.yml \
    -o ./acq_name_stitched.zarr/*/*/*
```

## Additional utilities

We have also developed the following CLI utilities to help with data acquisition and reconstruction

```sh
# Estimate fluorescence bleaching
mantis estimate-bleaching --help

# Update zarr scale metadata
mantis update-scale-metadata --help

# Generate an HTML report with PSF measurements
mantis characterize-psf --help
```

## Data and metadata format

The format of the raw and reconstructed data and associated metadata is documented [here](/docs/data_structure.md).

## Contributing
We are updating the code to enable smart high throughput microscopy on any Micro-Manager controlled microscope. The code will have rough edges for the next several months. We appreciate the bug reports and code contributions if you use this package. If you would like to contribute to this package, please read the [contributing guide](CONTRIBUTING.md).
