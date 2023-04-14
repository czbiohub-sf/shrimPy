# Acquisition and analysis platform for the mantis microscope

This package facilitates acquiring and analyzing data from the mantis microscope. The microscope simultaneously acquires label-free and fluorescence light-sheet data using two independent arms. The acquisition module coordinates data collection using [Micro-manager](https://micro-manager.org/) and [pycromanager](https://pycro-manager.readthedocs.io/). The analysis module is used to process raw data (e.g. reconstruct, deskew, deconvolve, register) to enable biological interpretation.

## Installation

**mantis** can be installed as follows:

1. Create a new Python 3.9 virtual environment using conda:

```sh
conda create -y --name mantis python=3.9
conda activate mantis
```

2. Clone the repo and install this package and the [napari](https://napari.org/) viewer:

```sh
pip install ".[napari]"
```

## Usage

Data are acquired using the `run-mantis-acquisition` command. A list of the command line argument can be obtained with:

```
run-mantis-acquisition --help
``` 

The mantis acquisition is configures using a YAML settings file. An example of a settings file can be found [here](mantis/acquisition/settings/example_acquisition_settings.yaml).

This is an example of a command which will start an acquisition on the mantis microscope:

```
run-mantis-acquisition \\
    --data--dirpath ./test \\
    --name test_acquisition \\
    --settings path/to/settings/file
```

The acquisition may also be run in "demo" mode with the Micro-manager `MMConfig_Demo.cfg` config. This does not require any microscope hardware. A demo run can be started with:

```
run-mantis-acquisition \\
    --data--dirpath ./test \\
    --name test_acquisition \\
    --mm-config-file path/to/MMConfig_Demo.cfg/file \\
    --settings path/to/settings/file
```

## Contributing

If you would like to contribute to this package, please read the [contributing guide](CONTRIBUTING.md)