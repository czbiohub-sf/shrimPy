
# Data format 

This document defines the standard for organizing data acquired by the mantis microscope.

## Raw directory organization 

Organization of the raw data is constrained by the `pycromanager`-based acquisitioon engine. Currently, we structure raw data in the following hierarchy:

```text

YYYY_MM_DD <experiment_description>
|--- <acq-name>_<n>
|    |--- mantis_acquisition_log_YYYYMMDDTHHMMSS.txt
|
|    |--- positions.csv
|
|    |--- platemap.csv
|
|    |--- <plate_metadata>.csv
|
|    |--- <acq-name>_labelfree_1  # contains PTCZYX dataset
|        |--- NDTiff.index
|        |--- <acq-name>_labelfree_NDTiffStack.tif
|        |--- <acq-name>_labelfree_NDTiffStack_1.tif
|        ...
|
|    |--- <acq-name>_lightsheet_1  # contains PTCZYX dataset
|        |--- NDTiff.index
|        |--- <acq-name>_lightsheet_NDTiffStack.tif
|        |--- <acq-name>_lightsheet_NDTiffStack_1.tif
|        ...
|
|--- <acq-name>_<n>  # one experiment folder may contain multiple acquisitions
|    ...
|
|
|--- calibration
|    |--- labelfree
|    |--- lightsheet

```

An example dataset is provided in: `//ESS/comp_micro/rawdata/mantis/2023_02_21_mantis_dataset_standard/`.

Each acquisition will contain a PTCZYX dataset; some dimensions may be singleton.

The structure of the mantis acquisition log file is not final and is subject to change. Input is welcome. Currently, acquisition script writes one log file per call.

A `positions.csv` file will accompany each acquisition. This file is needed as it carries information about the position labels, which is not saved by `pycromanager`. In the future, we may decide to manage that differently - see [pycro-manager#575](https://github.com/micro-manager/pycro-manager/issues/575). A template for this file is provided [here](positions.csv). 

A `platemap.csv` file will accompany each acquisition. This file carries information about the sample in each well and is populated by the user. Multiple wells may contain the same sample. A template for this file is provided [here](platemap.csv).

Other plate metadata CSV files may also be present. They should follow the structure of `platemap.csv` and contain information on other experimental variables per well.

## Format for storing deconvolved and registered volumes

Raw data files will be converted to [OME-Zarr v0.4](https://ngff.openmicroscopy.org/0.4/) for long-term storage and downstream processing using [iohub](https://github.com/czbiohub/iohub).

The algorithms for deconvolution and registration of label-free and light-sheet data are being developed.  We will organize the data by positions with a dedicated folder for calibrations considering the following:

* We can parallelize analysis by distributing the compute using `jobs` and `sbatch` commands on HPC.
* We match the directory structure of OME-NGFF format, to make it easier to use the reader and writer modules implemented in `iohub`.
* Backup and file i/o are performant when the data is stored in nested structure with few files per directory.
* Calibration data required by each analysis module (recOrder and dexp) is organized consistently and properly.

```text

YYYY_MM_DD_<experiment_description>
|--- <acq-name>
|    |--- <Row>
|       |--- <Col>
|           |--- <Pos_Label> 
|              |--- labelfree  # zarr dataset with TCZYX dimensions
|              |--- lightsheet  # zarr dataset with TCZYX dimensions
|           |--- <Pos_Label>  
|              |--- labelfree  # zarr dataset with TCZYX dimensions
|              |--- lightsheet  # zarr dataset with TCZYX dimensions
|    ...
|
|--- mantis_acquisition_log_YYYYMMDDTHHMMSS.txt
|--- platemap.csv
|--- positions.csv
|--- <plate_metadata>.csv
|
|--- calibration
|    |--- labelfree
|    |--- lightsheet

```

We will also store position metadata within [ome-zarr metadata](https://github.com/czbiohub/iohub/issues/103)

## Constraints and flexibilities in data hierarchy

* The modalities (channels) contained in one acquisition folder are identical, i.e., all positions and calibration folders contain either labelfree, lightsheet, or labelfree + lightsheet stacks.
* The names of Positions (`Pos`) may be renamed to reflect the condition or perturbation.
* Question: what constraints are imposed by recOrder and dexp analysis pipelines?