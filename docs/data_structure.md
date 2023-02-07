
# Data format 


## Directory heirarchy 

We organize the data by positions with a dedicated folder for calibrations considering the following:

* We can parallelize analysis by distributing the compute using `jobs` and `sbatch` commands on HPC.
* We match the directory structure of OME-NGFF format, to make it easier to use the reader and writer modules implemented in `iohub`.
* Backup and file i/o are performant when the data is stored in nested structure with few files per directory.
* Calibration data required by each anlaysis module (recOrder and dexp) is organized consistently and properly.

```markdown

experiment (datestamp_sample)
|--- acq_<short description>_<nn>
|    |--- PositionList.csv (Write out positions from micro-manager to facilitate parsing by iohub and document the cell condition in each well)
|    |--- <Row>
|       |--- <Col>
|           |--- Pos000 
|              |--- labelfree
|                  |--- ndtiff stacks  # all label-free images
|              |--- lightsheet
|                  |--- ndtiff stacks  # all light-sheet images
|           |--- Pos<nnn> 
|              |--- labelfree
|                  |--- ndtiff stacks  # all label-free images
|              |--- lightsheet
|                  |--- ndtiff stacks  # all light-sheet images
|           ...
|--- calib_<short description>_nn
|    |--- labelfree
|           |--- ndtiff stacks  # all label-free images  
|           |---- recOrder calibration metadata.    
|    |--- lightsheet
|           |--- ndtiff stacks  # all light-sheet images
|           |--- dexp calibration metadata.
|   ....
|--- acq_<short description>_nn

```

## Constraints and flexibilities in data heirarchy

* The modalities (channels) contained in one acquisition folder are identical, i.e., all positions and calibration folders contain either labelfree, lightsheet, or labelfree + lightsheet stacks.
* The names of Positions (`Pos`) may be renamed to reflect the condition or perturbation.
* Question: what constraints are imposed by recOrder and dexp analysis pipelines?