#!/bin/bash

module load anaconda
conda activate iohub

DATA_DIR=/hpc/projects/comp.micro/mantis/2023_05_10_PCNA_RAC1/0-crop-convert-zarr
INPUT_DATA=$DATA_DIR/sample.zarr

PROCESSED_DIR=/hpc/mydata/ziwen.liu
TEMP_DIR=$PROCESSED_DIR/demo_processed_sample_tmp
OUTPUT_DIR=$PROCESSED_DIR/demo_processed_sample.zarr
mkdir $TEMP_DIR

POSITION_INFO=$(iohub info -v $INPUT_DATA | grep "Positions")

POSITIONS=${POSITION_INFO: -1}

SCATTER_JOB_ID=$(sbatch --parsable --array=0-$(($POSITIONS-1)) process.sh $INPUT_DATA $TEMP_DIR)
echo $SCATTER_JOB_ID
GATHER_JOB_ID=$(sbatch --parsable -d afterok:$SCATTER_JOB_ID gather.sh $TEMP_DIR $OUTPUT_DIR)
echo $GATHER_JOB_ID
# rm -rf $TEMP_DIR
