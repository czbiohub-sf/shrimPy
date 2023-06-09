#!/bin/bash

INPUT_DATA="/hpc/instruments/cm.mantis/2023_05_10_PCNA_RAC1/timelapse_2_3/timelapse_2_lightsheet_1.zarr/*/*/*"
OUTPUT_PATH="./timelapse_2_lightsheet_1_deskewed.zarr"
DESKEW_PARAMS="./deskew_settings.yml"

# Make an array of positions
POSPATHS=($(natsort -p $INPUT_DATA))
POSITIONS=${#POSPATHS[@]}

# Get the Zarrstore name
IFS='/' read -ra path <<< "${POSPATHS[0]}"
ZARR_STORE=$(IFS='/'; echo "${path[*]:0:${#path[@]}-3}")
echo "Zarr Store: $ZARR_STORE"

rm ./output/deskew/*.out

# Create an empty array to pre-initialize positions taking pos 0 as sample for the shape
ZARR_JOB_ID=$(sbatch --parsable empty_zarr.sh "$INPUT_DATA" $DESKEW_PARAMS $OUTPUT_PATH)
echo "DONE ZARR JOB: $ZARR_JOB_ID"
DESKEW_JOB_ID=$(sbatch --parsable --array=0-$((POSITIONS-1)) -d after:$ZARR_JOB_ID deskew.sh "$INPUT_DATA" $OUTPUT_PATH)
