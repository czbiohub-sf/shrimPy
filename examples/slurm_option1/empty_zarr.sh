#!/bin/bash

#SBATCH --job-name=ZARR_INIT
#SBATCH --time=0:10:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --output=./output/deskew/empty_zarr_%j.out
env | grep "^SLURM" | sort

module load anaconda
module load comp_micro
conda activate pyplay

IN_DATA=$1
DESKEW_PARAMS=$2
OUT_DATA=$3

# Logging parameters
rm -r $OUT_DATA
now=$(date '+%y-%m-%d')
logpath=./logs/$now/deskew
mkdir -p $logpath
rm $logpath/*.out
logfile="$logpath/empty_zarr.out"

echo  "raw: $IN_DATA" >> ${logfile}
echo "out: $OUT_DATA " >> ${logfile}
echo "deskew params: $DESKEW_PARAMS " >> ${logfile}

python -u ./empty_zarr.py $IN_DATA $DESKEW_PARAMS -o $OUT_DATA &>> ${logfile}
 