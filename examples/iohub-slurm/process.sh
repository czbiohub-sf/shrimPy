#!/bin/bash

#SBATCH --job-name=demo_iohub_scatter
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=./output/process_%A-%a.out
env | grep "^SLURM" | sort

#For saving the files stdouts
now=$(date '+%y-%m-%d')
logpath=./logs/$now
mkdir -p $logpath
logfile="$logpath/process_$SLURM_ARRAY_TASK_ID.out"

FOV_NAME=/0/0/$SLURM_ARRAY_TASK_ID

module load anaconda/2022.05
conda activate iohub
python -u process.py --input $1$FOV_NAME --output $2$FOV_NAME
