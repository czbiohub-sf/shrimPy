#!/bin/bash

#SBATCH --job-name=demo_iohub_gather
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=./output/gather_%j.out
env | grep "^SLURM" | sort

#For saving the files stdouts
now=$(date '+%y-%m-%d')
logpath=./logs/$now
mkdir -p $logpath
logfile="$logpath/gather.out"

python -u gather.py --input $1 --output $2 &> ${logfile}