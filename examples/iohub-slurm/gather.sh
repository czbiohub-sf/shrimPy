#!/bin/bash

#SBATCH --job-name=demo_iohub_gather
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G

python -u gather.py --input $1 --output $2