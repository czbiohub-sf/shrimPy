#!/bin/bash
#SBATCH --job-name=deskew
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=64G
#SBATCH --output=./output/deskew/deskew-%A-%a.out
env | grep "^SLURM" | sort

module load anaconda
module load comp_micro
conda activate pyplay

now=$(date '+%y-%m-%d')
logpath=./logs/$now/deskew
mkdir -p $logpath
logfile="$logpath/deskew_$SLURM_ARRAY_TASK_ID.out"

INPUT_PATH=$1
OUTPUT_PATH=$2
NUM_CORES=128

echo  "in:  $INPUT_PATH " >> ${logfile}
echo "out: $OUTPUT_PATH " >> ${logfile}
echo "idx: $SLURM_ARRAY_TASK_ID " >> ${logfile}

#Get the array of positions
POSPATHS=($(natsort -p $INPUT_PATH)) >> ${logfile}
echo "pospaths: ${POSPATHS[$SLURM_ARRAY_TASK_ID]}" >> ${logfile}

# Create an array to store the last three directories for each path
OUTPUT_FOVS=()

# Extract the last three directories for each path
for path in "${POSPATHS[@]}"; do
  IFS='/' read -ra last_three_dirs <<< "$path"
  OUTPUT_FOVS+=("${last_three_dirs[-3]}/${last_three_dirs[-2]}/${last_three_dirs[-1]}")
done

# Start measuring the execution time
start_time=$(date +%s.%N) 
echo "Starting the deskew" >> ${logfile}

# Key Code
mantis deskew ${POSPATHS[${SLURM_ARRAY_TASK_ID}]} ./deskew_settings.yml -o "$OUTPUT_PATH/${OUT_FOV_DIR[SLURM_ARRAY_TASK_ID]}" -j $NUM_CORES --slurm >> ${logfile}

# End measuring the execution time
end_time=$(date +%s.%N)
# Calculate the elapsed time
elapsed_time=$(echo "$end_time - $start_time" | bc)

echo "Script execution time: $elapsed_time seconds" >> ${logfile}