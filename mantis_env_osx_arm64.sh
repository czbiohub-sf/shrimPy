# Run this script to create a conda environment for MANTIS on OSX arm64
# This script assumes that you have already installed miniconda3

# Set the environment variable 
export CONDA_SUBDIR=osx-64 

# Create the conda environment
conda env create --file mantis_env_osx_arm64.yml --name mantis
    
