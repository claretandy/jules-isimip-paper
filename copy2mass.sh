#!/bin/bash -l
#SBATCH --qos=long  #high #long
#SBATCH --mem=20000  # 50000
#SBATCH --ntasks=16  # 16
#SBATCH --output=batch_output/copy2mass_%j_%N.out
#SBATCH --time=4320  #360 #4320

# module load scitools
conda activate impacts_toolbox

python copy2mass.py
