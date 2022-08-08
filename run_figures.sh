#!/bin/bash -l
#SBATCH --qos=high  #high #long
#SBATCH --mem=200000  # 50000
#SBATCH --ntasks=2  # 16
#SBATCH --output=controller_%j_%N.out
#SBATCH --time=360  #360 #4320

# module load scitools
conda activate isimip

python figures.py
#python projections_analysis.py
