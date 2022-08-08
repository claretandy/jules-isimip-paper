#!/bin/bash

variables=( rflow frac tstar_gb gpp gpp_gb npp npp_gb lai lai_gb harvest_gb burnt_area burnt_area_gb runoff precip q1p5m_gb t1p5m_gb )
#variables=( ftl_gb latent_heat )
regions=( brazil southafrica )

for reg in "${regions[@]}"; do
for var in "${variables[@]}"; do
  cat <<EOF > batch_output/preprocessing_${var}_${reg}.sh
#!/bin/bash -l
#SBATCH --qos=long
#SBATCH --mem=20000
#SBATCH --ntasks=2
#SBATCH --output=batch_output/batch_output_${var}_${reg}_%j_%N.out
#SBATCH --time=4320

conda activate isimip

echo ${var} ${reg}
python projections_analysis.py ${var} ${reg}

EOF

  echo "Running: batch_output/preprocessing_${var}_${reg}.sh"
  sbatch batch_output/preprocessing_${var}_${reg}.sh
  sleep 5

done
done
