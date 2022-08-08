#!/bin/bash -l

# Remove prompts ...
export CONDA_ALWAYS_YES="true"

# Make sure we have the most recent version of conda ...
conda update -n base -c defaults conda

test=$(conda info --envs | grep 'isimip' | cut -d' ' -f1)
if [ $test == 'isimip' ]; then

  conda activate isimip

else

  # Create a new conda environment
  conda create -n isimip python=3.8
  conda config --prepend channels conda-forge
  conda config --set channel_priority strict

  # Install some important packages
  conda install -c conda-forge cftime
  conda install -c conda-forge -n isimip iris
  conda install -c conda-forge -n isimip mo_pack
  conda install -c conda-forge -n isimip h5py
  conda install -c conda-forge -n isimip gdal
  conda install -c conda-forge -n isimip geopandas
  conda install -c conda-forge -n isimip rasterstats
  conda install -c conda-forge -n isimip mapclassify
  conda install -c conda-forge -n isimip seaborn
  conda install -c conda-forge -n isimip regionmask cartopy pygeos

#  conda install -c plotly -n isimip plotly
#  conda install -c plotly -n isimip plotly_express

  # For access to ERA5 data
  # ERA5 access also requires registration at https://cds.climate.copernicus.eu/
  conda install -c conda-forge -n isimip cdsapi

  conda activate isimip

fi
