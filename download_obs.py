import cdsapi
import numpy as np

def lai_c3s(odir):

    # V0: 01/1999 to 06/2020
    # V1: 04/1998 to 05/2014
    # V2: 12/2013 to 06/2020
    version = 'V2'

    for year in np.arange(2013, 2021):
        for month in np.arange(1, 13):
            print(year, month)

            c = cdsapi.Client()

            c.retrieve(
                'satellite-lai-fapar',
                {
                    'format': 'tgz',
                    'variable': 'lai',
                    'satellite': 'proba',
                    'sensor': 'vgt',
                    'horizontal_resolution': '1km',
                    'product_version': version,
                    'year': f"{year}",
                    'month': f"{month:02d}",
                    'nominal_day': [
                        '03', '13', '24',
                    ],
                },
                odir.rstrip('/') + '/lai_'+f'{year}-{month:02d}'+'.tar.gz')


def soilmoisture_c3s(odir):

    version = 'v202012.0.0'

    for year in np.arange(1992, 2021):
        for month in np.arange(1, 13):
            print(year, month)

            c = cdsapi.Client()

            c.retrieve(
                'satellite-soil-moisture',
                {
                    'format': 'tgz',
                    'variable': 'soil_moisture_saturation',
                    'type_of_sensor': 'active',
                    'time_aggregation': 'day_average',
                    'version': version,
                    'type_of_record': 'cdr',
                    'day': [f"{x:02d}" for x in np.arange(1, 32)],
                    'month': f"{month:02d}",
                    'year': f"{year}",
                },
                odir.rstrip('/') + '/soilmoisture_'+f'{year}-{month:02d}'+'.tar.gz')


def albedo_c3s(odir):

    version = 'V1'

    for year in np.arange(1998, 2021):
        for month in np.arange(1, 13):

            print(year, month)

            c = cdsapi.Client()

            try:
                c.retrieve(
                    'satellite-albedo',
                    {
                        'format': 'tgz',
                        'month': f'{month:02d}',
                        'nominal_day': ['10', '20', '28', '29', '30', '31'],
                        'year': f"{year}",
                        'horizontal_resolution': '1km',
                        'product_version': version,
                        'sensor': 'vgt',
                        'satellite': 'spot',
                        'variable': 'albb_bh',
                    },
                    odir.rstrip('/') + '/albedo_'+f'{year:04}-{month:02}'+'.tar.gz')
            except:
                pass


def main():

    # Run all download functions
    albedo_c3s(odir='/scratch/hadhy/Obs/Albedo/')
    soilmoisture_c3s(odir='/scratch/hadhy/Obs/SoilMoisture/CCI/')
    lai_c3s(odir='/scratch/hadhy/Obs/LAI/')


if __name__ == '__main__':
    main()
