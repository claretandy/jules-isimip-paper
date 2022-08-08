import iris
import iris.coord_categorisation
import numpy as np
import cf_units
import glob
import subprocess
import datetime as dt
import pandas as pd
import os
import warnings
import rioxarray as rx
import sys
# sys.path.append('/home/h02/hadhy/GitHub/wcssp_casestudies')
import std_functions as sf

fcm_update = False

try:
    # Checkout Karina's JULES branch that contains helper functions such as parallelise
    if os.path.isdir('r6715_python_packages/share'):
        if fcm_update:
            print('Updating Karina\'s JULES branch that contains helper functions such as parallelise')
            out = subprocess.run(['fcm', 'update', 'r6715_python_packages/share'], stdout=subprocess.PIPE)
        import r6715_python_packages.share.jules as jules
    else:
        print('Checking out Karina\'s JULES branch that contains helper functions such as parallelise')
        out = subprocess.run(['fcm', 'checkout', 'fcm:jules.x_br/pkg/karinawilliams/r6715_python_packages/share', 'r6715_python_packages/share'], stdout=subprocess.PIPE)
        import r6715_python_packages.share.jules as jules
        # if out.returncode == 0:
        #     import r6715_python_packages.share.jules as jules

except:
    print('Unable to load jules.py')

import pdb


def extract_from_mass(dest='/scratch/hadhy/ISIMIP/2b/', wildcard=None):

    models = ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
    for model in models:
        masspath = 'moose:/adhoc/users/camilla.mathison/isimip2b/postprocessed/'
        mass_src = masspath + model + '/*'
        if wildcard:
            mass_src = mass_src + wildcard + '*'

        massfiles = subprocess.run(['moo', 'ls', mass_src], capture_output=True, text=True).stdout.split('\n')
        massfiles = [x for x in massfiles if not x == '' ]

        odir = dest + model
        if not os.path.isdir(odir):
            os.makedirs(odir)

        for mf in massfiles:
            ofile = odir + '/' + os.path.basename(str(mf))
            if not os.path.isfile(ofile):
                print('Retrieving', mf)
                subprocess.run(['moo', 'get', mf, ofile])
            # else:
            #     print('File exists', os.path.basename(str(mf)))


def get_region_bbox(region):

    reg_dict = {'global': [-180, -90, 180, 90],  # None, #
                'southafrica': [10, -36, 40, -19],
                'brazil': [-85, -36, -32, 10],
                'london': [-0.58, 51.3, -0.26, 51.62],  # [-1.12193, 50.48168, 0.92705, 52.53066],
                'amsterdam': [4.69, 52.2, 5.06, 52.57]  # [4.33217, 51.71872, 5.561558, 52.948108]
                }

    try:
        return reg_dict[region]
    except:
        return reg_dict['global']


def aggregator_multiyear(cube, var, agg_period='year'):
    '''
    Creates a multi-year mean of annual mean or sum (depending on what's appropriate for the variable). It also ensures that only full years are included in the aggregation
    :param cube: regular cube (tested on monthly data)
    :param var: jules variable name
    :param period: 'year', 'djf', 'mam', 'jja', 'son' or 'all_seas'
    :return: collapsed 2D cube
    '''

    # Define the aggregation function to use over the span of 1 year
    # i.e. do we want an annual SUM or an annual MEAN?
    agg_funcs = {'gpp': iris.analysis.MEAN,
                 'albedo': iris.analysis.MEAN,
                 'et': iris.analysis.MEAN}

    if agg_period == 'year':
        # Year constraint only accepts years where number of months = 12
        yearcon = iris.Constraint(time=lambda t: (t.bound[1] - t.bound[0]) > dt.timedelta(hours=360 * 24.0))
        # First, aggregate by year
        cube_agg = cube.aggregated_by('year', agg_funcs[var])
        # Then, remove years with < 12 month data
        cube_agg = cube_agg.extract(yearcon)
        # Finally, collapse to make a multi-year mean
        cube_ym = cube_agg.collapsed('time', iris.analysis.MEAN)
    else:
        # Year constraint only accepts years where number of months = 12
        seascon = iris.Constraint(time=lambda t: (t.bound[1] - t.bound[0]) > dt.timedelta(hours=85 * 24.0))
        # First, aggregate by season
        cube_agg = cube.aggregated_by(['season', 'season_year'], agg_funcs[var])
        # Then, remove seasons with < 3 months data
        cube_agg = cube_agg.extract(seascon)
        # Finally, collapse to make a multi-year mean
        cube_ym = cube_agg.aggregated_by('season', iris.analysis.MEAN)
        # ... and extract the requested season
        if not agg_period == 'all_seas':
            constr = iris.Constraint(season=agg_period.lower())
            cube_ym = cube_ym.extract(constr)

    return cube_ym


def cci_pft_onload(cube, field, filename):

    import numpy as np

    pft_lut = {'TREES_BE': 0,
               'TREES_BD': 1,
               'TREES_NE': 2,
               'TREES_ND': 3,
               'SHRUBS_BE': 4,
               'SHRUBS_BD': 5,
               'SHRUBS_NE': 6,
               'SHRUBS_ND': 7,
               'GRASS_NAT': 8,
               'GRASS_MAN': 9,
               'BUILT': 10,
               'INLAND_WATER': 11,
               'BARE': 12,
               'SNOWICE': 13}

    major_veg_lut = {'TREES_BE': 0,
               'TREES_BD': 0,
               'TREES_NE': 0,
               'TREES_ND': 0,
               'SHRUBS_BE': 1,
               'SHRUBS_BD': 1,
               'SHRUBS_NE': 1,
               'SHRUBS_ND': 1,
               'GRASS_NAT': 2,
               'GRASS_MAN': 2,
               'BUILT': 4,
               'INLAND_WATER': 4,
               'BARE': 3,
               'SNOWICE': 4}

    leaf_phenol_lut = {'TREES_BE': 0,
               'TREES_BD': 1,
               'TREES_NE': 0,
               'TREES_ND': 1,
               'SHRUBS_BE': 0,
               'SHRUBS_BD': 1,
               'SHRUBS_NE': 0,
               'SHRUBS_ND': 1,
               'GRASS_NAT': 2,
               'GRASS_MAN': 2,
               'BUILT': 4,
               'INLAND_WATER': 4,
               'BARE': 3,
               'SNOWICE': 4}

    pft_lut_5pft = {'Broadleaf_Tree': 0,
               'Needleleaf_Tree': 1,
               'C3Grass': 2,
               'C4Grass': 3,
               'SHRUB': 4,
               'BUILT': 5,
               'INLAND_WATER': 6,
               'BARE': 7,
               'SNOWICE': 8}

    major_veg_lut_5pft = {'Broadleaf_Tree': 0,
               'Needleleaf_Tree': 0,
               'C3Grass': 2,
               'C4Grass': 2,
               'SHRUB': 1,
               'BUILT': 4,
               'INLAND_WATER': 4,
               'BARE': 3,
               'SNOWICE': 4}


    # pft = os.path.basename(filename).split('_0.25_')[0]
    cube.rename('CCI-LC PFTs')
    cube.attributes = {'pft_lut': pft_lut, 'major_veg_lut': major_veg_lut, 'leaf_phenol_lut': leaf_phenol_lut, 'pft_lut_5pft': pft_lut_5pft, 'major_veg_lut_5pft': major_veg_lut_5pft}

    # Uses the filename to identify the pseudo coord value
    if not cube.coords('pseudo_level'):
        for key in pft_lut.keys():
            if key in filename:
                realization_number = pft_lut[key]
                import iris.coords
                realization_coord = iris.coords.AuxCoord(np.int32(realization_number), long_name='pseudo_level')
                cube.add_aux_coord(realization_coord)



def cci_pft_fracs(output_major_veg=False, bbox=None, ver='v1'):
    '''
    Loads the 2010 PFT fractions derived from high resolution observations
    :param output_major_veg: says whether to aggregate to major veg types or just output all pfts
    :param bbox: list containing xmin, ymin, xmax, ymax
    :param ver: string. One of ['cwt', 'v1', 'v2_old', 'v2']
    :return: cube containing tree, shrub, grass, bare dimensions
    '''

    from iris.util import equalise_attributes
    import cf_units

    cubev1 = iris.load_cube('/project/LandCoverCCI/V2/PFT_frac_300m/netcdf_0_25/*.nc', callback=cci_pft_onload)
    pft_lut = cubev1.attributes['pft_lut']
    major_veg_lut = cubev1.attributes['major_veg_lut']
    # leaf_phenol_lut = cubev1.attributes['leaf_phenol_lut']

    if ver == 'v1':
        cube = cubev1.copy()

    elif ver == 'v2_old':
        ocubes = iris.cube.CubeList([])
        for pft, i in pft_lut.items():
            print(pft, i)
            pftcube = sf.gdalds2cube('/project/LandCoverCCI/V2/PFT_frac_300m/PFT_v2/PFT_'+pft+'_0.25_x_0.25_2010_GLOBAL_v2.tif')
            pseudocoord = iris.coords.DimCoord(i, long_name='pseudo_level', units=cf_units.Unit(1))
            pftcube.add_aux_coord(pseudocoord)
            pftcube.rename(cubev1.name())
            ocubes.append(pftcube)
        cube = ocubes.merge_cube()
        cube.attributes = cubev1.attributes

    elif ver == 'v2':
        #NB:  5 PFTs only!!!
        cube = iris.load_cube('/project/LandCoverCCI/V2/PFT_frac_300m/vegfrac_lc-cci_5PFTs_2010_2.nc', callback=cci_pft_onload)
        pft_lut = cube.attributes['pft_lut_5pft']
        major_veg_lut = cube.attributes['major_veg_lut_5pft']
        # iris.coord_categorisation.add_categorised_coord(cube, 'major_veg_class', 'pseudo_level',
        #                         lambda coord, x: major_veg_lut[[k for k, v in pft_lut.items() if x == v][0]])

    elif ver == 'cwt':
        # NB: 5 PFTs only!!!
        cube = iris.load_cube('/project/LandCoverCCI/V2/PFT_frac_300m/vegfrac_lc-cci_5PFTs_2010_CWT.nc', callback=cci_pft_onload)
        pft_lut = cube.attributes['pft_lut_5pft']
        major_veg_lut = cube.attributes['major_veg_lut_5pft']

    elif ver == '300m_CWT':
        ocubes = iris.cube.CubeList([])
        for pft, i in pft_lut.items():
            pft = pft.split('_')[1] + '_TREE' if 'TREE' in pft else pft
            pft = pft.split('_')[1] + '_SHRUB' if 'SHRUB' in pft else pft
            print(pft, i)
            pftxarr = rx.open_rasterio('/project/LandCoverCCI/V2/PFT_FOR_MODELING/300m_CWT_based/CWT_ORCHIDEE_'+pft+'_300m_2010_GLOBAL_v2.0.8.tif', chunks={'band': 1, 'x': 256, 'y': 256})
            pftcube = pftxarr.to_iris()
            pftcube.coord('band').rename('pseudo_level')
            pftcube.coord('x').rename('longitude')
            pftcube.coord('y').rename('latitude')
            pftcube.coord('pseudo_level').units = cf_units.Unit(1)
            pftcube.coord('pseudo_level').points = [i]
            pftcube.rename(cubev1.name())
            ocubes.append(pftcube)

        equalise_attributes(ocubes)
        cube = ocubes.concatenate_cube()
        cube.coord('longitude').units = cf_units.Unit('degrees')
        cube.coord('latitude').units = cf_units.Unit('degrees')
        cube.attributes = cubev1.attributes

    elif ver == '300m_PFT':
        ocubes = iris.cube.CubeList([])
        for pft, i in pft_lut.items():
            pft = 'TREE-' + pft.split('_')[1] if 'TREE' in pft else pft
            pft = 'SHRUB-' + pft.split('_')[1] if 'SHRUB' in pft else pft
            pft = 'WATER-INLAND' if 'INLAND_WATER' in pft else pft
            pft = pft.replace('_', '-')
            ver = 'v0.1' if pft != 'WATER-INLAND' else 'v0.2'
            print(pft, i)
            pftxarr = rx.open_rasterio('/project/LandCoverCCI/V2/PFT_FOR_MODELING/300m_PFT_product/ESACCI-LC-L4-'+pft+'-PFT-Map-300m-P1Y-2010-'+ver+'.tif', chunks={'band': 1, 'x': 256, 'y': 256})
            pftcube = pftxarr.to_iris()
            pftcube.coord('band').rename('pseudo_level')
            pftcube.coord('x').rename('longitude')
            pftcube.coord('y').rename('latitude')
            pftcube.coord('pseudo_level').units = cf_units.Unit(1)
            pftcube.coord('pseudo_level').points = [i]
            pftcube.rename(cubev1.name())
            ocubes.append(pftcube)

        equalise_attributes(ocubes)
        cube = ocubes.concatenate_cube()
        cube.coord('longitude').units = cf_units.Unit('degrees')
        cube.coord('latitude').units = cf_units.Unit('degrees')
        cube.attributes = cubev1.attributes

    elif ver == '0.25deg_CWT':
        ocubes = iris.cube.CubeList([])
        for pft, i in pft_lut.items():
            pft = pft.split('_')[1] + '_TREE' if 'TREE' in pft else pft
            pft = pft.split('_')[1] + '_SHRUB' if 'SHRUB' in pft else pft
            pft = 'WATER_INLAND' if 'INLAND_WATER' in pft else pft
            print(pft, i)
            pftxarr = rx.open_rasterio('/project/LandCoverCCI/V2/PFT_FOR_MODELING/0.25deg_CWT_based/ORCHIDEE_'+pft+'_FRAC_OF_LAND_0.25x0.25_2010_v2.0.8.tif', chunks={'band': 1, 'x': 256, 'y': 256})
            pftcube = pftxarr.to_iris()
            pftcube.coord('band').rename('pseudo_level')
            pftcube.coord('x').rename('longitude')
            pftcube.coord('y').rename('latitude')
            pftcube.coord('pseudo_level').units = cf_units.Unit(1)
            pftcube.coord('pseudo_level').points = [i]
            pftcube.rename(cubev1.name())
            ocubes.append(pftcube)
        cube = ocubes.concatenate_cube()
        cube.attributes = cubev1.attributes

    elif ver == '0.25deg_PFT':
        ocubes = iris.cube.CubeList([])
        for pft, i in pft_lut.items():
            pft = pft.split('_')[1] + '_TREE' if 'TREE' in pft else pft
            pft = pft.split('_')[1] + '_SHRUB' if 'SHRUB' in pft else pft
            pft = 'WATER_INLAND' if 'INLAND_WATER' in pft else pft
            ver = 'v0.1' if pft != 'WATER_INLAND' else 'v0.2'
            print(pft, i)
            pftxarr = rx.open_rasterio('/project/LandCoverCCI/V2/PFT_FOR_MODELING/0.25deg_PFT_product/'+pft+'_FRAC_OF_LAND_0.25x0.25_2010_'+ver+'.tif', chunks={'band': 1, 'x': 256, 'y': 256})
            pftcube = pftxarr.to_iris()
            pftcube.coord('band').rename('pseudo_level')
            pftcube.coord('x').rename('longitude')
            pftcube.coord('y').rename('latitude')
            pftcube.coord('pseudo_level').units = cf_units.Unit(1)
            pftcube.coord('pseudo_level').points = [i]
            pftcube.rename(cubev1.name())
            ocubes.append(pftcube)
        cube = ocubes.concatenate_cube()
        cube.attributes = cubev1.attributes
        cube.coord('longitude').units = cf_units.Unit('degrees')

    else:
        cube = cubev1.copy()

    if bbox and not (bbox == [-180, -90, 180, 90]):
        cube = cube.intersection(longitude=(bbox[0], bbox[2]), latitude=(bbox[1], bbox[3]))

    # Categorise trees, shrub, grass, bare
    try:
        iris.coord_categorisation.add_categorised_coord(cube, 'major_veg_class', 'pseudo_level', lambda coord, x: major_veg_lut[[k for k, v in pft_lut.items() if x == v][0]])
    except:
        pdb.set_trace()

    if output_major_veg:
        major_veg = cube.aggregated_by('major_veg_class', iris.analysis.SUM)
        return major_veg
    else:
        return cube


def observations_runoff():
    '''
    Loads runoff for river basins from Eddy's iLAMB data store
    :return:
    '''
    import netCDF4 as nc
    import pandas as pd
    import geopandas as gpd
    import cf_units
    import numpy as np
    import datetime as dt

    obspath = '/data/users/eroberts/ILAMB_2.5/ILAMB-Data/DATA/runoff/Dai/'
    basins = obspath + 'basins_0.5x0.5.nc'
    runoff = obspath + 'runoff.nc'

    basinsnc = nc.Dataset(basins)
    runoffnc = nc.Dataset(runoff)

    # Load the gridded catchment data
    basins_index = np.flip(basinsnc['basin_index'][:, :], axis=0)
    bid = np.unique(basins_index)
    bid = bid.data[~bid.mask]
    # Make an empty numpy array for the data
    basin_masks = np.empty((len(bid), basins_index.shape[0], basins_index.shape[1]))
    # For each basin, make a 1/0 mask in the 0 axis
    for b in bid:
        tmp = np.where(basins_index == b, 1, 0)
        basin_masks[b, :, :] = tmp

    # Now, make a cube from the basins masks
    lons = basinsnc['lon']
    lats = np.flip(basinsnc['lat'], axis=0)
    lon2d, lat2d = np.meshgrid(lons, lats)
    loncoord = iris.coords.DimCoord(lon2d[0, :], standard_name='longitude', units='degrees')
    loncoord.guess_bounds(0.5)
    latcoord = iris.coords.DimCoord(lat2d[:, 0], standard_name='latitude', units='degrees')
    latcoord.guess_bounds(0.5)
    basincoord = iris.coords.DimCoord(bid, long_name='Basins_ID')
    basins_cube = iris.cube.Cube(data=basin_masks, long_name='Basins index', dim_coords_and_dims=[(basincoord, 0), (latcoord, 1), (loncoord, 2)])

    # Load the timeseries for each catchment into pandas dataframe
    site_names = runoffnc.getncattr('site_name').split(',')
    site_names = [sn.split(' (')[0] for sn in site_names]
    times = runoffnc['time'][:]
    times_dt = cf_units.num2date(times, 'days since 1850-01-01 00:00:00', cf_units.CALENDAR_NO_LEAP)
    times_dt_new = [dt.datetime(mydt.year, mydt.month, 1) for mydt in times_dt]

    mouth_lat = runoffnc['lat'][:]
    mouth_lon = runoffnc['lon'][:]
    coords = [(x, y) for x, y in zip(mouth_lon, mouth_lat)]
    mouth_points = pd.DataFrame({'basin': site_names, 'coords': coords})
    # Create geopandas dataframe of the mouth points
    mouth_points['x'] = mouth_lon
    mouth_points['y'] = mouth_lat
    mouth_points = gpd.GeoDataFrame(mouth_points, geometry=gpd.points_from_xy(mouth_lon, mouth_lat))

    runoff_data = runoffnc['runoff'][:, :]  # ['time', 'data'] assume each row equates to site_name position and value in basins index
    runoff_data = runoff_data.T * 1000
    df = pd.DataFrame(runoff_data, index=site_names, columns=times_dt_new)
    df_long = pd.melt(df, ignore_index=False)
    df_long['basin'] = df_long.index
    df_long = df_long.reset_index()
    df_long.rename(columns={'variable': 'Date'}, inplace=True)
    df_long.index = df_long['Date']
    # df_long_mon = df_long.groupby(by=[df_long.basin, df_long.index.month])

    return {'mouth_points': mouth_points, 'runoff_df': df_long, 'basins_cube': basins_cube}


def observations(var, src='all', start=dt.datetime(1861, 1, 1), end=dt.datetime(2021, 1, 1), bbox=None, aggregation='monthly'):
    '''
    Loads observations from Eddy's ILAMB data store.
    In future, we will add new / updated data sources that can be changed easily
    :param var: string. Same name as the JULES output variable
    :param src: string. Name of the individual dataset or 'all'
    :param start: datetime
    :param end: datetime
    :param bbox: [xmin, ymin, xmax, ymax]
    :param aggregation: 'monthly' or 'annual', but in future could include 'daily'
    :return: cube
    '''

    obspath = '/data/users/eroberts/ILAMB_2.5/ILAMB-Data/DATA/'
    files = glob.glob(obspath + '*/*/*.nc')
    data = [[f.split('/')[-3], f.split('/')[-2], f] for f in files]
    df = pd.DataFrame(data, columns=['variable', 'source', 'filename'])
    vardf = df[df['variable'] == var]
    if not src == 'all':
        vardf = vardf[vardf['source'] == src]

    # Remove Dai dataset because it doesn't work!
    vardf = vardf.loc[vardf['source'] != 'Dai', :]

    # Loop through all records and load the file
    out_dict = {}
    for i, row in vardf.iterrows():
        rsrc, rfn = [row['source'], row['filename']]
        print('   Loading', rsrc, rfn)
        try:
            cube = iris.load_cube(rfn, callback=isimip_output_onload)
        except:
            cube = iris.load_cube(rfn)
        tcon = iris.Constraint(time=lambda cell: start <= cell.point < end)
        cube = cube.extract(tcon)
        if bbox:
            cube = cube.intersection(longitude=(bbox[0], bbox[2]), latitude=(bbox[1], bbox[3]))
        cube = correct_units(var, cube)
        myu = cube.coord('time').units
        ds_start, ds_end = [myu.num2date(cube.coord('time').bounds[0][0]), myu.num2date(cube.coord('time').bounds[-1][1])]
        out_dict[rsrc] = {'data': cube, 'start': ds_start, 'end': ds_end}

    return out_dict


def isimip_output(var, start=dt.datetime(1861, 1, 1), end=dt.datetime(2006, 1, 1), model='all', rcp=None, bbox=None, aggregation='monthly'):
    '''
    Load ISIMIP (post-processed) JULES output
    :param var: as declared in the filename. For a pandas dataframe of the available variables use load_data.list_available_vars()
    :param start: datetime
    :param end: datetime. Note that if a date after 12/2005 is specified, an RCP needs to be given
    :param model: 'all' or name of an individual model
    :param rcp: historical, or rcp26 or rcp60
    :param aggregation: monthly or annual
    :return:
    '''
    warnings.filterwarnings("ignore")
    # '/scratch/hadhy/ISIMIP/2b/'
    datadir = '/data/users/hadcam/ISIMIP/isimip2b_postproc_data/ALL_u-bk886_isimip_0p5deg_origsoil_dailytrif/'

    if model not in ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5', 'all']:
        print('model not in list of models')
        return

    if end <= dt.datetime(2006, 1, 1):
        rcp = 'historical'
    elif not rcp or rcp not in ['historical', 'rcp26', 'rcp60']:
        # print('Assuming RCP6.0 ...')
        rcp = 'rcp60'
    else:
        print('Selected RCP is ' + rcp)

    # Assuming we test date using the logic start <= a_date < end
    hist = {'start': dt.datetime(1860, 1, 1), 'end': dt.datetime(2006, 1, 1)}
    futr = {'start': dt.datetime(2006, 1, 1), 'end': dt.datetime(2100, 1, 1)}

    # Decide if we are straddling a historical / future range
    if start < hist['end'] and futr['start'] < end:
        # Then we have hist and futr data to merge together
        rcp_list = ['historical', rcp]
    else:
        rcp_list = [rcp]

    if model == 'all':
        models = ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
    else:
        models = [model]

    # Use this for building the filename
    time_string = {'historical': '1861_2005', 'rcp26': '2006_2099', 'rcp60': '2006_2099'}
    conc_string = {'historical': 'histsoc_co2', 'rcp26': 'rcp26soc_co2', 'rcp60': 'rcp60soc_co2'}

    out_dict = {}
    for mod in models:
        print('   Loading ' + var.upper() + ': ' + mod)
        outcubelist = iris.cube.CubeList([])

        for onercp in rcp_list:
            # print(mod, onercp)
            # Example filename jules-es-55_miroc5_ewembi_historical_histsoc_co2_trans-ndlevg_global_monthly_1861_2005.nc4
            infile = datadir + mod + '/jules-es-55_'+ mod.lower() +'_ewembi_'+onercp+'_'+conc_string[onercp]+'_'+var+'_global_'+aggregation+'_'+time_string[onercp]+'.nc4'
            try:
                cube = iris.load_cube(infile, callback=isimip_output_onload)
                outcubelist.append(cube)
            except:
                continue

        if len(outcubelist) == 1:
            cube = outcubelist[0]
            tcon = iris.Constraint(time=lambda cell: start <= cell.point < end)
            cube = cube.extract(tcon)
            if bbox:
                cube = cube.intersection(longitude=(bbox[0], bbox[2]), latitude=(bbox[1], bbox[3]))
            out_dict[mod] = correct_units(var, cube)
        elif len(outcubelist) == 2:
            cube = outcubelist.concatenate_cube()
            tcon = iris.Constraint(time=lambda cell: start <= cell.point < end)
            cube = cube.extract(tcon)
            if bbox:
                cube = cube.intersection(longitude=(bbox[0], bbox[2]), latitude=(bbox[1], bbox[3]))
            out_dict[mod] = correct_units(var, cube)
        else:
            out_dict[mod] = None

    return out_dict


def correct_units(var, cube):
    '''
    Given a variable name and cube, converts the data to the desired units, returning a cube in the new units
    Usage: cube = correct_units('gpp', cube)
    :param var: currently 'gpp', 'et' and 'albedo'
    :param cube: cube to plot
    :return: cube in new units and string of the units
    '''

    units_lut = {'gpp': 'g m-2 d-1', 'et': 'kg m-2 d-1', 'evap': 'kg m-2 d-1', 'albedo': '1'}
    unit_str = cube.units.title(1).replace('1 ', '')

    try:
        desired_units = units_lut[var]
    except KeyError:
        desired_units = unit_str

    if (not cube.units.is_unknown()) and (not cube.units.is_dimensionless()):
        if not unit_str == desired_units:
            try:
                cube.convert_units(units_lut[var])
            except:
                # pdb.set_trace()
                if cube.units == 'W m-2' and var == 'et':
                    # See table 1 in the following:
                    # http://www.fao.org/3/x0490e/x0490e04.htm#units
                    # Assuming that the temporal frequency of the units is already in days
                    cube *= 0.408
                    cube.units = cf_units.Unit('kg m-2 d-1')

    if cube.units.is_unknown():
        # Probably Albedo ...
        cube.units = cf_units.Unit(1)

    return cube


def add_frac_metadata(cube):
    '''
    Adds pft names and major vegetation classes to the pft cube
    :param cube:
    :return:
    '''

    pft_lut = {'pft-bdlevgtrop': 0,
                'pft-bdlevgtemp': 1,
                'pft-bdldcd': 2,
                'pft-ndlevg': 3,
                'pft-ndldcd': 4,
                'pft-shrubevg': 5,
                'pft-shrubdcd': 6,
                'pft-c3crop': 7,
                'pft-c3grass': 8,
                'pft-c3pasture': 9,
                'pft-c4crop': 10,
                'pft-c4grass': 11,
                'pft-c4pasture': 12,
                'pft-urban': 13,
                'pft-lake': 14,
                'pft-soil': 15,
                'pft-ice': 16
                }

    major_veg_lut = {'pft-bdlevgtrop': 0,
                'pft-bdlevgtemp': 0,
                'pft-bdldcd': 0,
                'pft-ndlevg': 0,
                'pft-ndldcd': 0,
                'pft-shrubevg': 1,
                'pft-shrubdcd': 1,
                'pft-c3crop': 2,
                'pft-c3grass': 2,
                'pft-c3pasture': 2,
                'pft-c4crop': 2,
                'pft-c4grass': 2,
                'pft-c4pasture': 2,
                'pft-urban': 4,
                'pft-lake': 4,
                'pft-soil': 3,
                'pft-ice': 4
                }

    leaf_phenol_lut = {'pft-bdlevgtrop': 0,
                'pft-bdlevgtemp': 0,
                'pft-bdldcd': 1,
                'pft-ndlevg': 0,
                'pft-ndldcd': 1,
                'pft-shrubevg': 0,
                'pft-shrubdcd': 1,
                'pft-c3crop': 2,
                'pft-c3grass': 2,
                'pft-c3pasture': 2,
                'pft-c4crop': 2,
                'pft-c4grass': 2,
                'pft-c4pasture': 2,
                'pft-urban': 4,
                'pft-lake': 4,
                'pft-soil': 3,
                'pft-ice': 4
                }

    import iris.coords
    cube.attributes = {'pft_lut': pft_lut, 'major_veg_lut': major_veg_lut, 'leaf_phenol_lut': leaf_phenol_lut}
    major_veg_number = [major_veg_lut[list(major_veg_lut.keys())[list(pft_lut.values()).index(pt)]] for pt in cube.coord('type').points]
    leafphenol_number = [leaf_phenol_lut[list(leaf_phenol_lut.keys())[list(pft_lut.values()).index(pt)]] for pt in cube.coord('type').points]
    major_veg_class_coord = iris.coords.AuxCoord(np.int32(major_veg_number), long_name='major_veg_class')
    leaf_phenol_class_coord = iris.coords.AuxCoord(np.int32(leafphenol_number), long_name='leaf_phenol_class')
    cube.add_aux_coord(major_veg_class_coord, data_dims=0)
    cube.add_aux_coord(leaf_phenol_class_coord, data_dims=0)

    return cube


def jules_output(jobid, model='all', var='npp_gb', stream='ilamb', rcp='rcp60', start=dt.datetime(1861, 1, 1), end=dt.datetime(2100, 1, 1), bbox=None):
    '''
    Load raw JULES output
    :param jobid: suite id
    :param model: either 'all' or 'GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5'
    :param var: JULES output variable name
    :param stream: JULES output profile name
    :param rcp:
    :param start:
    :param end:
    :param bbox: list containing xmin, ymin, xmax, ymax
    :return: dictionary of cubes
    '''

    import r6715_python_packages.share.jules as jules
    from iris.util import unify_time_units
    import numpy.ma as ma

    # chantelle_path = '/hpc/data/d05/cburton/jules_output/u-cf137'
    # camilla_data = '/hpc/data/d01/hadcam/jules_output/ALL_u-bk886_isimip_0p5deg_origsoil_dailytrif/'

    path_jobid_lut = {'u-bk886': '/hpc/data/d01/hadcam/jules_output/ALL_u-bk886_isimip_0p5deg_origsoil_dailytrif/',
                      'u-cf137': '/hpc/data/d05/cburton/jules_output/u-cf137/'}
    # var_lut = {'rflow': 'Routing gridbox flow',
    #            'frac': 'Fractional cover of each surface type'}
    # stream_lut = {'rflow': 'gen_mon_gb',
    #               'frac': 'gen_ann_pftlayer',
    #               'tstar_gb': 'gen_mon_gb',
    #               'gpp': 'gen_mon_pft',
    #               'npp': 'gen_mon_pft',
    #               'lai': 'ilamb',
    #               'lai_gb': 'ilamb',
    #               'harvest_gb': 'ilamb',
    #               'burnt_area': 'c_ann_pftlayer',
    #               'burnt_area_gb': 'ilamb',
    #               'runoff': 'ilamb',
    #               'precip': 'ilamb',
    #               'q1p5m_gb': 'ilamb',
    #               't1p5m_gb': 'ilamb'}
    rcp_lut = {'historical': 'c20c',
               'rcp26': 'rcp2p6',
               'rcp60': 'rcp6p0'}
    years = np.arange(start.year, end.year + 1)  # NB: JULES output in yearly streams

    # Could write something here to read /home/h02/hadhy/roses/u-bk886/app/jules/rose-app.conf
    # and parse all of the variable names into a dictionary
    # Might also need a variable name LUT
    # rflow is in 'gen_mon_gb' and is called 'Routing gridbox flow'

    if model not in ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5', 'all']:
        print('model not in list of models')
        return

    if end <= dt.datetime(2006, 1, 1):
        rcp = 'historical'
    elif not rcp or rcp not in ['historical', 'rcp26', 'rcp60']:
        rcp = 'rcp60'
    else:
        print('Selected RCP is ' + rcp)

    # Assuming we test date using the logic start <= a_date < end
    hist = {'start': dt.datetime(1860, 1, 1), 'end': dt.datetime(2006, 1, 1)}
    futr = {'start': dt.datetime(2006, 1, 1), 'end': dt.datetime(2100, 1, 1)}

    # Decide if we are straddling a historical / future range
    if start < hist['end'] and futr['start'] < end:
        # Then we have hist and futr data to merge together
        rcp_list = ['historical', rcp]
    else:
        rcp_list = [rcp]

    if model == 'all':
        models = ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
    else:
        models = [model]

    out_dict = {}
    for mod in models:
        print('   Loading ' + var.upper() + ': ' + mod)
        outcubelist = iris.cube.CubeList([])

        for onercp in rcp_list:
            path = path_jobid_lut[jobid] + mod + '/'
            for yr in years:
                infile = path + mod.lower() + '_' + rcp_lut[onercp] + '.' + stream + '.' + str(yr) + '.nc'
                if os.path.isfile(infile):
                    try:
                        cube = jules.load_cube(infile, var)
                        if not cube.coord('latitude').has_bounds():
                            cube.coord('latitude').guess_bounds(bound_position=0.5)
                        if not cube.coord('longitude').has_bounds():
                            cube.coord('longitude').guess_bounds(bound_position=0.5)
                        if bbox:
                            cube = cube.intersection(latitude=(bbox[1], bbox[3]), longitude=(bbox[0], bbox[2]))
                        cube.data = ma.masked_invalid(cube.data)
                        outcubelist.append(cube)
                        print(onercp, yr)
                    except:
                        continue

        unify_time_units(outcubelist)
        cube = outcubelist.concatenate_cube()
        # Note about the time coordinate:
        #   By default, the stream is saved at the end of the aggregation period (month in this case)
        #   Meaning that the date point is the end of the period, and the bounds refer to the preceding month
        tcoord = cube.coord('time')
        myu = tcoord.units
        bnd_start = [myu.num2date(bnd[0]) for bnd in cube.coord('time').bounds]  # Get the first bound point
        tcoord.points = myu.date2num(bnd_start)
        tcoord.bounds = None
        cube.coord('time').guess_bounds(0)
        cube = cube.extract(iris.Constraint(time=lambda cell: start <= cell.point < end))
        if not cube.coord('latitude').has_bounds():
            cube.coord('latitude').guess_bounds(0.5)
        if not cube.coord('longitude').has_bounds():
            cube.coord('longitude').guess_bounds(0.5)
        out_dict[mod] = cube

    return out_dict


def isimip_driving(model, start=dt.datetime(1861, 1, 1), end=dt.datetime(2100, 1, 1), var='all', rcp=None, bbox=None):
    '''
    Load ISIMIP driving data from HPC
    :param model: One of 4 ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
    :param start: datetime
    :param end: datetime
    :param var: choice of ['hurs', 'huss', 'pr', 'prsn', 'ps', 'psl',  'rlds', 'rsds', 'sfcWind', 'tas', 'tas_range', 'tasmax', 'tasmin']
    :param rcp: choice of ['rcp26', 'rcp60', 'rcp85']
    :return:
    '''

    if model not in ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']:
        print('model not in list of models')
        return

    if var not in ['hurs', 'huss', 'pr', 'prsn', 'ps', 'psl',  'rlds', 'rsds', 'sfcWind', 'tas', 'tas_range', 'tasmax', 'tasmin']:
        print('var not in list of variables')
        return

    if rcp and rcp not in ['rcp26', 'rcp60', 'rcp85']:
        print('rcp not in list of rcps')
        return

    path = '/hpc/data/d00/hadea/isimip2b/'

    # Iris constraint for the start and end date range
    dtrange = iris.Constraint(time=lambda cell: start <= cell.point < end)

    hist_period = {'start': dt.datetime(1861, 1, 1), 'end': dt.datetime(2006, 1, 1)}
    futr_period = {'start': dt.datetime(2006, 1, 1), 'end': dt.datetime(2100, 1, 1)}

    include_historical = True if (hist_period['start'] < start) and (hist_period['end'] > start) else False
    include_future = True if (futr_period['start'] < end) and (futr_period['end'] > end) else False

    if include_historical:
        search_str = path + 'historical/' + model + '/' + var + '_*.nc'
        h_files = sorted(glob.glob(search_str))
        # NB: The last driving data file for 2006-01-01 onwards, is just a symbolic link to the previous period, so ignore
        h_files = [f for f in h_files if not os.path.basename(f).split('_')[-1].split('-')[0] == '20060101']
        h_cubes = iris.load(h_files, callback=isimip_driving_onload)
        h_cube = h_cubes.concatenate_cube()
        h_cube = h_cube.extract(dtrange)
        if bbox:
            h_cube = h_cube.intersection(longitude=(bbox[0], bbox[2]), latitude=(bbox[1], bbox[3]))

    if include_future:
        search_str = path + rcp + '/' + model + '/' + var + '_*.nc'
        f_files = sorted(glob.glob(search_str))
        # NB: The last driving data file for 2100-01-01 onwards, is just a symbolic link to the previous period, so ignore
        f_files = [f for f in f_files if not os.path.basename(f).split('_')[-1].split('-')[0] == '21000101']
        f_cubes = iris.load(f_files, callback=isimip_driving_onload)
        f_cube = f_cubes.concatenate_cube()
        f_cube = f_cube.extract(dtrange)
        if bbox:
            f_cube = f_cube.intersection(longitude=(bbox[0], bbox[2]), latitude=(bbox[1], bbox[3]))

    if include_historical and include_future:
        # h_cube.attributes['title'] = f_cube.attributes['title']
        cubes = iris.cube.CubeList([h_cube, f_cube])
        cube = cubes.concatenate_cube()
        return cube
    elif include_historical:
        return h_cube
    else:
        return f_cube


def isimip_driving_onload(cube, field, filename):

    import cf_units
    import iris.coord_categorisation

    cube.coord("time").bounds = None
    tcoord = cube.coord("time")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="gregorian")
    tcoord.convert_units("days since 1661-01-01 00:00:00")

    # Replace the time coordinate with the corrected one
    cube.remove_coord("time")
    cube.add_dim_coord(tcoord, 0) # might need to find this dimension

    # Guess some bounds
    cube.coord('time').guess_bounds(0)
    cube.coord('latitude').guess_bounds(0.5)
    cube.coord('longitude').guess_bounds(0.5)

    # Add some year and month categorisation
    iris.coord_categorisation.add_year(cube, 'time')
    iris.coord_categorisation.add_month(cube, 'time')
    iris.coord_categorisation.add_season(cube, 'time')
    iris.coord_categorisation.add_season_year(cube, 'time')

    cube.attributes['title'] = None

    return cube


def jules_output_onload(cube, field, filename):
    '''
    Applies some datetime corrections to make reading the ISIMIP data more consistent
    :param cube: cube before the callback is applied
    :param field: unused
    :param filename: unused
    :return: cube with corrected time units
    '''

    import cf_units
    import datetime as dt
    import iris.coord_categorisation
    from dateutil import relativedelta

    cube.coord("time").bounds = None
    tcoord = cube.coord("time")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="gregorian")
    tcoord.convert_units("days since 1661-01-01 00:00:00")
    # Proleptic gregorian goes backwards from when the gregorian calendar was first implemented (1582)
    # tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="proleptic_gregorian")

    # This is monthly data, so make the point refer to the first of the month
    myu = tcoord.units
    dates = myu.num2date(tcoord.points)
    pts = [cf_units.date2num(dt.datetime(timeval.year, timeval.month, timeval.day), 'days since 1661-01-01',
                             calendar='gregorian') for timeval in dates]
    diff = [(myu.num2date(pts[di]) - myu.num2date(pts[di - 1])).days for di in np.arange(len(pts)) if di > 0]
    mndiff = np.mean(diff)
    tstep = 'daily' if mndiff < 2 else 'monthly' if mndiff < 350 else 'annual'

    if 'daily' in filename or tstep == 'daily':
        tcoord.points = [
            cf_units.date2num(dt.datetime(timeval.year, timeval.month, timeval.day), 'days since 1661-01-01',
                              calendar='gregorian') for timeval in dates]

    elif 'monthly' in filename or tstep == 'monthly':
        tcoord.points = [cf_units.date2num(dt.datetime(timeval.year, timeval.month, 1), 'days since 1661-01-01',
                                           calendar='gregorian') for timeval in dates]

    elif 'annual' in filename or tstep == 'annual':
        tcoord.points = [
            cf_units.date2num(dt.datetime(timeval.year, 1, 1), 'days since 1661-01-01', calendar='gregorian') for
            timeval in dates]

    else:
        # Assume daily
        tcoord.points = [
            cf_units.date2num(dt.datetime(timeval.year, timeval.month, timeval.day), 'days since 1661-01-01',
                              calendar='gregorian') for timeval in dates]
    # pdb.set_trace()
    # Replace the time coordinate with the corrected one
    cube.remove_coord("time")
    cube.add_dim_coord(tcoord, 0)  # might need to find this dimension

    return cube


def isimip_output_onload(cube, field, filename):
    '''
    Applies some datetime corrections to make reading the ISIMIP data more consistent
    :param cube: cube before the callback is applied
    :param field: unused
    :param filename: unused
    :return: cube with corrected time units
    '''

    import cf_units
    import datetime as dt
    import iris.coord_categorisation
    from dateutil import relativedelta

    cube.coord("time").bounds = None
    tcoord = cube.coord("time")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="gregorian")
    tcoord.convert_units("days since 1661-01-01 00:00:00")
    # Proleptic gregorian goes backwards from when the gregorian calendar was first implemented (1582)
    # tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="proleptic_gregorian")

    # This is monthly data, so make the point refer to the first of the month
    myu = tcoord.units
    dates = myu.num2date(tcoord.points)
    pts = [cf_units.date2num(dt.datetime(timeval.year, timeval.month, timeval.day), 'days since 1661-01-01', calendar='gregorian') for timeval in dates]
    diff = [(myu.num2date(pts[di]) - myu.num2date(pts[di-1])).days for di in np.arange(len(pts)) if di > 0]
    mndiff = np.mean(diff)
    tstep = 'daily' if mndiff < 2 else 'monthly' if mndiff < 350 else 'annual'

    if 'daily' in filename or tstep == 'daily':
        tcoord.points = [cf_units.date2num(dt.datetime(timeval.year, timeval.month, timeval.day), 'days since 1661-01-01', calendar='gregorian') for timeval in dates]

    elif 'monthly' in filename or tstep == 'monthly':
        tcoord.points = [cf_units.date2num(dt.datetime(timeval.year, timeval.month, 1), 'days since 1661-01-01', calendar='gregorian') for timeval in dates]

    elif 'annual' in filename or tstep == 'annual':
        tcoord.points = [cf_units.date2num(dt.datetime(timeval.year, 1, 1), 'days since 1661-01-01', calendar='gregorian') for timeval in dates]

    else:
        # Assume daily
        tcoord.points = [cf_units.date2num(dt.datetime(timeval.year, timeval.month, timeval.day), 'days since 1661-01-01', calendar='gregorian') for timeval in dates]
    # pdb.set_trace()
    # Replace the time coordinate with the corrected one
    cube.remove_coord("time")
    cube.add_dim_coord(tcoord, 0) # might need to find this dimension

    # Guess some bounds
    cube.coord('time').guess_bounds(0)
    cube.coord('latitude').guess_bounds(0.5)
    cube.coord('longitude').guess_bounds(0.5)

    # Add some year and month categorisation
    iris.coord_categorisation.add_year(cube, 'time')
    if 'daily' in filename or 'monthly' in filename or mndiff < 40:
        iris.coord_categorisation.add_month(cube, 'time')
        iris.coord_categorisation.add_season(cube, 'time')
        iris.coord_categorisation.add_season_year(cube, 'time')

    pft_lut = {'pft-bdlevgtrop': 0,
                'pft-bdlevgtemp': 1,
                'pft-bdldcd': 2,
                'pft-ndlevg': 3,
                'pft-ndldcd': 4,
                'pft-shrubevg': 5,
                'pft-shrubdcd': 6,
                'pft-c3crop': 7,
                'pft-c3grass': 8,
                'pft-c3pasture': 9,
                'pft-c4crop': 10,
                'pft-c4grass': 11,
                'pft-c4pasture': 12,
                'pft-urban': 13,
                'pft-lake': 14,
                'pft-soil': 15,
                'pft-ice': 16
                }

    major_veg_lut = {'pft-bdlevgtrop': 0,
                'pft-bdlevgtemp': 0,
                'pft-bdldcd': 0,
                'pft-ndlevg': 0,
                'pft-ndldcd': 0,
                'pft-shrubevg': 1,
                'pft-shrubdcd': 1,
                'pft-c3crop': 2,
                'pft-c3grass': 2,
                'pft-c3pasture': 2,
                'pft-c4crop': 2,
                'pft-c4grass': 2,
                'pft-c4pasture': 2,
                'pft-urban': 4,
                'pft-lake': 4,
                'pft-soil': 3,
                'pft-ice': 4
                }

    leaf_phenol_lut = {'pft-bdlevgtrop': 0,
                'pft-bdlevgtemp': 0,
                'pft-bdldcd': 1,
                'pft-ndlevg': 0,
                'pft-ndldcd': 1,
                'pft-shrubevg': 0,
                'pft-shrubdcd': 1,
                'pft-c3crop': 2,
                'pft-c3grass': 2,
                'pft-c3pasture': 2,
                'pft-c4crop': 2,
                'pft-c4grass': 2,
                'pft-c4pasture': 2,
                'pft-urban': 4,
                'pft-lake': 4,
                'pft-soil': 3,
                'pft-ice': 4
                }

    if 'pft' in filename:
        import iris.coords
        cube.var_name = None
        cube.attributes = {'pft_lut': pft_lut, 'major_veg_lut': major_veg_lut, 'leaf_phenol_lut': leaf_phenol_lut}
        if not cube.coords('realization'):
            for key in major_veg_lut.keys():
                if key in filename:
                    realization_number = pft_lut[key]
                    majorveg_number = major_veg_lut[key]
                    leafphenol_number = leaf_phenol_lut[key]
                    realization_coord = iris.coords.AuxCoord(np.int32(realization_number), long_name='pft')
                    major_veg_class_coord = iris.coords.AuxCoord(np.int32(majorveg_number), long_name='major_veg_class')
                    leaf_phenol_class_coord = iris.coords.AuxCoord(np.int32(leafphenol_number), long_name='leaf_phenol_class')
                    cube.add_aux_coord(realization_coord)
                    cube.add_aux_coord(major_veg_class_coord)
                    cube.add_aux_coord(leaf_phenol_class_coord)

    # Convert the variable's units to g/m2/s if kg/m2/s
    if cube.units == cf_units.Unit('kg m-2 s-1'):
        cube.convert_units('g m-2 d-1')

    return cube


def list_available_vars(var_string=None, aggregation='monthly'):
    '''
    Get a dataframe containing all the available data
    :param var_string: optional string to limit the dataframe to
    :param aggregation: monthly or annual
    :return: pandas dataframe
    '''
    import glob
    # Get list of variables available
    # scratchdir = '/scratch/hadhy/ISIMIP/2b/'
    scratchdir = '/data/users/hadcam/ISIMIP/isimip2b_postproc_data/ALL_u-bk886_isimip_0p5deg_origsoil_dailytrif/'
    wildcard = scratchdir + '*/jules-es-55_*_ewembi_*_global_'+aggregation+'*.nc4'
    files = sorted(glob.glob(wildcard))
    vars = [f.split('co2_')[1].split('_global_monthly_')[0] for f in files]
    scen = [f.split('_ewembi_')[1].split('_')[0] for f in files]
    models = [os.path.dirname(f).split(scratchdir)[1] for f in files]
    df = pd.DataFrame({'filename': [os.path.basename(f) for f in files], 'var': vars, 'model': models, 'scenario': scen})

    if var_string:
        df = df[df['var'].str.contains(var_string)]

    return df


def testing():
    ''' Read in ISIMIP data and do some initial plots '''
    # moo ls moose:/adhoc/users/camilla.mathison/isimip2b/postprocessed
    # HPC ...
    # No fire : /hpc/data/d01/hadcam/jules_output/ALL_u-bk886_isimip_0p5deg_origsoil_dailytrif
    # Fire : /hpc/data/d01/hadcam/jules_output/RUNC20C_u-by276_isimip_0p5deg_origsoil_dailytrif_fire

    scratchdir = '/scratch/hadhy/ISIMIP/2b/'
    path = '/hpc/data/d01/hadcam/jules_output/RUNFUT60_u-bk886_isimip_0p5deg_origsoil_dailytrif/'
    histrun = 'RUNC20C_u-bk886_isimip_0p5deg_origsoil_dailytrif'
    futureruns = {'rcp26': 'RUNFUT26_u-bk886_isimip_0p5deg_origsoil_dailytrif', 'rcp60': 'RUNFUT60_u-bk886_isimip_0p5deg_origsoil_dailytrif'}
    models = ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']

    vars = ['npp_global_monthly', 'c4grass']

    # Some testing ...
    import iris.quickplot as qplt
    import matplotlib.pyplot as plt
    infile = '/scratch/hadhy/ISIMIP/2b/HADGEM2-ES/jules-es-55_hadgem2-es_ewembi_historical_histsoc_co2_npp_global_monthly_1861_2005.nc4'
    cube = iris.load_cube(infile, callback=isimip_output_onload)
    cube_ts = cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=iris.analysis.cartography.area_weights(cube))

    # Get list of variables available
    wildcard = scratchdir + '*/jules-es-55_*_ewembi_*_global_monthly_*.nc4'
    files = sorted(glob.glob(wildcard))
    vars = [f.split('co2_')[1].split('_global_monthly_')[0] for f in files]
    scen = [f.split('_ewembi_')[1].split('_')[0] for f in files]
    models = [os.path.dirname(f).split(scratchdir)[1] for f in files]
    df = pd.DataFrame({'filename': [os.path.basename(f) for f in files], 'var': vars, 'model': models, 'scenario': scen})

    # Select things
    df[(df['model'] == 'HADGEM2-ES') & (df['var'] == 'albedo')]


if __name__ == '__main__':
    testing()
