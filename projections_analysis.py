import pdb
import sys
import os.path
import pickle
import iris
import pandas as pd
import numpy as np
import numpy.ma as ma
import load_data
import datetime as dt
import itertools
import warnings
import seaborn as sns
import itertools

import projections_analysis


def getGwls(model=None, rcp=None, gwl=None, ens='r1i1p1'):

    # File taken from https://github.com/mathause/cmip_warming_levels
    infile = 'cmip5_warming_levels_all_ens_1861_1900.csv'
    df = pd.read_csv(infile, header=4, skipinitialspace=True)

    if model:
        model = 'HadGEM2-ES' if model == 'HADGEM2-ES' else model
        df = df.loc[(df['model'] == model) & (df['ensemble'] == ens)]

    if rcp:
        df = df.loc[(df['exp'] == rcp)]

    if gwl:
        df = df.loc[df['warming_level'] == gwl]

    return df


def change_units(var, cube):
    '''
    Function for correcting units depending on variable name
    :param cube:
    :return:
    '''

    if var in ['precip', 'runoff']:
        cube.convert_units('kg m-2 d-1')

    if var == 't1p5m_gb':
        cube.convert_units('deg_C')

    if var in ['npp_gb', 'gpp_gb']:
        cube.convert_units('g m-2 d-1')

    if var == 'burnt_area_gb':
        # % per year
        cube.data *= 3110400000  # 360 days * 24h * 60mins * 60secs * 100
        cube.units = '% per year'
        # cube.convert_units('d-1')

    return cube


def aggregateTimeseries_by_basin(in_dict=None, var=None, region='southafrica', overwrite=False):
    '''
    Takes a dictionary of cubes, and aggregates by major drainage basin within the country
    :param in_dict:
    :param var:
    :param region:
    :return: pandas dataframe
    '''

    # Read dictionary keys to get region, var, etc
    # model, fireflag, rcp, var = [key.split('-') for key in out_dict.keys()]

    csv_basins_file = f'/net/data/users/hadhy/Projects/ISIMIP/timeseries_on_basins-{region}-{var}.csv'

    if not os.path.isfile(csv_basins_file) or overwrite:

        # Create a pandas dataframe with the following columns
        cols = ['model', 'fire', 'rcp', 'variable', 'basin_id', 'time', 'value']
        df = pd.DataFrame(columns=cols)

        # Get a cube from the dictionary as a template for the river basins mask
        template = in_dict[list(in_dict.keys())[0]]

        # Get a river basin mask, with a unique ID for each basin
        # Also, only river basins that intersect the region polygon (i.e. brazil or southafrica) will be selected.
        # This is used later to aggregate out_dict by river basins
        basins_mask = get_river_basins(template, region=region)
        # Get a list of unique values from the river basins mask
        vals = np.unique(basins_mask)

        # Now loop through those unique values
        for bas in vals[~vals.mask]:
            print('Basin:', bas)
            # For this basin, create a temporary mask so that all the other basin grid cells are removed
            this_basin_mask = ma.masked_where(basins_mask != bas, basins_mask)

            # Loop through all the cubes in the dictionary
            for k, cube in in_dict.items():

                # The key has a few important variables in that describe the data.
                # They're separated by '-', so we can use that to split the string
                model, fire, rcp = k.rsplit('-', 2)
                # This just corrects for some of the model names containing a '-', which messes up the splitting
                if not 'fire' in fire:
                    model, fire, rcp, var = k.rsplit('-', 3)

                # If the mask shape doesn't match cube shape, then nothing will work,
                # so I've put it around a try / except statement. Continue means it jumps to the next item in the loop
                try:
                    cube.data.mask = this_basin_mask.mask
                except:
                    continue

                # Create a pandas datetime series from the cube time coordinate
                myu = cube.coord('time').units
                myudt = myu.num2date(cube.coord('time').points)
                # Need to subtract 1 month (JULES values are for end of month)
                pddt = pd.to_datetime([dt.datetime(x.year, x.month, x.day) for x in myudt]) - pd.DateOffset(months=1)
                if pddt[-1].day != 1:
                    pddt[-1] + pd.DateOffset(days=2)

                # Do some common transformations of the data units
                cube = change_units(var, cube)

                # Calculate global weighted mean
                if not cube.coord('latitude').has_bounds():
                    cube.coord('latitude').guess_bounds()
                if not cube.coord('longitude').has_bounds():
                    cube.coord('longitude').guess_bounds()
                grid_areas = iris.analysis.cartography.area_weights(cube)
                tmp_data = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=grid_areas).data.data

                # Some JULES output might have a 4th dimension (such as veg_frac)
                # The following line gets the names of all the dims that are not lat or lon ...
                dims = [cc.name() for cc in cube.coords() if not cc.name() in ['latitude', 'longitude']]
                if len(dims) > 1:
                    print('More than 1 dimension to read ...')
                    # Get the names of the 4th dimension
                    nontime, = [d for d in dims if d != 'time']
                    # Create a temporary dataframe to put the data in
                    tmpcols = ['time', 'value', 'variable']
                    tmp_df = pd.DataFrame(columns=tmpcols)
                    # Loop through each point in the nontime dimension, and add data to the tmp_df
                    for i, x in enumerate(cube.coord(nontime).points):
                        dimtmp_df = pd.DataFrame({'time': pddt, 'value': tmp_data[:, i]})
                        dimtmp_df['variable'] = f"{var}_{x}"
                        tmp_df = pd.concat([tmp_df, dimtmp_df[tmpcols]], ignore_index=True)
                else:
                    # If we only have a 3rd dimension (and it's "time"), then put that into a tmp_df
                    tmp_df = pd.DataFrame({'time': pddt, 'value': tmp_data})
                    tmp_df['variable'] = var

                # Add some extra metadata to the tmp_df (the constant is replicated for each row of the dataframe)
                tmp_df['model'] = model
                tmp_df['fire'] = fire
                tmp_df['rcp'] = rcp
                tmp_df['basin_id'] = int(bas)
                # Concatenate the tmp_df to the dataframe that we'll output (df)
                df = pd.concat([df, tmp_df[cols]], ignore_index=True)

        # Reset the index so it is sequential from 1 upwards
        df.reset_index(inplace=True)
        # Save the file to a csv text file
        df.to_csv(csv_basins_file, index=False)
    else:
        # Retrieve the csv file if it already exists on disk
        df = pd.read_csv(csv_basins_file)
        df['time'] = pd.to_datetime(df['time'])

    return df


def getVar_timeseries(fire=True, nofire=True, var='frac', stream='ilamb', region='southafrica'):
    '''
    Extracts time series of isimip data on large scale drainage basins
    :param fire:
    :param nofire:
    :param var:
    :param region:
    :return:
    '''

    pkl_file = f'/net/data/users/hadhy/Projects/ISIMIP/timeseries-{region}-{var}.pkl'

    start = dt.datetime(1860, 1, 1)
    end = dt.datetime(2100, 1, 1)
    bbox = load_data.get_region_bbox(region)
    rcps = ['rcp26', 'rcp60']
    models = ['HADGEM2-ES', 'GFDL-ESM2M', 'IPSL-CM5A-LR', 'MIROC5']

    if not os.path.isfile(pkl_file):
        out_dict = {}

        for mod, rcp in itertools.product(models, rcps):
            print(region, mod, rcp)

            if nofire:
                try:
                    out_dict[f"{mod}-nofire-{rcp}-{var}"] = load_data.jules_output(jobid='u-bk886', model=mod, var=var, stream=stream, rcp=rcp, start=start, end=end, bbox=bbox)[mod]
                except:
                    out_dict[f"{mod}-nofire-{rcp}-{var}"] = None

            if fire:
                try:
                    out_dict[f"{mod}-fire-{rcp}-{var}"] = load_data.jules_output(jobid='u-cf137', model=mod, var=var, rcp=rcp, start=start, end=end, bbox=bbox)[mod]
                except:
                    out_dict[f"{mod}-fire-{rcp}-{var}"] = None

                    # Save out_dict to a pickle file
        with open(pkl_file, 'wb') as f:
            pickle.dump(out_dict, f)

    else:
        # Retrieve the pickled data
        with open(pkl_file, 'rb') as f:
            out_dict = pickle.load(f)

    return out_dict


def getVar_on_gwls(fire=True, nofire=True, var='frac', stream='ilamb', region='southafrica'):
    '''
    Gets a dictionary containing GWLS , RCPs and models, for a given region and variable

    :return:
    '''

    pkl_file = f'/net/data/users/hadhy/Projects/ISIMIP/gwls-{region}-{var}.pkl'

    if not os.path.isfile(pkl_file):

        gwls = [1.0, 1.5, 2.0, 3.0, 4.0]
        rcps = ['rcp26', 'rcp60']
        models = ['HADGEM2-ES', 'GFDL-ESM2M', 'IPSL-CM5A-LR', 'MIROC5']
        # regions = ['southafrica', 'brazil']

        out_dict = {}

        for gwl, mod, rcp in itertools.product(gwls, models, rcps):
            print(region, gwl, mod, rcp)
            bbox = load_data.get_region_bbox(region)

            df = getGwls(model=mod, rcp=rcp, gwl=gwl, ens='r1i1p1')
            if df.empty:
                print('   No Data')
                continue
            start = dt.datetime(df.start_year.values[0], 1, 1)
            end = dt.datetime(df.end_year.values[0] + 1, 1, 1)

            if nofire:
                try:
                    out_dict[f"{mod}-nofire-{gwl}-{rcp}-{var}"] = load_data.jules_output(jobid='u-bk886', model=mod, var=var, stream=stream, rcp=rcp, start=start, end=end, bbox=bbox)[mod]
                except:
                    out_dict[f"{mod}-nofire-{gwl}-{rcp}-{var}"] = None

            if fire:
                try:
                    out_dict[f"{mod}-fire-{gwl}-{rcp}-{var}"] = load_data.jules_output(jobid='u-cf137', model=mod, var=var, stream=stream, rcp=rcp, start=start, end=end, bbox=bbox)[mod]
                except:
                    out_dict[f"{mod}-fire-{gwl}-{rcp}-{var}"] = None

                    # Save out_dict to a pickle file
        with open(pkl_file, 'wb') as f:
            pickle.dump(out_dict, f)

    else:
        # Retrieve the pickled data
        with open(pkl_file, 'rb') as f:
            out_dict = pickle.load(f)

    return out_dict


def get_river_basins(cube=None, region='southafrica', plot=False):

    # from std_functions import cube2gdalds
    import geopandas as gpd
    import regionmask
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    country_file = '/data/users/hadhy/GIS_Data/Global/NaturalEarth_10m/10m_cultural/ne_10m_admin_0_countries.shp'
    cntry = gpd.read_file(country_file)

    if region == 'southafrica':
        cntry_name = 'South Africa'
        shpfile = '/data/users/hadhy/GIS_Data/Global/RiverBasins/HydroBasins/hybas_af_lev03_v1c.shp'

    if region == 'brazil':
        cntry_name = 'Brazil'
        shpfile = '/data/users/hadhy/GIS_Data/Global/RiverBasins/HydroBasins/hybas_sa_lev03_v1c.shp'
        basins_subset = pd.DataFrame({'PFAF_ID': [642, 635, 633, 634, 622, 624],
                                      'basin_name': ['Parana', 'Atlantico Leste & Sudeste', 'Atlantico NE Oriental',
                                               'Sao Francisco', 'Amazonica', 'Tocantins & Araguaia']})

    basins = gpd.read_file(shpfile)
    cntry_poly = cntry.loc[cntry['ADMIN'] == cntry_name].geometry
    cntry_basins_i = basins.geometry.map(lambda x: x.intersects(cntry_poly.geometry.any()))
    cntry_basins = basins.loc[cntry_basins_i]
    if region == 'brazil':
        cntry_basins = pd.merge(cntry_basins, basins_subset, on='PFAF_ID')

    # Plot Basins
    if plot:
        fig, ax = plt.subplots(figsize=(9, 9))
        cntry_basins.plot(ax=ax, column='PFAF_ID', categorical=True)
        cntry_poly.plot(ax=ax, edgecolor='black', facecolor="none")
        if region == 'brazil':
            cntry_basins.apply(lambda x: ax.annotate(text=x['basin_name'], xy=x.geometry.centroid.coords[0], ha='center', color='black', path_effects=[pe.withStroke(linewidth=2, foreground="white")]), axis=1)
        else:
            cntry_basins.apply(
                lambda x: ax.annotate(text=x['PFIF_ID'], xy=x.geometry.centroid.coords[0], ha='center', color='black',
                                      path_effects=[pe.withStroke(linewidth=2, foreground="white")]), axis=1)
        plt.savefig("plots/basins_map_"+region+".png")

    # Creates a region from the geometry
    if cube:
        cntry_basins_regions = regionmask.Regions(cntry_basins.geometry, numbers=cntry_basins.PFAF_ID, names=cntry_basins.basin_name)
        lon = cube.coord('longitude').points
        lat = cube.coord('latitude').points
        # Rasterises the regions to the lon, lat grid
        mask = cntry_basins_regions.mask(lon, lat)
        mask_cube = mask.to_iris()
        # Broadcast the mask to the shape of the input cube
        mask_array = np.broadcast_to(mask_cube.data, cube.shape)
        mask_array = ma.masked_invalid(mask_array)

        return mask_array

    else:
        return


def biome_shifts():
    '''
    1. Fire vs no fire
    2. Land use transitions
    3. Biome changes by river catchment
    4. GPP, NPP and LAI changes by PFT and catchment
    :return:
    '''

    start = dt.datetime(1860, 1, 1)
    end = dt.datetime(2100, 1, 1)
    bbox = load_data.get_region_bbox('southafrica')

    models = ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
    frac_nofire_rcp26 = {}
    frac_fire_rcp26 = {}
    frac_nofire_rcp60 = {}
    frac_fire_rcp60 = {}

    for mod in models:
        print(mod)
        frac_nofire_rcp26[mod] = load_data.jules_output(jobid='u-bk886', model=mod, var='frac', rcp="rcp26", start=start, end=end, bbox=bbox)[mod]
        frac_fire_rcp26[mod] = load_data.jules_output(jobid='u-cf137', model=mod, var='frac', rcp="rcp26", start=start, end=end, bbox=bbox)[mod]
        frac_nofire_rcp60[mod] = load_data.jules_output(jobid='u-bk886', model=mod, var='frac', rcp="rcp60", start=start, end=end, bbox=bbox)[mod]
        frac_fire_rcp60[mod] = load_data.jules_output(jobid='u-cf137', model=mod, var='frac', rcp="rcp60", start=start, end=end, bbox=bbox)[mod]

    # Save the full time series
    frac = {}
    frac['nofire_rcp26'] = frac_nofire_rcp26
    frac['fire_rcp26'] = frac_fire_rcp26
    frac['nofire_rcp60'] = frac_nofire_rcp60
    frac['fire_rcp60'] = frac_fire_rcp60

    with open('frac_cubes.pkl', 'wb') as f:
        pickle.dump(frac, f)

    # Retrieve the pickled data
    with open('frac_cubes.pkl', 'rb') as f:
        frac_full_ts = pickle.load(f)


def figure3(region='southafrica'):
    '''
    change in PFT fractions per catchment
    :return:
    '''
    # frac
    # Checked these mappings compared to the post-processed output
    lut = {'frac_5': 'C3-Grass-Nat',
           'frac_6': 'C3-Grass-Crop',
           'frac_7': 'C3-Grass-Pasture',
           'frac_8': 'C4-Grass-Nat',
           'frac_9': 'C4-Grass-Crop',
           'frac_10': 'C4-Grass-Pasture',
           'frac_15': 'Bare',
           'frac_1': 'BLE-Tr'}
    pft_lut = pd.read_csv('pft_numbers_and_names.csv')
    df_ann = figure1_preprocessing(var='frac', region=region)
    top5 = df_ann.loc[(df_ann['fire'] == 'fire')].groupby('variable').mean().sort_values(by='value', ascending=False)[:7].index.to_list()
    df_ann = df_ann.loc[df_ann.variable.isin(top5)]  # South Africa : ['frac_1', 'frac_5', 'frac_6', 'frac_7', 'frac_8', 'frac_9', 'frac_10', 'frac_15']

    for k, v in pft_lut.iterrows():
        print(k, v['shortname'])
        df_ann['variable'] = df_ann['variable'].str.replace(fr'^(frac_{k})$', v['shortname'], regex=True)

    sns.set(rc={'figure.figsize': (15, 9)})
    sns.set_context("paper")
    with sns.axes_style("whitegrid"):
        g = sns.relplot(data=df_ann.loc[df_ann['model'] == 'HADGEM2-ES'], x='time', y='filtered_values', row='rcp', col='basin_name', hue='variable', style='fire', kind='line')
        g.set(xlim=(1860, 2100))
        g.set(xlabel='Year', ylabel='PFT Fraction')
        g.savefig("plots/frac_projections_"+region+".png")


def fig2_preprocessing(var='precip', region='southafrica'):

    var_gwl_dict = getVar_on_gwls(fire=True, nofire=True, var=var, region=region)
    abs_vals = {}
    anom_wrt_gwl1 = {}
    # Loop through keys to get absolute values
    for key, cube in var_gwl_dict.items():
        print(key)
        model, fire, gwl, rcp, var = key.rsplit('-', 4)
        cube_mean = cube.collapsed(['time'], iris.analysis.MEAN)
        abs_vals[key] = change_units(var, cube_mean)

    # Loop through abs_vals to get anomalies wrt_gwl1
    for key, cube in abs_vals.items():
        model, fire, gwl, rcp, var = key.rsplit('-', 4)
        if float(gwl) > 1.0:
            k_gwl1 = f"{model}-{fire}-1.0-{rcp}-{var}"
            anom_wrt_gwl1[key] = cube - abs_vals[k_gwl1]

    return abs_vals, anom_wrt_gwl1


def get_min_max(in_dict, extend='both'):

    values = []
    for k, cube in in_dict.items():
        values.extend(ma.getdata(cube.data).flatten())

    if extend == 'centre_zero':
        tmin = np.nanpercentile(values, 1)
        tmax = np.nanpercentile(values, 99)
        fmax = max(abs(tmin), abs(tmax))
        fmin = -fmax
    elif extend == 'neither':
        fmin = np.nanmin(values)
        fmax = np.nanmax(values)
    elif extend == 'min':
        fmin = np.nanpercentile(values, 1)
        fmax = np.nanmax(values)
    elif extend == 'max':
        fmin = np.nanmin(values)
        fmax = np.nanpercentile(values, 99)
    elif extend == 'both':
        fmin = np.nanpercentile(values, 1)
        fmax = np.nanpercentile(values, 99)
    else:
        fmin = np.nanmin(values)
        fmax = np.nanmax(values)

    return fmin, fmax


def figure2(var='precip', region='southafrica'):
    '''
    Plot postage stamps of driving data
    4 columns of GWL maps
    4 rows models
    :return:
    '''
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import iris.plot as iplt

    abs, anom = fig2_preprocessing(var=var, region='brazil')
    ofile = f'plots/gwl_maps_{var}_{region}.png'

    # Plot PFTs as COLUMNS and Obs+Models as ROWS
    col_vars = ['1.0', '1.5', '2.0', '3.0']
    row_vars = ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
    nrows = len(row_vars)
    ncols = len(col_vars)
    # Make an index for each position in the plot matrix
    ind = np.reshape(1 + np.arange(ncols * nrows), (nrows, ncols))
    x0, y0, x1, y1 = load_data.get_region_bbox(region)

    # Get mins and maxes for abs and anom data
    absmin, absmax = get_min_max(abs, extend='both')
    anomin, anomax = get_min_max(anom, extend='centre_zero')
    abs_cmap = plt.get_cmap('gray')
    anom_cmap = plt.get_cmap('RdBu')
    anom_leg_done = False

    # Make the figure
    # width, height
    fig = plt.figure(figsize=(15, 13), dpi=150)
    plt.gcf().subplots_adjust(hspace=0.1, wspace=0.01, top=0.93, bottom=0.1, left=0.025, right=0.93)

    for ir, rvar in enumerate(row_vars):
        for ic, cvar in enumerate(col_vars):
            print(ind[ir, ic], rvar, cvar)

            ax = fig.add_subplot(nrows, ncols, ind[ir, ic], projection=ccrs.PlateCarree())
            ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

            if ir == 0:
                ax.set_title(f"${cvar}^\circ$C", fontsize=10)

            if ic == 0:
                l, b, w, h = ax.get_position().bounds
                plt.figtext(l - (w / 15), b + (h / 2), rvar, horizontalalignment='left', verticalalignment='center',
                            rotation='vertical', fontsize=10)

            if ic == 0:
                print('Absolute: GWL 1.0')
                cube2plot = abs[f"{rvar}-fire-1.0-rcp26-{var}"].copy(data=abs[f"{rvar}-fire-1.0-rcp26-{var}"].data)
                cmabs = iplt.pcolormesh(cube2plot, cmap=abs_cmap, vmin=absmin, vmax=absmax)
                units = cube2plot.units
            else:
                print(f'Anomaly: GWL {cvar}')
                # gwl1 = abs[f"{rvar}-fire-1.0-rcp26-{var}"].copy(data=(abs[f"{rvar}-fire-1.0-rcp26-{var}"].data + abs[f"{rvar}-fire-1.0-rcp60-{var}"].data) / 2)
                k26 = f"{rvar}-fire-{cvar}-rcp26-{var}"
                k60 = f"{rvar}-fire-{cvar}-rcp60-{var}"
                try:
                    # Mess around with possibility of not all data being available
                    if (k60 in list(anom.keys())) and (k26 in list(anom.keys())):
                        cube2plot = anom[k60].copy(data=(anom[k26].data + anom[k60].data) / 2)
                    elif (k60 in list(anom.keys())) and not (k26 in list(anom.keys())):
                        cube2plot = anom[k60].copy()
                    elif (k26 in list(anom.keys())) and not (k60 in list(anom.keys())):
                        cube2plot = anom[k26].copy()
                    else:
                        continue
                except:
                    print(f'No data for {rvar}, gwl{cvar}')
                    continue
                cmanom = iplt.pcolormesh(cube2plot, cmap=anom_cmap, vmin=anomin, vmax=anomax)

            borderlines = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
            ax.add_feature(borderlines, edgecolor='black')
            ax.coastlines(resolution='50m', color='black')

            # Make legends
            if ir == (len(row_vars) - 1):
                if ic == 0:
                    l, b, w, h = ax.get_position().bounds
                    ## Values legend
                    # [left, bottom, width, height]
                    abs_ax = plt.gcf().add_axes(rect=[l + (w * 0.1), b - (2. * b / 5.), w * 0.8, 0.02])
                    abs_colorbar = plt.colorbar(cmabs, cax=abs_ax, orientation='horizontal', extend='neither')
                    abs_colorbar.ax.set_xticklabels(["{:d}".format(int(x)) for x in abs_colorbar.get_ticks()])
                    abs_colorbar.set_label(f"{var.title()} ({units})")
                if ic > 0 and not anom_leg_done:
                    l, b, w, h = ax.get_position().bounds
                    ## Anomaly legend
                    # acm_ax = plt.gcf().add_axes(rect=[l + (w * 0.1), 0, w * 0.8, 0.02])
                    acm_ax = plt.gcf().add_axes(rect=[l + (w * 0.1), b - (2. * b / 5.), w * 0.8, 0.02])
                    acm_colorbar = plt.colorbar(cmanom, acm_ax, orientation='horizontal', extend='both')
                    acm_colorbar.ax.set_xticklabels(["{:.2f}".format(float(x)) for x in acm_colorbar.get_ticks()])
                    acm_colorbar.set_label('Anomaly ' + var.title() + ' (model - obs)')
                    anom_leg_done = True

    fig.savefig(ofile, bbox_inches='tight')
    # plt.close()


def figure1_preprocessing(var='precip', region='southafrica'):

    from statsmodels.nonparametric.smoothers_lowess import lowess

    # csv_basins_file = f'/net/data/users/hadhy/Projects/ISIMIP/timeseries_on_basins-{region}-{var}.csv'
    csv_basins_file = f'/net/data/users/hadhy/Projects/ISIMIP/timeseries_on_basins-{region}-{var}.csv'
    if not os.path.isfile(csv_basins_file):
        create_dicts(var=var, region=region)

    df = pd.read_csv(csv_basins_file)
    df['time'] = pd.to_datetime(df['time'])
    if region == 'brazil':
        basins_subset = pd.DataFrame({'basin_id': [642, 635, 633, 634, 622, 624],
                                      'basin_name': ['Parana', 'Atlantico Leste & Sudeste', 'Atlantico NE Oriental',
                                               'Sao Francisco', 'Amazonica', 'Tocantins & Araguaia']})
        df = pd.merge(df, basins_subset, on='basin_id')
    else:
        df['basin_name'] = df['basin_id']

    # Calculate annual mean
    df_ann = df.groupby(by=[df.time.dt.year, df.model, df.fire, df.rcp, df.variable, df.basin_name]).mean()
    df_ann = df_ann.reset_index(level=['time', 'model', 'fire', 'rcp', 'variable', 'basin_name'])
    df_ann = df_ann.drop(['index'], axis=1)

    # Smooth values for plotting
    # By model, rcp, basin_id, fire
    dfout = []
    for model, rcp, basin_name, fire, v in itertools.product(pd.unique(df_ann.model),
                                                             pd.unique(df_ann.rcp),
                                                             pd.unique(df_ann.basin_name),
                                                             pd.unique(df_ann.fire),
                                                             pd.unique(df_ann.variable)):

        dfss = df_ann.loc[(df_ann.model == model) & (df_ann.rcp == rcp) & (df_ann.basin_name == basin_name) & (df_ann.fire == fire) & (df_ann.variable == v), ('value', 'time')]
        dfss_filt = pd.DataFrame(lowess(dfss.value, dfss.time, frac=20/240, missing='drop'), columns=['time', 'filtered_values']).sort_values(by='time', ascending=True)
        dfss_new = dfss.join(dfss_filt.set_index('time'), on='time')
        dfss_new['model'] = model
        dfss_new['rcp'] = rcp
        dfss_new['basin_name'] = basin_name
        dfss_new['fire'] = fire
        dfss_new['variable'] = v
        dfout.append(dfss_new)

    df_new = pd.concat(dfout)

    return df_new


def figure1(region='southafrica'):
    '''
    Plot facets of time series for each basin and rcp
    :return:
    '''

    # Precip
    df_ann = figure1_preprocessing(var='precip', region=region)
    sns.set(rc={'figure.figsize': (15, 9), 'figure.dpi': 150})
    sns.set_context("paper")
    with sns.axes_style("whitegrid"):
        g = sns.relplot(data=df_ann.loc[df_ann['fire'] == 'fire'], x='time', y='filtered_values', row='rcp', col='basin_name', hue='model', kind='line')
        # g.data = df_ann.loc[df_ann['fire'] == 'fire']
        # g.map(sns.lineplot, 'time', 'value', color='black', lw=2)
        g.set(xlim=(1860, 2100))
        g.set(xlabel='Year', ylabel='Precipitation (mm/day)')
        g.savefig("plots/precip_projections_"+region+".png")

    # 1.5m Air Temp
    df_ann = figure1_preprocessing(var='t1p5m_gb', region=region)
    df_ann['filtered_values'] = df_ann['filtered_values'] - 273.15
    sns.set(rc={'figure.figsize': (15, 9)})
    sns.set_context("paper")
    with sns.axes_style("whitegrid"):
        g = sns.relplot(data=df_ann.loc[df_ann['fire'] == 'fire'], x='time', y='filtered_values', row='rcp', col='basin_name', hue='model', kind='line')
        # g.data = df_ann.loc[df_ann['fire'] == 'fire']
        # g.map(sns.lineplot, 'time', 'value', color='black', lw=2)
        g.set(xlim=(1860, 2100))
        g.set(xlabel='Year', ylabel='1.5m Temperature (C)')
        g.savefig("plots/temp_projections_"+region+".png")

    # Runoff
    df_ann = figure1_preprocessing(var='runoff', region=region)
    df_ann['filtered_values'] = df_ann['filtered_values'] * 60 * 60 * 24
    sns.set(rc={'figure.figsize': (15, 9)})
    sns.set_context("paper")
    with sns.axes_style("whitegrid"):
        g = sns.relplot(data=df_ann, x='time', y='filtered_values', row='rcp', col='basin_name', hue='model', style='fire', kind='line')
        # g.data = df_ann.loc[df_ann.fire == 'fire']
        # g.map(sns.lineplot, 'time', 'filtered_values', color='black', lw=2, ci=None, label='Ens mean (fire)')
        # g.data = df_ann.loc[df_ann.fire == 'nofire']
        # g.map(sns.lineplot, 'time', 'filtered_values', data=df_ann.loc[df_ann.fire == 'nofire'], color='black', lw=2, ci=None, label='Ens mean (nofire)')
        g.set(xlim=(1860, 2100))
        g.set(xlabel='Year', ylabel='Runoff (mm/day)')
        g.savefig("plots/runoff_projections_"+region+".png")

    # Harvest_gb
    df_ann = figure1_preprocessing(var='harvest_gb', region=region)
    df_ann['filtered_values'] = df_ann['filtered_values']
    sns.set(rc={'figure.figsize': (15, 9)})
    sns.set_context("paper")
    with sns.axes_style("whitegrid"):
        g = sns.relplot(data=df_ann, x='time', y='filtered_values', row='rcp', col='basin_name', hue='model', style='fire', kind='line')
        g.set(xlim=(1860, 2100))
        g.set(xlabel='Year', ylabel='Harvest (kg C  m-2 per year)')
        g.savefig("plots/harvest_projections_"+region+".png")

    # npp_gb
    df_ann = figure1_preprocessing(var='npp_gb', region=region)
    df_ann['filtered_values'] = df_ann['filtered_values'] * 60 * 60 * 24
    sns.set(rc={'figure.figsize': (15, 9)})
    sns.set_context("paper")
    with sns.axes_style("whitegrid"):
        g = sns.relplot(data=df_ann, x='time', y='filtered_values', row='rcp', col='basin_name', hue='model', style='fire', kind='line')
        g.set(xlim=(1860, 2100))
        g.set(xlabel='Year', ylabel='Harvest (kg C m-2 per day)')
        g.savefig("plots/npp_gb_projections_"+region+".png")


def create_dicts(var=None, region=None):

    warnings.filterwarnings("ignore")

    # Pre-process results into easy-to-use dictionaries

    stream_lut = {'rflow': 'gen_mon_gb',
                  'frac': 'gen_ann_pftlayer',
                  'tstar_gb': 'gen_mon_gb',
                  'gpp': 'gen_mon_pft',
                  'gpp_gb': 'ilamb',
                  'npp': 'gen_mon_pft',
                  'npp_gb': 'ilamb',
                  'lai': 'ilamb',
                  'lai_gb': 'ilamb',
                  'harvest_gb': 'ilamb',
                  'burnt_area': 'c_ann_pftlayer',
                  'burnt_area_gb': 'ilamb',
                  'runoff': 'ilamb',
                  'precip': 'ilamb',
                  'q1p5m_gb': 'ilamb',
                  't1p5m_gb': 'ilamb',
                  'ftl_gb': 'ilamb',
                  'latent_heat': 'ilamb'}

    if var:
        lutss, = [{k: v} for k, v in stream_lut.items() if k == var]
    else:
        lutss = stream_lut

    if region:
        region_list = [region]
    else:
        region_list = ['brazil', 'southafrica']

    basin_df = {}
    for var, stream in lutss.items():
        for reg in region_list:
            print(var, stream, reg)
            print('   Extracting timeseries')
            var_ts_dict = getVar_timeseries(fire=True, nofire=True, var=var, stream=stream, region=reg)
            print('   Calculating basin aggregation')
            var_ts_basins_df = aggregateTimeseries_by_basin(in_dict=var_ts_dict, var=var, region=reg)
            # basin_df[f"{var}_{reg}"] = var_ts_basins_df
            # var_ts_basins_df = aggregateTimeseries_by_basin(in_dict=None, var=var, region=region)
            print('   Extracting data on GWLs')
            var_gwl_dict = getVar_on_gwls(fire=True, nofire=True, var=var, stream=stream, region=reg)

    # return basin_df


def main():
    # Run plots
    regions = ['brazil']  # , 'southafrica'

    for region in regions:
        print(region)

        # Plot basin map ...
        get_river_basins(cube=None, region=region, plot=True)

        # Timeseries plots by basin
        print('Figure 1')
        figure1(region=region)

        # Postage stamp map plots
        print('Figure 2')
        vars = ['rflow', 'tstar_gb', 'gpp_gb', 'npp_gb', 'lai_gb', 'harvest_gb', 'burnt_area_gb', 'runoff', 'precip', 'q1p5m_gb', 't1p5m_gb']  #, 'ftl_gb', 'latent_heat']
        for var in vars:
            figure2(var=var, region=region)

        # Veg frac by basin
        print('Figure 3')
        figure3(region=region)


if __name__ == '__main__':
    try:
        # This allows me pre-process the dictionaries and csv files on spice
        var = sys.argv[1]
        region = sys.argv[2]
        create_dicts(var=var, region=region)
    except:
        main()
