import os, sys
import iris
import cartopy.feature as cfeature
import cf_units
import numpy as np
import numpy.ma as ma
import pandas as pd
import datetime as dt
import pdb
import load_data
import dask.array as da
import warnings
import std_functions as sf

def figure7():
    print('Runoff')
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy.ma as ma
    import geopandas as gpd

    warnings.filterwarnings("ignore")

#    fig7_dict = figure7_preprocessing()
    fig7_dict = figure7_preprocessing_nic()
#    fig7_dict = figure7_preprocessing_nic2()
    df = fig7_dict['df']
    mouth_pts = fig7_dict['mouth_pts']
    obs_treatment = fig7_dict['obs_treatment']
    obs_string = '-and-'.join([x.replace('_', '-') for x in obs_treatment])

    # Plot locations of points in the Dai & Trenberth dataset
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, color='white', edgecolor='black')
    ax.set_xlim((-180, 180))
    ax.set_ylim((-90, 90))
    mouth_pts.plot(ax=ax, color='red')
    for x, y, label in zip(mouth_pts.geometry.x, mouth_pts.geometry.y, mouth_pts.basin):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", size=8, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    fig.tight_layout()
    fig.savefig('plots/river_flow_basins_map_'+obs_string+'.png', dpi=200)
    plt.clf()  # Closes figure

    # Plot monthly timeseries for a subset of basins
    # fig, ax = plt.subplots(1, 1, figsize=(10, 12.5))
    df_subset = df.loc[(df['basin'] == 'Amazon') | (df['basin'] == 'Congo') | (df['basin'] == 'Orinoco') | (
                df['basin'] == 'Changjiang') | (df['basin'] == 'Brahmaputra') | (df['basin'] == 'Mississippi'), :]
    df_subset = df_subset.loc[pd.notna(df_subset['value'])]
    sns.color_palette("bright")
    sns.set(rc={'figure.figsize': (10, 12.5)}, font_scale=1.5)

    model_list = [x for x in pd.unique(df_subset['model']) if x != 'Dai & Trenberth']
    for mod in model_list:
        with sns.axes_style("whitegrid"):
            g = sns.relplot(data=df_subset.loc[df_subset['model'].isin([mod, 'Dai & Trenberth'])] \
                            , x='month', y='value', hue='model', col='basin', style='fireflag' \
                            , col_wrap=2, kind='line', facet_kws={'sharey': False}, height=4, aspect=1.5)
            g.set(xlim=(1, 12), xticks=np.arange(1, 13, 1), xticklabels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            g.set_axis_labels("Month", r"River Flow $(m^3 s^{-1})$")
            g.set_titles("{col_name}")
            for ax in g.axes.flatten():
                ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')
            #     ax.set_yticklabels(ax.get_yticklabels())
            #     ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: "{:.1e}".format(x)))
            g.savefig('plots/river_flow_facets_'+mod+'_'+obs_string+'.png', dpi=200)
    plt.clf()  # Closes figure

    # Plot 6 rivers for all models, but with fire off
    with sns.axes_style("whitegrid"):
        g = sns.relplot(data=df_subset.loc[df_subset['fireflag'] == 'No Fire'] \
                        , x='month', y='value', hue='model', col='basin', style='fireflag' \
                        , col_wrap=2, kind='line', facet_kws={'sharey': False}, height=4, aspect=1.5)

        g.set(xlim=(1, 12), xticks=np.arange(1, 13, 1) \
              , xticklabels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        g.set_axis_labels("Month", r"River Flow $(m^3 s^{-1})$")
        g.set_titles("{col_name}")
        for ax in g.axes.flatten():
            ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')
        #     ax.set_yticklabels(ax.get_yticklabels())
        #     ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: "{:.1e}".format(x)))
        g.savefig('plots/river_flow_facets_'+obs_string+'.png', dpi=200)
    plt.clf()  # Closes figure

    # Plot all basins individually
    sns.set(rc={'figure.figsize': (10, 8)})
    for basin in pd.unique(df['basin']):
        print(basin)
        tmp = df.loc[(df['basin'] == basin)]
        print('A tmp',tmp)
        tmp = tmp['value']
        print('B tmp',tmp)
        ymax = 1.05*np.max(np.asarray(tmp))
        print('ymax',ymax)
        with sns.axes_style("whitegrid"):
            g = sns.lineplot(data=df.loc[df['basin'] == basin], x='month', y='value', hue='model', estimator=ma.mean)
#            g.set(xlim=(1, 12), xticks=np.arange(1, 13, 1) \
#              , xticklabels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            g.set(title=basin.title(), xlim=(1, 12), xticks=np.arange(1, 13, 1) \
                  , xticklabels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'] \
                  , ylabel=r"River Flow $(m^3 s^{-1})$")
#            g.set_axis_labels("Month", r"River Flow $(m^3 s^{-1})$")
            fig = g.get_figure()
            fig.savefig('plots/river_flow_'+basin.replace(' ', '_')+'_'+obs_string+'.png', dpi=200)
            plt.clf()  # Closes figure


def figure7_preprocessing():

    import datetime as dt
    from dateutil.relativedelta import relativedelta
    import numpy.ma as ma
    import geopandas as gpd

    # Flag to modify how we treat the observations
    # I *think* unit_and_cell_area is the correct method, but the basin area correction may look better
    obs_treatment = ['basin_area', 'units', 'cell_area']  # and/or 'units' and/or 'cell_area'

    obs = load_data.observations_runoff()
    mouth_pts = obs['mouth_points']
    runoff_df = obs['runoff_df']
    runoff_df = runoff_df.reset_index(drop=True)
    basins_cube = obs['basins_cube']

    start = runoff_df['Date'].min()
    start = dt.datetime(start.year, start.month, start.day)
    end = runoff_df['Date'].max()
    end = dt.datetime(end.year, end.month, end.day)
    # NB: Obs refer to a period +1 month after the date stamp, so the actual end is +1 month
    end = end + relativedelta(months=+1)

    out_dict = load_data.jules_output(jobid='u-bk886', var='rflow', stream='gen_mon_gb', start=start, end=end)

    # Unit conversion
    # JULES output is in Kg m-2 s-1
    # Need output to be in m-3 s-1
    # 1Kg water = 1cm-3
    # Dai&Trenberth obs are also in Kg m-2 s-1
    # ... but they might be averages over the whole catchment, so multiply basin area (sqkm) by obs value
    grid_areas = iris.analysis.cartography.area_weights(basins_cube)  # Square Metres
    basin_area_sqm = basins_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=grid_areas)
    basin_area_sqkm = basin_area_sqm / (1000*1000)  # sq metres to sq kms

    # Extract model data at mouth_pts, and put into dataframe
    df = pd.DataFrame(columns=['date', 'value', 'basin', 'model'])
    for i, row in mouth_pts.iterrows():
        sn = row['basin']
        print(i, sn, row['coords'])

        obs_treatment = ['units', 'cell_area']  # 'basin_area', 'units', 'cell_area'
        obsval = runoff_df.loc[runoff_df['basin'] == sn, 'value']

        if 'basin_area' in obs_treatment:
            print('Basin area correction')
            # If we think that the value in the data is an average over the catchment
            obsval = obsval * basin_area_sqkm[i].data  # This gets us closer to the correct value, but not sensible
            # obsval = obsval * basin_area_sqm[i].data  # More sensible, but nowhere near the correct value
            print(obsval.max())

        if 'units' in obs_treatment:
            print('Units to m3')
            # Convert units to m3
            obsval = obsval / 1000  # Kg to m3
            print(obsval.max())

        if 'cell_area' in obs_treatment:
            print('Grid cell area correction')
            # Get area of grid cell at the mouth point
            x, y = row['coords']
            xi = basins_cube.coord('longitude').nearest_neighbour_index(x)
            yi = basins_cube.coord('latitude').nearest_neighbour_index(y)
            pt_cell_area_sqm = grid_areas.data[0, yi, xi]
            # Convert units to refer to whole grid cell area
            obsval = obsval * pt_cell_area_sqm
            print(obsval.max())

        obs_df = pd.DataFrame({'date': runoff_df.loc[runoff_df['basin'] == sn, 'Date'],
                               'value': obsval})
        obs_df['basin'] = sn
        obs_df['model'] = 'Dai & Trenberth'
        df = pd.concat([df, obs_df], axis=0, ignore_index=True)

        for key, cube in out_dict.items():
            print(key)
            # This extracts the grid cell value at the point ...
            # Might be better to take the max of a 2x2 window around the point?
            x, y = row['coords']
            # xi = cube.coord('longitude').nearest_neighbour_index(x)
            # yi = cube.coord('latitude').nearest_neighbour_index(y)
            # pt_ts = cube.data[:, yi, xi]
            # Here's the alternative ...
            b = 0.5
            subset = cube.intersection(longitude=(x - b, x + b), latitude=(y - b, y + b))
            subset_grid_areas = iris.analysis.cartography.area_weights(subset)  # Weights msq
            # Multiply the grid areas by the value to get the value per grid cell (as opposed to m-2)
            val = ma.masked_invalid(subset.data) * subset_grid_areas
            pt_ts = ma.max(val, axis=(1, 2))
            # Change units from Kg Water to m3
            pt_ts = pt_ts / 1000
            myu = cube.coord('time').units
            my_dates = [dt.datetime(mydt.year, mydt.month, mydt.day) for mydt, mydt_end in myu.num2date(cube.coord('time').bounds)]
            river_df = pd.DataFrame({'date': my_dates, 'value': pt_ts})
            river_df['basin'] = row['basin']
            river_df['model'] = key
            df = pd.concat([df, river_df], axis=0, ignore_index=True)

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

    return {'df': df, 'mouth_pts': mouth_pts, 'obs_treatment': obs_treatment}

def figure7_preprocessing_nic():
    from netCDF4 import Dataset
    import numpy as np
    import load_data
    import std_functions as sf
    import iris
    import pandas as pd
    import geopandas as gpd
    import datetime as dt


    # Load Nic's river dataset
#    fn = 'coastal-stns-Vol-monthly.updated-Aug2014.nc'
    fn = '/project/jchmr/hadng/dai_2014_data/coastal-stns-Vol-monthly.updated-Aug2014.nc'
    fid = Dataset(fn, "r")
    lon = np.array(fid.variables["lon"])
    lat = np.array(fid.variables["lat"])
    areao = np.array(fid.variables["area_stn"])
    alt = np.array(fid.variables["elev"])
    river = np.array(fid.variables["riv_name"])
    stn_name = np.array(fid.variables["stn_name"])
    ct_name = np.array(fid.variables["ct_name"])
    ratio_m2s = np.array(fid.variables["ratio_m2s"])

    # Make the station, river and country names into readable character strings
    criv_name_list = [''.join([c.decode('ISO-8859-1') for c in x]).rstrip(' ') for x in river]
    criv_name_list = [x[:x.find('(')].rstrip(' ') if x.find('(') > 0 else x for x in criv_name_list]
    cstn_name_list = [''.join([c.decode('ISO-8859-1') for c in x]).rstrip(' ') for x in stn_name]
    ct_name_list = [''.join([c.decode('ISO-8859-1') for c in x]).rstrip(' ') for x in ct_name]

    # Extract data
    rflow = np.transpose(np.array(fid.variables["FLOW"]))
    ncdate = np.array([f"{str(x)[:4]}-{str(x)[-2:]}" for x in fid.variables["time"][:].data], dtype='datetime64[D]')
    # ncdate = np.array([f"{str(x)[:4]}-{str(x)[-2:]}" for x in fid.variables["time"][:].data], dtype='datetime64[s]')
    dtdate = [dt.datetime(int(str(x)[:4]), int(str(x)[-2:]), 1) for x in fid.variables["time"][:].data]
    fid.close()

    print('ncdate',ncdate)
    print('dtdate',dtdate)

    # Put the river station metadata into a pandas dataframe
    stn_points_df = pd.DataFrame({'coords': list(zip(lon, lat)), 'mouth_lon': lon, 'mouth_lat': lat, 'ratio_m2s': ratio_m2s, 'area': areao, 'alt': alt, 'basin': criv_name_list, 'stn_name': cstn_name_list, 'country': ct_name_list})
    # Sort by area, but keep the index values (so we can refer to the data)
    stn_points_df.sort_values(by="area", ascending=False, inplace=True, ignore_index=False)
    # Get the index values of the top 50 by area
    stn_points_df_top50 = stn_points_df.iloc[:50, :]
## ***temporary***   
#    stn_points_df_top50 = stn_points_df.iloc[:15, :]
    # mouth_points = gpd.GeoDataFrame(mouth_points, geometry=gpd.points_from_xy(mouth_lon, mouth_lat))
    mouth_pts = gpd.GeoDataFrame(stn_points_df_top50, geometry=gpd.points_from_xy(stn_points_df_top50['mouth_lon'], stn_points_df_top50['mouth_lat']))

    # Put the river data into a pandas dataframe
    rflow = np.where(rflow == -999., np.nan, rflow)
    rflow_df = pd.DataFrame(rflow, columns=ncdate, index=range(rflow.shape[0]))
    rflow_df_top50 = rflow_df.iloc[stn_points_df_top50.index, :]

    # River flow from JULES
    maxdt = ncdate.max().astype(dt.datetime)
    maxdt2 = dt.datetime(maxdt.year, maxdt.month, maxdt.day)

#    print('maxdt2',maxdt2)
#    sys.exit()

    jules_fire = load_data.jules_output(jobid='u-cf137', var='rflow', stream='gen_mon_gb', start=dt.datetime(1980, 1, 1), end=maxdt2)
    jules_nofire = load_data.jules_output(jobid='u-bk886', var='rflow', stream='gen_mon_gb', start=dt.datetime(1980, 1, 1), end=maxdt2)
## ***temporary***
#    jules_fire = load_data.jules_output(jobid='u-cf137', var='rflow', stream='gen_mon_gb', start=dt.datetime(2012, 1, 1), end=maxdt2)
#    jules_nofire = load_data.jules_output(jobid='u-bk886', var='rflow', stream='gen_mon_gb', start=dt.datetime(2012, 1, 1), end=maxdt2)

    # Upstream area
    river_ancfile = '/hpc/data/d00/hadea/isimip3a/jules_ancils/rivers.latlon_fixed.nc' # -89.75 -> 89.75 so yp=yi+1,ym=yi-1
#    river_ancfile = '/hpc/data/d00/hadea/isimip2b/jules_ancils/rivers.latlon_fixed.nc' # 89.75 -> -89.75 so yp=yi-1,ym=yi+1
#    river_ancfile = '/hpc/data/d01/hadcam/helix_jules/ancil_links/river_ancil_global05deg_MASTER.nc' # 89.75 -> -89.75 so yp=yi-1,ym=yi+1
    ja_aream = iris.load_cube(river_ancfile, "mystery1")
    ja_rivseqm = iris.load_cube(river_ancfile, "rivseq")

    # Calculate the area weighting (sqkm)
    area2d = iris.analysis.cartography.area_weights(jules_fire['HADGEM2-ES'][0, ...], normalize=False)  # / (1000*1000)
    # wgt2d = iris.analysis.cartography.area_weights(jules_fire['HADGEM2-ES'][0, ...], normalize=True)

    odf = pd.DataFrame(columns=['basin', 'model', 'fireflag', 'date', 'month', 'value'])  # 'cell_index',
    odf = odf.astype({'date': 'datetime64'})
    for k in jules_fire.keys():
        for id, row in stn_points_df_top50.iterrows():
#            if((row['basin'] == 'Amazon') | (row['basin'] == 'Ganges') | (row['basin'] == 'Niger') | (
#                    row['basin'] == 'Amur') | (row['basin'] == 'Nile') | (row['basin'] == 'Parana') | (
#                    row['basin'] == 'Congo')):
            if((row['basin'] == 'Amazon') | (row['basin'] == 'Congo') | (row['basin'] == 'Orinoco') | (
                row['basin'] == 'Changjiang') | (row['basin'] == 'Brahmaputra') | (row['basin'] == 'Mississippi')):
                print(k, id, row['basin'])

            # We have different grids, so need to get grid indices for each
            coords = row['coords']
            cubes_dict = {'ancil_aream': ja_aream, 'ancil_rivseqm': ja_rivseqm, 'jules_output_fire': jules_fire[k], 'jules_output_nofire': jules_nofire[k]}
            data_on_indices = sf.extract_grid_coords(cubes_dict, coords)
            anc_aream = data_on_indices['ancil_aream']
            anc_rivseqm = data_on_indices['ancil_rivseqm']
            jules_fire_data = data_on_indices['jules_output_fire']
            jules_nofire_data = data_on_indices['jules_output_nofire']
#            max_in_3x3 = anc_aream.loc[anc_aream['value'] == anc_aream['value'].max(), 'cell_index'].values[0]
# ***NEW***:
# *** choose grid box with value closest to obs upstream area for the following basins:
            dareamo = np.abs(areao[id]-anc_aream['value'])
            if((row['basin'] == 'Orinoco') | (row['basin'] == 'Mississippi') | (
                row['basin'] == 'Yenisey') | (row['basin'] == 'St Lawrence') | (
                row['basin'] == 'Amur') | (row['basin'] == 'Mackenzie') | (
                row['basin'] == 'Xijiang')):
                max_in_3x3 = anc_aream.loc[dareamo == dareamo.min(), 'cell_index'].values[0]
            else:
                max_in_3x3 = 'xc_yc'


#            if((row['basin'] == 'Amazon') | (row['basin'] == 'Ganges') | (row['basin'] == 'Niger') | (
#                row['basin'] == 'Amur') | (row['basin'] == 'Nile') | (row['basin'] == 'Parana')| (
#                row['basin'] == 'Congo')):
#            if((row['basin'] == 'Mackenzie') | (row['basin'] == 'Amur')):
            if((row['basin'] == 'Amur')):
                print(k, id, row['basin'])
                print('coords',coords)
                print('areao[id]',areao[id])
                print('anc_aream[value]',anc_aream['value'])
                print('anc_rivseqm[value]',anc_rivseqm['value'])
#                print('areao[id],anc_aream[value]',areao[id],anc_aream['value'])
#                print('areao[id]-anc_aream[value]',areao[id]-anc_aream['value'])
#                print('np.abs(areao[id]-anc_aream[value])',np.abs(areao[id]-anc_aream['value']))

#                print('***ANDYs')
#                print('anc_aream[value] == anc_aream[value].max()',anc_aream['value'] == anc_aream['value'].max())
#                print('anc_aream[value] == anc_aream[value].max(), cell_index',anc_aream['value'] == anc_aream['value'].max(), 'cell_index')
                print('***NICs')
                print('dareamo',dareamo)
                print('dareamo == dareamo.min()',dareamo == dareamo.min())
                print('anc_aream[value].dareamo == dareamo.min()',anc_aream['value'][dareamo == dareamo.min()])
                print('max_in_3x3',max_in_3x3)

            # Extract data for cell with max contributing area
            jules_fire_data_ind = jules_fire_data.loc[jules_fire_data['cell_index'] == max_in_3x3, :]
            jules_nofire_data_ind = jules_nofire_data.loc[jules_nofire_data['cell_index'] == max_in_3x3, :]

            jules_fire_data_ind_xc_yc = jules_fire_data.loc[jules_fire_data['cell_index'] == 'xc_yc', :]
            jules_fire_data_ind_xc_yp = jules_fire_data.loc[jules_fire_data['cell_index'] == 'xc_yp', :]
            jules_fire_data_ind_xp_yp = jules_fire_data.loc[jules_fire_data['cell_index'] == 'xp_yp', :]
            jules_fire_data_ind_xp_yc = jules_fire_data.loc[jules_fire_data['cell_index'] == 'xp_yc', :]
            jules_fire_data_ind_xp_ym = jules_fire_data.loc[jules_fire_data['cell_index'] == 'xp_ym', :]
            jules_fire_data_ind_xc_ym = jules_fire_data.loc[jules_fire_data['cell_index'] == 'xc_ym', :]
            jules_fire_data_ind_xm_ym = jules_fire_data.loc[jules_fire_data['cell_index'] == 'xm_ym', :]
            jules_fire_data_ind_xm_yc = jules_fire_data.loc[jules_fire_data['cell_index'] == 'xm_yc', :]
            jules_fire_data_ind_xm_yp = jules_fire_data.loc[jules_fire_data['cell_index'] == 'xm_yp', :]


            this_aream = anc_aream.loc[anc_aream['cell_index'] == max_in_3x3, 'value'].values[0]

            # Rescale river flow to allow for river ancil having a different upstream area to the observation
            if((row['basin'] == 'Amur')):
                print('rflow max jules_fire_data_ind.loc',np.max(jules_fire_data_ind.loc[:, 'value'][0:12]))
                print('rflow max jules_fire_data_ind_xc_yc.loc',np.max(jules_fire_data_ind_xc_yc.loc[:, 'value'][0:12]))
                print('rflow max jules_fire_data_ind_xc_yp.loc',np.max(jules_fire_data_ind_xc_yp.loc[:, 'value'][0:12]))
                print('rflow max jules_fire_data_ind_xp_yp.loc',np.max(jules_fire_data_ind_xp_yp.loc[:, 'value'][0:12]))
                print('rflow max jules_fire_data_ind_xp_yc.loc',np.max(jules_fire_data_ind_xp_yc.loc[:, 'value'][0:12]))
                print('rflow max jules_fire_data_ind_xp_ym.loc',np.max(jules_fire_data_ind_xp_ym.loc[:, 'value'][0:12]))
                print('rflow max jules_fire_data_ind_xc_ym.loc',np.max(jules_fire_data_ind_xc_ym.loc[:, 'value'][0:12]))
                print('rflow max jules_fire_data_ind_xm_ym.loc',np.max(jules_fire_data_ind_xm_ym.loc[:, 'value'][0:12]))
                print('rflow max jules_fire_data_ind_xm_yc.loc',np.max(jules_fire_data_ind_xm_yc.loc[:, 'value'][0:12]))
                print('rflow max jules_fire_data_ind_xm_yp.loc',np.max(jules_fire_data_ind_xm_yp.loc[:, 'value'][0:12]))

            rflow_fire = jules_fire_data_ind.copy()
            rflow_fire['value'] = areao[id] * (jules_fire_data_ind.loc[:, 'value'] / this_aream)
            rflow_nofire = jules_nofire_data_ind.copy()
            rflow_nofire['value'] = areao[id] * (jules_nofire_data_ind.loc[:, 'value'] / this_aream)
            if((row['basin'] == 'Amur')):
                print('A max rflow_fire',np.max(rflow_fire['value'][0:12]))

            # Change units to m3/sec
            indices = sf.get_grid_indices(cubes_dict, coords)
            ix, iy = indices['jules_output_fire'][max_in_3x3]
            area2d_cell = area2d[iy, ix]
            rflow_fire['rflow_m3persec'] = rflow_fire['value'] * area2d_cell / 1000.
            rflow_nofire['rflow_m3persec'] = rflow_nofire['value'] * area2d_cell / 1000.

            if((row['basin'] == 'Amur')):
                print('B max rflow_fire',np.max(rflow_fire['rflow_m3persec'][0:12]))

            # Get model obs
            obs = rflow_df_top50.iloc[rflow_df_top50.index == id].dropna(axis='columns').T

            # Get matching dates between model and obs
            model_dates = jules_fire_data_ind['time']
            obs_dates = obs.index
            common_dates = obs_dates.intersection(model_dates)
            ifire = rflow_fire['time'].isin(common_dates)
            inofire = rflow_nofire['time'].isin(common_dates)
            iobs = obs.index.isin(common_dates)

            # Put everything into a dataframe to plot
            temp_fire = pd.DataFrame({'basin': [row['basin']]*sum(ifire),
                                      'model': [k]*sum(ifire),
                                      'fireflag': ['Fire']*sum(ifire),
                                      'date': rflow_fire['time'][ifire].values,
                                      # 'cell_index': rflow_fire['cell_index'][ifire],
                                      'value': rflow_fire['rflow_m3persec'][ifire]})
            temp_nofire = pd.DataFrame({'basin': [row['basin']] * sum(inofire),
                                        'model': [k] * sum(inofire),
                                        'fireflag': ['No Fire'] * sum(inofire),
                                        'date': rflow_nofire['time'][inofire].values,
                                        # 'cell_index': rflow_nofire['cell_index'][inofire],
                                        'value': rflow_nofire['rflow_m3persec'][inofire]})
            temp_obs = pd.DataFrame({'basin': [row['basin']] * sum(iobs),
                                     'model': ['Dai & Trenberth'] * sum(iobs),
                                     'fireflag': ['Fire'] * sum(iobs),
                                     'date': obs.index[iobs].values,
                                     # 'cell_index': rflow_nofire['cell_index'][inofire],
                                     'value': obs[iobs][id]})
            odf = pd.concat([odf, temp_fire, temp_nofire, temp_obs], ignore_index=True)

    odf['month'] = pd.DatetimeIndex(odf['date']).month

    # return odf
    return {'df': odf, 'mouth_pts': mouth_pts, 'obs_treatment': ['Gedney-method']}

#-----------------------------------------------------------------

def figure7_preprocessing_nic2():
    from netCDF4 import Dataset
    import numpy as np
    import load_data
    import std_functions as sf
    import iris
    import pandas as pd
    import geopandas as gpd
    import datetime as dt


    # Load Nic's river dataset
#    fn = 'coastal-stns-Vol-monthly.updated-Aug2014.nc'
    fn = '/project/jchmr/hadng/dai_2014_data/coastal-stns-Vol-monthly.updated-Aug2014.nc'
    fid = Dataset(fn, "r")
    lon = np.array(fid.variables["lon"])
    lat = np.array(fid.variables["lat"])
    areao = np.array(fid.variables["area_stn"])
    alt = np.array(fid.variables["elev"])
    river = np.array(fid.variables["riv_name"])
    stn_name = np.array(fid.variables["stn_name"])
    ct_name = np.array(fid.variables["ct_name"])
    ratio_m2s = np.array(fid.variables["ratio_m2s"])

    # Make the station, river and country names into readable character strings
    criv_name_list = [''.join([c.decode('ISO-8859-1') for c in x]).rstrip(' ') for x in river]
    criv_name_list = [x[:x.find('(')].rstrip(' ') if x.find('(') > 0 else x for x in criv_name_list]
    cstn_name_list = [''.join([c.decode('ISO-8859-1') for c in x]).rstrip(' ') for x in stn_name]
    ct_name_list = [''.join([c.decode('ISO-8859-1') for c in x]).rstrip(' ') for x in ct_name]

    # Extract data
    rflow = np.transpose(np.array(fid.variables["FLOW"]))
    ncdate = np.array([f"{str(x)[:4]}-{str(x)[-2:]}" for x in fid.variables["time"][:].data], dtype='datetime64[D]')
    # ncdate = np.array([f"{str(x)[:4]}-{str(x)[-2:]}" for x in fid.variables["time"][:].data], dtype='datetime64[s]')
    dtdate = [dt.datetime(int(str(x)[:4]), int(str(x)[-2:]), 1) for x in fid.variables["time"][:].data]
    fid.close()

    # Put the river station metadata into a pandas dataframe
    stn_points_df = pd.DataFrame({'coords': list(zip(lon, lat)), 'mouth_lon': lon, 'mouth_lat': lat, 'ratio_m2s': ratio_m2s, 'area': areao, 'alt': alt, 'basin': criv_name_list, 'stn_name': cstn_name_list, 'country': ct_name_list})
    # Sort by area, but keep the index values (so we can refer to the data)
    stn_points_df.sort_values(by="area", ascending=False, inplace=True, ignore_index=False)
    # Get the index values of the top 50 by area
    stn_points_df_top50 = stn_points_df.iloc[:50, :]
## ***temporary***   
#    stn_points_df_top50 = stn_points_df.iloc[:15, :]
    # mouth_points = gpd.GeoDataFrame(mouth_points, geometry=gpd.points_from_xy(mouth_lon, mouth_lat))
    mouth_pts = gpd.GeoDataFrame(stn_points_df_top50, geometry=gpd.points_from_xy(stn_points_df_top50['mouth_lon'], stn_points_df_top50['mouth_lat']))

    # Put the river data into a pandas dataframe
    rflow = np.where(rflow == -999., np.nan, rflow)
    rflow_df = pd.DataFrame(rflow, columns=ncdate, index=range(rflow.shape[0]))
    rflow_df_top50 = rflow_df.iloc[stn_points_df_top50.index, :]

    # River flow from JULES
    maxdt = ncdate.max().astype(dt.datetime)
    maxdt2 = dt.datetime(maxdt.year, maxdt.month, maxdt.day)

    jules_fire = load_data.jules_output(jobid='u-cf137', var='rflow', stream='gen_mon_gb', start=dt.datetime(1980, 1, 1), end=maxdt2)
    jules_nofire = load_data.jules_output(jobid='u-bk886', var='rflow', stream='gen_mon_gb', start=dt.datetime(1980, 1, 1), end=maxdt2)
## ***temporary***
#    jules_fire = load_data.jules_output(jobid='u-cf137', var='rflow', stream='gen_mon_gb', start=dt.datetime(2012, 1, 1), end=maxdt2)
#    jules_nofire = load_data.jules_output(jobid='u-bk886', var='rflow', stream='gen_mon_gb', start=dt.datetime(2012, 1, 1), end=maxdt2)

    # Upstream area
    river_ancfile = '/hpc/data/d00/hadea/isimip3a/jules_ancils/rivers.latlon_fixed.nc' # -89.75 -> 89.75 so yp=yi+1,ym=yi-1
#    river_ancfile = '/hpc/data/d00/hadea/isimip2b/jules_ancils/rivers.latlon_fixed.nc' # 89.75 -> -89.75 so yp=yi-1,ym=yi+1
    ja_aream = iris.load_cube(river_ancfile, "mystery1")
    ja_rivseqm = iris.load_cube(river_ancfile, "rivseq")

    # Calculate the area weighting (sqkm)
    area2d = iris.analysis.cartography.area_weights(jules_fire['HADGEM2-ES'][0, ...], normalize=False)  # / (1000*1000)
    # wgt2d = iris.analysis.cartography.area_weights(jules_fire['HADGEM2-ES'][0, ...], normalize=True)

    odf = pd.DataFrame(columns=['basin', 'model', 'fireflag', 'date', 'month', 'value'])  # 'cell_index',
    odf = odf.astype({'date': 'datetime64'})
    for k in jules_fire.keys():
        for id, row in stn_points_df_top50.iterrows():
            if((row['basin'] == 'Amazon') | (row['basin'] == 'Congo') | (row['basin'] == 'Orinoco') | (
                row['basin'] == 'Changjiang') | (row['basin'] == 'Brahmaputra') | (row['basin'] == 'Mississippi')):
                print(k, id, row['basin'])

            # We have different grids, so need to get grid indices for each
            coords = row['coords']
            cubes_dict = {'ancil_aream': ja_aream, 'ancil_rivseqm': ja_rivseqm, 'jules_output_fire': jules_fire[k], 'jules_output_nofire': jules_nofire[k]}
            data_on_indices = sf.extract_grid_coords(cubes_dict, coords)
            anc_aream = data_on_indices['ancil_aream']
            anc_rivseqm = data_on_indices['ancil_rivseqm']
            jules_fire_data = data_on_indices['jules_output_fire']
            jules_nofire_data = data_on_indices['jules_output_nofire']
#            max_in_3x3 = anc_aream.loc[anc_aream['value'] == anc_aream['value'].max(), 'cell_index'].values[0]
# ***NEW***:
# *** choose grid box with value closest to obs upstream area for the following basins:
            dareamo = np.abs(areao[id]-anc_aream['value'])
            if((row['basin'] == 'Orinoco') | (row['basin'] == 'Mississippi') | (
                row['basin'] == 'Yenisey') | (row['basin'] == 'St Lawrence') | (
                row['basin'] == 'Amur') | (row['basin'] == 'Mackenzie') | (
                row['basin'] == 'Xijiang')):
                max_in_3x3 = anc_aream.loc[dareamo == dareamo.min(), 'cell_index'].values[0]
            else:
                max_in_3x3 = 'xc_yc'

            if((row['basin'] == 'Amur')):
                print(k, id, row['basin'])
                print('coords',coords)
                print('anc_aream[value]',anc_aream['value'])
                print('anc_rivseqm[value]',anc_rivseqm['value'])
                print('areao[id]',areao[id])
#                print('areao[id],anc_aream[value]',areao[id],anc_aream['value'])
#                print('areao[id]-anc_aream[value]',areao[id]-anc_aream['value'])
#                print('np.abs(areao[id]-anc_aream[value])',np.abs(areao[id]-anc_aream['value']))
                print('***NICs')
                print('dareamo',dareamo)
                print('dareamo == dareamo.min()',dareamo == dareamo.min())
                print('***ANDYs')
                print('anc_aream[value] == anc_aream[value].max()',anc_aream['value'] == anc_aream['value'].max())
                print('anc_aream[value] == anc_aream[value].max(), cell_index',anc_aream['value'] == anc_aream['value'].max(), 'cell_index')
#                val = sf.get_grid_indices(cubes_dict, coords)
                print('max_in_3x3',max_in_3x3)

            # Extract data for cell with max contributing area
            jules_fire_data_ind = jules_fire_data.loc[jules_fire_data['cell_index'] == max_in_3x3, :]
            jules_nofire_data_ind = jules_nofire_data.loc[jules_nofire_data['cell_index'] == max_in_3x3, :]
            this_aream = anc_aream.loc[anc_aream['cell_index'] == max_in_3x3, 'value'].values[0]

            # Rescale river flow to allow for river ancil having a different upstream area to the observation
            if((row['basin'] == 'Amur')):
                print('rflow',jules_fire_data_ind.loc[:, 'value'])

            rflow_fire = jules_fire_data_ind.copy()
#            rflow_fire['value'] = areao[id] * (jules_fire_data_ind.loc[:, 'value'] / this_aream)
            rflow_nofire = jules_nofire_data_ind.copy()
#            rflow_nofire['value'] = areao[id] * (jules_nofire_data_ind.loc[:, 'value'] / this_aream)

            # Change units to m3/sec
            indices = sf.get_grid_indices(cubes_dict, coords)
            ix, iy = indices['jules_output_fire'][max_in_3x3]
            area2d_cell = area2d[iy, ix]
            rflow_fire['rflow_m3persec'] = rflow_fire['value'] * area2d_cell / 1000.
            rflow_nofire['rflow_m3persec'] = rflow_nofire['value'] * area2d_cell / 1000.

            # Get model obs
            obs = rflow_df_top50.iloc[rflow_df_top50.index == id].dropna(axis='columns').T

            # Get matching dates between model and obs
            model_dates = jules_fire_data_ind['time']
            obs_dates = obs.index
            common_dates = obs_dates.intersection(model_dates)
            ifire = rflow_fire['time'].isin(common_dates)
            inofire = rflow_nofire['time'].isin(common_dates)
            iobs = obs.index.isin(common_dates)

            # Put everything into a dataframe to plot
            temp_fire = pd.DataFrame({'basin': [row['basin']]*sum(ifire),
                                      'model': [k]*sum(ifire),
                                      'fireflag': ['Fire']*sum(ifire),
                                      'date': rflow_fire['time'][ifire].values,
                                      # 'cell_index': rflow_fire['cell_index'][ifire],
                                      'value': rflow_fire['rflow_m3persec'][ifire]})
            temp_nofire = pd.DataFrame({'basin': [row['basin']] * sum(inofire),
                                        'model': [k] * sum(inofire),
                                        'fireflag': ['No Fire'] * sum(inofire),
                                        'date': rflow_nofire['time'][inofire].values,
                                        # 'cell_index': rflow_nofire['cell_index'][inofire],
                                        'value': rflow_nofire['rflow_m3persec'][inofire]})
            temp_obs = pd.DataFrame({'basin': [row['basin']] * sum(iobs),
                                     'model': ['Dai & Trenberth'] * sum(iobs),
                                     'fireflag': ['Fire'] * sum(iobs),
                                     'date': obs.index[iobs].values,
                                     # 'cell_index': rflow_nofire['cell_index'][inofire],
                                     'value': obs[iobs][id]})
            odf = pd.concat([odf, temp_fire, temp_nofire, temp_obs], ignore_index=True)

    odf['month'] = pd.DatetimeIndex(odf['date']).month

    # return odf
    return {'df': odf, 'mouth_pts': mouth_pts, 'obs_treatment': ['Gedney-method2']}

#-----------------------------------------------------------------
def main():

    # Annual and seasonal GPP, ET and albedo plots
    # fig1(region='global')
    # fig1(region='southafrica')
    # fig1(region='brazil')

    # Veg fraction evaluation using CCI
    # fig2()
    # fig2(region='southafrica')
    # fig2(region='brazil')

    # Plot for the CCI PFT paper
    # fig2_cci_comparison()

    # Latitudinal Variation in major PFTs
    # fig3()
    # For the CCI PFT paper
    # fig3_cwt_vs_pft_CAR()

    # fig4(region='brazil')
    # fig5
    figure7()


if __name__ == '__main__':
    main()
