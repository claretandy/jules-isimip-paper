import os, sys
import pickle

import matplotlib as mpl
mpl.use('agg')
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


def fig1_preprocessing(agg_period='year', fire=False, region='global'):
    '''
    Do the pre-processing so that we have a ncie dictionary that we can easily plot
    :param agg_period: can be 'year', 'djf', 'mam', 'jja', 'son' or 'all_seas'
    :param region: can be 'global', 'southafrica' or 'brazil'
    :return: two dictionaries - one for observations, second for model anomalies
    '''

    bbox = load_data.get_region_bbox(region=region)
    if fire:
        jobid = 'u-cf137'
    else:
        jobid = 'u-bk886'

    print('Loading observations ...')
    # Load observations
    # NB: Chose the longest time series available from ILAMB folder
    obs = {}
    obs['et'] = load_data.observations('et', src='GLEAM', bbox=bbox) # ['DOLCE', 'GLEAM', 'MODIS']
    obs['gpp'] = load_data.observations('gpp', src='GBAF', bbox=bbox) # ['FLUXNET', 'GBAF']
    obs['albedo'] = load_data.observations('albedo', src='GEWEX.SRB', bbox=bbox) # ['CERES', 'GEWEX.SRB', 'MODIS']

    print('Loading model data ...')
    # Load model data
    modeldata = {}
    modeldata['et'] = load_data.jules_output(jobid=jobid, var='evap', stream='ilamb', start=obs['et']['GLEAM']['start'], end=obs['et']['GLEAM']['end'], bbox=bbox)
    modeldata['gpp'] = load_data.jules_output(jobid=jobid, var='gpp', stream='ilamb', start=obs['gpp']['GBAF']['start'], end=obs['gpp']['GBAF']['end'], bbox=bbox)
    modeldata['albedo'] = load_data.jules_output(jobid=jobid, var='gpp', stream='ilamb', start=obs['albedo']['GEWEX.SRB']['start'], end=obs['albedo']['GEWEX.SRB']['end'], bbox=bbox)

    # modeldata['et'] = load_data.isimip_output('evap', start=obs['et']['GLEAM']['start'],
    #                                             end=obs['et']['GLEAM']['end'], bbox=bbox)
    # modeldata['gpp'] = load_data.isimip_output('gpp', start=obs['gpp']['GBAF']['start'], end=obs['gpp']['GBAF']['end'], bbox=bbox)
    # modeldata['albedo'] = load_data.isimip_output('albedo', start=obs['albedo']['GEWEX.SRB']['start'],
    #                                               end=obs['albedo']['GEWEX.SRB']['end'], bbox=bbox)

    # Aggregate stuff
    print('Aggregating observational data')
    obs_agged = {'et': load_data.aggregator_multiyear(obs['et']['GLEAM']['data'], 'et', agg_period=agg_period),
                'gpp': load_data.aggregator_multiyear(obs['gpp']['GBAF']['data'], 'gpp', agg_period=agg_period),
                'albedo': load_data.aggregator_multiyear(obs['albedo']['GEWEX.SRB']['data'], 'albedo', agg_period=agg_period)
                }

    print('Aggregating model data')
    models2plot = {'et': {}, 'gpp': {}, 'albedo': {}}
    for k in modeldata['et'].keys():
        print('   ' + k.upper())
        models2plot['et'][k] = load_data.aggregator_multiyear(modeldata['et'][k], 'et', agg_period=agg_period)
        models2plot['gpp'][k] = load_data.aggregator_multiyear(modeldata['gpp'][k], 'gpp', agg_period=agg_period)
        models2plot['albedo'][k] = load_data.aggregator_multiyear(modeldata['albedo'][k], 'albedo', agg_period=agg_period)

    # Regrid Observations to JULES grid so that we can calculate anomalies
    # Mask out grid cells not in the JULES simulations
    # NB: All JULES output is the same, so just using one cube
    print('Regridding observations')
    obs2plot = {}
    mod = list(models2plot['gpp'].keys())[0]
    for k in obs_agged.keys():
        print('   ' + k.upper())
        obs2plot[k] = obs_agged[k].regrid(models2plot[k][mod], iris.analysis.Linear())
        obs2plot[k].data = ma.masked_array(obs2plot[k].data.data, mask=models2plot[k][mod].data.mask)

    # Calculate the model anomalies first (needed so that we can get the legend mins and maxs
    print('Calculating anomalies')
    modanom2plot = {}
    for v in models2plot.keys():
        print('   ' + v.upper())
        modanom2plot[v] = {}
        for m in models2plot[v].keys():
            modanom2plot[v][m] = models2plot[v][m].copy(data=models2plot[v][m].data - obs2plot[v].data)

    return obs2plot, modanom2plot



def fig1_improved_preprocessing(var, agg_period='year', source='Observations', fire=False, region='global'):
    '''
    Do the pre-processing so that we have a ncie dictionary that we can easily plot
    :param var: variable name can be 'et', 'gpp' or 'albedo'
    :param agg_period: can be 'year', 'djf', 'mam', 'jja', 'son' or 'all_seas'
    :param source: can be 'Observations' or any one of 4 ISIMIP models
    :param region: can be 'global', 'southafrica' or 'brazil'
    :return: two dictionaries - one for observations, second for model anomalies
    '''

    bbox = load_data.get_region_bbox(region=region)

    if fire:
        jobid = 'u-cf137'
    else:
        jobid = 'u-bk886'

    var_src_lut = {'et': 'GLEAM',
                   'gpp': 'GBAF',
                   'albedo': 'GEWEX.SRB'}
    mod_var_lut = {'et': {'varname': 'fqw_gb', 'stream': 'ilamb'},
                   'gpp': {'varname': 'gpp_gb', 'stream': 'ilamb'},
                   'albedo': {'varname': 'albedo_land', 'stream': 'gen_mon_gb'}}

    model = 'HADGEM2-ES' if source == 'Observations' else source
    # Need this for the start and end time coords even if loading model data
    obs = load_data.observations(var, src=var_src_lut[var], bbox=bbox)
    # Need this for the model anomalies if loading model data
    obs_agged = load_data.aggregator_multiyear(obs[var_src_lut[var]]['data'], var, agg_period=agg_period)
    # Need this as a template for regridding obs to model grid
    modeldata = load_data.jules_output(jobid=jobid, var=mod_var_lut[var]['varname'], stream=mod_var_lut[var]['stream'], start=obs[var_src_lut[var]]['start'], end=obs[var_src_lut[var]]['end'], model=model, bbox=bbox)[model]
    # modeldata = load_data.isimip_output(mod_var_lut[var], start=obs[var_src_lut[var]]['start'], end=obs[var_src_lut[var]]['end'], model=model, bbox=bbox)[model]

    # Regrid to model grid
    obs2plot = obs_agged.regrid(modeldata, iris.analysis.Linear())
    # Mask the array
    obs2plot.data = ma.masked_array(obs2plot.data.data, mask=modeldata[0].data.mask)

    if source == 'Observations':
        # return 2D cube
        return obs2plot
    else:
        # Load the model data
        # modeldata = load_data.isimip_output(mod_var_lut[var], start=obs[var_src_lut[var]]['start'], end=obs[var_src_lut[var]]['end'], model=source, bbox=bbox)[source]
        # Aggregate it for the desired period
        mod_agged = load_data.aggregator_multiyear(modeldata, var, agg_period=agg_period)
        # Calculate obs - model anomalies
        mod_agged.data = ma.masked_array(mod_agged.data.data, mask=modeldata[0].data.mask)
        modanom2plot = mod_agged.copy(data=mod_agged.data - obs2plot.data)

        return modanom2plot


def get_contour_levels(data2plot, extend='neither', level_num=200):
    '''
    Returns contour levels for plotting
    :param data2plot: dictionary with a key for each variable, and within that a key for each model pointing to a cube
    :param extend: choose from 'centre_zero', 'neither', 'min', 'max', 'both'
    :param level_num: integer
    :return: dictionary containing contour levels, vmin and vmax for each variable
    '''

    minpc = 5
    maxpc = 95

    def _get_min_max():
        if extend == 'centre_zero':
            tmin = np.nanpercentile(values, minpc)
            tmax = np.nanpercentile(values, maxpc)
            fmax = max(abs(tmin), abs(tmax))
            fmin = -fmax
        elif extend == 'neither':
            fmin = np.nanmin(values)
            fmax = np.nanmax(values)
        elif extend == 'min':
            fmin = np.nanpercentile(values, minpc)
            fmax = np.nanmax(values)
        elif extend == 'max':
            fmin = np.nanmin(values)
            fmax = np.nanpercentile(values, maxpc)
        elif extend == 'both':
            fmin = np.nanpercentile(values, minpc)
            fmax = np.nanpercentile(values, maxpc)
        else:
            fmin = np.nanmin(values)
            fmax = np.nanmax(values)
        return fmin, fmax

    outdict = {}
    label_format = {'gpp': '{:.1f}', 'et': '{:.1f}', 'albedo': '{:.2f}'}

    for v, it in data2plot.items():
        try:
            labelf = label_format[v]
        except KeyError:
            labelf = '{:.1f}'
        values = ma.masked_array([])
        if isinstance(it, iris.cube.Cube):
            values = ma.append(values, it.data[~it.data.mask])
            fmin, fmax = _get_min_max()
            unit_str = it.units.title(1).replace('1 ', '')
        else:
            for m, cube in data2plot[v].items():
                values = ma.append(values, cube.data[~cube.data.mask])
                unit_str = cube.units.title(1).replace('1 ', '')
            fmin, fmax = _get_min_max()

        contour_levels = np.linspace(fmin, fmax, level_num)
        outdict[v] = {'cl': contour_levels, 'vmin': fmin, 'vmax': fmax, 'unit_string': unit_str, 'label_format': labelf}

    return outdict


def fig1(region='global'):
    '''
    Plot figure 1
    :param region: string. Either 'global', 'southafrica' or 'brazil'
    :return:
    '''

    # import matplotlib
    # matplotlib.use('agg')
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import iris.plot as iplt
    import itertools

    x0, y0, x1, y1 = load_data.get_region_bbox(region)
    agg_periods = {'year': 'Annual', 'djf': 'DJF', 'mam': 'MAM', 'jja': 'JJA', 'son': 'SON'}
    legend_type = ['varying', 'fixed']

    # Get the fixed legend values
    obs2plot, modanom2plot = fig1_preprocessing(agg_period='year', region=region)
    fixed_obs_levels = get_contour_levels(obs2plot, extend='max', level_num=200)
    fixed_anom_levels = get_contour_levels(modanom2plot, extend='centre_zero', level_num=7)

    # for agg in agg_periods.keys():
    for agg, leg in itertools.product(agg_periods.keys(), legend_type):

        print('Plotting: ' + agg_periods[agg] + ' with ' + leg + ' legend')

        ofile = 'plots/multi-year_'+agg_periods[agg].lower()+'-mean_'+leg+'-legend_'+region+'.png'

        obs2plot, modanom2plot = fig1_preprocessing(agg_period=agg)

        # Plot Variables as COLUMNS and Obs+Models as ROWS
        nrows = 5
        ncols = 3
        col_vars = ['gpp', 'et', 'albedo']
        row_vars = ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
        # Make an index for each position in the plot matrix
        ind = np.reshape(1 + np.arange(ncols * nrows), (nrows, ncols))

        # Set up the colour maps and legends
        obs_cmap = {'gpp': plt.get_cmap('Greens'), 'et': plt.get_cmap('Blues'), 'albedo': plt.get_cmap('magma')}
        anom_cmap = {'gpp': plt.get_cmap('RdYlGn'), 'et': plt.get_cmap('BrBG'), 'albedo': plt.get_cmap('PiYG')}

        ####

        if leg == 'varying':
            obs_levels = get_contour_levels(obs2plot, extend='max', level_num=200)
            anom_levels = get_contour_levels(modanom2plot, extend='centre_zero', level_num=7)

        if leg == 'fixed':
            obs_levels = fixed_obs_levels
            anom_levels = fixed_anom_levels

        # Make the figure
        fig = plt.figure(figsize=(15.5, 13), dpi=300) # width, height
        plt.gcf().subplots_adjust(hspace=0.1, wspace=0.01, top=0.93, bottom=0.1, left=0.025, right=0.93)

        ocm = {}
        acm = {}
        for iv, var in enumerate(col_vars):
            # 1. Plot all observations as absolute values
            ## (nrows, ncols, index) index starts at 1 in top left, increases to the right
            if region == 'global':
                ax = fig.add_subplot(nrows, ncols, ind[0, iv], projection=ccrs.Robinson())
            else:
                ax = fig.add_subplot(nrows, ncols, ind[0, iv], projection=ccrs.PlateCarree()) #  ccrs.RotatedPole(pole_longitude=x0+((x1-x0)/2), pole_latitude=y0+((y1-y0)/2))
                ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

            ocm[var] = iplt.pcolormesh(obs2plot[var], cmap=obs_cmap[var], norm=colors.BoundaryNorm(obs_levels[var]['cl'], ncolors=obs_cmap[var].N, clip=True))

            if region == 'global':
                ax.set_global()
                ax.coastlines()
                ax.gridlines(color="gray", alpha=0.2, draw_labels=False)
            else:
                # lakelines = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale='10m', edgecolor=cfeature.COLORS['water'], facecolor='none')
                # ax.add_feature(lakelines)
                borderlines = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
                ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
                ax.coastlines(resolution='50m', color='black')
                gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
                gl.top_labels = False
                gl.left_labels = False
                gl.bottom_labels = False
                if iv < len(col_vars)-1:
                    gl.right_labels = False

            ax.set_title(var.upper(), fontsize=14)
            if iv == 0:
                l, b, w, h = ax.get_position().bounds
                plt.figtext(l - (w / 15), b + (h / 2), 'Observations', horizontalalignment='left', verticalalignment='center',
                            rotation='vertical', fontsize=14)

            # 2. Plot model results as anomalies
            for im, mod in enumerate(row_vars):
                if region == 'global':
                    ax = fig.add_subplot(nrows, ncols, ind[im+1, iv], projection=ccrs.Robinson())
                else:
                    ax = fig.add_subplot(nrows, ncols, ind[im + 1, iv], projection=ccrs.PlateCarree())  # ccrs.RotatedPole(pole_longitude=x0+((x1-x0)/2), pole_latitude=y0+((y1-y0)/2))
                    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
                try:
                    acm[var] = iplt.pcolormesh(modanom2plot[var][mod], cmap=anom_cmap[var], vmin=anom_levels[var]['vmin'], vmax=anom_levels[var]['vmax'])  # , cmap=abs_cmap, norm=abs_norm) anom_levels
                except:
                    continue

                if region == 'global':
                    ax.set_global()
                    ax.coastlines()
                    ax.gridlines(color="gray", alpha=0.2, draw_labels=False)
                else:
                    # lakelines = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale='10m',
                    #                                          edgecolor=cfeature.COLORS['water'], facecolor='none')
                    # ax.add_feature(lakelines)
                    borderlines = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land',
                                                               scale='50m', facecolor='none')
                    ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
                    ax.coastlines(resolution='50m', color='black')
                    gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
                    gl.top_labels = False
                    gl.left_labels = False
                    if im < len(row_vars) - 1:
                        gl.bottom_labels = False
                    if iv < len(col_vars) - 1:
                        gl.right_labels = False

                if iv == 0:
                    l, b, w, h = ax.get_position().bounds
                    plt.figtext(l - (w / 15), b + (h / 2), mod, horizontalalignment='left', verticalalignment='center', rotation='vertical', fontsize=14)

                if im == 3:
                    l, b, w, h = ax.get_position().bounds
                    ## Values legend
                    # [left, bottom, width, height]
                    obs_ax = plt.gcf().add_axes([l, b - (2*b/5), w, 0.02])
                    obs_colorbar = plt.colorbar(ocm[var], obs_ax, orientation='horizontal', extend='max')
                    obs_colorbar.ax.set_xticklabels([obs_levels[var]['label_format'].format(x) for x in obs_colorbar.get_ticks()])
                    obs_colorbar.set_label('Observed ' + var.upper() + ' ('+obs_levels[var]['unit_string']+')')

                    ## Anomaly legend
                    acm_ax = plt.gcf().add_axes([l, 0, w, 0.02])
                    acm_colorbar = plt.colorbar(acm[var], acm_ax, orientation='horizontal', extend='both')
                    acm_colorbar.ax.set_xticklabels([anom_levels[var]['label_format'].format(x) for x in acm_colorbar.get_ticks()])
                    acm_colorbar.set_label('Anomaly ' + var.upper() + ' (model - obs)')

        plt.suptitle('ISIMIP: Multi-year '+agg_periods[agg]+' Mean', fontsize=20)

        fig.savefig(ofile, bbox_inches='tight')
        plt.close()



def paper_fig3(region='global'):
    '''
    Plot figure 3
    *** Used in the Paper ***
    :param region: string. Either 'global', 'southafrica' or 'brazil'
    :return:
    '''

    import matplotlib
    matplotlib.use('agg')
    import iris.quickplot as qplt
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import iris.plot as iplt
    import itertools

    x0, y0, x1, y1 = load_data.get_region_bbox(region)

    # Get the fixed legend values
    fixed_obs_levels_file = "fixed_obs_levels_fig3.pkl"
    fixed_anom_levels_file = "fixed_anom_levels_fig3.pkl"

    if not (os.path.isfile(fixed_obs_levels_file) and os.path.isfile(fixed_anom_levels_file)):
        obs2plot, modanom2plot = fig1_preprocessing(agg_period='year', region='global')

    if os.path.isfile(fixed_obs_levels_file):
        with open(fixed_obs_levels_file, "rb") as f:
            obs_levels = pickle.load(f)
    else:
        with open(fixed_obs_levels_file, "wb") as f:
            obs_levels = get_contour_levels(obs2plot, extend='max', level_num=200)
            pickle.dump(obs_levels, f)

    if os.path.isfile(fixed_anom_levels_file):
        with open(fixed_anom_levels_file, "rb") as f:
            anom_levels = pickle.load(f)
    else:
        with open(fixed_anom_levels_file, "wb") as f:
            anom_levels = get_contour_levels(modanom2plot, extend='centre_zero', level_num=8)
            pickle.dump(anom_levels, f)

    # Set up the colour maps and legends
    obs_cmap = {'gpp': plt.get_cmap('Greens'), 'et': plt.get_cmap('Blues'), 'albedo': plt.get_cmap('magma')}
    anom_cmap = {'gpp': plt.get_cmap('RdYlGn'), 'et': plt.get_cmap('BrBG'), 'albedo': plt.get_cmap('PiYG')}

    col_vars = ['gpp', 'et', 'albedo']
    row_vars = ['Observations', 'GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
    # Plot seasons as COLUMNS and Obs/Models as ROWS. Legend on RIGHT
    nrows = len(row_vars)
    ncols = len(col_vars)
    # Make an index for each position in the plot matrix
    ind = np.reshape(1 + np.arange(ncols * nrows), (nrows, ncols))

    ofile = 'plots/fig3_global_annual-mean.png'

    # Make the figure
    fig = plt.figure(figsize=(14.5, 13), dpi=300)  # width, height
    fig.subplots_adjust(hspace=0.1, wspace=0.01, top=0.93, bottom=0.03, left=0.025, right=0.99)

    for irow, row in enumerate(row_vars):
        for icol, col in enumerate(col_vars):

            print(irow, row, icol, col)

            # Add a subplot
            ax = fig.add_subplot(nrows, ncols, ind[irow, icol], projection=ccrs.Robinson())

            # Get the 2D cube to plot
            cube2plot = fig1_improved_preprocessing(col.lower(), agg_period='year', source=row, region='global')

            if row == 'Observations':
                cmap = obs_cmap[col]
                norm = colors.BoundaryNorm(obs_levels[col]['cl'], ncolors=cmap.N)  # , clip=True)
                # Plot cube
                ocm = iplt.contourf(cube2plot, axes=ax, cmap=cmap, norm=norm, extend='max')
                ax.set_title(col.upper() + ' ('+obs_levels[col]['unit_string']+')', fontsize=16)
            else:
                cmap = anom_cmap[col]
                norm = colors.BoundaryNorm(anom_levels[col]['cl'], extend='both', ncolors=cmap.N)  # , clip=True)
                vmin = anom_levels[col]['vmin']
                vmax = anom_levels[col]['vmax']
                # Plot cube
                acm = iplt.contourf(cube2plot, axes=ax, cmap=cmap, levels=anom_levels[col]['cl'], extend='both')  # norm=norm, vmin=vmin, vmax=vmax
                # colors.CenteredNorm() # works, but can't standardise across the 4 model plots
                # acm = iplt.pcolormesh(cube2plot, cmap=cmap, vmin=vmin, vmax=vmax)  # ,

            # Add some stuff to the plot
            ax.set_global()
            ax.coastlines()
            ax.gridlines(color="gray", alpha=0.2, draw_labels=False)

            l, b, w, h = ax.get_position().bounds

            if icol == 0:
                # NB: ax.text is the ONLY WAY to set the row labels with cartopy axes
                # ax.text(x0, 0.5 * (y0 + y1), f'{row}', va='center', ha='right', rotation='vertical', fontsize=16)
                plt.figtext(l - (w / 15), b + (h / 2), row, ha='left', va='center',
                            rotation='vertical', fontsize=16)

            if row == 'Observations':
                ## Values legend
                # [left, bottom, width, height]
                obs_ax = fig.add_axes([l + (w/4), b+0.005, w/2, 0.01])
                obs_colorbar = plt.colorbar(ocm, obs_ax, orientation='horizontal', extend='max')
                if col == 'albedo':
                    obs_colorbar.set_ticklabels(
                        ['{:.1f}'.format(x) for x in obs_colorbar.get_ticks()])
                else:
                    obs_colorbar.set_ticklabels([obs_levels[col]['label_format'].format(x) for x in obs_colorbar.get_ticks()])
                obs_colorbar.set_label(obs_levels[col]['unit_string'])

            if irow == len(row_vars)-1:
                ## Anomaly legend
                print("Anomaly legend")
                # [left, bottom, width, height]
                # acm_ax = plt.gcf().add_axes([l + w + 0.05, b - h, 0.02, h*2])
                acm_ax = fig.add_axes([l + (w/8), 0, 3*w/4, 0.01])
                acm_colorbar = plt.colorbar(acm, acm_ax, orientation='horizontal', extend='both')  # , ticks=anom_levels[var]['cl'])
                acm_colorbar.set_ticklabels([anom_levels[col]['label_format'].format(x) for x in acm_colorbar.get_ticks()])
                acm_colorbar.set_label(col.upper() + ' anomaly (model - obs)')

        plt.suptitle('ISIMIP: Multi-year Annual Mean', fontsize=20)
        # fig.tight_layout()
        fig.savefig(ofile, bbox_inches='tight')
    # fig.savefig(ofile)
    plt.close()


def paper_figS4(region='brazil', fire=False):
    '''
    Plot figure S4
    *** Used in the Supplementary info ***
    :param region: string. Either 'global', 'southafrica' or 'brazil'
    :return:
    '''

    import matplotlib
    matplotlib.use('agg')
    import iris.quickplot as qplt
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import iris.plot as iplt
    import itertools

    x0, y0, x1, y1 = load_data.get_region_bbox(region)
    legend_type = ['fixed']

    # Get the fixed legend values
    fixed_obs_levels_file = "fixed_obs_levels_figS4.pkl"
    fixed_anom_levels_file = "fixed_anom_levels_figS4.pkl"

    if not (os.path.isfile(fixed_obs_levels_file) and os.path.isfile(fixed_anom_levels_file)):
        obs2plot, modanom2plot = fig1_preprocessing(agg_period='all_seas', fire=fire, region=region)

    if os.path.isfile(fixed_obs_levels_file):
        with open(fixed_obs_levels_file, "rb") as f:
            fixed_obs_levels = pickle.load(f)
    else:
        with open(fixed_obs_levels_file, "wb") as f:
            fixed_obs_levels = get_contour_levels(obs2plot, extend='max', level_num=200)
            pickle.dump(fixed_obs_levels, f)

    if os.path.isfile(fixed_anom_levels_file):
        with open(fixed_anom_levels_file, "rb") as f:
            fixed_anom_levels = pickle.load(f)
    else:
        with open(fixed_anom_levels_file, "wb") as f:
            fixed_anom_levels = get_contour_levels(modanom2plot, extend='centre_zero', level_num=8)
            pickle.dump(fixed_anom_levels, f)

    # Set up the colour maps and legends
    obs_cmap = {'gpp': plt.get_cmap('Greens'), 'et': plt.get_cmap('Blues'), 'albedo': plt.get_cmap('magma')}
    anom_cmap = {'gpp': plt.get_cmap('RdYlGn'), 'et': plt.get_cmap('BrBG'), 'albedo': plt.get_cmap('PiYG')}

    # Loop through GPP, ET, and albedo
    for var, leg in itertools.product(['albedo', 'et', 'gpp'], legend_type):

        print('Plotting: ' + var + ' with ' + leg + ' legend')
        fire_string = 'fire' if fire else 'nofire'
        ofile = 'plots/seasonal-mean_'+var+'_'+leg+'-legend_'+region+'_'+fire_string+'.png'
        print(ofile)

        # if leg == 'varying':
        #     obs_levels = get_contour_levels(obs2plot, extend='max', level_num=200)
        #     anom_levels = get_contour_levels(modanom2plot, extend='centre_zero', level_num=10)

        if leg == 'fixed':
            obs_levels = fixed_obs_levels
            anom_levels = fixed_anom_levels

        # Make the figure
        fig = plt.figure(figsize=(15.5, 13), dpi=300)  # width, height
        fig.subplots_adjust(hspace=0.1, wspace=0.01, top=0.93, bottom=0.01, left=0.12, right=0.82)

        col_vars = ['DJF', 'MAM', 'JJA', 'SON']
        row_vars = ['Observations', 'GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
        # Plot seasons as COLUMNS and Obs/Models as ROWS. Legend on RIGHT
        nrows = len(row_vars)
        ncols = len(col_vars)
        # Make an index for each position in the plot matrix
        ind = np.reshape(1 + np.arange(ncols * nrows), (nrows, ncols))

        for irow, row in enumerate(row_vars):
            for icol, col in enumerate(col_vars):

                print(irow, row, icol, col)

                # Add a subplot
                if region == 'global':
                    ax = fig.add_subplot(nrows, ncols, ind[irow, icol], projection=ccrs.Robinson())
                else:
                    ax = fig.add_subplot(nrows, ncols, ind[irow, icol], projection=ccrs.PlateCarree())
                    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

                # Get the 2D cube to plot
                cube2plot = fig1_improved_preprocessing(var, agg_period=col, source=row, fire=fire, region='brazil')

                if row == 'Observations':
                    cmap = obs_cmap[var]
                    norm = colors.BoundaryNorm(obs_levels[var]['cl'], ncolors=obs_cmap[var].N)  # , clip=True)
                    # Plot cube
                    ocm = iplt.contourf(cube2plot, axes=ax, cmap=cmap, norm=norm, extend='max')
                    ax.set_title(col.upper(), fontsize=14)
                else:
                    cmap = anom_cmap[var]
                    norm = colors.BoundaryNorm(anom_levels[var]['cl'], extend='both', ncolors=cmap.N)  # , clip=True)
                    vmin = anom_levels[var]['vmin']
                    vmax = anom_levels[var]['vmax']
                    # Plot cube
                    acm = iplt.contourf(cube2plot, axes=ax, cmap=cmap, levels=anom_levels[var]['cl'], extend='both')  # norm=norm, vmin=vmin, vmax=vmax
                    # colors.CenteredNorm() # works, but can't standardise across the 4 model plots
                    # acm = iplt.pcolormesh(cube2plot, cmap=cmap, vmin=vmin, vmax=vmax)  # ,

                # Add some stuff to the plot
                if region == 'global':
                    ax.set_global()
                    ax.coastlines()
                    ax.gridlines(color="gray", alpha=0.2, draw_labels=False)
                else:
                    borderlines = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
                    ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
                    ax.coastlines(resolution='50m', color='black')
                    # ax.set_xticks(ax.get_xticks())
                    # ax.set_yticks(ax.get_yticks())
                    # plt.grid()
                    gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
                    gl.top_labels = False
                    gl.left_labels = False
                    # if icol > 0:
                    #     gl.left_labels = False
                    if irow < len(row_vars) - 1:
                        gl.bottom_labels = False
                    if icol < len(col_vars) - 1:
                        gl.right_labels = False

                l, b, w, h = ax.get_position().bounds

                if icol == 0:
                    # NB: ax.text is the ONLY WAY to set the row labels with cartopy axes
                    ax.text(x0, 0.5 * (y0 + y1), f'{row}', va='center', ha='right', rotation='vertical', fontsize=16)

                if (irow == 0) and (icol == len(col_vars)-1):
                    ## Values legend
                    # [left, bottom, width, height]
                    obs_ax = fig.add_axes([l + w + 0.05, b, 0.02, h])
                    obs_colorbar = plt.colorbar(ocm, obs_ax, orientation='vertical', extend='max')
                    obs_colorbar.ax.set_yticklabels([obs_levels[var]['label_format'].format(x) for x in obs_colorbar.get_ticks()])
                    obs_colorbar.set_label('Observed ' + var.upper() + ' ('+obs_levels[var]['unit_string']+')')

                if (irow == 1) and (icol == len(col_vars)-1):
                    ## Anomaly legend
                    print("Anomaly legend")
                    # [left, bottom, width, height]
                    # acm_ax = plt.gcf().add_axes([l + w + 0.05, b - h, 0.02, h*2])
                    acm_ax = fig.add_axes([l + w + 0.05, b, 0.02, h])
                    acm_colorbar = plt.colorbar(acm, acm_ax, orientation='vertical', extend='both')  # , ticks=anom_levels[var]['cl'])
                    acm_colorbar.ax.set_yticklabels([anom_levels[var]['label_format'].format(x) for x in acm_colorbar.get_ticks()])
                    acm_colorbar.set_label('Anomaly ' + var.upper() + ' (model - obs)')

            plt.suptitle('ISIMIP: Seasonal Mean ' + var.upper(), fontsize=20)
            # fig.tight_layout()
            fig.savefig(ofile, bbox_inches='tight')
        # fig.savefig(ofile)
        plt.close()


def fig2_preprocessing(region='global', cci_version='0.25deg_PFT', fire=True, regrid=True, calc_anom=False):
    '''
    Load and pre-process PFT fractions from CCI and model output
    :param region:
    :param cci_version:
    :param fire: Do we want the runs with fire on or off?
    :param regrid:
    :return:
    '''

    bbox = load_data.get_region_bbox(region=region)
    if fire:
        jobid = 'u-cf137'
    else:
        jobid = 'u-bk886'

    # Get CCI observations (for 2010)
    print('   Loading CCI observations '+cci_version)
    obs_cci = load_data.cci_pft_fracs(output_major_veg=True, bbox=bbox, ver=cci_version)
    if not cci_version in ['300m_CWT', '300m_PFT']:
        obs_cci *= 100.
    obs_cci.units = cf_units.Unit('%')

    # Get model data (for 2010)
    print('   Loading model data')
    # modeldata = load_data.isimip_output('pft*', start=dt.datetime(2010, 1, 1), end=dt.datetime(2011, 1, 1), bbox=bbox, aggregation='annual')
    modeldata = load_data.jules_output(jobid=jobid, var='frac', stream='gen_ann_pftlayer', start=dt.datetime(2010, 1, 1), end=dt.datetime(2011, 1, 1), bbox=bbox)

    for k in modeldata.keys():
        temp = load_data.add_frac_metadata(modeldata[k])
        temp *= 100
        modeldata[k] = temp.aggregated_by('major_veg_class', iris.analysis.SUM)

    if regrid:
        # Regrid the observations
        # First remove all the coord info from the obs ...
        obs_cci.coord('longitude').coord_system = iris.coord_systems.GeogCS(6371229)
        obs_cci.coord('latitude').coord_system = iris.coord_systems.GeogCS(6371229)
        obs_cci.coord_system = iris.coord_systems.GeogCS(6371229)
        obs_cci.coord('longitude').units = cf_units.Unit('degrees')
        obs_cci.coord('latitude').units = cf_units.Unit('degrees')
        mod = list(modeldata.keys())[0]  # Gets a model as a template for regridding
        modeldata[mod].coord('longitude').coord_system = iris.coord_systems.GeogCS(6371229)
        modeldata[mod].coord('latitude').coord_system = iris.coord_systems.GeogCS(6371229)
        modeldata[mod].coord_system = iris.coord_systems.GeogCS(6371229)
        # Now do the regridding
        obs_cci_rg = obs_cci.regrid(modeldata[mod], iris.analysis.Linear())
        obs_cci_rg.data = ma.masked_array(obs_cci_rg.data.data, mask=modeldata[mod].data.mask)
    else:
        obs_cci_rg = obs_cci

    if calc_anom & regrid:
        # Now calculate the anomaly
        print('   Calculating anomalies')
        modanom = {}
        for m in modeldata.keys():
            modanom[m] = modeldata[m].copy(data=modeldata[m].data - obs_cci_rg.data)

    # Finally, pull out each major veg type into a separate dictionary item
    obs2plot = {}
    mod2plot = {}
    modanom2plot = {}

    for i, veg in enumerate(['Tree', 'Shrub', 'Grass', 'Bare']):
        obs2plot[veg] = obs_cci_rg.extract(iris.Constraint(major_veg_class=i))
        if calc_anom & regrid:
            modanom2plot[veg] = {}
        mod2plot[veg] = {}
        for im, mod in enumerate(modeldata.keys()):
            mod2plot[veg][mod] = modeldata[mod].extract(iris.Constraint(major_veg_class=i))
            if calc_anom & regrid:
                try:
                    modanom2plot[veg][mod] = mod2plot[veg][mod].copy(data=(mod2plot[veg][mod].data - obs2plot[veg].data))
                except:
                    pdb.set_trace()

    return obs2plot, modanom2plot, mod2plot


def fig2(region='global', fire=False):
    '''
    Plant functional type distributions from TRIFFID compared to CCI observations from 2010
    :return:
    '''
    # import matplotlib
    # matplotlib.use('agg')
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import iris.plot as iplt

    # Other options: 'cwt', 'v1', 'v2_old', 'v2', '0.25deg_CWT',
    for cciv in ['0.25deg_PFT', 'cwt', 'v1', 'v2_old', 'v2', '0.25deg_CWT']:

        ofile = 'plots/pfts_vs_cci_'+region+'_'+['fire' if fire else 'nofire'][0]+'_cci-'+cciv+'.png'

        x0, y0, x1, y1 = load_data.get_region_bbox(region)

        obs2plot, modanom2plot, mod2plot = fig2_preprocessing(region=region, cci_version=cciv, fire=fire, calc_anom=True)
        print('Plotting ' + cciv)

        # Plot PFTs as COLUMNS and Obs+Models as ROWS
        nrows = 5
        ncols = 4
        col_vars = ['Tree', 'Shrub', 'Grass', 'Bare']
        row_vars = ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
        # Make an index for each position in the plot matrix
        ind = np.reshape(1 + np.arange(ncols * nrows), (nrows, ncols))

        # Set up the colour maps and legends
        obs_cmap = {'Tree': plt.get_cmap('Greens'), 'Shrub': plt.get_cmap('Oranges'), 'Grass': plt.get_cmap('Blues'), 'Bare': plt.get_cmap('pink_r')}
        anom_cmap = {'Tree': plt.get_cmap('PiYG'), 'Shrub': plt.get_cmap('PuOr_r'), 'Grass': plt.get_cmap('RdBu'), 'Bare': plt.get_cmap('BrBG_r')}
        # obs_levels = get_contour_levels(obs2plot, extend='max', level_num=200)
        anom_levels = get_contour_levels(modanom2plot, extend='centre_zero', level_num=7)

        # Make the figure
        # width, height
        fig = plt.figure(figsize=(17, 13), dpi=300)
        plt.gcf().subplots_adjust(hspace=0.1, wspace=0.01, top=0.93, bottom=0.1, left=0.025, right=0.93)

        ocm = {}
        acm = {}
        for iv, var in enumerate(col_vars):
            # 1. Plot all observations as absolute values
            ## (nrows, ncols, index) index starts at 1 in top left, increases to the right
            if region == 'global':
                ax = fig.add_subplot(nrows, ncols, ind[0, iv], projection=ccrs.Robinson())
            else:
                ax = fig.add_subplot(nrows, ncols, ind[0, iv], projection=ccrs.PlateCarree())
                ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

            ocm[var] = iplt.pcolormesh(obs2plot[var], cmap=obs_cmap[var], norm=colors.BoundaryNorm(np.arange(0, 120, 20), ncolors=obs_cmap[var].N, clip=True))

            if region == 'global':
                ax.set_global()
                ax.coastlines()
                gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=False)
            else:
                # lakelines = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale='10m',
                #                                          edgecolor=cfeature.COLORS['water'], facecolor='none')
                # ax.add_feature(lakelines)
                borderlines = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land',
                                                           scale='50m', facecolor='none')
                ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
                ax.coastlines(resolution='50m', color='black')
                gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
                gl.top_labels = False
                gl.left_labels = False
                gl.bottom_labels = False
                if iv < len(col_vars) - 1:
                    gl.right_labels = False

            ax.set_title(var.upper(), fontsize=14)
            if iv == 0:
                l, b, w, h = ax.get_position().bounds
                plt.figtext(l - (w / 15), b + (h / 2), 'CCI_LC', horizontalalignment='left',
                            verticalalignment='center',
                            rotation='vertical', fontsize=14)

            # 2. Plot model results as anomalies
            for im, mod in enumerate(row_vars):
                if region == 'global':
                    ax = fig.add_subplot(nrows, ncols, ind[im+1, iv], projection=ccrs.Robinson())
                else:
                    ax = fig.add_subplot(nrows, ncols, ind[im + 1, iv], projection=ccrs.PlateCarree())
                    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
                try:
                    acm[var] = iplt.pcolormesh(modanom2plot[var][mod], cmap=anom_cmap[var], vmin=-100, vmax=100)  # norm=colors.BoundaryNorm(np.arange(-60, 80, 20), ncolors=obs_cmap[var].N, clip=True))  #
                except:
                    continue

                if region == 'global':
                    ax.set_global()
                    ax.coastlines()
                    gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=False)
                else:
                    # lakelines = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale='10m', edgecolor=cfeature.COLORS['water'], facecolor='none')
                    # ax.add_feature(lakelines)
                    borderlines = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
                    ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
                    ax.coastlines(resolution='50m', color='black')
                    gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
                    gl.top_labels = False
                    gl.left_labels = False
                    if im < len(row_vars) - 1:
                        gl.bottom_labels = False
                    if iv < len(col_vars) - 1:
                        gl.right_labels = False

                if iv == 0:
                    l, b, w, h = ax.get_position().bounds
                    plt.figtext(l - (w / 15), b + (h / 2), mod, horizontalalignment='left', verticalalignment='center', rotation='vertical', fontsize=14)

                if im == 3:
                    l, b, w, h = ax.get_position().bounds
                    ## Values legend
                    # [left, bottom, width, height]
                    obs_ax = plt.gcf().add_axes(rect=[l + (w*0.1), b - (2. * b / 5.), w * 0.8, 0.02])
                    obs_colorbar = plt.colorbar(ocm[var], cax=obs_ax, orientation='horizontal', extend='neither')
                    obs_colorbar.ax.set_xticklabels(["{:d}".format(int(x)) for x in obs_colorbar.get_ticks()])
                    obs_colorbar.set_label('Observed ' + var.upper() + ' Cover %')
                    ## Anomaly legend
                    acm_ax = plt.gcf().add_axes(rect=[l + (w*0.1), 0, w * 0.8, 0.02])
                    acm_colorbar = plt.colorbar(acm[var], acm_ax, orientation='horizontal', extend='both')
                    acm_colorbar.ax.set_xticklabels(["{:d}".format(int(x)) for x in acm_colorbar.get_ticks()])
                    acm_colorbar.set_label('Anomaly ' + var.upper() + ' (model - obs)')

        plt.suptitle('ISIMIP: PFT Fractions ('+['Fire On' if fire else 'Fire Off'][0]+')', fontsize=20)

        fig.savefig(ofile, bbox_inches='tight')
        plt.close()


def fig2_cci_comparison(region='global'):
    '''
    Plant functional type distributions from TRIFFID compared to CCI observations from 2010
    For the PFT paper
    :return:
    '''
    # import matplotlib
    # matplotlib.use('agg')
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import iris.plot as iplt

    cci_pft = '0.25deg_PFT'
    cci_cwt = '0.25deg_CWT'

    ofile = 'plots/pft-comparison-maps_' + region +'_cci-cwt-vs-pft.png'

    x0, y0, x1, y1 = load_data.get_region_bbox(region)

    obs2plot, modanom2plot, mod2plot = fig2_preprocessing(region=region, cci_version=cci_pft, calc_anom=True)
    obs2plot_cwt, modanom2plot_cwt, mod2plot_cwt = fig2_preprocessing(region=region, cci_version=cci_cwt)

    print('Plotting ' + cci_pft)

    # Plot PFTs as COLUMNS and Obs+Models as ROWS
    nrows = 3
    ncols = 4
    col_vars = ['Tree', 'Shrub', 'Grass', 'Bare']
    row_vars = ['0.25deg_CWT', 'HADGEM2-ES']
    # Make an index for each position in the plot matrix
    ind = np.reshape(1 + np.arange(ncols * nrows), (nrows, ncols))

    # Set up the colour maps and legends
    obs_cmap = {'Tree': plt.get_cmap('Greens'), 'Shrub': plt.get_cmap('Oranges'), 'Grass': plt.get_cmap('Blues'), 'Bare': plt.get_cmap('pink_r')}
    anom_cmap = {'Tree': plt.get_cmap('PiYG'), 'Shrub': plt.get_cmap('PuOr_r'), 'Grass': plt.get_cmap('RdBu'), 'Bare': plt.get_cmap('BrBG_r')}
    # obs_levels = get_contour_levels(obs2plot, extend='max', level_num=200)
    anom_levels = get_contour_levels(modanom2plot, extend='centre_zero', level_num=7)

    # Make the figure
    # width, height
    fig = plt.figure(figsize=(17, 10), dpi=300)
    plt.gcf().subplots_adjust(hspace=0.1, wspace=0.01, top=0.93, bottom=0.1, left=0.025, right=0.93)

    ocm = {}
    acm = {}
    for iv, var in enumerate(col_vars):
        # 1. Plot all observations as absolute values
        ## (nrows, ncols, index) index starts at 1 in top left, increases to the right
        if region == 'global':
            ax = fig.add_subplot(nrows, ncols, ind[0, iv], projection=ccrs.Robinson())
        else:
            ax = fig.add_subplot(nrows, ncols, ind[0, iv], projection=ccrs.PlateCarree())
            ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

        ocm[var] = iplt.pcolormesh(obs2plot[var], cmap=obs_cmap[var], norm=colors.BoundaryNorm(np.arange(0, 120, 20), ncolors=obs_cmap[var].N, clip=True))

        if region == 'global':
            ax.set_global()
            ax.coastlines()
            gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=False)
        else:
            # lakelines = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale='10m',
            #                                          edgecolor=cfeature.COLORS['water'], facecolor='none')
            # ax.add_feature(lakelines)
            borderlines = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land',
                                                       scale='50m', facecolor='none')
            ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
            ax.coastlines(resolution='50m', color='black')
            gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
            gl.top_labels = False
            gl.left_labels = False
            gl.bottom_labels = False
            if iv < len(col_vars) - 1:
                gl.right_labels = False

        ax.set_title(var.upper(), fontsize=14)
        if iv == 0:
            l, b, w, h = ax.get_position().bounds
            plt.figtext(l - (w / 15), b + (h / 2), 'CCI_LC ' + cci_pft, horizontalalignment='left',
                        verticalalignment='center',
                        rotation='vertical', fontsize=14)

        # 2. Plot model results as anomalies
        for im, mod in enumerate(row_vars):
            if region == 'global':
                ax = fig.add_subplot(nrows, ncols, ind[im+1, iv], projection=ccrs.Robinson())
            else:
                ax = fig.add_subplot(nrows, ncols, ind[im + 1, iv], projection=ccrs.PlateCarree())
                ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
            # pdb.set_trace()
            if mod == '0.25deg_CWT':
                cube2plot = obs2plot_cwt[var] - obs2plot[var]
            else:
                cube2plot = modanom2plot[var][mod]

            try:
                acm[var] = iplt.pcolormesh(cube2plot, cmap=anom_cmap[var], vmin=-100, vmax=100)  # norm=colors.BoundaryNorm(np.arange(-60, 80, 20), ncolors=obs_cmap[var].N, clip=True))  #
            except:
                continue

            if region == 'global':
                ax.set_global()
                ax.coastlines()
                gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=False)
            else:
                # lakelines = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale='10m', edgecolor=cfeature.COLORS['water'], facecolor='none')
                # ax.add_feature(lakelines)
                borderlines = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
                ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
                ax.coastlines(resolution='50m', color='black')
                gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
                gl.top_labels = False
                gl.left_labels = False
                if im < len(row_vars) - 1:
                    gl.bottom_labels = False
                if iv < len(col_vars) - 1:
                    gl.right_labels = False

            if iv == 0:
                l, b, w, h = ax.get_position().bounds
                plt.figtext(l - (w / 15), b + (h / 2), mod, horizontalalignment='left', verticalalignment='center', rotation='vertical', fontsize=14)

            if im == 1:
                l, b, w, h = ax.get_position().bounds
                ## Values legend
                # [left, bottom, width, height]
                obs_ax = plt.gcf().add_axes(rect=[l + (w*0.1), b - (2. * b / 5.), w * 0.8, 0.02])
                obs_colorbar = plt.colorbar(ocm[var], cax=obs_ax, orientation='horizontal', extend='neither')
                obs_colorbar.ax.set_xticklabels(["{:d}".format(int(x)) for x in obs_colorbar.get_ticks()])
                obs_colorbar.set_label('Observed ' + var.upper() + ' Cover %')
                ## Anomaly legend
                acm_ax = plt.gcf().add_axes(rect=[l + (w*0.1), 0, w * 0.8, 0.02])
                acm_colorbar = plt.colorbar(acm[var], acm_ax, orientation='horizontal', extend='both')
                acm_colorbar.ax.set_xticklabels(["{:d}".format(int(x)) for x in acm_colorbar.get_ticks()])
                acm_colorbar.set_label('Anomaly ' + var.upper() + ' (model - obs)')

    plt.suptitle('ISIMIP: PFT Fractions', fontsize=20)

    fig.savefig(ofile, bbox_inches='tight')
    plt.close()



def fig3_preprocessing(cci_version='v2', regrid=True):

    import iris.analysis.cartography

    obs2plot, anom2plot, mod2plot = fig2_preprocessing(region='global', cci_version=cci_version, regrid=regrid)
    # Now, aggregate by latitude
    obs_bylat = {}
    anom_bylat = {}
    mod_bylat = {}
    df = pd.DataFrame(columns=['pft', 'model', 'latitude', 'value'])
    for pft in obs2plot.keys():
        print(pft)
        anom_bylat[pft] = {}
        mod_bylat[pft] = {}
        obs2plot[pft].data = da.ma.masked_where(da.isnan(obs2plot[pft].data), obs2plot[pft].core_data())
        grid_areas = iris.analysis.cartography.area_weights(obs2plot[pft])
        obs_bylat[pft] = obs2plot[pft].collapsed('longitude', iris.analysis.MEAN, weights=grid_areas, mdtol=1)
        dftemp = pd.DataFrame(dict(pft=pft, model='observation', latitude=obs_bylat[pft].coord('latitude').points, value=obs_bylat[pft].data))
        df = df.append(dftemp, ignore_index=True)
        for mod in mod2plot[pft].keys():
            mod2plot[pft][mod].data = ma.masked_where(np.isnan(mod2plot[pft][mod].data), mod2plot[pft][mod].data)
            grid_areas = iris.analysis.cartography.area_weights(mod2plot[pft][mod])
            mod_bylat[pft][mod] = mod2plot[pft][mod].collapsed('longitude', iris.analysis.MEAN, weights=grid_areas)
            dftemp = pd.DataFrame(dict(pft=pft, model=mod, latitude=mod_bylat[pft][mod].coord('latitude').points, value=mod_bylat[pft][mod].data))
            df = df.append(dftemp, ignore_index=True)

    # Put everything into a pandas dataframe for plotting

    return df


def fig3():
    '''
    Latitudinal variation in PFT fractions in TRIFFID compared to CCI
    :return:
    '''

    import seaborn as sns
    import matplotlib.pyplot as plt

    for cciv in ['0.25deg_CWT', '0.25deg_PFT']: # ['cwt', 'v1', 'v2_old', 'v2', '300m_PFT', '300m_CWT']:
        regrid = False if '300m' in cciv else True
        df = fig3_preprocessing(cci_version=cciv, regrid=regrid)
        df.loc[df['model'] == 'observation', 'model'] = cciv
        sns.set(rc={'figure.figsize': (12, 12)})
        sns.set_context("paper")
        with sns.axes_style("whitegrid"):

            # fig = plt.figure(figsize=(12, 12), dpi=300)
            g = sns.relplot(data=df, kind='line', sort=False, x='value', y='latitude', hue='model', col='pft', col_wrap=2)
            g.set(xlim=(0, 90))
            g.set(ylim=(-60, 85))
            (g.map(plt.axhline, y=0, color=".9", zorder=0),
             g.map(plt.axhline, y=23.5, color=".7", dashes=(2, 1), zorder=1),
             g.map(plt.axhline, y=-23.5, color=".7", dashes=(2, 1), zorder=2)
             .set_axis_labels("PFT % Coverage", "Latitude")
             .set_titles("{col_name}"))
            # plt.show()
            g.tight_layout()
            g.savefig("plots/pft_latitude_cross-section_cci-"+cciv+".png")


def fig3_cwt_vs_pft(df=None):
    '''
    Latitudinal variation in PFT fractions in TRIFFID compared to CCI
    :return:
    '''

    import seaborn as sns
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if not isinstance(df, pd.DataFrame):
        df = fig3_preprocessing(cci_version='0.25deg_CWT', regrid=True)
        df.loc[df['model'] == 'observation', 'model'] = 'CWT'
        df = df.loc[(df['model'] == 'CWT') | (df['model'] == 'HADGEM2-ES')]
        df.loc[df['model'] == 'HADGEM2-ES', 'model'] = 'JULES-TRIFFID'
        df_pft = fig3_preprocessing(cci_version='0.25deg_PFT', regrid=True)
        df_pft = df_pft.loc[df_pft['model'] == 'observation']
        df_pft.loc[df_pft['model'] == 'observation', 'model'] = 'PFT'
        df = df_pft.append(df, ignore_index=True)

    dfout = []
    for pft in pd.unique(df['pft']):
        for model in pd.unique(df['model']):
            print(pft, model)
            dfss = df.loc[(df['pft'] == pft) & (df['model'] == model), ('latitude', 'value')]
            dfss_filt = pd.DataFrame(lowess(dfss.value, dfss.latitude, frac=10/180, missing='drop'), columns=['latitude', 'filtered_values']).sort_values(by='latitude', ascending=False)
            dfss_new = dfss.join(dfss_filt.set_index('latitude'), on='latitude')
            dfss_new['pft'] = pft
            dfss_new['model'] = model
            dfout.append(dfss_new)
    df_new = pd.concat(dfout)

    # Blue = old (CWT), orange = new (PFT), green = model
    sns.set(rc={'figure.figsize': (5, 10)})
    sns.set_context("paper", font_scale=1.5)
    sns.color_palette("bright")
    with sns.axes_style("whitegrid"):

        g = sns.relplot(data=df_new, kind='line', sort=False, x='filtered_values', y='latitude', hue='model', col='pft', hue_order=['CWT', 'PFT', 'JULES-TRIFFID'], col_wrap=1, height=3, aspect=2)
        g.set(xlim=(0, 80))
        g.set(ylim=(-90, 90))
        g.set(yticks=np.arange(-90, 120, 30))
        g.set_axis_labels('PFT %', '')
        sns.move_legend(g, 'upper right', bbox_to_anchor=(.79, .99), title=None, frameon=True),
        (g.map(plt.axhline, y=0, color=".9", zorder=0),
         g.map(plt.axhline, y=23.5, color=".7", dashes=(2, 1), zorder=1),
         g.map(plt.axhline, y=-23.5, color=".7", dashes=(2, 1), zorder=2)
         .set_titles(''))
        # plt.show()
        g.tight_layout()
        g.savefig("plots/pft_latitude_cross-section_cci-cwt-vs-pft.png", dpi=200)


def fig3_cwt_vs_pft_CAR(df=None):
    '''
    Latitudinal variation in PFT fractions in TRIFFID compared to CCI
    :return:
    '''

    import seaborn as sns
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if not isinstance(df, pd.DataFrame):
        df = fig3_preprocessing(cci_version='0.25deg_CWT', regrid=True)
        df.loc[df['model'] == 'observation', 'model'] = 'CWT'
        df = df.loc[(df['model'] == 'CWT') | (df['model'] == 'HADGEM2-ES')]
        df.loc[df['model'] == 'HADGEM2-ES', 'model'] = 'JULES-TRIFFID'
        df_pft = fig3_preprocessing(cci_version='0.25deg_PFT', regrid=True)
        df_pft = df_pft.loc[df_pft['model'] == 'observation']
        df_pft.loc[df_pft['model'] == 'observation', 'model'] = 'PFT'
        df = df_pft.append(df, ignore_index=True)

    dfout = []
    for pft in pd.unique(df['pft']):
        for model in pd.unique(df['model']):
            print(pft, model)
            dfss = df.loc[(df['pft'] == pft) & (df['model'] == model), ('latitude', 'value')]
            dfss_filt = pd.DataFrame(lowess(dfss.value, dfss.latitude, frac=10/180, missing='drop'), columns=['latitude', 'filtered_values']).sort_values(by='latitude', ascending=False)
            dfss_new = dfss.join(dfss_filt.set_index('latitude'), on='latitude')
            dfss_new['pft'] = pft
            dfss_new['model'] = model
            dfout.append(dfss_new)
    df_new = pd.concat(dfout)

    # Blue = old (CWT), orange = new (PFT), green = model
    sns.set(rc={'figure.figsize': (14, 12)})
    sns.set_context("paper", font_scale=1.1)
    sns.color_palette("bright")
    with sns.axes_style("whitegrid"):

        g = sns.relplot(data=df_new, kind='line', sort=False, x='filtered_values', y='latitude', hue='model', col='pft', hue_order=['CWT', 'PFT', 'JULES-TRIFFID'], col_wrap=2, height=3, aspect=1)
        g.set(xlim=(0, 80))
        g.set(ylim=(-60, 85))
        g.set(yticks=np.arange(-60, 100, 20))
        g.set_axis_labels('PFT %', 'Latitude')
        sns.move_legend(g, 'upper right', bbox_to_anchor=(.79, .95), title=None, frameon=True),
        (g.map(plt.axhline, y=0, color=".9", zorder=0),
         g.map(plt.axhline, y=23.5, color=".7", dashes=(2, 1), zorder=1),
         g.map(plt.axhline, y=-23.5, color=".7", dashes=(2, 1), zorder=2)
         .set_titles("{col_name}"))
        g.tight_layout()
        g.savefig("plots/pft_latitude_cross-section_cci-cwt-vs-pft_2x2.png", dpi=200)


def fig4_preprocessing(region='brazil'):

    bbox = load_data.get_region_bbox(region=region)

    print('Loading observations ...')
    # Load observations
    # NB: Chose the longest time series available from ILAMB folder
    obs = {}
    obs['et'] = load_data.observations('et', src='all') # ['DOLCE', 'GLEAM', 'MODIS']
    obs['gpp'] = load_data.observations('gpp', src='GBAF') # ['FLUXNET', 'GBAF']
    obs['albedo'] = load_data.observations('albedo', src='all') # ['CERES', 'GEWEX.SRB', 'MODIS']

    print('Loading model data ...')
    start_min = min([obs[k][ds]['start'] for k in obs.keys() for ds in obs[k].keys()])
    end_max = max([obs[k][ds]['end'] for k in obs.keys() for ds in obs[k].keys()])

    # Load model data
    modeldata = {}
    modeldata['et'] = load_data.isimip_output('evap', start=start_min, end=end_max)
    modeldata['gpp'] = load_data.isimip_output('gpp', start=start_min, end=end_max)
    modeldata['albedo'] = load_data.isimip_output('albedo', start=start_min, end=end_max)

    # Get regions
    import geopandas as gpd
    from rasterstats import zonal_stats
    inshp = '/data/users/hadhy/Projects/ISIMIP/IPCC-WGI-reference-regions-v4.shp'
    ipcc = gpd.read_file(inshp)
    ipcc_land = ipcc[ipcc['Type'].isin(['Land', 'Land-Ocean'])]

    # ipcc_regions = sf.poly2cube(inshp, 'Acronym', modeldata['gpp']['HADGEM2-ES'][0])
    statlist = ['count', 'sum', 'median', 'mean']  # , 'median', 'mean', 'min', 'max', 'percentile_10', 'percentile_90']
    ogpdf = []
    for var in ['et', 'gpp', 'albedo']:
        for i, x in ipcc_land.iterrows():
            # Observations
            for ok in obs[var].keys():
                iris.save(obs[var][ok]['data'], 'tmp.nc')
                zstats = zonal_stats(x.geometry, 'tmp.nc', stats=statlist)[0]
                myseries = x.append(pd.Series(zstats).rename({'sum': 'hist_sum', 'count': 'hist_count'}))
            if not isinstance(ogpdf, pd.DataFrame):
                ogpdf = gpd.GeoDataFrame(myseries).transpose()
            else:
                ogpdf = pd.concat([ogpdf, pd.DataFrame(myseries).transpose()])

def ipcc_region_stats(cube):
    import geopandas as gpd
    import rioxarray
    import xarray
    from shapely.geometry import mapping
    from geocube.api.core import make_geocube
    import regionmask
    # See https://regionmask.readthedocs.io/en/stable/defined_scientific.html

    inshp = '/data/users/hadhy/Projects/ISIMIP/IPCC-WGI-reference-regions-v4.shp'

    # iris.save(cube, 'tmp.nc')
    # xarr_cube = xarray.open_dataarray('tmp.nc')
    xarr_cube = xarray.DataArray.from_iris(cube)
    xarr_cube.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    xarr_cube.rio.write_crs("epsg:4326", inplace=True)
    ipcc = gpd.read_file(inshp, crs="epsg:4326")
    ipcc_land = ipcc[ipcc['Type'].isin(['Land', 'Land-Ocean'])]

    ogpdf = []
    for i, x in ipcc_land.iterrows():
        clipped = xarr_cube.rio.clip(x.geometry.apply(mapping), x.crs, drop=False)


def geom_to_masked_cube(cube, geometry, x_coord, y_coord,
                        mask_excludes=False):
    """
    Convert a shapefile geometry into a mask for a cube's data.

    Args:

    * cube:
        The cube to mask.
    * geometry:
        A geometry from a shapefile to define a mask.
    * x_coord: (str or coord)
        A reference to a coord describing the cube's x-axis.
    * y_coord: (str or coord)
        A reference to a coord describing the cube's y-axis.

    Kwargs:

    * mask_excludes: (bool, default False)
        If False, the mask will exclude the area of the geometry from the
        cube's data. If True, the mask will include *only* the area of the
        geometry in the cube's data.

    .. note::
        This function does *not* preserve lazy cube data.

    """
    from shapely.geometry import Point
    import geopandas as gpd

    # Get horizontal coords for masking purposes.
    lats = cube.coord(y_coord).points
    lons = cube.coord(x_coord).points
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Reshape to 1D for easier iteration.
    lon2 = lon2d.reshape(-1)
    lat2 = lat2d.reshape(-1)

    mask = []
    # Iterate through all horizontal points in cube, and
    # check for containment within the specified geometry.
    for lat, lon in zip(lat2, lon2):
        this_point = gpd.GeoSeries([Point(lon, lat)])
        res = geometry.contains(this_point)
        mask.append(res.values[0])

    mask = np.array(mask).reshape(lon2d.shape)
    if mask_excludes:
        # Invert the mask if we want to include the geometry's area.
        mask = ~mask
    # Make sure the mask is the same shape as the cube.
    dim_map = (cube.coord_dims(y_coord)[0],
               cube.coord_dims(x_coord)[0])
    cube_mask = iris.util.broadcast_to_shape(mask, cube.shape, dim_map)

    # Apply the mask to the cube's data.
    data = cube.data
    masked_data = np.ma.masked_array(data, cube_mask)
    cube.data = masked_data
    return cube


def fig4(region='global'):
    # Seasonal cycle
    print(region)


def fig5():
    # Figure for Land cover CCI PFT paper
    # Rows are PFTs
    # Col 1 = HG2 anomaly vs CWT
    # Col 2 = HG2 anomaly vs PFT
    # Col 3 = Change in anomaly
    # Col 4 = Latitudinal plot HG2 vs CWT vs PFT

    print('CCI paper plot')
    from matplotlib.ticker import MaxNLocator
    import matplotlib.cm as mpl_cm
    from matplotlib import colors
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from matplotlib.transforms import Bbox
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    import numpy as np

    mydict = fig5_preprocessing()
    cwt_anom = mydict['cwt_anom']  # Dictionary containing Tree, Shrub, Grass, Bare
    pft_anom = mydict['pft_anom']  # Same
    anom_diff = mydict['anom_diff']  # Same
    lat_abs = mydict['lat_abs']  # Pandas dataframe for plotting
    data_list = [cwt_anom, pft_anom, anom_diff]
    veg_list = list(cwt_anom.keys())

    levels = MaxNLocator(nbins=16).tick_values(-40, 40)
    cmap = mpl_cm.get_cmap('PiYG')
    norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, {'map_projection': projection})

    nrows, ncols = (4, 3)
    fig = plt.figure(figsize=(12.8, 9.6))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,  # Could change 111 to 121 for line plots
                    nrows_ncols=(nrows, ncols),
                    direction='row',
                    axes_pad=0.1,
                    cbar_location='bottom',
                    cbar_mode='edge',
                    cbar_pad=0.3,
                    cbar_size='4%',
                    label_mode='')  # note the empty label_mode

    lons, lats = np.meshgrid(cwt_anom['Tree'].coord('longitude').points, cwt_anom['Tree'].coord('latitude').points)
    plotgrid = np.arange(nrows * ncols).reshape([nrows, ncols])
    titles = ['CWT: HadGEM2-ES anomaly', 'PFT: HadGEM2-ES anomaly', 'Difference']

    for i, ax in enumerate(axgr):
        print(i)
        # Get the plot position in the matrix
        ypos, xpos = [int(val) for val in np.where(plotgrid == i)]
        data2plot = data_list[xpos]
        veg = veg_list[ypos]

        ax.coastlines()
        if ypos == 0:
            ax.set_title(titles[xpos])
        if ypos == (nrows - 1):
            ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
        if xpos == 0:
            # Set y-label
            ax.set_ylabel(veg)
            ax.set_yticks(np.linspace(-90, 90, 7), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        if xpos <= 1:
            p_anom = ax.pcolormesh(lons, lats, data2plot[veg].data, transform=projection, norm=norm, cmap='RdBu')
        else:
            p_diff = ax.pcolormesh(lons, lats, data2plot[veg].data, transform=projection, norm=norm, cmap=cmap)

    # Get colorbar positions
    cbar0_pos = axgr.cbar_axes[0].get_position()
    cbar1_pos = axgr.cbar_axes[1].get_position()
    axgr.cbar_axes[1].set_visible(False)
    ww10pc = (cbar1_pos.xmax - cbar1_pos.xmin) / 10
    # fig.add_axes()
    axgr.cbar_axes[0].set_position(Bbox([[cbar0_pos.xmin + ww10pc, cbar0_pos.ymin], [cbar1_pos.xmax - ww10pc, cbar1_pos.ymax]]))

    axgr.cbar_axes[0].colorbar(p_anom)
    axgr.cbar_axes[2].colorbar(p_diff)

    plt.tight_layout()

    plt.savefig('plots/cci_plots.png', dpi=200)
    # qplt.pcolormesh(diff_dict['Tree'])
    # plt.figure(4); qplt.pcolormesh(outcube, cmap=cmap, norm=norm)


def fig5_preprocessing():

    import pandas as pd

    obs2plot_cwt, anom2plot_cwt, mod2plot_cwt = fig2_preprocessing(region='global', cci_version='0.25deg_CWT', regrid=True, calc_anom=True)
    obs2plot_pft, anom2plot_pft, mod2plot_pft = fig2_preprocessing(region='global', cci_version='0.25deg_PFT', regrid=True, calc_anom=True)

    # Get latitudinal means
    print('Get latitudinal means')
    df_all = fig3_preprocessing(cci_version='0.25deg_CWT', regrid=True)
    df_all.loc[df_all['model'] == 'observation', 'model'] = '0.25deg_CWT'
    df = df_all.loc[(df_all['model'] == '0.25deg_CWT') | (df_all['model'] == 'HADGEM2-ES')]
    df_pft = fig3_preprocessing(cci_version='0.25deg_PFT', regrid=True)
    df_pft = df_pft.loc[df_pft['model'] == 'observation']
    df_pft.loc[df_pft['model'] == 'observation', 'model'] = '0.25deg_PFT'
    df = df_pft.append(df, ignore_index=True)

    # Calculate anomaly differences
    print('Calculate anomaly differences')
    anom_diff = {}
    hg2_pft_anom = {}
    hg2_cwt_anom = {}
    for veg in anom2plot_pft.keys():
        print(veg)
        veg_pft = anom2plot_pft[veg]['HADGEM2-ES']
        veg_cwt = anom2plot_cwt[veg]['HADGEM2-ES']

        # Save anomalies
        hg2_pft_anom[veg] = veg_pft
        hg2_cwt_anom[veg] = veg_cwt

        outdata_improvement = np.where(
            ((veg_cwt.data < 0) & (veg_pft.data < 0) & (veg_pft.data > veg_cwt.data)) |
            ((veg_cwt.data > 0) & (veg_pft.data > 0) & (veg_pft.data < veg_cwt.data)),
            np.abs(veg_pft.data - veg_cwt.data), 0)

        outdata_worsening = np.where(
            ((veg_cwt.data < 0) & (veg_pft.data < 0) & (veg_pft.data < veg_cwt.data)) |
            ((veg_cwt.data > 0) & (veg_pft.data > 0) & (veg_pft.data > veg_cwt.data)),
            np.abs(veg_pft.data - veg_cwt.data) * -1, 0)

        outdata_same_sign = np.where(outdata_improvement > 0, outdata_improvement, outdata_worsening)

        outdata = np.where(
            ((outdata_improvement == 0) & (outdata_worsening == 0)) &
            (((veg_cwt.data < 0) & (veg_pft.data > 0)) |
            ((veg_cwt.data > 0) & (veg_pft.data < 0))),
            -1 * (np.abs(veg_pft.data) - np.abs(veg_cwt.data)), outdata_same_sign)

        outdata_masked = ma.masked_array(data=outdata, mask=veg_cwt.data.mask)

        # cube = veg_pft.copy(data=outdata)
        cube_masked = veg_pft.copy(data=outdata_masked)
        cube_masked.rename(veg + ' change vs JULES')
        anom_diff[veg] = cube_masked

    outdict = {
        'cwt_anom': hg2_cwt_anom,
        'pft_anom': hg2_pft_anom,
        'anom_diff': anom_diff,
        'lat_abs': df
    }

    return outdict


def fig6():
    # Urban zoom with the CCI 300m product
    print('Figure 6')
    from matplotlib.ticker import MaxNLocator
    import matplotlib.cm as mpl_cm
    from matplotlib import colors
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from matplotlib.transforms import Bbox
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    import numpy as np

    cci_dict = fig6_preprocessing()

    levels = MaxNLocator(nbins=20).tick_values(0, 100)
    cmap = mpl_cm.get_cmap('YlOrRd')
    newcolors = cmap(np.linspace(0, 1, 256))
    grey = np.array([220/256, 220/256, 220/256, 1])
    newcolors[:12, :] = grey
    newcmap = colors.ListedColormap(newcolors)
    norm = colors.BoundaryNorm(levels, ncolors=newcmap.N, clip=True)

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, {'map_projection': projection})

    nrows, ncols = (2, 2)
    fig = plt.figure(figsize=(12.8, 9.6))  # or 9.6, 7.2
    axgr = AxesGrid(fig, 111, axes_class=axes_class,  # Could change 111 to 121 for line plots
                    nrows_ncols=(nrows, ncols),
                    share_all=False,
                    # aspect=False,
                    direction='row',
                    axes_pad=0.1,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0.3,
                    cbar_size='4%',
                    label_mode='')  # note the empty label_mode

    plotgrid = np.arange(nrows * ncols).reshape([nrows, ncols])
    titles = ['CWT', 'PFT']  # , 'Difference' or histogram?

    for i, k in enumerate(cci_dict.keys()):

        print(i)
        data2plot = cci_dict[k]
        lons, lats = np.meshgrid(data2plot.coord('longitude').points, data2plot.coord('latitude').points)
        ax = axgr[i]
        # Get the plot position in the matrix
        ypos, xpos = [int(val) for val in np.where(plotgrid == i)]

        rowname = k.split('_')[1].title()

        ax.coastlines(resolution='10m')
        xmin, ymin, xmax, ymax = [data2plot.coord('longitude').points.min(),
                                  data2plot.coord('latitude').points.min(),
                                  data2plot.coord('longitude').points.max(),
                                  data2plot.coord('latitude').points.max()
                                  ]
        ax.set_extent([xmin, xmax, ymin, ymax], ccrs.PlateCarree())
        if ypos == 0:
            ax.set_title(titles[xpos])
        # if ypos == (nrows - 1):
            # ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
        if xpos == 0:
            # Set y-label
            ax.set_ylabel(rowname)
            # ax.set_yticks(np.linspace(-90, 90, 7), crs=projection)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        # if xpos <= 1:
        p = ax.pcolormesh(lons, lats, data2plot[10].data, transform=projection, norm=norm, cmap=newcmap)
        pdb.set_trace()
        # else:
        #     p_diff = ax.pcolormesh(lons, lats, data2plot[veg].data, transform=projection, norm=norm, cmap=cmap)

    axgr.cbar_axes[0].colorbar(p)

    # plt.tight_layout()

    plt.savefig('plots/cci_urban.png', dpi=200)


def fig6_alt():
    print('Figure 6 alternative')
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    import itertools
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.ticker import MaxNLocator
    import matplotlib.cm as mpl_cm
    from matplotlib import colors
    import seaborn as sns
    import itertools

    # See https://matplotlib.org/3.5.1/gallery/subplots_axes_and_figures/subfigures.html#sphx-glr-gallery-subplots-axes-and-figures-subfigures-py

    levels = MaxNLocator(nbins=20).tick_values(0, 100)
    cmap = mpl_cm.get_cmap('YlOrRd')
    newcolors = cmap(np.linspace(0, 1, 256))
    grey = np.array([220/256, 220/256, 220/256, 1])
    newcolors[:12, :] = grey
    newcmap = colors.ListedColormap(newcolors)
    norm = colors.BoundaryNorm(levels, ncolors=newcmap.N, clip=True)

    def annotate_axes(ax, text, fontsize=18):
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=fontsize, color="darkgrey")

    cci_dict = fig6_preprocessing()
    row_list = ['london', 'amsterdam']
    col_list = ['cwt300', 'pft300', 'Histogram']

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, {'map_projection': projection})

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=(5, 5, 5), height_ratios=(3, 3))

    for [gsy, gsx] in itertools.product([0, 1], [0, 1, 2]):
        print(gsx, gsy)

        city = row_list[gsy]
        col = col_list[gsx]

        if gsx < 2:
            data2plot = cci_dict[city][col]
            if col == 'cwt300':
                data2plot[10].data = np.where(data2plot[10].data == 100, 75, data2plot[10].data)
            lons, lats = np.meshgrid(data2plot.coord('longitude').points, data2plot.coord('latitude').points)
            ax = fig.add_subplot(gs[gsy, gsx], projection=projection)
            ax.coastlines(resolution='10m')
            xmin, ymin, xmax, ymax = [data2plot.coord('longitude').points.min(),
                                      data2plot.coord('latitude').points.min(),
                                      data2plot.coord('longitude').points.max(),
                                      data2plot.coord('latitude').points.max()
                                      ]
            ax.set_extent([xmin, xmax, ymin, ymax], ccrs.PlateCarree())
            if gsy == 0:
                ax.set_title(col_list[gsx])
            if gsx == 0:
                ax.set_ylabel(city.title())
            ax.pcolormesh(lons, lats, data2plot[10].data, transform=projection, norm=norm, cmap=newcmap)
        else:
            ax = fig.add_subplot(gs[gsy, gsx])
            cwt_urb = cci_dict[city]['cwt300'][10]
            pft_urb = cci_dict[city]['pft300'][10]
            # cwturb_msk = ma.masked_where((cwt_urb.data == 0) & (pft_urb.data == 0), cwt_urb.data)
            pfturb_msk = ma.masked_where((cwt_urb.data == 0) & (pft_urb.data == 0), pft_urb.data)
            sns.histplot(pfturb_msk[pfturb_msk.mask == False], bins=np.linspace(0, 100, 21))
            plt.axvline(75, color='red')

    plt.tight_layout()

    plt.savefig('plots/cci_urban2.png', dpi=200)


def fig6_alt2():
    print('Figure 6 alternative 2')
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    import itertools
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.ticker import MaxNLocator
    import matplotlib.cm as mpl_cm
    from matplotlib import colors
    import seaborn as sns
    import numpy.ma as ma
    import itertools

    # See https://matplotlib.org/3.5.1/gallery/subplots_axes_and_figures/subfigures.html#sphx-glr-gallery-subplots-axes-and-figures-subfigures-py

    levels = MaxNLocator(nbins=20).tick_values(0, 100)
    cmap = mpl_cm.get_cmap('YlOrRd')
    newcolors = cmap(np.linspace(0, 1, 256))
    grey = np.array([220 / 256, 220 / 256, 220 / 256, 1])
    newcolors[:12, :] = grey
    newcmap = colors.ListedColormap(newcolors)
    norm = colors.BoundaryNorm(levels, ncolors=newcmap.N, clip=True)

    def get_bb(data2plot):
        xmin, ymin, xmax, ymax = [data2plot.coord('longitude').points.min(),
                                  data2plot.coord('latitude').points.min(),
                                  data2plot.coord('longitude').points.max(),
                                  data2plot.coord('latitude').points.max()
                                  ]
        return [xmin, xmax, ymin, ymax]

    # subfigs[0, 0].set_facecolor('0.25')
    cci_dict = fig6_preprocessing()

    cwt_lon = cci_dict['london']['cwt300']
    cwt_lon[10].data = np.where(cwt_lon[10].data == 100, 75, cwt_lon[10].data)
    pft_lon = cci_dict['london']['pft300']
    cwt_ams = cci_dict['amsterdam']['cwt300']
    cwt_ams[10].data = np.where(cwt_ams[10].data == 100, 75, cwt_ams[10].data)
    pft_ams = cci_dict['amsterdam']['pft300']

    projection = ccrs.PlateCarree()

    fig = plt.figure(constrained_layout=True, figsize=(12, 8))  #
    subfigs = fig.subfigures(3, 2, wspace=0.05, width_ratios=[2, 1], height_ratios=[5, 5, 1])

    # London maps
    lons, lats = np.meshgrid(cwt_lon.coord('longitude').points, cwt_lon.coord('latitude').points)
    ## CWT
    axsTopLeft = subfigs[0, 0].add_subplot(121, projection=projection)
    axsTopLeft.set_extent(get_bb(cwt_lon), projection)
    axsTopLeft.coastlines(resolution='10m')
    # axsTopLeft.set_title('CWT 300m')
    axsTopLeft.pcolormesh(lons, lats, cwt_lon[10].data * 0.75, transform=projection, norm=norm, cmap=newcmap)
    axsTopLeft.set_ylabel('London')
    ## PFT
    axsTopRight = subfigs[0, 0].add_subplot(122, projection=projection)
    axsTopRight.set_extent(get_bb(cwt_lon), projection)
    axsTopRight.coastlines(resolution='10m')
    # axsTopRight.set_title('PFT 300m')
    axsTopRight.pcolormesh(lons, lats, pft_lon[10].data, transform=projection, norm=norm, cmap=newcmap)
    # Amsterdam maps
    lons, lats = np.meshgrid(cwt_ams.coord('longitude').points, cwt_ams.coord('latitude').points)
    ## CWT
    axsBtmLeft = subfigs[1, 0].add_subplot(121, projection=projection)
    axsBtmLeft.set_extent(get_bb(cwt_ams), projection)
    axsBtmLeft.coastlines(resolution='10m')
    axsBtmLeft.pcolormesh(lons, lats, cwt_ams[10].data * 0.75, transform=projection, norm=norm, cmap=newcmap)
    axsBtmLeft.set_ylabel('Amsterdam')
    ## PFT
    axsBtmRight = subfigs[1, 0].add_subplot(122, projection=projection)
    axsBtmRight.set_extent(get_bb(cwt_ams), projection)
    pc = axsBtmRight.pcolormesh(lons, lats, pft_ams[10].data, transform=projection, norm=norm, cmap=newcmap)
    axsBtmRight.coastlines(resolution='10m')

    # fig.subplots_adjust(bottom=0.1)
    # cbar_ax = fig.add_axes([0, 0, 0.5, 0.07])
    # cbar_ax.set_facecolor('coral')
    # fig.colorbar(pc, cax=cbar_ax, orientation='horizontal')

    cbarax = subfigs[2, 0].add_subplot(111)
    fig.colorbar(pc, shrink=0.6, pad=0.2, aspect=10, ticks=np.linspace(0, 100, 11), orientation='horizontal', cax=cbarax)  # ax=np.array([axsTopLeft, axsTopRight, axsBtmLeft, axsBtmRight]),

    pfturblon_msk = ma.masked_where((cwt_lon[10].data == 0) & (pft_lon[10].data == 0), pft_lon[10].data)
    pfturbams_msk = ma.masked_where((cwt_ams[10].data == 0) & (pft_ams[10].data == 0), pft_ams[10].data)

    # London Histogram
    axsLon = subfigs[0, 1].add_subplot(111)
    sns.histplot(pfturblon_msk[pfturblon_msk.mask == False], bins=np.linspace(0, 100, 21), ax=axsLon)
    plt.axvline(75, color='red')

    # Amsterdam histogram
    axsAms = subfigs[1, 1].add_subplot(111)
    sns.histplot(pfturbams_msk[pfturbams_msk.mask == False], bins=np.linspace(0, 100, 21), ax=axsAms)
    plt.axvline(75, color='red')

    plt.savefig('plots/cci_urban3.png', dpi=200)


def fig6_preprocessing():
    # Load the 300m products subset for a bounding box
    print('Figure 6 pre-processing')

    # Get CCI observations (for 2010)
    bbox = load_data.get_region_bbox(region='london')
    cwt300_london = load_data.cci_pft_fracs(output_major_veg=False, bbox=bbox, ver='300m_CWT')
    cwt300_london.units = cf_units.Unit('%')
    pft300_london = load_data.cci_pft_fracs(output_major_veg=False, bbox=bbox, ver='300m_PFT')
    pft300_london.units = cf_units.Unit('%')

    # Amsterdam
    bbox = load_data.get_region_bbox(region='amsterdam')
    cwt300_amsterdam = load_data.cci_pft_fracs(output_major_veg=False, bbox=bbox, ver='300m_CWT')
    cwt300_amsterdam.units = cf_units.Unit('%')
    pft300_amsterdam = load_data.cci_pft_fracs(output_major_veg=False, bbox=bbox, ver='300m_PFT')
    pft300_amsterdam.units = cf_units.Unit('%')


    return {'london': {'cwt300': cwt300_london, 'pft300': pft300_london},
     'amsterdam': {'cwt300': cwt300_amsterdam, 'pft300': pft300_amsterdam}}


def figure7():
    print('Runoff')
    # import matplotlib as mpl
    # mpl.use('AGG')
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy.ma as ma
    import geopandas as gpd
    warnings.filterwarnings("ignore")

    # Rivers that we want to plot in a facet grid
    # rivers = ['Orinoco', 'Mississippi', 'Yenisey', 'St Lawrence', 'Amur', 'Mackenzie', 'Xijiang']
    rivers = ['Amazon', 'Congo', 'Orinoco', 'Changjiang', 'Brahmaputra', 'Mississippi']

    fig7_dict = figure7_preprocessing_nic()
    # fig7_dict = figure7_preprocessing()
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
    plt.close()  # Closes figure

    # Plot monthly timeseries for a subset of basins
    # fig, ax = plt.subplots(1, 1, figsize=(10, 12.5))
    df_subset = df.loc[df['basin'].isin(rivers)]
    df_subset = df_subset.loc[pd.notna(df_subset['value'])]
    df_subset = df_subset.loc[(df_subset['fireflag'] == 'No Fire') | (df_subset['model'] == 'Dai & Trenberth')]
    sns.color_palette("bright")
    sns.set(rc={'figure.figsize': (10, 12.5)}, font_scale=1.5)

    # Plot the subset of models in a facet grid
    mypal = sns.color_palette()[:4]
    mypal.extend([(0, 0, 0)])
    hue_order = [ho for ho in pd.unique(df_subset['model']) if ho != 'Dai & Trenberth']
    hue_order.extend(['Dai & Trenberth'])
    with sns.axes_style("whitegrid"):
        g = sns.relplot(data=df_subset, x='month', y='value', hue='model', hue_order=hue_order, palette=mypal, col='basin', col_wrap=2, kind='line', facet_kws={'sharey': False}, height=4, aspect=1.5)  # style='fireflag',
        g._legend.set_title(None)
        g.set(xlim=(1, 12), xticks=np.arange(1, 13, 1), xticklabels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        g.set_axis_labels("Month", r"River Flow $(m^3 s^{-1})$")
        g.set_titles("{col_name}")
        for ax in g.axes.flatten():
            ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')
        g.savefig('plots/river_flow_facets_'+obs_string+'.png', dpi=200)
        plt.close()

    # Plot all basins individually
    sns.set(rc={'figure.figsize': (10, 8)})
    for basin in pd.unique(df['basin']):
        print(basin)
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots()
            g = sns.lineplot(data=df.loc[df['basin'] == basin], x='month', y='value', hue='model', estimator=ma.mean, hue_order=hue_order, palette=mypal, ax=ax)
            g.legend_.set_title(None)
            g.set(xlim=(1, 12), xticks=np.arange(1, 13, 1),
                  xticklabels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            g.set_xlabel("Month")
            g.set_ylabel(r"River Flow $(m^3 s^{-1})$")
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles=handles[1:], labels=labels[1:])
            g.set(title=basin.title())
            fig.savefig('plots/river_flow_'+basin.replace(' ', '_')+'_'+obs_string+'.png', dpi=200)
            plt.close()  # Closes figure


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
    fn = 'coastal-stns-Vol-monthly.updated-Aug2014.nc'
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
    stn_points_df_top50 = stn_points_df.iloc[:100, :]  # Changed top 50 to top 100. Hoping to capture Orinoco
    # mouth_points = gpd.GeoDataFrame(mouth_points, geometry=gpd.points_from_xy(mouth_lon, mouth_lat))
    mouth_pts = gpd.GeoDataFrame(stn_points_df_top50, geometry=gpd.points_from_xy(stn_points_df_top50['mouth_lon'], stn_points_df_top50['mouth_lat']))

    # Put the river data into a pandas dataframe
    rflow = np.where(rflow == -999., np.nan, rflow)
    rflow_df = pd.DataFrame(rflow, columns=ncdate, index=range(rflow.shape[0]))
    rflow_df_top50 = rflow_df.iloc[stn_points_df_top50.index, :]


    ofile = '/data/users/hadhy/Projects/ISIMIP/river_flow_df.csv'
    if os.path.isfile(ofile):
        print('Reading', ofile)
        odf = pd.read_csv(ofile)
    else:
        print('Creating', ofile)
        # River flow from JULES
        maxdt = ncdate.max().astype(dt.datetime)
        maxdt2 = dt.datetime(maxdt.year, maxdt.month, maxdt.day)
        # pdb.set_trace()
        jules_fire = load_data.jules_output(jobid='u-cf137', var='rflow', stream='gen_mon_gb', start=dt.datetime(1980, 1, 1), end=maxdt2)
        jules_nofire = load_data.jules_output(jobid='u-bk886', var='rflow', stream='gen_mon_gb', start=dt.datetime(1980, 1, 1), end=maxdt2)

        # Upstream area
        river_ancfile = '/hpc/data/d00/hadea/isimip3a/jules_ancils/rivers.latlon_fixed.nc'
        # river_ancfile = '/hpc/data/d00/hadea/isimip2b/jules_ancils/rivers.latlon_fixed.nc'
        ja_aream = iris.load_cube(river_ancfile, "mystery1")
        ja_rivseqm = iris.load_cube(river_ancfile, "rivseq")

        # Calculate the area weighting (sqkm)
        area2d = iris.analysis.cartography.area_weights(jules_fire['HADGEM2-ES'][0, ...], normalize=False)  # / (1000*1000)

        odf = pd.DataFrame(columns=['basin', 'model', 'fireflag', 'date', 'month', 'value'])  # 'cell_index',
        odf = odf.astype({'date': 'datetime64'})
        for k in jules_fire.keys():
            for id, row in stn_points_df_top50.iterrows():
                print(k, id, row['basin'])

                # We have different grids, so need to get grid indices for each
                coords = row['coords']
                cubes_dict = {'ancil_aream': ja_aream, 'ancil_rivseqm': ja_rivseqm, 'jules_output_fire': jules_fire[k], 'jules_output_nofire': jules_nofire[k]}
                data_on_indices = sf.extract_grid_coords(cubes_dict, coords)
                anc_aream = data_on_indices['ancil_aream']
                anc_rivseqm = data_on_indices['ancil_rivseqm']
                jules_fire_data = data_on_indices['jules_output_fire']
                jules_nofire_data = data_on_indices['jules_output_nofire']
                # This is my method for getting the cell coordinate
                # Has since been edited and removed by Nic
                # max_in_3x3 = anc_aream.loc[anc_aream['value'] == anc_aream['value'].max(), 'cell_index'].values[0]
                # Nic's method:
                # *** choose grid box with value closest to obs upstream area for the following basins:
                # Absolute difference between the observed catchment area and the ancillary catchment area
                dareamo = np.abs(areao[id] - anc_aream['value'])
                # Where the absolute difference is lowest, return the cell index name (e.g. xc_yc)
                max_in_3x3 = anc_aream.loc[dareamo == dareamo.min(), 'cell_index'].values[0]

                # Extract data for cell with max contributing area
                jules_fire_data_ind = jules_fire_data.loc[jules_fire_data['cell_index'] == max_in_3x3, :]
                jules_nofire_data_ind = jules_nofire_data.loc[jules_nofire_data['cell_index'] == max_in_3x3, :]
                this_aream = anc_aream.loc[anc_aream['cell_index'] == max_in_3x3, 'value'].values[0]

                # Rescale river flow to allow for river ancil having a different upstream area to the observation
                rflow_fire = jules_fire_data_ind.copy()
                rflow_nofire = jules_nofire_data_ind.copy()
                # Nic removed the area correction ...
                # rflow_fire['value'] = areao[id] * (jules_fire_data_ind.loc[:, 'value'] / this_aream)
                # rflow_nofire['value'] = areao[id] * (jules_nofire_data_ind.loc[:, 'value'] / this_aream)

                # Change units to m3/sec
                indices = sf.get_grid_indices(cubes_dict, coords)
                ix, iy = indices['jules_output_fire'][max_in_3x3]
                area2d_cell = area2d[iy, ix]
                rflow_fire['rflow_m3persec'] = rflow_fire['value'] * area2d_cell / 1000.
                rflow_nofire['rflow_m3persec'] = rflow_nofire['value'] * area2d_cell / 1000.

                # Get model obs
                # obs = rflow_df_top50.iloc[rflow_df_top50.index == id].dropna(axis='columns').T
                obs = rflow_df_top50.loc[id].dropna(axis='rows')

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
                                         'value': obs[iobs]})  # obs[iobs][id]
                odf = pd.concat([odf, temp_fire, temp_nofire, temp_obs], ignore_index=True)

        odf['month'] = pd.DatetimeIndex(odf['date']).month
        odf.to_csv(ofile)

    return {'df': odf, 'mouth_pts': mouth_pts, 'obs_treatment': ['Gedney-method']}


def fig8():
    # Make Taylor diagram for runoff results
    print('Making Taylor diagrams')
    import os
    if not os.path.isdir('ycpoin'):
        os.system("git clone https://gist.github.com/3342888.git ycopin")

    from ycopin.taylorDiagram import TaylorDiagram
    fig7_dict = figure7_preprocessing()
    df = fig7_dict['df']
    mouth_pts = fig7_dict['mouth_pts']
    a = df.groupby(['basin', 'month', 'model'])['value'].mean()
    a = a.reset_index()

    for basin in pd.unique(a['basin']):
        obs = a.loc[(a['model'] == 'Dai & Trenberth') & (a['basin'] == basin)]
        models = pd.unique(a['model'])
        # for model in models:



def fig8_preprocessing():
    import os
    if not os.path.isdir('ycpoin'):
        os.system("git clone https://gist.github.com/3342888.git ycopin")

    from ycopin.taylorDiagram import TaylorDiagram



def main():

    # Annual and seasonal GPP, ET and albedo plots
    # fig1(region='global')
    # fig1(region='southafrica')
    # fig1(region='brazil')

    # Veg fraction evaluation using CCI
    # fig2()
    # fig2(region='southafrica')
    # fig2(region='brazil')

    # Plots for the CCI PFT paper
    # fig2_cci_comparison()
    # fig3_cwt_vs_pft_CAR()
    # fig3()  # Latitudinal Variation in major PFTs
    # fig3_cwt_vs_pft()
    # fig3_cwt_vs_pft_CAR()
    # fig5()

    # Plots for Brazil evaluation
    # fig4(region='brazil')
    # fig5

    # ISIMIP paper plots
    # paper_fig3(region='global')
    # paper_figS4(region='brazil')
    paper_figS4(region='brazil', fire=True)
    paper_figS4(region='brazil', fire=False)

    # fig1(region='global')  # Annual and seasonal GPP, ET and albedo plots
    # fig2(region='global', fire=False)  # Veg fraction evaluation using CCI
    # fig2(region='global', fire=True)  # Veg fraction evaluation using CCI
    # figure7()  # River flow plots


if __name__ == '__main__':
    main()
