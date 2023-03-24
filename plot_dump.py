import sys
sys.path.append('/home/h02/hadhy/cylc-run/u-cu360/app/postprocess/file')
import jules
import cf_units
import iris
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.plot as iplt
import cartopy


def get_colorbar_min_max_intervals(cube1, cube2, num_legend_colours, lower_pc, upper_pc):
    '''
    Works out sensible colorbar min, max and interval values for cubes 1 & 2, and diff
    :param cube1:
    :param cube2:
    :param num_legend_colours:
    :param lower_pc: lower percentile value
    :param upper_pc: upper percentile value
    :return: dictionary of values
    '''

    # Calculate min, max and interval values for the absolute value colorbars
    pctile1 = cube1.collapsed(['latitude', 'longitude'], iris.analysis.PERCENTILE, percent=[lower_pc, upper_pc])
    pctile2 = cube2.collapsed(['latitude', 'longitude'], iris.analysis.PERCENTILE, percent=[lower_pc, upper_pc])
    leg_min = np.int32(np.rint(np.append(pctile1[0, :].data, pctile2[0, :].data).mean()))
    leg_max = np.int32(np.rint(np.append(pctile1[1, :].data, pctile2[1, :].data).mean()))
    # if leg_min == leg_max, we'll need to increase the number of decimal points, so that there's a difference in the legend
    minmaxdiff = leg_max - leg_min
    dpts = 0
    while minmaxdiff == 0:
        leg_min = np.round(np.append(pctile1[0, :].data, pctile2[0, :].data).mean(), dpts)
        leg_max = np.round(np.append(pctile1[1, :].data, pctile2[1, :].data).mean(), dpts)
        minmaxdiff = leg_max - leg_min
        dpts += 1

    leg_interval = np.int32(np.rint((leg_max - leg_min) / num_legend_colours))
    # If leg_interval is 0, this will gradually increase the number of decimal points
    dpts = 0
    while leg_interval == 0:
        leg_interval = np.round((leg_max - leg_min) / num_legend_colours, decimals=dpts)
        dpts += 1

    # Calculate min, max and interval values for the difference colorbars
    diff = cube2 - cube1
    pctile_diff = diff.collapsed(['latitude', 'longitude'], iris.analysis.PERCENTILE, percent=[lower_pc, upper_pc])
    legdif_min = np.int32(np.rint(pctile_diff[0, :].data.mean()))
    legdif_max = np.int32(np.rint(pctile_diff[1, :].data.mean()))
    # If legdif_max - legdif_min == 0, then we need to add more decimal points ...
    minmaxdiff = legdif_max - legdif_min
    dpts = 0
    while minmaxdiff == 0:
        legdif_min = np.round(pctile_diff[0, :].data.mean(), dpts)
        legdif_max = np.round(pctile_diff[1, :].data.mean(), dpts)
        minmaxdiff = legdif_max - legdif_min
        dpts += 1

    # Now, make sure the diff is centred around 0
    legdif_min = np.max([np.abs(legdif_min), np.abs(legdif_max)]) * -1.
    legdif_max = np.max([np.abs(legdif_min), np.abs(legdif_max)])

    legdif_interval = np.int32(np.rint((legdif_max - legdif_min) / num_legend_colours))
    # If legdif_interval is 0, this will gradually increase the number of decimal points
    dpts = 0
    while legdif_interval == 0:
        legdif_interval = np.round((legdif_max - legdif_min) / num_legend_colours, decimals=dpts)
        dpts += 1

    odict = {'leg_min': leg_min, 'leg_max': leg_max, 'leg_interval': leg_interval,
             'legdif_min': legdif_min, 'legdif_max': legdif_max, 'legdif_interval': legdif_interval}

    return odict


def plot_2cubes(cubedict, name1, name2, outfile=None, coastline=False, num_legend_colours=20, lower_pc=15, upper_pc=85):
    '''
    Compares 2 cubes and creates a difference column
    :param cubedict: dictionary containing cubes with key names to include in the plot
    :param name1: key name to use to select cube1 (NB: Must contain units)
    :param name2: key name to use to select cube2 (NB: Must contain units)
    :param outfile: optional output filename
    :param coastline: optional boolean whether to print coastlines
    :param num_legend_colours: approximate number of colours we want in the legend - this helps decide the spacing between legend ticks
    :return:
    '''

    cube1 = cubedict[name1]
    cube2 = cubedict[name2]

    geogcs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    cube1.coord_system = geogcs
    cube1.coord('latitude').coord_system = geogcs
    cube1.coord('longitude').coord_system = geogcs
    cube2.coord_system = geogcs
    cube2.coord('latitude').coord_system = geogcs
    cube2.coord('longitude').coord_system = geogcs

    # First dimension name
    coordname = [coord.name() for coord in cube1.coords()][0]

    nrows = len(cube1.coord(coordname).points)
    indices = np.arange(1, 1 + (3 * nrows))

    cb = get_colorbar_min_max_intervals(cube1, cube2, num_legend_colours, lower_pc, upper_pc)

    fig = plt.figure(figsize=(10, 100))

    for x in cube1.coord(coordname).points:

        print(x)

        # Plot #1: 1850 spinup
        ax1 = plt.subplot(nrows, 3, indices[(3 * x)], projection=cartopy.crs.PlateCarree())
        ax1.axis('off')
        ax1plt = iplt.contourf(cube1[x], levels=np.arange(cb['leg_min'], cb['leg_max'] + cb['leg_interval'], cb['leg_interval']), extend='both', axes=ax1)
        if coastline:
            plt.gca().coastlines()
        if x == 0:
            ax1.set_title(f"A. {name1}", fontsize=8)
        if x == (nrows - 1):
            pos1 = ax1.get_position()

        # Plot #2: 1856 spinup
        ax2 = plt.subplot(nrows, 3, indices[(3 * x) + 1], projection=cartopy.crs.PlateCarree())
        ax2.axis('off')
        ax2plt = iplt.contourf(cube2[x], levels=np.arange(cb['leg_min'], cb['leg_max'] + cb['leg_interval'], cb['leg_interval']), extend='both', axes=ax2)
        if coastline:
            plt.gca().coastlines()
        if x == 0:
            ax2.set_title(f"B. {name2}", fontsize=8)
        if x == (nrows - 1):
            pos2 = ax2.get_position()
            trim_x = (pos2.xmax - pos1.xmin) / 4.
            colorbar_axes = fig.add_axes([pos1.xmin + trim_x, pos2.ymin - 0.05, (pos2.xmax - trim_x) - (pos1.xmin + trim_x), 0.02])
            # colorbar_axes = fig.add_axes(
            #     [pos2.xmin, pos2.ymin - 0.05, pos2.xmax - pos2.xmin, 0.02])  # Left, bottom, width, height
            colorbar = plt.colorbar(ax2plt, colorbar_axes, orientation="horizontal")
            colorbar.ax.tick_params(labelsize=8)
            colorbar.set_label(cube2.units)

        # Plot #3: 1856 minus 1850 diff
        diff = cube2[x] - cube1[x]
        ax3 = plt.subplot(nrows, 3, indices[(x * 3) + 2], projection=cartopy.crs.PlateCarree())
        ax3.axis('off')
        ax3plt = iplt.contourf(diff, levels=np.arange(cb['legdif_min'], cb['legdif_max'] + cb['legdif_interval'], cb['legdif_interval']), extend='both', cmap="RdBu", axes=ax3)
        if coastline:
            plt.gca().coastlines()
        if x == 0:
            ax3.set_title("Difference: B - A", fontsize=8)
        if x == (nrows - 1):
            pos3 = ax3.get_position()
            colorbar_axes = fig.add_axes(
                [pos3.xmin, pos3.ymin - 0.05, pos3.xmax - pos3.xmin, 0.02])  # Left, bottom, width, height
            colorbar = plt.colorbar(ax3plt, colorbar_axes, orientation="horizontal")
            colorbar.ax.tick_params(labelsize=8)
            colorbar.set_label(cube1.units)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def load_dump(var):

    loc = '/hpc/data/d05/hadhy/jules_output/u-cu360/'
    units_lut = {'frac': cf_units.Unit('%'),
                 'tstar_tile': cf_units.Unit('K')}

    cubedict = {}
    for sup in ['01', '02', '03']:
        for yr in ['1850', '1856']:
            print(sup, yr)
            cube = jules.load_cube(f'{loc}isimip3b_fire_spinup_{sup}.dump.{yr}0101.0.nc', var)  # tstar_tile
            cube.data = ma.masked_invalid(cube.data)  # Needed for plot colorbar limits estimation
            try:
                cube.units = units_lut[var]
            except:
                print('unable to work out units')
                cube.units = cf_units.Unit(1)
            thiskey = f'Spinup{sup} {yr}'
            cubedict[thiskey] = cube

    return cubedict


def main():

    cubedict = load_dump('frac')
    plot_2cubes(cubedict, 'Spinup01 1850', 'Spinup03 1856', num_legend_colours=10, lower_pc=5, upper_pc=95)

    cubedict = load_dump('tstar_tile')
    plot_2cubes(cubedict, 'Spinup01 1850', 'Spinup03 1856', num_legend_colours=10, lower_pc=5, upper_pc=95)


if __name__ == '__main__':
    main()
