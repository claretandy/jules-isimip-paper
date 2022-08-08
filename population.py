import iris
import iris.analysis.cartography
import os, sys

sys.path.append('/home/h02/hadhy/GitHub/wcssp_casestudies')
import std_functions as sf


def main():
    # Read TRENDY pop data (pop density = pop per sqkm)
    tpop = iris.load('/data/cr1/cburton/Population/TRENDY2020/popd_trendyv9.nc')
    landfrac = tpop[0]
    tpop_density = tpop[1]

    # Read ISIMIP pop data (pop per grid cell)
    ipop = iris.load_cube(
        '/home/h02/hadcam/isimip/ISIMIP/population/isimip2b/histsoc/population_histsoc_0p5deg_annual_1861-2005.nc4')

    # Convert ISIMIP data to ppl / sqkm
    for i in ['latitude', 'longitude']:
        ipop.coord(i).guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(ipop)  # square metres
    cube_areas = ipop.copy(grid_areas) / 1000000  # to sqkm
    ipop_density = ipop / cube_areas

    # Plot 2000
    diff = tpop_density[300].copy(tpop_density[300].data - ipop_density[139].data)
    sf.plot_cube(ipop_density[139], title='ISIMIP population density, 2000', stretch='low',
                 ofile='isimip_ppl-per-sqkm_2000.png')  # Year 2000
    sf.plot_cube(tpop_density[300], title='TRENDY population density, 2000', stretch='low',
                 ofile='trendy_ppl-per-sqkm_2000.png')
    sf.plot_cube(diff, title='TRENDY minus ISIMIP population density, 2000',
                 ofile='trendy-isimip_ppl-per-sqkm_2000.png')

    # Plot 1950
    diff = tpop_density[250].copy(tpop_density[250].data - ipop_density[89].data)
    sf.plot_cube(ipop_density[89], title='ISIMIP population density, 1950', stretch='low',
                 ofile='isimip_ppl-per-sqkm_1950.png')  # Year 2000
    sf.plot_cube(tpop_density[250], title='TRENDY population density, 1950', stretch='low',
                 ofile='trendy_ppl-per-sqkm_1950.png')
    sf.plot_cube(diff, title='TRENDY minus ISIMIP population density, 1950',
                 ofile='trendy-isimip_ppl-per-sqkm_1950.png')


if __name__ == '__main__':
    main()
