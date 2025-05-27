import os
import glob
import tarfile
import subprocess
from itertools import product
from multiprocessing import Pool, freeze_support


def run_model_driving(model, drv, stream, data_path, output_path, moose_path):

    print(f"{model} | {drv} | {stream}")
    files = glob.glob(f"{os.path.join(data_path, model)}/*.nc")
    files2archive = sorted([file for file in files if (drv in file) and (stream in file)])

    # Write the list to a file
    with open(f"files2archive_{model}_{drv}_{stream}.txt", 'w') as f:
        for file in files2archive:
            f.write(file + '\n')

    # Make tarball to save to, and check dir exists
    ofile = f"{'.'.join(os.path.basename(files2archive[0]).split('.')[:2])}.tar.gz"
    tarball = f"{os.path.join(output_path, model)}/{ofile}"
    if not os.path.isdir(os.path.dirname(tarball)):
        os.makedirs(os.path.dirname(tarball))

    # Check if the tarball is already on MASS
    mass_tarball = f"{moose_path}/{model}/{os.path.basename(tarball)}"
    output = subprocess.run(['moo', 'ls', mass_tarball], capture_output=True, text=True)
    already_there = True if mass_tarball in output.stdout else False

    # Check if it's already a folder on MASS, if not, create one
    dir_test = subprocess.run(['moo', 'ls', os.path.dirname(mass_tarball)], capture_output=True, text=True)
    create_dir = False if dir_test.returncode == 0 else True
    if create_dir:
        subprocess.run(['moo', 'mkdir', '-p', os.path.dirname(mass_tarball)])

    if not already_there:
        # Create a compressed tar file, only adding existing files
        if not os.path.isfile(tarball):
            with tarfile.open(tarball, 'w:gz') as tar:
                for file in files2archive:
                    if os.path.exists(file):
                        tar.add(file)
                    else:
                        print(f"File {file} does not exist and will not be added to the archive.")

        # Once that's completed, put the file on MASS
        print(f"Uploading to MASS: {os.path.basename(tarball)}")
        subprocess.run(['moo', 'put', tarball, mass_tarball])


def main():
    # For each model and each stream, create a *.tar.gz file
    # data_path = '/hpc/data/d01/hadcam/jules_output/ALL_u-bk886_isimip_0p5deg_origsoil_dailytrif/'
    data_path = '/hpc/data/d05/cburton/jules_output/u-cf137'
    output_path = '/scratch/hadhy/u-cf137/'
    moose_path = 'moose:/adhoc/projects/isi-mip/isimip_2b/u-cf137'

    models = ['MIROC5', 'IPSL-CM5A-LR', 'GFDL-ESM2M', 'HADGEM2-ES']

    driving = []  # 'c20c', 'rcp26', or 'rcp60'
    streams = []  # list of 20 streams
    for model in models:
        files = glob.glob(f"{os.path.join(data_path, model)}/*.nc")
        driving.extend(
            sorted(list(set(['_'.join(os.path.basename(file).split('.')[0].split('_')[1:]) for file in files]))))
        streams.extend(sorted(list(set([os.path.basename(file).split('.')[1] for file in files]))))

    # Get a unique list of values
    driving = sorted(list(set(driving)))
    streams = sorted(list(set(streams)))

    # Exclude spinup streams from the upload
    driving = [drv for drv in driving if 'spinup' not in drv]

    with Pool(processes=16) as pool:
        pool.starmap(run_model_driving, product(models, driving, streams, [data_path], [output_path], [moose_path]))


def main_orig():
    # For each model and each stream, create a *.tar.gz file
    data_path = '/hpc/data/d01/hadcam/jules_output/ALL_u-bk886_isimip_0p5deg_origsoil_dailytrif/'
    output_path = '/scratch/hadhy/ALL_u-bk886_isimip_0p5deg_origsoil_dailytrif/'
    moose_path = 'moose:/adhoc/projects/isi-mip/isimip_2b/ALL_u-bk886_isimip_0p5deg_origsoil_dailytrif'
    models = ['MIROC5', 'IPSL-CM5A-LR', 'GFDL-ESM2M', 'HADGEM2-ES']

    for model in models:
        files = glob.glob(f"{os.path.join(data_path, model)}/*.nc")
        driving = sorted(list(set([os.path.basename(file).split('.')[0] for file in files])))
        for drv in driving:
            print(f"{model} | {drv}")
            driving_files = sorted([file for file in files if drv in file])
            streams = sorted(list(set([os.path.basename(file).split('.')[1] for file in driving_files])))
            for stream in streams:
                files2archive = sorted([file for file in driving_files if stream in file])

                # Write the list to a file
                with open(f"files2archive_{model}_{drv}_{stream}.txt", 'w') as f:
                    for file in files2archive:
                        f.write(file + '\n')

                # Make tarball to save to, and check dir exists
                ofile = f"{'.'.join(os.path.basename(files2archive[0]).split('.')[:2])}.tar.gz"
                tarball = f"{os.path.join(output_path, model)}/{ofile}"
                if not os.path.isdir(os.path.dirname(tarball)):
                    os.makedirs(os.path.dirname(tarball))

                # Create a compressed tar file, only adding existing files
                with tarfile.open(tarball, 'w:gz') as tar:
                    for file in files2archive:
                        if os.path.exists(file):
                            tar.add(file)
                        else:
                            print(f"File {file} does not exist and will not be added to the archive.")

                # Once that's completed, put the file on MASS
                print(f"Uploading to MASS: {os.path.basename(tarball)}")
                subprocess.run(['moo', 'put', tarball, f"{moose_path}/{os.path.basename(tarball)}"])


if __name__ == '__main__':
    freeze_support()
    main()
