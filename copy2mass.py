import os
import glob
import tarfile
import subprocess


def main():
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
                with open('files2archive.txt', 'w') as f:
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
    main()
