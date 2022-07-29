# Specific script name.
SCRIPT_NAME = 'concatenate.py'

# Specify script version.
VERSION = 1.0

import argparse
import pandas as pd
from glob import glob
import os
import locsutil as lu

MESSAGE = f"""
%s version %s. Requires a path to a directory containing Z-series of DNA-PAINT localization files (HDF5)
to be concatenated, Z step size, and an output file name. Returns a single DNA-PAINT localization file
(HDF5 and YAML). Optionally, localizations of low quality can be omitted.
""" % (SCRIPT_NAME, VERSION)


def main():
    """
    Concatenates a series of DNA-PAINT localization files in Z to a single file.

    Parameters (user input)
    -------
    input : str
    zcalib : float, optional
    lp : float, optional
    minz : float, optional
    maxz : float, optional
    maxphoton : int, optional
    output : str, optional

    Returns
    -------
    None

    """

    # Allows user to input parameters on command line.
    user_input = argparse.ArgumentParser(description=MESSAGE)

    # Inputs the directory name.
    required_named = user_input.add_argument_group('required arguments')
    required_named.add_argument('-d', '--dir', action='store', required=True, type=str,
                                help='Input directory name which contains locs data to be concatenated.')
    required_named.add_argument('-s', '--zstep', action='store', required=True, type=int,
                                help='Z step size (nm)')
    

    # Optional parameters to filter out bad localizations.
    user_input.add_argument('-c', '--zcalib', action='store', type=float, default=0.0,
                            help='Filtering threshold for d_zcalib. Default = 0.0 (disabled)')
    user_input.add_argument('-l', '--lp', action='store', type=float, default=0.0,
                            help='Filtering threshold for lpx and lpy. Default = 0.0 (disabled)')
    user_input.add_argument('-z', '--minz', action='store', type=float, default=-9999.9,
                            help='Filtering threshold for minimum z. Default = -9999.9 (disabled)')
    user_input.add_argument('-Z', '--maxz', action='store', type=float, default=9999.9,
                            help='Filtering threshold for maximum z. Default = 9999.9 (disabled)')
    user_input.add_argument('-p', '--maxphoton', action='store', type=int, default=-1,
                            help='Filtering threshold for maximum photon counts. Default = -1 (disabled)')
    user_input.add_argument('-o', '--output', action='store', type=str, default='concatenated',
                            help='Output file prefix. Default = None')

    args = user_input.parse_args()

    input_path = args.dir
    z_step = args.zstep
    max_dzcalib = args.zcalib
    max_locs_precision = args.lp
    min_z = args.minz
    max_z = args.maxz
    max_photons = args.maxphoton
    out_name = args.output
    
    os.chdir(input_path)
    hdf5_list = lu.get_hdf_list()
    print(hdf5_list)

    outdata = pd.DataFrame()

    off_val = 0
    z_val = 0

    for item in hdf5_list:
        print(item)
        data = lu.read_locs(item)
        data = lu.clean_locs(data, max_dzcalib, max_locs_precision, min_z, max_z, max_photons)

        outdata, _off = concat_z(outdata, data, off_val, z_val)
        off_val = off_val + _off
        z_val = z_val + z_step

    # Imports a yaml file
    yaml_in = glob('*.yaml')[0]
    frame_val, height_val, width_val = lu.read_yaml(yaml_in)
    
    total_frame_val = str(int(frame_val) * int(len(hdf5_list)))

    # Generates output file names
    out_hdf5_name = out_name + '.hdf5'
    out_yaml_name = out_name + '.yaml'

    # Outputs drift concatenated files
    lu.write_locs(outdata, out_hdf5_name)
    lu.write_yaml(total_frame_val, height_val, width_val, out_yaml_name)


def concat_z(df1, df2, frame_offset, z_offset):
    # Input form: frame, x, y, photons, sx, sy, bg, lpx, lpy, ellipticity, net_gradient, z, d_zcalib
    
    # Reads in from HDF5
    data1 = df1.copy()
    data2 = df2.copy()
    
    # Reads in from HDF5
    last_frame = int(data2['frame'].iloc[-1]) + 1
    
    data2['frame'] = data2['frame'].astype('int') + int(frame_offset)
    data2['z'] = data2['z'].astype('float') + int(z_offset)
    
    data_out = pd.concat([data1, data2], ignore_index=True)
    
    return data_out, last_frame


if __name__ == '__main__':
    main()
