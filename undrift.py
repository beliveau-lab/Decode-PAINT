# Specific script name.
SCRIPT_NAME = 'undrift.py'

# Specify script version.
VERSION = 1.0

import argparse
import pandas as pd
import os
import locsutil as lu

# Translates DNA-PAINT localization files in XYZ to roughly align the very first frame of
# each recording to the initial position of the experiment for further image registration.
# Takes drift information file and output file name as command line arguments.
# Drift information file (csv format):
#     Line 1: DNA-PAINT localization file (HDF5) names to be undrifted
#     Line 2: X shift values
#     Line 3: Y shift values
#     Line 4: Z shift values
#
# Details:
# A series of DNA-PAINT recordings are done in a following manner:
# Surface_1 (~100 frames) -> Recording_1 (Z-position 1, ~ 5000 frames)
# -> Surface_2 (~100 frames) -> Recording_2 (Z-position 2, ~ 5000 frames)
# -> Surface_3 (~100 frames) -> ...
# First, surface images are registered with Picasso to compute XYZ drifts over time.
# Then, using the drift values, this script translates DNA-PAINT recording files
# (HDF5 format) in XYZ to roughly align the very first frame of each recording to
# the initial position (e.g. Surface_1).

MESSAGE = f"""
%s version %s. Requires a csv drift file containing DNA-PAINT localization file (HDF5)
names to be undrifted, X shift values, Y shift values, Z shift values, and an output file name.
Returns a series of undrifted DNA-PAINT localization files (HDF5 and YAML).
Optionally, localizations of low quality can be omitted.
""" % (SCRIPT_NAME, VERSION)


def main():
    """
    Translates DNA-PAINT localization files in XYZ to roughly align the very first
    frame of each recording to the initial position of the experiment using the
    information in the input drift file.
    
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
    
    # Inputs file name.
    required_named = user_input.add_argument_group('required arguments')
    required_named.add_argument('-f', '--input', action='store',
                        required=True, type=str, help='Input csv drift file name.')

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
    user_input.add_argument('-o', '--output', action='store', type=str, default=None,
                            help='Output file prefix. Default = None')
    

    # Imports user-specified command line values.
    args = user_input.parse_args()
    drift_file = args.input
    output_file = args.output
    max_dzcalib = args.zcalib
    max_locs_precision = args.lp
    min_z = args.minz
    max_z = args.maxz
    max_photons = args.maxphoton

    # Gets working directory path.
    work_dir = os.path.dirname(drift_file)

    # Imports a csv drift file.
    # The first line containing localization file names is going to be columns of DataFrame
    drift = pd.read_csv(drift_file)
    print(drift)
    locs_file_names = drift.columns.values
    print(locs_file_names)
    print(drift[locs_file_names[1]])

    # Makes a directory for output.
    output_dir = os.path.join(work_dir, 'undrifted')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Processes each file.
    for item in locs_file_names:
        
        print(item)
        
        # Imports a hdf5 file as pd.DataFrame.
        locs_file = os.path.join(work_dir, item)
        locs_data = lu.read_locs(locs_file + '.hdf5')
        
        # Imports a yaml file
        yaml_in = os.path.join(work_dir, (item + '.yaml'))
        frame_val, height_val, width_val = lu.read_yaml(yaml_in)

        # Processes an imported hdf5 file
        out_data = lu.clean_locs(
            locs_data, max_dzcalib, max_locs_precision, min_z, max_z, max_photons)
        out_data = translate(out_data, drift[item])

        # Generates output file name prefix.
        file_base = os.path.basename(locs_file)
        file_stem = lu.clean_filename(file_base)
        if output_file:
            outname = f'{file_stem}_{output_file}'
        else:
            outname = f'{file_stem}_undrifted'
        out_hdf5_name = os.path.join(output_dir, outname) + '.hdf5'
        out_yaml_name = os.path.join(output_dir, outname) + '.yaml'

        # Outputs drift-corrected files.
        lu.write_locs(out_data, out_hdf5_name)
        lu.write_yaml(frame_val, height_val, width_val, out_yaml_name)
        
    return None


def translate(locs_data: pd.DataFrame, drift: pd.Series) -> pd.DataFrame:
    """
    Translates localization data
    
    Parameters
    ----------
    locs_data : pd.DataFrame
        Localization data containing x, y, and z (optional)
    drift : pd.Series
        Drift values for x, y, and z (optional)
    
    Returns
    -------
    data : pd.DataFrame
        Localization data with drift-corrected x, y, and z (optional)
    
    """
    drift_list = drift.values.tolist()

    locs_data['x'] -= float(drift_list[1])
    locs_data['y'] -= float(drift_list[0])

    if len(drift_list) == 3:
        locs_data['z'] -= float(drift_list[2])

    return locs_data


if __name__ == '__main__':
    main()
