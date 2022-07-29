# Specific script name.
SCRIPT_NAME = 'crop.py'

# Specify script version.
VERSION = 1.0

import argparse
import os
import pandas as pd
import locsutil as lu

MESSAGE = f"""
%s version %s. Requires a path to DNA-PAINT localization file (HDF5), output path,
the coordinates of the top-left corner, and the length of the square side.
Returns cropped DNA-PAINT localization file.
""" % (SCRIPT_NAME, VERSION)


def main():
    """
    Crops DNA-PAINT localization with HDBSCAN.
    
    Parameters (user input)
    -------
    input : str
    mask : str
    dapi : str, optional
    idcol : str, optional
    output : str, optional
    
    Returns
    -------
    None

    """
    
    # Allows user to input parameters on command line.
    user_input = argparse.ArgumentParser(description=MESSAGE)

    # Inputs file names.
    required_named = user_input.add_argument_group('required arguments')
    required_named.add_argument('-f', '--input', action='store', required=True, type=str,
                                help='Input csv drift file name.')
    required_named.add_argument('-o', '--output', action='store', required=False, type=str,
                                help='output file prefix')
    # Defines ROI.
    required_named.add_argument('-x', '--minx', action='store', required=True, type=int,
                                help='The X coordinate of the top left corner')
    required_named.add_argument('-y', '--miny', action='store', required=True, type=int,
                                help='The Y coordinate of the top left corner')
    required_named.add_argument('-s', '--size', action='store', required=True, type=int,
                                help='The width/height of the cropping region')

    # Optional parameters to crop out.
    user_input.add_argument('-z', '--minz', action='store', type=float, default=-9999.9,
                            help='The minimum z-height of the cropping region, default = disabled')
    user_input.add_argument('-Z', '--maxz', action='store', type=float, default=9999.9,
                            help='The maximum z-height of the cropping region, default = disabled')
    
    args = user_input.parse_args()
    in_file = args.input
    out_name = args.output
    min_x = args.minx
    min_y = args.miny
    min_z = args.minz
    max_z = args.maxz
    size = args.size
    
    # Imports locs file.
    data = lu.read_locs(in_file)
    
    # Gets working directory path.
    work_dir = os.path.dirname(in_file)
    
    # Crops locs file.
    out_data = crop_xy(data, min_x, min_y, size)
    
    if min_z and max_z:
        out_data = crop_z(out_data, min_z, max_z)

    # Imports a yaml file.
    yaml_in = in_file.rstrip('hdf5') + 'yaml'
    frame_val, height_val, width_val = lu.read_yaml(yaml_in)

    # Generates output file name prefix.
    file_base = os.path.basename(in_file)
    print(file_base)
    file_stem = lu.clean_filename(file_base)

    if out_name:
        out_name = f'{file_stem}_{out_name}'
    else:
        out_name = f'{file_stem}_cropped'
        
    out_hdf5_name = os.path.join(work_dir, out_name) + '.hdf5'
    out_yaml_name = os.path.join(work_dir, out_name) + '.yaml'

    # Outputs cropped files.
    lu.write_locs(out_data, out_hdf5_name)
    lu.write_yaml(frame_val, size, size, out_yaml_name)


def crop_xy(locs_data: pd.DataFrame, min_x: float, min_y: float, width: float) -> pd.DataFrame:
    """
    Crops localization data using given x & y coordinates and width.
    For Picasso localization data, x and y are in camera-pixel scale.
        top-left : (mix_x, min_y),
        top-right : (mix_x + width, min_y),
        bottom-left : (mix_x, min_y + width),
        bottom-right : (mix_x + width, min_y + width)

    Parameters
    ----------
    locs_data : pd.DataFrame
        Localization data.
    min_x : float
        x of the top-left corner of the cropping square.
    min_y : float
        y of the top-left corner of the cropping square.
    width : float
        Width of the cropping square.

    Returns
    -------
    cropped : pd.DataFrame

    """
    
    max_x = min_x + width
    max_y = min_y + width
    
    cropped = locs_data.copy()
    
    cropped = cropped[(cropped['x'] >= min_x) & (cropped['x'] <= max_x)]
    cropped = cropped[(cropped['y'] >= min_y) & (cropped['y'] <= max_y)]
    
    cropped['x'] -= min_x
    cropped['y'] -= min_y
    
    return cropped


def crop_z(locs_data: pd.DataFrame, min_z: float, max_z: float) -> pd.DataFrame:
    """
    Crops localization data using given a z height range.

    Parameters
    ----------
    locs_data : pd.DataFrame
        Localization data.
    min_z : float
    max_z : float

    Returns
    -------
    cropped : pd.DataFrame

    """
    
    cropped = locs_data.copy()
    cropped = cropped[(cropped['z'] >= min_z) & (cropped['z'] <= max_z)]
    
    return cropped


if __name__ == '__main__':
    main()
