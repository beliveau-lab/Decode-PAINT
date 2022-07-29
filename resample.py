# Specific script name.
SCRIPT_NAME = 'resample.py'

# Specify script version.
VERSION = 1.0

import argparse
import os
import locsutil as lu
import time
import pandas as pd
from glob import glob
_PIXEL_SIZE = 65.0

MESSAGE = f"""
%s version %s. Requires a path to DNA-PAINT localization file (HDF5),
downsampling frequency, and output file name. Returns cropped DNA-PAINT
localization file.
""" % (SCRIPT_NAME, VERSION)


def main():
    """
    
    Parameters (user input)
    ----------
    input : str
    frequency : int
    output : str
    
    Returns
    -------
    None
    
    """
    # Counts process time
    start = time.time()
    
    # Takes in input and output filenames
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input', required=True, action='store',
                        type=str, help='Input file name.')
    parser.add_argument('-q', '--frequency', action='store', default=4,
                        type=int, help='How many series of resampled data generated.')
    parser.add_argument('-o', '--output', action='store', default='analyzed',
                        type=str, help='Specify the output file postfix.')
    
    args = parser.parse_args()
    input_path = args.input
    freq = args.frequency

    # Makes a directory for output
    wdpath = os.path.dirname(input_path)
    out_path = os.path.join(wdpath, 'downsampled')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Performs conversion iteratively if input is a path to a directory.
    if os.path.isdir(input_path):
        query = os.path.join(input_path, '*.hdf5')
        locs_files = glob(query)
        for file in locs_files:
            print('Converting...' + file)
            resample(file, freq, out_path)
    else:
        resample(input_path, freq, out_path)

    print(f'Total_time: {time.time() - start} [sec]')


def resample(file: str, freq: int, out_path: str) -> None:
    """
    
    Parameters
    ----------
    file : str
    freq ; int
    out_path : str

    Returns
    -------
    None

    """
    # Imports a hdf5 file
    data = lu.read_locs(file)

    # Imports a yaml file
    yaml_in = file.rstrip('hdf5') + 'yaml'
    frame_val, height_val, width_val = lu.read_yaml(yaml_in)
    
    resampled = make_resampled_list(data, freq)
    
    # Creates output prefix
    file_base = os.path.basename(file)
    file_stem = lu.clean_filename(file_base)
    
    for i in range(len(resampled)):
        out_name = f'{file_stem}_{i + 1}'
        
        out_hdf5_name = os.path.join(out_path, out_name) + '.hdf5'
        out_yaml_name = os.path.join(out_path, out_name) + '.yaml'
        
        # Output cropped files
        lu.write_locs(resampled[i], out_hdf5_name)
        lu.write_yaml(str(len(resampled[i])), height_val, width_val, out_yaml_name)


def make_resampled_list(locs_data: pd.DataFrame, freq: int) -> list:
    """
    Makes series of localizations resampled.
    
    Parameters
    ----------
    locs_data : pd.DataFrame
    freq : int

    Returns
    -------
    list

    """
    out_list = []
    
    for i in range(1, freq):
        fraction = i / freq
        out_list.append(locs_data.sample(frac=fraction, random_state=1))
    
    return out_list


if __name__ == '__main__':
    main()
