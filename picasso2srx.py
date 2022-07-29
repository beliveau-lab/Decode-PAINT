# Specific script name.
SCRIPT_NAME = 'picasso2srx.py'

# Specify script version.
VERSION = 1.0

from glob import glob
import argparse
import pandas as pd
import os
import locsutil as lu
import time

MESSAGE = f"""
%s version %s. Requires a path to DNA-PAINT localization file (HDF5) or a directory containing
localization files. Optionally takes in other parameters.
Returns SRX compatible CSV files.
""" % (SCRIPT_NAME, VERSION)

# SRX .csv file format
SRX_COL = [
    'image-ID', 'time-point', 'cycle', 'z-step', 'frame', 'accum', 'probe',
    'photon-count', 'photon-count11', 'photon-count12', 'photon-count21', 'photon-count22',
    'psfx', 'psfy', 'psfz', 'psf-photon-count', 'x', 'y', 'z', 'stdev', 'amp',
    'background11', 'background12', 'background21', 'background22', 'maxResidualSlope',
    'chisq', 'log-likelihood', 'llr', 'accuracy', 'fiducial', 'valid',
    'precisionx', 'precisiony', 'precisionz', 'cluster-ID'
]


def main():
    """
    Converts DNA-PAINT localization files to SRX readable CSV files.
    
    Parameters (user input)
    -------
    input : str
    frames : int
    pixel : int, optional
    segmented : bool, optional
    decoded : bool, optional
    
    Returns
    -------
    None

    """
    
    # Allow user to input parameters on command line.
    user_input = argparse.ArgumentParser(description=MESSAGE)

    # Input file name.
    required_named = user_input.add_argument_group('required arguments')
    required_named.add_argument('-f', '--input', required=True, action='store',
                                type=str, help='Input file name or directory name which includes input files.')
    required_named.add_argument('-a', '--frames', required=True, action='store',
                                type=int, help='Number of frames per section acquired.')
    
    # Take in optional parameters.
    user_input.add_argument('-p', '--pixel', action='store', type=int,
                        default=65.0, help='Pixel size in nm scale. Default = 65 nm.')
    user_input.add_argument('-s', '--segmented', action='store_true',
                        default=False, help='Whether file was segmented. Default = disabled.')
    user_input.add_argument('-d', '--decoded', action='store_true',
                        default=False, help='Whether file was decoded. Default = disabled.')
    
    args = user_input.parse_args()
    input_path = args.input
    pixel_size = args.pixel
    frames_per_section = args.frames
    segmented = args.segmented
    decoded = args.decoded
    
    # Starts a timer.
    start = time.time()

    # Performs conversion iteratively if input is a path to a directory.
    if os.path.isdir(input_path):
        query = os.path.join(input_path, '*.hdf5')
        locs_files = glob(query)
        for file in locs_files:
            print('Converting...' + file)
            picasso_to_srx(file, pixel_size, frames_per_section, segmented, decoded)
    else:
        picasso_to_srx(input_path, pixel_size, frames_per_section, segmented, decoded)
    
    print(f'Total_time: {time.time() - start} [sec]')


def picasso_to_srx(input_file: str,
                   pixel_size: float,
                   frames_per_section: int,
                   segmented: bool,
                   decoded: bool
                   ) -> None:
    
    # Converts format
    converted = convert_to_srx(input_file, pixel_size, frames_per_section, segmented, decoded)

    # Makes an output directory.
    work_dir = os.path.dirname(input_file)
    out_path = os.path.join(work_dir, 'srx')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # Generates output file name prefix
    file_base = os.path.basename(input_file)
    file_stem = lu.clean_filename(file_base)
    out_name = f'{file_stem}.csv'

    out_name = os.path.join(out_path, out_name)

    # Outputs a converted file.
    converted.to_csv(out_name, index=False)


def convert_to_srx(input_file: str,
                   pixel_size: float,
                   frames_per_section: int,
                   segmented: bool,
                   decoded: bool
                   ) -> pd.DataFrame:
    
    locs_data = lu.read_locs(input_file)
    locs_data = locs_data.reset_index()
    converted = pd.DataFrame(columns=SRX_COL)
    
    if segmented:
        if 'hdbscan' in locs_data.columns:
            locs_data = locs_data[locs_data['hdbscan'] != -1]
            converted['cluster-ID'] = locs_data['hdbscan']
        else:
            locs_data = locs_data[locs_data['dbscan'] != -1]
            converted['cluster-ID'] = locs_data['dbscan']
    
    locs_data = lu.convert_xy(locs_data, pixel_size)
    
    converted['image-ID'] = locs_data['index']
    converted['z-step'] = locs_data['frame'] // frames_per_section
    converted['frame'] = locs_data['frame'] % frames_per_section
    
    if 'n' in locs_data.columns:
        converted['accum'] = locs_data['n']
    converted['photon-count'] = locs_data['photons']
    converted['photon-count11'] = locs_data['photons'] / 2.0
    converted['photon-count12'] = locs_data['photons'] / 2.0
    converted['x'] = locs_data['x_nm']
    converted['y'] = locs_data['y_nm']
    converted['z'] = locs_data['z']
    converted['background11'] = locs_data['bg'] / 2.0
    converted['background12'] = locs_data['bg'] / 2.0
    converted['valid'] = 1
    converted['precisionx'] = locs_data['lpx_nm']
    converted['precisiony'] = locs_data['lpy_nm']
    converted['precisionz'] = locs_data['d_zcalib']
    
    if decoded:
        converted['probe'] = locs_data['id']
        
    converted.fillna(0, inplace=True)
    
    return converted


if __name__ == '__main__':
    main()
