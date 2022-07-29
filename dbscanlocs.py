# Specific script name.
SCRIPT_NAME = 'dbscanlocs.py'

# Specify script version.
VERSION = 1.0

from glob import glob
import argparse
import os
import pandas as pd
from sklearn.cluster import DBSCAN
import locsutil as lu
import time
import random

random.seed(0)

MESSAGE = f"""
%s version %s. Requires a path to DNA-PAINT localization file (HDF5) or a directory containing
localization files. Optionally takes in the DBSCAN parameters.
Returns segmented DNA-PAINT localization file(s).
""" % (SCRIPT_NAME, VERSION)


def main():
    """
    Segments DNA-PAINT localization with DBSCAN.

    Parameters (user input)
    -------
    input : str
    pixel : float, optional
    minSize : int, optional
    minSamples : int, optional
    eps : float, optional
    memory : bool, optional

    Returns
    -------
    None
    """
    
    # Allows user to input parameters on command line.
    user_input = argparse.ArgumentParser(description=MESSAGE)
    
    # Inputs file name.
    required_named = user_input.add_argument_group('required arguments')
    required_named.add_argument('-f', '--input', required=True, action='store', type=str,
                                help='Inputs file name or directory name which includes input files.')

    # Optional parameters to filter out bad localizations.
    user_input.add_argument('-p', '--pixel', action='store', type=float, default=65.0,
                            help='Camera pixel size in nm scale, default = 65 nm.')
    user_input.add_argument('-s', '--minSamples', action='store', type=int, default=None,
                            help='DBSCAN parameter: min_samples, default=None.')
    user_input.add_argument('-e', '--eps', action='store', type=float, default=0,
                            help='DBSCAN parameter: cluster_selection_epsilon, default=0.')
    user_input.add_argument('-t', '--threshold', action='store', type=int, default=0,
                            help='Minimum threshold of locs in cluster.')

    args = user_input.parse_args()
    input_path = args.input
    pixel_size = args.pixel
    epsilon = args.eps
    min_samples = args.minSample
    min_threshold = args.threshold
    
    start = time.time()
    
    # Performs DBSCAN iteratively if input is a directory
    if os.path.isdir(input_path):
        query = os.path.join(input_path, '*.hdf5')
        locs_files = glob(query)

        for item in locs_files:
            print('Processing...' + item)
            find_cluster(item, pixel_size, epsilon, min_samples, min_threshold)
    else:
        find_cluster(input_path, pixel_size, epsilon, min_samples, min_threshold)
    
    elapsed_time = time.time() - start
    print('Elapsed_time:{0}'.format(elapsed_time) + '[sec]')


def find_cluster(file: str, pixel_size: float, epsilon: float, min_sample: int, threshold: int):
    """
    Segments localization data into clusters using the DBSCAN algorithm.
    
    Parameters
    ----------
    file : str
        File path to localization data.
    pixel_size : float
        Camera pixel size (nm).
    epsilon : float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_sample : int
        The desired minimum cluster size.
    threshold : int
        Cutoff of the cluster size after segmentation.
        
    Returns
    -------
    None

    """

    # Imports files.
    data = lu.read_locs(file)
    yaml_in = file.rstrip('hdf5') + 'yaml'
    frame_val, height_val, width_val = lu.read_yaml(yaml_in)

    # Applies DBSCAN.
    out_data = dbscan_locs(data, pixel_size, epsilon, min_sample, threshold)

    # Shuffles dbscan ids. This helps to assign different colors to spatially close segments
    # when segments are visualized with Picasso Render.
    unique_ids = out_data['dbscan'].unique().tolist()
    unique_ids_random = random.sample(range(len(unique_ids)), len(unique_ids))
    convert_id = dict(zip(unique_ids, unique_ids_random))
    out_data['dbscan'] = [convert_id[i] for i in out_data['hdbscan']]

    # Makes a directory for output.
    wd_path = os.path.dirname(file)
    out_path = os.path.join(wd_path, 'dbscan')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Creates output prefix.
    file_base = os.path.basename(file)
    file_stem = lu.clean_filename(file_base)
    out_name = f'{file_stem}_dbscan_{epsilon}_{min_sample}_{threshold}'

    out_hdf5_name = os.path.join(out_path, out_name) + '.hdf5'
    out_yaml_name = os.path.join(out_path, out_name) + '.yaml'

    # Outputs cropped files.
    lu.write_locs(out_data, out_hdf5_name)
    lu.write_yaml(frame_val, height_val, width_val, out_yaml_name)


def dbscan_locs(locs_data: pd.DataFrame,
                pixel_size: float,
                epsilon: float,
                min_sample: int,
                threshold: int
                ):
    """
    Apply the DBSCAN algorithm to 3D DNA-PAINT localization data.

    Parameters
    ----------
    locs_data : pd.DataFrame
        Localization data of 3D DNA-PAINT, should have 'z' column.
    pixel_size : float
        Camera pixel size (nm).
    epsilon : float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_sample : int
        The desired minimum cluster size.
    threshold : int
        Cutoff of the cluster size after segmentation.

    Returns
    -------
    data : pd.DataFrame
        Segmented localization data, the 'dbscan' column added

    """
    
    data_col = [
        'frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg',
        'lpx', 'lpy', 'ellipticity', 'net_gradient', 'z',
        'd_zcalib', 'len', 'n', 'photon_rate', 'dbscan'
    ]
    
    # Sets DBSCAN parameters.
    clusterer = DBSCAN(eps=epsilon, min_samples=min_sample)
    
    data = locs_data.copy()
    
    # Converts data_xyz to numpy.array.
    xyz = lu.hdf2xyz(data, pixel_size)
    
    # Applies DBSCAN.
    clusterer.fit(xyz)
    labels = clusterer.labels_
    
    # Merges the DBSCAN labels into the original DataFrame.
    data['dbscan'] = pd.Series(labels)
    data = data[data_col]
    
    # Drops 'noise' localizations.
    # Localizations which don't belong to clusters are labeled '-1'
    data = data[data.dbscan != -1]
    
    # Drops clusters with locs below the threshold.
    if threshold > 0:
        dbscan_ids = data.dbscan.unique()
        for id_ in dbscan_ids:
            if len(data[data.dbscan == id_]) < threshold:
                data = data[data.dbscan != id_]
    
    return data


if __name__ == '__main__':
    main()
