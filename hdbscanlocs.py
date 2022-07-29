# Specific script name.
SCRIPT_NAME = 'hdbscanlocs.py'

# Specify script version.
VERSION = 1.0

from glob import glob
import argparse
import os
import pandas as pd
from hdbscan import HDBSCAN
import locsutil as lu
import time
import random

MESSAGE = f"""
%s version %s. Requires a path to DNA-PAINT localization file (HDF5) or a directory containing
localization files. Optionally takes in the HDBSCAN parameters.
Returns segmented DNA-PAINT localization file(s).
""" % (SCRIPT_NAME, VERSION)


def main():
    """
    Segments DNA-PAINT localization with HDBSCAN.

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
    required_named.add_argument('-f', '--input', required=True, action='store',
                                type=str, help='Input file name or directory name which includes input files.')

    # Optional parameters to filter out bad localizations.
    user_input.add_argument('-p', '--pixel', action='store',
                        type=float, default=65.0, help='Camera pixel size in nm scale, default = 65 nm.')
    user_input.add_argument('-c', '--minSize', action='store',
                        type=int, default=40, help='HDBSCAN parameter: min_cluster_size.')
    user_input.add_argument('-s', '--minSamples', action='store',
                        type=int, default=None, help='HDBSCAN parameter: min_samples, default=None.')
    user_input.add_argument('-e', '--eps', action='store',
                        type=float, default=0, help='HDBSCAN parameter: cluster_selection_epsilon, default=0.')
    user_input.add_argument('-m', '--memory', action='store_true',
                        default=False, help='HDBSCAN parameter: use memory or not, default=False.')
    
    args = user_input.parse_args()
    input_path = args.input
    pixel_size = args.pixel
    min_cluster_size = args.minSize
    min_samples = args.minSamples
    cluster_selection_epsilon = args.eps
    if args.memory:
        _MEMORY = 'cache'
    else:
        _MEMORY = None
    
    # Starts a timer.
    start = time.time()
    
    # Performs HDBSCAN iteratively if input is a path to a directory.
    if os.path.isdir(input_path):
        query = os.path.join(input_path, '*.hdf5')
        locs_files = glob(query)
        
        for file in locs_files:
            print('Processing...' + file)
            find_cluster(file, pixel_size, min_cluster_size, min_samples, cluster_selection_epsilon, _MEMORY)
    else:
        find_cluster(input_path, pixel_size, min_cluster_size, min_samples, cluster_selection_epsilon, _MEMORY)
    
    print(f'Total_time: {time.time() - start} [sec]')


def find_cluster(locs_file: str,
                 pixel_size: float,
                 min_cluster_size: int,
                 min_samples: int,
                 cluster_selection_epsilon: float,
                 memory: str
                 ) -> None:
    """
    Segments localization data into clusters using the HDBSCAN algorithm and outputs them.
    
    Parameters
    ----------
    locs_file : str
        A path to the DNA-PAINT data file
    pixel_size : float
        Camera pixel size (nm).
    min_cluster_size : int
        The minimum size of clusters;
        single linkage splits that contain fewer points than this will be
        considered points “falling out” of a cluster rather than a cluster
        splitting into two new clusters.
    min_samples : int, optional (default = None)
        The number of samples in a neighbourhood for a point to be considered a core point.
    cluster_selection_epsilon : float, optional (default = None)
        A distance threshold. Clusters below this value will be merged.
    memory : str
        A path to the caching directory if a string is given.

    Returns
    -------
    None

    """
    
    # Imports files.
    data = lu.read_locs(locs_file)
    yaml_in = locs_file.rstrip('hdf5') + 'yaml'
    frame_val, height_val, width_val = lu.read_yaml(yaml_in)
    
    # Applies HDBSCAN.
    out_data = hdbscan_locs(data, pixel_size, min_cluster_size, min_samples, cluster_selection_epsilon, memory)
    
    # Shuffles hdbscan ids. This helps to assign different colors to spatially close segments
    # when segments are visualized with Picasso Render.
    unique_ids = out_data['hdbscan'].unique().tolist()
    random.seed(0)  # Fixes random seed.
    unique_ids_randomized = random.sample(unique_ids, len(unique_ids))
    new_ids = dict(zip(unique_ids, unique_ids_randomized))
    out_data['hdbscan'] = [new_ids[i] for i in out_data['hdbscan']]
    
    # Makes an output directory.
    work_dir = os.path.dirname(locs_file)
    out_path = os.path.join(work_dir, 'hdbscan', f'hdbscan_{min_cluster_size}_{min_samples}_{cluster_selection_epsilon}')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # Generates output file name prefix.
    file_base = os.path.basename(locs_file)
    file_stem = lu.clean_filename(file_base)
    out_name = f'{file_stem}_hdbscan_{min_cluster_size}_{min_samples}_{cluster_selection_epsilon}'
    out_hdf5_name = os.path.join(out_path, out_name) + '.hdf5'
    out_yaml_name = os.path.join(out_path, out_name) + '.yaml'
    
    # Outputs files.
    lu.write_locs(out_data, out_hdf5_name)
    lu.write_yaml(frame_val, height_val, width_val, out_yaml_name)


def hdbscan_locs(locs_data: pd.DataFrame,
                 pixel_size: float,
                 min_cluster_size: int,
                 min_samples: int,
                 cluster_selection_epsilon: float,
                 memory: str,
                 ) -> pd.DataFrame:
    """
    Apply the HDBSCAN algorithm to 3D DNA-PAINT localization data.

    Parameters
    ----------
    locs_data : pd.DataFrame
        Localization data of 3D DNA-PAINT, should have 'z' column.
    pixel_size : float
        Camera pixel size (nm).
    min_cluster_size : int
        The minimum size of clusters;
        single linkage splits that contain fewer points than this will be
        considered points “falling out” of a cluster rather than a cluster
        splitting into two new clusters.
    min_samples : int, optional (default = None)
        The number of samples in a neighbourhood for a point to be considered a core point.
    cluster_selection_epsilon : float, optional (default = None)
        A distance threshold. Clusters below this value will be merged.
    memory : str
        A path to the caching directory if a string is given.

    Returns
    -------
    data : pd.DataFrame
        Segmented localization data, the 'hdbscan' column added


    """
    
    data_col_linked = [
        'frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg',
        'lpx', 'lpy', 'ellipticity', 'net_gradient', 'z',
        'd_zcalib', 'len', 'n', 'photon_rate', 'hdbscan'
    ]

    data_col_unlinked = [
        'frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg',
        'lpx', 'lpy', 'ellipticity', 'net_gradient', 'z',
        'd_zcalib', 'hdbscan'
    ]
    
    # Sets HDBSCAN parameters
    if memory is not None:
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_epsilon=cluster_selection_epsilon,
                            memory=memory)
    else:
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_epsilon=cluster_selection_epsilon)
    
    data = locs_data.copy()
    
    # Converts xyz coordinates to numpy.array
    xyz = lu.hdf2xyz(data, pixel_size)
    
    # Applies HDBSCAN
    clusterer.fit(xyz)
    labels = clusterer.labels_
    prob = clusterer.probabilities_
    
    # Merges the DBSCAN labels into the original DataFrame
    data['hdbscan'] = pd.Series(labels)
    data['hdbscan_prob'] = pd.Series(prob)
    
    # Checks if locs_data is linked
    if 'len' in data.columns:
        data = data[data_col_linked]
    else:
        data = data[data_col_unlinked]
    
    # Drops 'noise' localizations
    data = data[data.hdbscan > -1]
    
    return data


if __name__ == '__main__':
    main()
