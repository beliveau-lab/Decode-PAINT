# Specific script name.
SCRIPT_NAME = 'decode.py'

# Specify script version.
VERSION = 1.0

import argparse
import pandas as pd
import os
import numpy as np
from PIL import Image, ImageSequence
import locsutil as lu
import time

MESSAGE = f"""
%s version %s. Requires a path to segmented DNA-PAINT localization file (HDF5) and SABER pre-decoding
binary mask file (tiff). Optionally takes in DAPI mask and other parameters.
Returns decoded DNA-PAINT localization file(s).
""" % (SCRIPT_NAME, VERSION)


def main():
    """
    Decodes DNA-PAINT localization with SABER pre-decoding masks.
    
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
                                help='Input DNA-PAINT locs segmented with HDBSCAN/DBSCAN.')
    required_named.add_argument('-m', '--mask', action='store', required=True, type=str,
                                help='Input SABER signal binary masks as a tiff stack.')

    # Optional parameters to filter out bad localizations.
    user_input.add_argument('-d', '--dapi', action='store', type=str,
                            help='Input a DAPI signal binary mask as a single tiff.')
    user_input.add_argument('-c', '--clustercol', action='store', type=str, default='hdbscan',
                            help='Specify cluster id col name, default=hdbscan')
    user_input.add_argument('-o', '--output', action='store', type=str,
                            help='Specify output file prefix')

    args = user_input.parse_args()
    input_locs = args.input
    input_mask = args.mask
    input_dapi = args.dapi
    cluster_col = args.clustercol
    out_name = args.output
    
    # Executes decoding
    decode_paint(input_locs, input_mask, input_dapi, out_name, cluster_col)


def decode_paint(locs_file: str, mask_file: str, dapi_file=None, out_name=None, cluster_col='hdbscan') -> None:
    """
    Decodes DNA-PAINT localization with SABER pre-decoding masks.
    
    Parameters
    ----------
    locs_file : str
    mask_file : str
    dapi_file : str
    out_name : str
    cluster_col : str

    Returns
    -------
    None

    """
    
    # Counts process time
    start = time.time()
    
    # Imports data and a yaml file
    data = lu.read_locs(locs_file)
    masks = read_masks(mask_file)
    
    yaml_in = locs_file.rstrip('hdf5') + 'yaml'
    frame_val, height_val, width_val = lu.read_yaml(yaml_in)
    
    # Filters out clusters outside nucleus based on DAPI signal if specified
    if dapi_file:
        nucl_mask = read_masks(dapi_file)[0]
        data = filter_clusters(data, nucl_mask, cluster_col)
        print(f'Elapsed_time: {time.time() - start} [sec]')
    
    # Gets working directory path
    work_dir = os.path.dirname(locs_file)
    out_path = os.path.join(work_dir, 'decoded')

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # Generates output file name prefix
    file_base = os.path.basename(locs_file)
    file_stem = lu.clean_filename(file_base)
    
    if out_name:
        out_name = f'{file_stem}_{out_name}'
    else:
        out_name = f'{file_stem}_decoded'
    
    out_hdf5_name = os.path.join(out_path, out_name) + '.hdf5'
    out_yaml_name = os.path.join(out_path, out_name) + '.yaml'
    
    # Decodes clusters
    decoded = decode_clusters(data, masks, cluster_col)
    decoded.dropna(inplace=True)
    
    # Outputs decoded files
    lu.write_locs(decoded, out_hdf5_name)
    lu.write_yaml(frame_val, height_val, width_val, out_yaml_name)


def read_masks(im_stack: str) -> list:
    """
    Parameters
    ----------
    im_stack : str
        A path to SABER pre-decoding masks

    Returns
    -------
    masks : list of np.array
        8-bit SABER pre-decoding image stack

    """
    im = Image.open(im_stack)
    masks = []
    for i, page in enumerate(ImageSequence.Iterator(im)):
        masks.append(np.array(page))
    
    return masks


def get_mask_val(masks: list, slice: int, coordinate: tuple) -> int:
    """
    Parameters
    ----------
    masks : list
    slice : int
    coordinate : tuple

    Returns
    -------
    the pixel intensity : int

    """
    mask = masks[slice]
    
    return get_pixel_val(mask, coordinate)


def get_pixel_val(image: np.array, coordinate: tuple) -> int:
    """
    Parameters
    ----------
    image : np.array
    coordinate : tuple

    Returns
    -------
    the pixel intensity : int

    """
    
    try:
        return image[coordinate[1], coordinate[0]]
    
    except IndexError:
        print(f'Out of bounds: {coordinate}')
        return 0


def filter_clusters(locs: pd.DataFrame, nucl_mask: np.array, cluster_col: str) -> pd.DataFrame:
    """
    Filters out clusters that don't overlap with nucl_mask.
    
    Parameters
    ----------
    locs : pd.DataFrame
    nucl_mask : np.array
    cluster_col : str

    Returns
    -------
    data without outside nucleus locs : pd.DataFrame

    """
    process_start = time.time()
    
    out_locs = pd.DataFrame()
    
    print('Filtering out clusters outside nucleus...')
    
    unique_dbscan = len(locs[cluster_col].unique())
    counter = 0
    
    for dbscan, locs_group in locs.groupby(cluster_col):
        min_x = int(locs_group['x'].min())
        max_x = int(locs_group['x'].max())
        min_y = int(locs_group['y'].min())
        max_y = int(locs_group['y'].max())
        
        top_left = get_pixel_val(nucl_mask, (min_x, min_y))
        top_right = get_pixel_val(nucl_mask, (max_x, min_y))
        bottom_left = get_pixel_val(nucl_mask, (min_x, max_y))
        bottom_right = get_pixel_val(nucl_mask, (max_x, max_y))
        
        if top_left > 0 or top_right > 0 or bottom_left > 0 or bottom_right > 0:
            out_locs = pd.concat([out_locs, locs_group], ignore_index=True)
        
        counter += 1
        if counter % 1000 == 0 or counter == unique_dbscan:
            elapsed_time = time.time() - process_start
            print(f'DBSCAN processed: {counter}/{unique_dbscan},'
                  f'process time: {elapsed_time} [sec]')
    
    print('Filtering done')
    
    return out_locs


def decode_clusters(locs: pd.DataFrame, masks: list, cluster_col: str) -> pd.DataFrame:
    """
    Returns decoded data.
    
    Parameters
    ----------
    locs : pd.DataFrame
    masks : list
    cluster_col : str

    Returns
    -------
    decoded locs : pd.DataFrame

    """
    
    process_start = time.time()
    print('Decoding clusters...')
    
    out_locs = pd.DataFrame()
    unique_clusters = len(locs[cluster_col].unique())
    counter = 0
    
    for _, locs_group in locs.groupby(cluster_col):
        
        locs_group['id'] = decode_ids(locs_group, masks)
        out_locs = pd.concat([out_locs, locs_group], ignore_index=True)
        
        counter += 1
        if counter % 1000 == 0 or counter == unique_clusters:
            print(f'DBSCAN decoded: {counter}/{unique_clusters},'
                  f'Process time: {time.time() - process_start} [sec]')
    
    return out_locs


def decode_ids(cluster: pd.DataFrame, masks: list):
    """
    Decodes a h/dbscaned cluster based on overlap scores.
    
    Parameters
    ----------
    cluster : pd.DataFrame
    masks : list

    Returns
    -------
    decoded : int or np.nan

    """
    # Input form: a group of locs sharing with the same 'h/dbscan' id
    mask_overlap = count_mask_overlap(cluster, masks)
    half_total_locs = len(cluster) / 2
    
    if mask_overlap.max() == 0: # Rejects if no overlap
        return np.nan
    
    elif mask_overlap.max() < half_total_locs: # Rejects if no more than 50% overlap
        return np.nan
    
    else:
        max_index = [i for i, x in enumerate(mask_overlap) if x == max(mask_overlap)]
        if len(max_index) == 1:
            return max_index[0] # Assigns id with
        else:
            return -1


def count_mask_overlap(cluster: pd.DataFrame, masks: list) -> int:
    """
    Counts the number of localizations overlapped with mask.
    
    Parameters
    ----------
    cluster : pd.DataFrame
    masks : list

    Returns
    -------
    number of locs overlapped with mask : int

    """
    # Input form: locs which have the same 'dbscan' id
    cluster = cluster[['x', 'y']].astype('uint32')
    cluster['count'] = 1
    
    len_masks = len(masks)
    mask_overlap = np.empty((0, len_masks), int)
    
    for key, selected in cluster.groupby(['x', 'y'], as_index=False):
        out_row = []
        for i in range(len_masks):
            # divide by 255 as binary masks have either 0 or 255 (8-bit)
            out_row.append(get_mask_val(masks, i, key) / 255 * len(selected))
        
        mask_overlap = np.append(mask_overlap, np.array([out_row]), axis=0)

    return mask_overlap.sum(axis=0)
    

if __name__ == '__main__':
    main()
