# Specific script name.
SCRIPT_NAME = 'locsutil.py'

# Specify script version.
VERSION = 1.0

import pandas as pd
import h5py
import re
import numpy as np


# Toolbox to handle localization files and SABER masks for Decode-PAINT.


def get_hdf_list() -> list:
    """
    Finds all HDF5 files in the directory and returns a list of them.
    
    Returns
    -------
    list
        List of .hdf5 files found in the directory, sorted.

    """
    
    from glob import glob
    
    return sorted(glob('*.hdf5'))


def read_locs(locs_file: str) -> pd.DataFrame:
    """
    Reads a Picasso-format DNA-PAINT localization file and returns as pd.DataFrame.
    
    Parameters
    ----------
    locs_file : str
        Path to DNA-PAINT localization file (HDF5 format).

    Returns
    -------
    data : pd.DataFrame
        Localization data sorted by 'frame'.

    """
    
    # Reads DNA-PAINT localization file
    data = pd.read_hdf(locs_file, key='locs', ignore_index=True)
    
    data = data.sort_values(by=['frame'])
    data = data.reset_index(drop=True)
    
    return data


def write_locs(locs_data: pd.DataFrame, out_path: str) -> None:
    """
    Writes localization data as a Picasso-format DNA-PAINT localization file.
    
    Parameters
    ----------
    locs_data : pd.DataFrame
    out_path : str

    Returns
    -------
    None

    """
    
    data_out = locs_data.copy()
    
    # Converts to original types
    data_out['frame'] = data_out['frame'].astype('uint32')
    data_out[data_out.columns[1:]] \
        = data_out[data_out.columns[1:]].astype('float32')
    
    if 'len' in data_out.columns:
        data_out['len'] = data_out['len'].astype('uint32')
    if 'n' in data_out.columns:
        data_out['n'] = data_out['n'].astype('uint32')
    if 'dbscan' in data_out.columns:
        data_out['dbscan'] = data_out['dbscan'].astype('uint32')
    if 'hdbscan' in data_out.columns:
        data_out['hdbscan'] = data_out['hdbscan'].astype('uint32')
    
    # Converts data for export
    rec_data = data_out.to_records(index=False)
    
    # Writes to hdf5 file
    with h5py.File(out_path, 'w') as locs_file:
        locs_file.create_dataset('locs', data=rec_data)


def read_yaml(yaml_file: str) -> None:
    """
    Reads a YAML file associated with a Picasso-format DNA-PAINT
    localization file and returns its information.
    
    Parameters
    ----------
    yaml_file : str
        Path to YAML file

    Returns
    -------
    frames : str
    height : str
    width : str

    """
    
    with open(yaml_file) as f:
        contents = f.readlines()
    
    for row in contents:
        if row.split(':')[0] == 'Frames':
            frames = row.split(':')[1].strip()
        elif row.split(':')[0] == 'Height':
            height = row.split(':')[1].strip()
        elif row.split(':')[0] == 'Width':
            width = row.split(':')[1].strip()
    
    try:
        return frames, height, width
    
    except UnboundLocalError:
        print('At least one of the image details is missing in in the YAML file.')


def write_yaml(frames: str,
               height: str,
               width: str,
               out_yaml_name: str
               ) -> None:
    """
    Writes a YAML file.
    
    Parameters
    ----------
    frames : str
    height : str
    width : str
    out_yaml_name : str

    Returns
    -------
    None

    """
    
    parameters = \
        f'''Byte Order: <
Data Type: uint16
Frames: {frames}
Height: {height}
Width: {width}
'''
    with open(out_yaml_name, 'w') as f:
        f.write(parameters)


def clean_locs(locs_data: pd.DataFrame,
               max_dzcalib: float,
               max_lp: float,
               min_z: float,
               max_z: float,
               max_photon: int
               ) -> pd.DataFrame:
    """
    Filters out 'bad' localizations from 3D DNA-PAINT data.
    
    Parameters
    ----------
    locs_data : pd.DataFrame
        Localization data of 3D DNA-PAINT, should have 'z' and 'd_zcalib' columns.
    max_dzcalib : float
    max_lp : float
    min_z : float
    max_z : float
    max_photon : int

    Returns
    -------
    data_out : pd.DataFrame
        Localization data of 3D DNA-PAINT, cleaned.

    """
    
    data_out = locs_data.copy()
    
    if max_dzcalib != 0:
        data_out = data_out[data_out['d_zcalib'] <= max_dzcalib]
    
    if max_lp != 0:
        data_out = data_out[data_out['lpx'] <= max_lp]
        data_out = data_out[data_out['lpy'] <= max_lp]
    
    if min_z != -9999.9:
        data_out = data_out[data_out['z'] >= min_z]
    
    if max_z != 9999.9:
        data_out = data_out[data_out['z'] <= max_z]
    
    if max_photon != -1:
        data_out = data_out[data_out['photons'] <= max_photon]
    
    return data_out


def clean_filename(file_name: str) -> str:
    """
    Removes some keywords from a file name.
    
    Parameters
    ----------
    file_name : str

    Returns
    -------
    new_name : str

    """
    new_name = file_name.replace('_render', '')
    new_name = new_name.replace('_arender', '')
    new_name = new_name.replace('_linked', '')
    new_name = new_name.replace('_filtered', '')
    new_name = new_name.replace('_cropped', '')
    new_name = new_name.replace('_corrected', '')
    new_name = new_name.replace('.hdf5', '')
    new_name = new_name.replace('.yaml', '')
    new_name = re.sub('_dbscan.*', '', new_name)
    
    return new_name


def hdf2xyz(locs_data: pd.DataFrame, pixel_size: float) -> np.array:
    """
    Converts localization data to simple xyz coordinates
    
    Parameters
    ----------
    locs_data : pd.DataFrame
        Localization data of 3D DNA-PAINT, should have 'z' column.
    pixel_size : float
        Camera pixel size (nm).

    Returns
    -------
    data[['new_x', 'new_y', 'z']].to_numpy() : numpy.array
    

    """
    data = locs_data.copy()
    data['new_x'] = data['x'].astype('float') * pixel_size
    data['new_y'] = data['y'].astype('float') * pixel_size
    
    return data[['new_x', 'new_y', 'z']].to_numpy()


def convert_xy(locs_data: pd.DataFrame, pixel_size: float) -> pd.DataFrame:
    """
    Converts xy coordinates to nanometer scale
    
    Parameters
    ----------
    locs_data : pd.DataFrame
    pixel_size
    
    Returns
    -------
    converted : pd.DataFrame

    """
    converted = locs_data.copy()
    
    converted['x_nm'] = converted['x'] * pixel_size
    converted['y_nm'] = converted['y'] * pixel_size
    converted['sx_nm'] = converted['sx'] * pixel_size
    converted['sy_nm'] = converted['sy'] * pixel_size
    converted['lpx_nm'] = converted['lpx'] * pixel_size
    converted['lpy_nm'] = converted['lpy'] * pixel_size
    
    return converted


def voxelize(locs_data: pd.DataFrame, bin_xy: float, bin_z: float) -> pd.DataFrame:
    """
    Voxelizes localization data using given bin sizes.
    
    Parameters
    ----------
    locs_data : pd.DataFrame
    bin_xy : float
    bin_z : float

    Returns
    -------
    voxelized : pd.DataFrame
    
    """
    
    voxelized = locs_data.copy()
    
    voxelized['x_vox'] = (voxelized['x_nm'] // bin_xy)
    voxelized['y_vox'] = (voxelized['y_nm'] // bin_xy)
    voxelized['z_vox'] = (voxelized['z'] // bin_z)
    
    return voxelized


def cam_pixelize(locs_data: pd.DataFrame) -> pd.DataFrame:
    """
    Rounds up 'x' and 'y' to pixelize the coordinate.
    
    Parameters
    ----------
    locs_data : pd.DataFrame

    Returns
    -------
    locs_data : pd.DataFrame

    """
    locs_data['x_px'] = locs_data['x'].astype(int)
    locs_data['y_px'] = locs_data['y'].astype(int)
    
    return locs_data


def get_status(file_base: str) -> str:

    file_info = file_base.strip('.hdf5').split('_')

    if 'xa' in file_info:
        status = 'active'
    elif 'xi' in file_info:
        status = 'inactive'
    else:
        status = 'unknown'

    return status
