# Specific script name.
SCRIPT_NAME = 'voxelutil.py'

# Specify script version.
VERSION = 1.0

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Toolbox to handle voxel data.


def extract_cluster(df):
    
    tmp_df = df.copy()
    tmp_df['count'] = 1
    
    return tmp_df.groupby('hdbscan')['count'].sum()


def extract_surface(df):
    
    return df[df['total_faces'] != 0]


def extract_core(df):
    
    return df[df['total_faces'] == 0]


def count_exposed_faces(locs_group: pd.DataFrame):
    
    locs_group = locs_group[~locs_group.duplicated(
        subset=['x_vox', 'y_vox', 'z_vox'])]
    
    locs_group = locs_group[['x_vox', 'y_vox', 'z_vox']].astype('int')
    locs_group = auto_origin_df(locs_group)
    
    out_data = scan_faces_all(locs_group)
    out_data['xy_faces'] = out_data['x_faces'] + out_data['y_faces']
    out_data['total_faces'] = out_data['x_faces'] + \
                              out_data['y_faces'] + out_data['z_faces']
    
    total_vox = len(out_data)
    surf_vox = len(out_data[out_data['total_faces'] != 0])
    xy_faces = out_data['xy_faces'].sum()
    z_faces = out_data['z_faces'].sum()
    
    return out_data, total_vox, surf_vox, xy_faces, z_faces


def scan_faces_all(df: pd.DataFrame) -> pd.DataFrame:
    
    scanned = scan_faces(df, 'x_vox')
    scanned = scan_faces(scanned, 'y_vox')
    scanned = scan_faces(scanned, 'z_vox')
    
    return scanned


def scan_faces(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    
    if axis == 'x_vox':
        fixed_columns = ['y_vox', 'z_vox']
        diff_col = 'x_diff'
        face_col = 'x_faces'
    elif axis == 'y_vox':
        fixed_columns = ['x_vox', 'z_vox']
        diff_col = 'y_diff'
        face_col = 'y_faces'
    else:
        fixed_columns = ['x_vox', 'y_vox']
        diff_col = 'z_diff'
        face_col = 'z_faces'
    
    return scan_faces_1d(df, axis, fixed_columns, diff_col, face_col)


def scan_faces_1d(df: pd.DataFrame, axis: str, fixed_columns: list, diff_col: str, face_col: str) -> pd.DataFrame:
    """
    Scans the volume in the one-dimensional manner to determine if the two faces of a voxel
    facing each other are exposed or not.
    E.g. fixed_columns = ['y_vox', 'z_vox']
    Scans each yz-fixed column: (n, 0, 0), (n, 1, 0), (n, 2, 0), etc...

    """
    
    scanned = pd.DataFrame()
    
    for _, locs_group in df.groupby(fixed_columns, as_index=False):
        # Selects a column of voxels.
        
        locs_group.sort_values(by=axis, inplace=True)  # axis = 'x_vox': sorts in order of the x_vox values
        locs_group.reset_index(drop=True, inplace=True)
        locs_group[face_col] = 0  # face_col = 'x_faces': 0 = not exposed, 1 or 2 = exposed
        
        locs_group[diff_col] = locs_group[axis].diff()  # diff_col = 'x_diff': checks the continuity, 1 = continuous
        locs_group[diff_col].fillna(0, inplace=True)
        
        for index, row in locs_group.iterrows():
            if index == 0:
                locs_group.at[index, face_col] = 1  # The first face is always exposed
            if locs_group.at[index, diff_col] > 1:  # If the two voxels are not neighbors
                locs_group.at[index - 1, face_col] += 1
                locs_group.at[index, face_col] += 1
            if index == len(locs_group) - 1:
                locs_group.at[index, face_col] += 1  # The last face is always exposed
        
        scanned = pd.concat([scanned, locs_group], ignore_index=True)
        scanned.drop(columns=[diff_col], inplace=True)
    
    return scanned


def auto_origin_df(df: pd.DataFrame) -> pd.DataFrame:
    
    df['x_vox'] = df['x_vox'] - df['x_vox'].min() + 1
    df['y_vox'] = df['y_vox'] - df['y_vox'].min() + 1
    df['z_vox'] = df['z_vox'] - df['z_vox'].min() + 1
    
    return df


def fit_sigmoid(_x: list, _y: list):
    
    p0 = [max(_y), np.median(_x), 1]
    popt, pcov = curve_fit(sigmoid, _x, _y, method='dogbox', p0=p0)
    
    return popt, pcov


def sigmoid(x, L, x0, k):
    
    return L / (1 + np.exp(-k * (x - x0)))
