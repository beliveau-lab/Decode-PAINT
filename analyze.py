# Specific script name.
SCRIPT_NAME = 'analyze.py'

# Specify script version.
VERSION = 1.0

import locsutil as lu
import math
import os
import pandas as pd
import argparse
from datetime import datetime
import time

COLUMNS = ['file', 'status', 'id', 'total_locs', 'com_x', 'com_y', 'com_z',
           'global_com_x', 'global_com_y', 'global_com_z',
           'rel_com_x', 'rel_com_y', 'rel_com_z', 'distance',
           'total_vox', 'total_surf_vox', 'vol_vox', 'surf_area', 'rg']

MESSAGE = f"""
%s version %s. Requires a path to a directory containing Dedode-PAINT localization files to be analyzed.
Optionally takes in other parameters. Returns a result table (CSV).
""" % (SCRIPT_NAME, VERSION)


def main():
    """
    Analyzes the geometry of decoded chromosome data.

    Parameters (user input)
    -------
    dir : str
    pixel : float, optional
    binsize_xy : float, optional
    binsize_z : float, optional
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
                                help='Input directory name which contains locs data to be analyzed. '
                                     'Each data name should include either "xa" or "xi" for a status specifier. '
                                     'E.g., 210513_roi2_xi.hdf5')

    # Optional parameters
    user_input.add_argument('-p', '--pixel', action='store', type=float, default=65.0,
                            help='Camera pixel size in nm scale, default = 65 nm.')
    user_input.add_argument('-x', '--binsize_xy', action='store', type=float, default=50.0,
                            help='The dimension of the XY bin.')
    user_input.add_argument('-z', '--binsize_z', action='store', type=float, default=50.0,
                            help='The dimension of the Z bin.')
    user_input.add_argument('-o', '--output', action='store', type=str, default='analyzed',
                            help='The output file name postfix.')

    args = user_input.parse_args()
    input_path = args.dir
    pixel_size = args.pixel
    bin_xy = args.binsize_xy
    bin_z = args.binsize_z
    out_postfix = args.output

    # Starts a timer.
    start = time.time()

    # Finds locs data to be analyzed.
    os.chdir(input_path)
    hdf5_list = lu.get_hdf_list()
    print(hdf5_list)

    # Analyzes each file and generates output table.
    all_data = pd.DataFrame()

    for item in hdf5_list:
        print('Analyzing...' + item)

        analyzed = analyze_data(item, pixel_size, bin_xy, bin_z)
        all_data = pd.concat([all_data, analyzed], ignore_index=True)

    # Normalizes the coordinate based on the global center of mass of each data.
    all_data = normalize_coms(all_data)
    all_data = calc_dist(all_data)

    print(all_data)

    # Organizes table.
    all_data = all_data[COLUMNS]

    all_data['s_to_v'] = all_data['surf_area'] / (all_data['vol_vox'] ** (2/3))  # Dimension-less S/V
    all_data['density'] = all_data['total_locs'] / all_data['vol_vox'] * 1e9

    # Outputs table.
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    out_name = f'{out_postfix}_{int(bin_xy)}xy_{int(bin_z)}z_{now}.csv'
    all_data.to_csv(out_name)

    print(f'Total_time: {time.time() - start} [sec]')


def analyze_data(file, pixel_size: float, bin_xy: float, bin_z: float) -> pd.DataFrame:

    status = lu.get_status(file)
    data = lu.read_locs(file)
    data = data[data['id'] != -1]
    data = lu.convert_xy(data, pixel_size)
    data = lu.voxelize(data, bin_xy, bin_z)

    analyzed = analyze_clusters(data, bin_xy, bin_z)

    analyzed['file'] = file
    analyzed['status'] = status

    return analyzed


def analyze_clusters(locs: pd.DataFrame, bin_xy: float, bin_z: float) -> pd.DataFrame:

    decode_id = []
    total_locs = []
    com_x = []
    com_y = []
    com_z = []
    num_vox = []
    surf_vox = []
    vol_vox = []
    surf_area = []
    rg = []

    for i, locs_group in locs.groupby('id'):

        decode_id += [i]
        total_locs += [len(locs_group)]

        com = calc_com(locs_group)
        com_x += [com[0]]
        com_y += [com[1]]
        com_z += [com[2]]

        total_vox = 0.0
        total_surf_vox = 0.0
        total_xy_faces = 0.0
        total_z_faces = 0.0

        for dbscan, dbscan_group in locs_group.groupby('hdbscan'):
            calculated, vol, surf, xy_faces, z_faces = lu.count_exposed_faces(dbscan_group)

            total_vox += vol
            total_surf_vox += surf
            total_xy_faces += xy_faces
            total_z_faces += z_faces

        num_vox += [total_vox]
        surf_vox += [total_surf_vox]

        vol_vox += [total_vox * (bin_xy ** 2) * bin_z]
        surf_area += [total_xy_faces * bin_xy *
                      bin_z + total_z_faces * (bin_xy ** 2)]

        rg += [gyration_radius(locs_group, com)]

    ret_df = pd.DataFrame(
        data={'id': decode_id, 'total_locs': total_locs,
              'com_x': com_x, 'com_y': com_y, 'com_z': com_z,
              'total_vox': num_vox, 'total_surf_vox': surf_vox,
              'vol_vox': vol_vox, 'surf_area': surf_area, 'rg': rg},
    )

    return ret_df


def normalize_coms(analyzed: pd.DataFrame) -> pd.DataFrame:
    # Should contain the following columns: com_x, com_y, com_z

    global_coms = calc_global_coms(analyzed)
    merged = analyzed.merge(global_coms, on='file')

    merged['rel_com_x'] = merged['com_x'] - merged['global_com_x']
    merged['rel_com_y'] = merged['com_y'] - merged['global_com_y']
    merged['rel_com_z'] = merged['com_z'] - merged['global_com_z']

    return merged


def calc_global_coms(analyzed: pd.DataFrame) -> pd.DataFrame:
    # Should contain the following columns: com_x, com_y, com_z

    file_list = []
    mean_comx_list = []
    mean_comy_list = []
    mean_comz_list = []

    for key, grouped in analyzed.groupby('file'):
        file_list.append(key)

        mean_comx_list.append(grouped['com_x'].mean())
        mean_comy_list.append(grouped['com_y'].mean())
        mean_comz_list.append(grouped['com_z'].mean())

    return pd.DataFrame(
        data={'file': file_list,
              'global_com_x': mean_comx_list,
              'global_com_y': mean_comy_list,
              'global_com_z': mean_comz_list}
    )


def calc_com(locs: pd.DataFrame) -> list:

    com_x = locs['x_nm'].mean()
    com_y = locs['y_nm'].mean()
    com_z = locs['z'].mean()

    return [com_x, com_y, com_z]


def calc_dist(analyzed: pd.DataFrame) -> pd.DataFrame:
    # Should contain the following columns: com_x, com_y, com_z

    out_list = []

    for key, grouped in analyzed.groupby('file'):

        grouped.sort_values(by='id', inplace=True)
        grouped.reset_index(drop=True, inplace=True)

        for index, row in grouped.iterrows():
            try:
                if grouped.at[index, 'id'] + 1 == grouped.at[index + 1, 'id']:
                    _x1 = grouped.at[index, 'com_x']
                    _y1 = grouped.at[index, 'com_y']
                    _z1 = grouped.at[index, 'com_z']
                    _x2 = grouped.at[index + 1, 'com_x']
                    _y2 = grouped.at[index + 1, 'com_y']
                    _z2 = grouped.at[index + 1, 'com_z']
                    _distance = (
                        (_x2 - _x1) ** 2
                        + (_y2 - _y1) ** 2
                        + (_z2 - _z1) ** 2
                    ) ** 0.5
                    out_list.append([key, grouped.at[index, 'id'], _distance])
            except:
                pass

    df_dist = pd.DataFrame(out_list, columns=['file', 'id', 'distance'])

    return analyzed.merge(df_dist, on=['file', 'id'], how='outer')


def gyration_radius(locs: pd.DataFrame, com: list) -> float:

    data = locs.copy()

    data['r^2'] = (data['x_nm'] - com[0]) ** 2 \
        + (data['y_nm'] - com[1]) ** 2 \
        + (data['z'] - com[2]) ** 2

    return math.sqrt(data['r^2'].sum() / data.shape[0])


if __name__ == '__main__':
    main()
