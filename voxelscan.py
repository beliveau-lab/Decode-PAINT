# Specific script name.
SCRIPT_NAME = 'voxelscan.py'

# Specify script version.
VERSION = 1.0

import argparse
import pandas as pd
import os
import locsutil as lu
from datetime import datetime
import time

COLUMNS = ['file', 'expdate', 'roi', 'subroi',
            'status', 'id', 'total_locs', 'clusters', 'vox_bins',
            'total_vol', 'total_surf', 's_to_v',
            'core_ratios', 'sigmoid_params']

MESSAGE = f"""
%s version %s. Requires a path to a directory containing Dedode-PAINT localization files to be analyzed.
Optionally takes in other parameters. Returns a result table (CSV).
""" % (SCRIPT_NAME, VERSION)


def main():
    """
    
    Parameters (user input)
    -------
    dir : str
    
    pixel :
    binsize_min :
    binsize_max :
    
    cutoff : int, optional
    output : str, optional
    
    Returns
    -------
    
    """

    # Allows user to input parameters on command line.
    user_input = argparse.ArgumentParser(description=MESSAGE)
    
    # Inputs the directory name.
    required_named = user_input.add_argument_group('required arguments')
    required_named.add_argument('-d', '--dir', action='store', required=True, type=str,
                                help='Input directory name which contains locs data to be analyzed.')

    # Optional parameters
    user_input.add_argument('-p', '--pixel', action='store', type=float, default=65.0,
                            help='Camera pixel size in nm scale, default = 65 nm.')
    user_input.add_argument('-m', '--binsize_min', action='store', type=int, default=10,
                            help='The minimum size of the bin window.')
    user_input.add_argument('-M', '--binsize_max', action='store', type=int, default=130,
                            help='The maximum size of the bin window.')
    user_input.add_argument('-s', '--binsize_step', action='store', type=int, default=5,
                            help='The step size of the bin scanning.')
    user_input.add_argument('-c', '--cutoff', action='store', default=500, type=int,
                            help='Localization number cutoff.')
    user_input.add_argument('-o', '--output', action='store', type=str, default='analyzed',
                            help='The output file name postfix.')
    
    args = user_input.parse_args()
    input_path = args.dir
    pixel_size = args.pixel
    bin_min = args.binsize_min
    bin_max = args.binsize_max
    bin_step = args.binsize_step
    cutoff = args.cutoff
    out_postfix = args.output

    # Count process time
    start = time.time()

    # Find data files
    os.chdir(input_path)
    hdf5_list = lu.get_hdf_list()
    print(hdf5_list)
    
    all_data = pd.DataFrame(columns=COLUMNS)

    # Analyze each file and generate output table
    for item in hdf5_list:
        print('Analyzing...' + item)

        analyzed = analyze_vox(item, bin_min, bin_max, bin_step, pixel_size, cutoff)
        all_data = pd.concat([all_data, analyzed], ignore_index=True)
        print(f'Elapsed_time: {time.time() - start} [sec]')

    # Outputs table.
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    outname = f'{out_postfix}_co{cutoff}_{now}.csv'
    all_data = all_data[COLUMNS]
    all_data.to_csv(outname)

    print(f'Total_time: {time.time() - start} [sec]')
    

def analyze_vox(file: str, bin_min: int, bin_max: int, step: int, pixel_size: float, cutoff: int):
    
    data = lu.read_locs(file)
    data = lu.convert_xy(data, pixel_size)
    
    if 'id' in data.columns.values:
        data = data[data['id'] != -1]
    else:
        data['id'] = -999
    
    if 'hdbscan' not in data.columns.values:
        data['hdbscan'] = 0
    
    analyzed = _analyze_vox(data, bin_min, bin_max, step, cutoff)

    analyzed['file'] = file
    analyzed['status'] = lu.get_status(file)
    
    return analyzed


def _analyze_vox(locs: pd.DataFrame, bin_min: int, bin_max: int, step: int, cutoff: int):
    
    _id = []
    total_locs = []
    num_clusters = []
    bins = []
    vols = []
    surfs = []
    stovs = []
    core_ratios = []
    params = []
    
    for i, locs_group in locs.groupby('id'):
        
        if len(locs_group) < cutoff:
            pass
        
        else:
            _id += [i]
            
            total_locs += [len(locs_group)]
            num_clusters += [len(locs_group['hdbscan'].unique())]
            bin_list, vol_list, surf_list, stov_list, core_ratio_list = voxel_scan(locs_group, bin_min, bin_max, step)
            bins += [int2str(bin_list)]
            vols += [int2str(vol_list)]
            surfs += [int2str(surf_list)]
            stovs += [int2str(stov_list)]
            core_ratios += [int2str(core_ratio_list)]
            
            try:
                popt, _ = lu.fit_sigmoid(bin_list, core_ratio_list)
                params += [int2str([popt[0], popt[1], popt[2]])]
                
            except RuntimeError:
                print('RuntimeErrorException: Fitting failed')
    
    ret_df = pd.DataFrame(
        data={'id': _id, 'total_locs': total_locs, 'clusters': num_clusters, 'vox_bins': bins,
              'total_vol': vols, 'total_surf': surfs, 's_to_v': stovs,
              'core_ratios': core_ratios, 'sigmoid_params': params},
    )
    
    return ret_df


def voxel_scan(locs: pd.DataFrame, bin_min: int, bin_max: int, step: int):
    
    bin_list = []
    vol_list = []
    surf_list = []
    stov_list = []
    core_ratio_list = []
    
    for bin in range(bin_min, bin_max, step):
        data = locs.copy()
        data = lu.voxelize(data, bin, bin)
        calculated, total_vox, surf_vox, xy_faces, z_faces = lu.count_exposed_faces(data)
        
        bin_list.append(bin)
        
        total_vol = total_vox * (bin ** 3)
        total_surf = (xy_faces + z_faces) * (bin ** 2)
        normalized_stov = total_surf / (total_vol ** (2 / 3))
        
        vol_list.append(total_vol)
        surf_list.append(total_surf)
        stov_list.append(normalized_stov)
        core_ratio_list.append((total_vox - surf_vox) / total_vox)
        
        print(f'Bin size: {bin} nm, Core vox ratio: {(total_vox - surf_vox) / total_vox}')
    
    return bin_list, vol_list, surf_list, stov_list, core_ratio_list


def int2str(in_list: list) -> list:
    
    return ",".join([str(i) for i in in_list])


if __name__ == '__main__':
    main()
