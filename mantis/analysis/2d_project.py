import os

import numpy as np
import tifffile

raw_data_path = r'D:\2023_02_20_LS_beads\deskew'
processed_data_path = r'D:\2023_02_20_LS_beads\deskew'
datasets = [i for i in os.listdir(raw_data_path) if 'beads' in i]

for dataset in datasets:
    # Load data
    data = tifffile.imread(os.path.join(raw_data_path, dataset, dataset + '.tif'))

    proj = np.amax(data, axis=0)

    os.remove(os.path.join(processed_data_path, dataset, dataset + '_max_proj.tif'))
    tifffile.imwrite(
        os.path.join(processed_data_path, dataset, dataset + '_max_proj.tif'), proj
    )
