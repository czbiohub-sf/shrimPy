#%%
import os

import napari
import numpy as np
import tifffile

from dexp.processing.deskew import yang_deskew

#%%

raw_data_path = r'D:\2023_02_16_LS_argolight'
processed_data_path = r'D:\2023_02_16_LS_argolight\deskew'
datasets = [i for i in os.listdir(raw_data_path) if 'rings' in i]

for dataset in datasets:
    # Load data
    data = tifffile.imread(
        os.path.join(raw_data_path, dataset, dataset + '_MMStack_Pos0.ome.tif')
    )

    # Deskew
    deskew = yang_deskew(
        image=data,
        depth_axis=0,
        lateral_axis=1,
        flip_depth_axis=True,
        dx=0.116,
        dz=0.333,
        angle=30.0,
    )

    proj = np.amax(deskew, axis=0)

    # Save data
    save_dir = os.path.join(processed_data_path, dataset)
    os.makedirs(save_dir)

    tifffile.imwrite(os.path.join(save_dir, dataset + '.tif'), deskew)
    tifffile.imwrite(os.path.join(save_dir, dataset + '_max_proj.tif'), proj)
