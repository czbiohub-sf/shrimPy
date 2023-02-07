# %% import modules
import os
import numpy as np
from waveorder.io.reader import WaveorderReader
from dexp.datasets import ZDataset

# %% load data
dataset = 'sparse-beads_ls_ap-scan-TS_1'
raw_data_path = '/hpc/projects/comp_micro/rawdata/mantis/2022_08_23 LS alignment'
processed_data_path = '/hpc/projects/comp_micro/projects/mantis/2022_08_23 LS alignment'

wo_dataset = WaveorderReader(os.path.join(raw_data_path, dataset))
T, C, Z, Y, X = wo_dataset.shape

y0 = 1115
roi = [slice(y0, y0+256), slice(0, 2048)]

print(f'Datset shape: {(T, C, Z, Y, X)}')
# %% convert to ZDataset
za = wo_dataset.get_zarr() # due to bug, this bring data into memory I think later tifffile version fixes that
ds = ZDataset(os.path.join(processed_data_path, dataset+'.zarr'), mode="w-")

for c in range(C):
    channel_name = wo_dataset.channel_names[c]
    ds.add_channel(name=channel_name, shape=(T, Z, 256, 2048), dtype=np.uint16)
    for t in range(T):
        ds.write_stack(channel_name, t, za[:,c,:,roi[0],roi[1]])

# %% deskew