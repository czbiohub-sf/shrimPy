import multiprocessing as mp

from pathlib import Path
from typing import List

import click
import numpy as np
import yaml

from iohub import open_ome_zarr
from iohub.ngff_meta import TransformationMeta

from scipy.ndimage import affine_transform

from mantis.analysis.AnalysisSettings import RegistrationSettings
from mantis.cli import utils
from mantis.cli.parsing import config_filepath, input_position_dirpaths, output_dirpath

t, c, z, y, x = (1,) * 2 + (0.8, 0.7, 0.7)
NEW_SCALE = (t, c, z, y, x)


@click.command()
@input_position_dirpaths()
def update_scale_metadata(
    input_position_dirpaths: List[str],
):
    for input_path in input_position_dirpaths:
        with open_ome_zarr(input_path) as input_dataset:
            input_dataset.metadata.
            transform = [TransformationMeta(type="scale", scale=input_dataset.scale)]


#%%

t, c, z, y, x = (1,) * 2 + (0.8, 0.7, 0.7)
new_scale = (t, c, z, y, x)
print(f'new scale {new_scale}')

with open_ome_zarr(input_path, mode='r+') as input_dataset:
    new_scale = (1,) * 2 + (0.8, 0.7, 0.7)
    transform=[
        TransformationMeta(type="scale", scale=new_scale)
    ]
    # Apply the transform to one image array
    input_dataset.set_transform("0",transform)

with open_ome_zarr(input_path, mode='r+') as input_dataset:
    print(input_dataset.scale)
    print(input_dataset.metadata.dict())
# %%
