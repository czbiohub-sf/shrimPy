from typing import List

import click

from iohub import open_ome_zarr
from iohub.ngff_meta import TransformationMeta

from mantis.cli.parsing import input_position_dirpaths


@click.command()
@input_position_dirpaths()
def update_scale_metadata(
    input_position_dirpaths: List[str],
):
    with open_ome_zarr(input_position_dirpaths[0]) as input_dataset:
        print(
            f"The first dataset in the list you provided has (t, c, z, y, x) scale {input_dataset.scale}"
        )

    print(
        "Please enter the new t, c, z, y, and x scales that you would like to apply to all of the positions in the list."
    )
    print(
        "The old scale will be saved in a metadata field named 'old_scale', and the new scale will adhere to the NGFF spec."
    )
    new_scale = []
    for character in "tczyx":
        new_scale.append(float(input(f"Enter a new {character} scale: ")))

    for input_position_dirpath in input_position_dirpaths:
        with open_ome_zarr(input_position_dirpath, layout="fov", mode="a") as input_dataset:
            input_dataset.zattrs['old_scale'] = input_dataset.scale
            transform = [TransformationMeta(type="scale", scale=new_scale)]
            input_dataset.set_transform("0", transform=transform)
            input_dataset.set_transform("*", transform=transform)

    print(f"The dataset now has (t, c, z, y, x) scale {new_scale}.")
