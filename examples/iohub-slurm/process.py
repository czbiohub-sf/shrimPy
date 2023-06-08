"""Demo for FOV processing on a single node. Only works for small arrays."""

import argparse
import os

import numpy as np

from iohub.ngff import ImageArray, open_ome_zarr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="input store path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output store path",
    )
    return parser.parse_args()


def process(image: ImageArray):
    return np.zeros_like(image) + int(os.environ.get("SLURM_JOB_ID"))


def main():
    args = parse_args()
    with open_ome_zarr(args.input) as input_fov:
        with open_ome_zarr(
            args.output,
            mode="w",
            layout="fov",
            channel_names=input_fov.channel_names,
            axes=input_fov.axes,
        ) as output_fov:
            output_fov["0"] = process(input_fov["0"])


if __name__ == "__main__":
    main()
