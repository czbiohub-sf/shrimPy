import argparse
import os
from glob import glob
import logging

from iohub.ngff import Plate, open_ome_zarr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="temp fovs path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output store path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fov_paths = glob(os.path.join(args.input, "*/*/*"))
    fovs = dict(
        (path[len(args.input) :], open_ome_zarr(path)) for path in fov_paths
    )
    logging.info(repr(fovs))
    _ = Plate.from_positions(args.output, fovs)


if __name__ == "__main__":
    main()
