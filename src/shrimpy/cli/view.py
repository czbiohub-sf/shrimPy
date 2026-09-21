"""``shrimpy view``: open an acquisition's OME-Zarr in the shrimPy napari viewer."""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.argument(
    "path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--deskew/--no-deskew",
    default=False,
    show_default=True,
    help=(
        "Offer deskewed display, for oblique-plane light-sheet data (e.g. the mantis "
        "light-sheet arm). Uses the scan step and pixel size recorded in the store."
    ),
)
def view(path: Path, deskew: bool):
    """View the acquisition stored at PATH (a ``.ome.zarr`` directory).

    The store may still be open for writing, so this also attaches to an acquisition
    already in progress -- from another terminal, or another machine that can see the
    same directory.

    Example:

        shrimpy view ./data/my_experiment_1.ome.zarr --deskew
    """
    from shrimpy.viewer._napari_process import run_viewer

    run_viewer(path, deskew=deskew)
