"""Tests for the FOV-selection worker's debug-artifact assembly.

`_assemble_debug_channels` stacks every reconstruction stage (deskew / phase /
VS volumes in 3D, plus the 2D projection + mask broadcast across Z) into one
`(C, Z, Y, X)` volume for the debug OME-Zarr. These pin the channel order, the
2D->3D broadcast, and the float cast without needing a GPU or iohub.
"""

from __future__ import annotations

import numpy as np

from shrimpy.fov_selection import worker


def _artifacts(nz=3, ny=4, nx=5):
    rng = np.random.default_rng(0)
    stacks = {
        "deskew": rng.normal(size=(nz, ny, nx)).astype(np.float32),
        "phase": rng.normal(size=(nz, ny, nx)).astype(np.float32),
        "nuclei": rng.normal(size=(nz, ny, nx)).astype(np.float32),
        "membrane": rng.normal(size=(nz, ny, nx)).astype(np.float32),
    }
    projections = {
        "nuclei": rng.normal(size=(ny, nx)).astype(np.float32),
        "membrane": rng.normal(size=(ny, nx)).astype(np.float32),
    }
    # Label masks as integers -> must be cast to float32 in the output.
    masks = {
        "nuclei": rng.integers(0, 7, size=(ny, nx)).astype(np.uint32),
        "membrane": rng.integers(0, 7, size=(ny, nx)).astype(np.uint32),
    }
    return {"stacks": stacks, "projections": projections, "masks": masks}


def test_assemble_channel_order_and_shape():
    names, czyx = worker._assemble_debug_channels(_artifacts())
    assert names == [
        "deskew",
        "phase",
        "nuclei_vs",
        "membrane_vs",
        "nuclei_projection",
        "membrane_projection",
        "nuclei_mask",
        "membrane_mask",
    ]
    assert czyx.shape == (8, 3, 4, 5)
    assert czyx.dtype == np.float32


def test_assemble_broadcasts_2d_across_z():
    art = _artifacts()
    names, czyx = worker._assemble_debug_channels(art)

    # The projection/mask channels are the same 2D plane on every Z slice.
    proj_idx = names.index("nuclei_projection")
    for z in range(czyx.shape[1]):
        np.testing.assert_array_equal(czyx[proj_idx, z], art["projections"]["nuclei"])

    # The 3D VS volume varies across Z (not broadcast).
    vs_idx = names.index("nuclei_vs")
    assert not np.array_equal(czyx[vs_idx, 0], czyx[vs_idx, 1])


def test_assemble_casts_mask_labels_to_float():
    art = _artifacts()
    names, czyx = worker._assemble_debug_channels(art)
    mask_idx = names.index("membrane_mask")
    np.testing.assert_array_equal(
        czyx[mask_idx, 0], art["masks"]["membrane"].astype(np.float32)
    )


def test_assemble_without_3d_stacks_returns_none():
    # No 3D stack -> nothing to anchor the (Z, Y, X) grid -> skip the store.
    names, czyx = worker._assemble_debug_channels(
        {"stacks": {}, "projections": {}, "masks": {}}
    )
    assert names == []
    assert czyx is None
