"""Tests for the source-agnostic deskew core (no napari/Qt required)."""

from __future__ import annotations

import numpy as np

from napari_deskew_preview import LS_ANGLE_DEG, DeskewProjector, array_gather, deskewed_layer


def _biahub_reference(raw: np.ndarray, angle: float, ratio: float) -> np.ndarray:
    """A numpy re-implementation of biahub's fast_deskew_zyx (N=1, keep_overhang=True)."""
    z, y, x = raw.shape
    ct = np.cos(np.deg2rad(angle))
    xp = int(np.ceil(z / ratio + y * ct))
    offset = ratio * ct * (y - 1) / 2 - ratio * (xp - 1) / 2 + (z - 1) / 2
    ra = np.transpose(raw, (1, 2, 0))[::-1, ::-1, :].astype(np.float32)  # (Y, X, Z)
    out = np.zeros((y, x, xp), np.float32)
    xi = np.arange(xp)
    last = z - 1
    for zo in range(y):
        s = ratio * xi - ratio * ct * zo + offset
        lo = np.floor(s).astype(int)
        w = (s - lo).astype(np.float32)
        lok = ((lo >= 0) & (lo <= last))[None, :]
        hok = ((lo + 1 >= 0) & (lo + 1 <= last))[None, :]
        col = ra[zo]
        out[zo] = (1 - w)[None, :] * col[:, np.clip(lo, 0, last)] * lok + w[None, :] * col[
            :, np.clip(lo + 1, 0, last)
        ] * hok
    return out


def test_geometry_matches_biahub_reference():
    rng = np.random.default_rng(0)
    z, y, x = 50, 14, 16
    raw = rng.integers(0, 40000, (z, y, x)).astype(np.uint16)
    scan_step, pixel = 0.30, 0.1133
    ratio = pixel / scan_step

    arr, proj = deskewed_layer(array_gather(raw), (z, y, x), scan_step, batch_sizes=())
    assert arr.shape == proj.output_shape
    mine = np.asarray(arr[:])
    ref = _biahub_reference(raw, LS_ANGLE_DEG, ratio)
    assert np.allclose(mine, ref, rtol=1e-4, atol=1e-1)


def test_batched_source():
    rng = np.random.default_rng(1)
    z, y, x = 40, 12, 18
    raw = rng.integers(0, 5000, (3, z, y, x)).astype(np.uint16)  # (P=3, Z, Y, X)
    arr, proj = deskewed_layer(array_gather(raw), (z, y, x), 0.31, batch_sizes=(3,))
    assert arr.shape == (3, *proj.output_shape)
    ratio = 0.1133 / 0.31
    expected = _biahub_reference(raw[2], LS_ANGLE_DEG, ratio)
    assert np.allclose(np.asarray(arr[2, y // 2]), expected[y // 2], rtol=1e-4, atol=1e-1)


def test_indexing_matches_numpy():
    # DeskewedArray must index like a real ndarray (int drops an axis, slice keeps it).
    rng = np.random.default_rng(2)
    z, y, x = 30, 10, 12
    raw = rng.integers(0, 1000, (2, z, y, x)).astype(np.uint16)
    arr, proj = deskewed_layer(array_gather(raw), (z, y, x), 0.3, batch_sizes=(2,))
    dense = np.asarray(arr[:])  # materialize once as ground truth
    for key in [(0, 0), (1, y // 2), (0, slice(None)), (slice(None), 0)]:
        assert np.array_equal(np.asarray(arr[key]), dense[key])
    assert np.asarray(arr[..., 1:3, 2:4]).shape == dense[..., 1:3, 2:4].shape


def test_editable_geometry_changes_output_shape():
    base = DeskewProjector((1068, 256, 1664), 0.30)
    wider = DeskewProjector((1068, 256, 1664), 0.50)  # larger scan step -> wider X_out
    assert wider.output_shape[2] > base.output_shape[2]
    assert base.output_shape[:2] == (256, 1664)
