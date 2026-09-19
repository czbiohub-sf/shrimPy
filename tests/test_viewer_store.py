"""Tests for reading an acquisition's OME-Zarr store, live or finished.

The stores here are written by ome-writers/acquire-zarr exactly as
:meth:`shrimpy.engines.base_engine.BaseEngine.acquire` writes them, so the layouts
(single image, bioformats2raw series, HCS plate) are the real thing rather than a
hand-rolled approximation.

Two behaviors of that writer shape most of what follows, and both are exercised here:
a position's array does not exist until the position is first written, and chunks are
flushed asynchronously once complete -- so an acquisition in progress is read by
leaving the stream open (:class:`Acquiring`) and waiting for the flush.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Channel, Dimension, Plate, Position, create_stream

from shrimpy.viewer.store import AcquisitionStore

FRAME = (8, 16)  # (y, x)
PIXEL_SIZE_UM = 0.1133
SCAN_STEP_UM = 0.313
FLUSH_TIMEOUT_S = 10.0


def _settings(root, *, n_t, n_c, n_z, positions=None, plate=None, z_chunk=None):
    """Acquisition settings mirroring what BaseEngine.acquire writes."""
    dims = [Dimension(name="t", count=n_t, type="time", unit="second", scale=1.0)]
    if positions:
        # After t, as in a real MDA: every position is visited at each timepoint.
        dims.append(Dimension(name="p", type="position", coords=positions))
    dims += [
        Dimension(
            name="c",
            count=n_c,
            type="channel",
            coords=[Channel(name=f"ch{i}") for i in range(n_c)],
        ),
        Dimension(
            name="z",
            count=n_z,
            type="space",
            unit="micrometer",
            scale=SCAN_STEP_UM,
            chunk_size=z_chunk,
        ),
        Dimension(name="y", count=FRAME[0], type="space", scale=PIXEL_SIZE_UM),
        Dimension(name="x", count=FRAME[1], type="space", scale=PIXEL_SIZE_UM),
    ]
    return AcquisitionSettings(
        root_path=str(root),
        dimensions=dims,
        dtype="uint16",
        compression="blosc-zstd",
        format="acquire-zarr",
        plate=plate,
        overwrite=True,
    )


def _frame(value: int) -> np.ndarray:
    """A frame identified by ``value``, its pixel at ``[0, 0]``.

    A ramp along x is added on top so that a frame is not a single constant: the viewer
    sets a layer's contrast limits from real data and needs a min below its max.
    """
    ramp = np.broadcast_to(np.arange(FRAME[1], dtype=np.uint16), FRAME)
    return (ramp + value).astype(np.uint16)


class Acquiring:
    """An acquisition in progress: append frames, then wait for them to hit disk.

    Frames are numbered from 1 in acquisition order, so a plane's value says which
    frame it is. acquire-zarr flushes on a writer thread once a chunk is complete, so
    :meth:`append` waits for the chunks it filled to appear before returning -- without
    that, a test would race the writer. Used as a context manager so the stream is
    always closed.
    """

    def __init__(self, root, *, n_z, z_chunk=None, **kwargs):
        self.root = root
        self._stream = create_stream(_settings(root, n_z=n_z, z_chunk=z_chunk, **kwargs))
        self._frames_per_chunk = z_chunk or 1
        self._written = 0

    def append(self, n: int = 1) -> None:
        """Append ``n`` frames and block until the writer has flushed them."""
        for _ in range(n):
            self._written += 1
            self._stream.append(_frame(self._written))
        deadline = time.monotonic() + FLUSH_TIMEOUT_S
        expected = self._written // self._frames_per_chunk
        while self._chunk_files() < expected:
            if time.monotonic() > deadline:
                raise TimeoutError(f"writer did not flush {self._written} frames")
            time.sleep(0.02)

    def _chunk_files(self) -> int:
        """Chunk files on disk; everything in the store that is not metadata."""
        return sum(1 for p in self.root.rglob("*") if p.is_file() and p.name != "zarr.json")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stream.close()


def write_store(root, *, n_t=2, n_c=2, n_z=4, positions=None, plate=None, z_chunk=None):
    """Write a complete store and return its path.

    Defaults to one chunk per z-stack, which is how shrimPy chunks its acquisitions
    (``dimension_overrides={"z": ...}`` in ``BaseEngine.acquire``).
    """
    n_pos = len(positions) if positions else 1
    with Acquiring(
        root,
        n_t=n_t,
        n_c=n_c,
        n_z=n_z,
        positions=positions,
        plate=plate,
        z_chunk=n_z if z_chunk is None else z_chunk,
    ) as stream:
        stream.append(n_t * n_pos * n_c * n_z)
    return root


@pytest.fixture
def single(tmp_path):
    """A single-position store: the root group is the image itself."""
    return write_store(tmp_path / "single.ome.zarr")


@pytest.fixture
def series(tmp_path):
    """A multi-position store in bioformats2raw layout."""
    return write_store(tmp_path / "series.ome.zarr", positions=["A", "B", "C"])


@pytest.fixture
def plate(tmp_path):
    """An HCS plate store, two wells with two fields of view each."""
    positions = [
        Position(name=f"fov{f}", plate_row="A", plate_column=str(col))
        for col in (1, 2)
        for f in (0, 1)
    ]
    return write_store(
        tmp_path / "plate.ome.zarr",
        positions=positions,
        plate=Plate(row_names=["A"], column_names=["1", "2"]),
    )


def chunk_files(store: AcquisitionStore, position: int, t: int, channel: int):
    """The files backing a ``(position, t, channel)`` stack on disk."""
    encode = store._arrays[position].metadata.chunk_key_encoding.encode_chunk_key
    base = store.path / store._array_paths[position]
    return [base / encode(coord) for coord in store._chunk_coords(position, t, channel)]


def chunk_file(store: AcquisitionStore, position: int, t: int, channel: int):
    """The one file backing a whole ``(position, t, channel)`` stack on disk."""
    files = chunk_files(store, position, t, channel)
    assert len(files) == 1, "these fixtures chunk a whole stack together"
    return files[0]


# -- layout discovery ----------------------------------------------------------


def test_single_position_layout(single):
    store = AcquisitionStore(single)
    assert store.position_names == (".",)
    assert store.axes == ("t", "c", "z", "y", "x")
    assert store.index_sizes == (1, 2, 4)
    assert store.channels == ("ch0", "ch1")
    assert store.frame_shape == FRAME
    assert store.dtype == np.uint16


def test_series_layout_is_ordered_by_metadata(series):
    store = AcquisitionStore(series)
    assert store.position_names == ("A", "B", "C")
    assert store.n_positions == 3


def test_plate_layout_is_ordered_well_then_field(plate):
    store = AcquisitionStore(plate)
    assert store.position_names == ("A/1/fov0", "A/1/fov1", "A/2/fov0", "A/2/fov1")


# -- geometry ------------------------------------------------------------------


def test_scan_step_comes_from_the_store(single):
    assert AcquisitionStore(single).scan_step_um == pytest.approx(SCAN_STEP_UM)


def test_scan_step_is_whatever_z_was_written_with(tmp_path):
    """An acquisition with no stepped z-plan leaves the writer's default, not None.

    OME-Zarr cannot distinguish "1 um" from "unset", so the Deskew widget -- which
    always shows the value in use -- is where a wrong one gets corrected.
    """
    root = tmp_path / "noscale.ome.zarr"
    settings = _settings(root, n_t=1, n_c=1, n_z=2)
    for dim in settings.dimensions:
        dim.scale = None
    with create_stream(settings) as stream:
        for i in range(2):
            stream.append(_frame(i + 1))

    assert AcquisitionStore(root).scan_step_um == 1.0


# -- reading -------------------------------------------------------------------


def test_planes_match_what_was_written(single):
    store = AcquisitionStore(single)
    # Frames are appended in (t, c, z) order and numbered from 1.
    for t in range(2):
        for c in range(2):
            for z in range(4):
                assert store.plane(0, t, c, z)[0, 0] == t * 8 + c * 4 + z + 1


def test_tilt_row_spans_the_scan_stack(single):
    store = AcquisitionStore(single)
    row = store.tilt_row(0, 1, 0, 3)
    assert row.shape == (4, FRAME[1])
    # (t=1, c=0) holds frames 9..12, one constant value per z slice.
    assert list(row[:, 0]) == [9, 10, 11, 12]


def test_reads_past_the_acquired_extent_are_blank(single):
    store = AcquisitionStore(single)
    assert not store.plane(0, 99, 0, 0).any()
    assert store.tilt_row(0, 99, 0, 0) is None


def test_each_position_has_its_own_data(series):
    store = AcquisitionStore(series)
    values = {store.plane(p, 0, 0, 0)[0, 0] for p in range(3)}
    assert len(values) == 3


# -- growth --------------------------------------------------------------------


def test_refresh_picks_up_new_timepoints(tmp_path):
    root = tmp_path / "growing.ome.zarr"
    with Acquiring(root, n_t=3, n_c=1, n_z=2, z_chunk=2) as stream:
        stream.append(2)  # all of t=0

        store = AcquisitionStore(root)
        assert store.n_timepoints == 1
        assert not store.plane(0, 1, 0, 0).any()

        stream.append(2)  # all of t=1
        assert store.refresh()
        assert store.n_timepoints == 2
        assert store.plane(0, 1, 0, 0)[0, 0] == 3


def test_refresh_finds_positions_that_had_no_array_yet(tmp_path):
    """acquire-zarr creates a position's array only when that position is written."""
    root = tmp_path / "late.ome.zarr"
    with Acquiring(root, n_t=1, n_c=1, n_z=2, z_chunk=2, positions=["A", "B"]) as stream:
        stream.append(2)  # all of position A

        store = AcquisitionStore(root)
        assert store.n_positions == 2
        assert store.t_extent(1) == 0
        assert not store.plane(1, 0, 0, 0).any()

        stream.append(2)  # all of position B
        assert store.refresh()
        assert store.t_extent(1) == 1
        assert store.plane(1, 0, 0, 0)[0, 0] == 3


def test_refresh_is_false_when_nothing_changed(single):
    store = AcquisitionStore(single)
    assert store.refresh() is True  # the first one confirms the newest volume
    assert store.refresh() is False


def test_layer_extent_spans_the_furthest_position(series):
    """A timepoint reached by any position must be reachable on the slider."""
    store = AcquisitionStore(series)
    assert store.n_timepoints == max(store.t_extent(p) for p in range(3))


# -- completeness --------------------------------------------------------------


def test_volume_is_not_ready_until_every_chunk_is_flushed(single, tmp_path):
    """A chunk the writer has not flushed yet reads back as zeros, not as an error.

    Simulated by removing the chunk file, which is precisely the on-disk state of a
    volume whose frames have been acquired but not yet written out.
    """
    store = AcquisitionStore(single)
    chunk = chunk_file(store, 0, 1, 0)
    held_back = chunk.read_bytes()
    chunk.unlink()

    assert store.volume_ready(0, 0, 0)
    assert not store.volume_ready(0, 1, 0)
    # A half-written stack must never be deskewed, however much of it is on disk.
    assert store.tilt_row(0, 1, 0, 0) is None
    assert not store.plane(0, 1, 0, 0).any()

    chunk.write_bytes(held_back)
    assert store.volume_ready(0, 1, 0)
    assert store.plane(0, 1, 0, 0)[0, 0] == 9


def test_unflushed_volume_is_never_cached(single):
    """Caching a mid-flush volume would pin its blank slices for the whole session."""
    store = AcquisitionStore(single)
    chunk = chunk_file(store, 0, 0, 0)
    held_back = chunk.read_bytes()
    chunk.unlink()

    assert not store.plane(0, 0, 0, 0).any()
    assert store._cache == {}

    chunk.write_bytes(held_back)
    assert store.plane(0, 0, 0, 0)[0, 0] == 1


def test_volume_spanning_several_chunks_needs_them_all(tmp_path):
    root = write_store(tmp_path / "split.ome.zarr", n_t=1, n_c=1, n_z=4, z_chunk=1)
    store = AcquisitionStore(root)
    assert store.volume_ready(0, 0, 0)

    files = chunk_files(store, 0, 0, 0)
    assert len(files) == 4  # one chunk per z slice
    files[-1].unlink()

    store._ready.clear()  # forget the earlier positive answer
    assert not store.volume_ready(0, 0, 0)


def test_follow_target_tracks_the_newest_complete_volume(tmp_path):
    """The viewer parks on whichever position most recently finished a volume."""
    root = tmp_path / "follow.ome.zarr"
    with Acquiring(root, n_t=3, n_c=1, n_z=2, z_chunk=2, positions=["A", "B"]) as stream:
        stream.append(4)  # t=0 at both positions

        store = AcquisitionStore(root)
        store.refresh()
        assert store.latest_complete == (1, 0)  # B was written after A

        stream.append(2)  # t=1 at position A
        store.refresh()
        assert store.latest_complete == (0, 1)

        stream.append(2)  # t=1 at position B
        store.refresh()
        assert store.latest_complete == (1, 1)


# -- cache ---------------------------------------------------------------------


def test_cache_serves_repeat_reads(single):
    store = AcquisitionStore(single)
    store.plane(0, 0, 0, 0)
    assert list(store._cache) == [(0, 0, 0)]
    store.plane(0, 0, 0, 3)
    assert list(store._cache) == [(0, 0, 0)]  # same volume, no second read


def test_cache_evicts_the_oldest_volumes_within_budget(single):
    volume_bytes = 4 * FRAME[0] * FRAME[1] * 2
    store = AcquisitionStore(single, cache_mb=(2 * volume_bytes) / 1e6)
    for key in [(0, 0), (1, 0), (0, 1), (1, 1)]:  # (t, channel)
        store.plane(0, key[0], key[1], 0)
    assert list(store._cache) == [(0, 0, 1), (0, 1, 1)]
    assert store._cache_bytes <= 2 * volume_bytes


def test_cache_keeps_the_volumes_most_recently_looked_at(single):
    """Scrubbing back and forth between two stacks must not evict the one on screen."""
    volume_bytes = 4 * FRAME[0] * FRAME[1] * 2
    store = AcquisitionStore(single, cache_mb=(2 * volume_bytes) / 1e6)
    store.plane(0, 0, 0, 0)
    store.plane(0, 1, 0, 0)
    store.plane(0, 0, 0, 1)  # back to the first: it is now the most recent
    store.plane(0, 1, 1, 0)  # a third volume evicts the least recent, not the first
    assert list(store._cache) == [(0, 0, 0), (0, 1, 1)]


def test_one_volume_is_always_cached_however_small_the_budget(single):
    store = AcquisitionStore(single, cache_mb=0.0)
    store.plane(0, 0, 0, 0)
    store.plane(0, 1, 1, 0)
    assert list(store._cache) == [(0, 1, 1)]
