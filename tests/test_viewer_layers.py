"""Tests for the viewer's layer logic, driven by a stand-in for the napari viewer.

``shrimpy.viewer._napari_process`` defers every napari/Qt import into
:func:`~shrimpy.viewer._napari_process.run_viewer`, so the layer bookkeeping around it
-- opening the store, growing the layers as the acquisition appends timepoints,
following the newest complete volume, and swapping between raw and deskewed views --
runs against the fake viewer below without a GUI.
"""

from __future__ import annotations

import numpy as np
import pytest

from shrimpy.viewer._napari_process import (
    LazyStoreArray,
    _ViewerState,
    store_gather,
)
from shrimpy.viewer.store import AcquisitionStore

from .test_viewer_store import FRAME, Acquiring, chunk_file, write_store


@pytest.fixture
def store_path(tmp_path):
    return write_store(tmp_path / "layers.ome.zarr", n_t=2, n_c=2, n_z=4)


# -- the fake viewer -----------------------------------------------------------


class FakeEvent:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self):
        for callback in self._callbacks:
            callback()


class FakeDims:
    """Just enough of ``napari.Viewer.dims``: a step tuple that emits on assignment."""

    def __init__(self, ndim=5):
        self.axis_labels = ()
        self._step = (0,) * ndim
        self.events = type("Events", (), {"current_step": FakeEvent()})()

    @property
    def current_step(self):
        return self._step

    @current_step.setter
    def current_step(self, value):
        self._step = tuple(value)
        self.events.current_step.emit()


class FakeLayer:
    def __init__(self, data, **kwargs):
        self.data = data
        self.name = kwargs.get("name")
        self.colormap = kwargs.get("colormap")
        self.contrast_limits = kwargs.get("contrast_limits")
        self.refreshes = 0

    def refresh(self):
        self.refreshes += 1


class FakeViewer:
    def __init__(self):
        self.dims = FakeDims()
        self.layers = []
        self.title = ""
        self.window = object()  # no _qt_viewer: the Home hookup degrades gracefully

    def add_image(self, data, **kwargs):
        layer = FakeLayer(data, **kwargs)
        self.layers.append(layer)
        return layer


def viewer_on(path, *, deskew=False):
    """A viewer state already opened on ``path``."""
    viewer = FakeViewer()
    state = _ViewerState(viewer, deskew=deskew)
    state.set_path(path)
    state.tick()  # opens the store and builds the layers
    return viewer, state


# -- raw layers ----------------------------------------------------------------


def test_lazy_array_indexes_like_numpy(store_path):
    store = AcquisitionStore(store_path)
    array = LazyStoreArray(store, 1)
    assert array.shape == (1, 2, 4, *FRAME)
    assert array.dtype == np.uint16
    # An integer drops its axis, a slice keeps it -- napari relies on both.
    assert array[0, 0, 2].shape == FRAME
    assert array[0, 0].shape == (4, *FRAME)
    assert array[0, 1, 0][0, 0] == 13  # (t=1, c=1, z=0) is the 13th frame written


def test_gather_returns_a_tilt_row_across_the_stack(store_path):
    store = AcquisitionStore(store_path)
    row = store_gather(store, 0)((0, 1), 2)
    assert row.shape == (4, FRAME[1])
    assert list(row[:, 0]) == [9, 10, 11, 12]


def test_gather_blanks_a_volume_that_is_not_all_on_disk(store_path):
    store = AcquisitionStore(store_path)
    chunk_file(store, 0, 1, 0).unlink()
    assert not store_gather(store, 0)((0, 1), 2).any()


# -- opening -------------------------------------------------------------------


def test_the_viewer_waits_for_the_acquisition_to_produce_data(tmp_path):
    """The window opens before the acquisition names its store, and retries.

    Nothing is lost in the meantime: the frames written while the viewer was still
    starting up are on disk, and appear as soon as it manages to open the store.
    """
    root = tmp_path / "pending.ome.zarr"
    viewer = FakeViewer()
    state = _ViewerState(viewer, deskew=False)

    state.tick()  # no path yet
    assert viewer.layers == []

    state.set_path(root)
    state.tick()  # path known, but the acquisition has not started
    assert viewer.layers == []

    with Acquiring(root, n_t=2, n_c=1, n_z=2, z_chunk=2) as stream:
        state.tick()  # store created, but no frame has been flushed
        assert viewer.layers == []

        stream.append(2)
        state.tick()
        assert len(viewer.layers) == 1
        assert viewer.layers[0].data[0, 0, 0][0, 0] == 1


def test_one_layer_per_channel(store_path):
    viewer, _ = viewer_on(store_path)
    assert [layer.name for layer in viewer.layers] == ["ch0", "ch1"]
    assert all(layer.data.shape == (1, 2, 4, *FRAME) for layer in viewer.layers)
    assert viewer.dims.axis_labels == ("p", "t", "z", "y", "x")


def test_contrast_limits_come_from_real_data(store_path):
    """Left at the dtype default, typical low-intensity frames would render black."""
    viewer, _ = viewer_on(store_path)
    for layer in viewer.layers:
        low, high = layer.contrast_limits
        assert (low, high) != (0.0, 65535.0)
        assert low < high


# -- following -----------------------------------------------------------------


def test_sliders_follow_the_newest_complete_volume(tmp_path):
    root = tmp_path / "follow.ome.zarr"
    with Acquiring(root, n_t=3, n_c=1, n_z=2, z_chunk=2, positions=["A", "B"]) as stream:
        stream.append(4)  # t=0 at both positions
        viewer, state = viewer_on(root)
        assert viewer.dims.current_step[:2] == (1, 0)

        stream.append(4)  # t=1 at both positions
        state.tick()
        assert viewer.dims.current_step[:2] == (1, 1)


def test_scrubbing_a_followed_axis_pauses_and_home_resumes(tmp_path):
    root = tmp_path / "pause.ome.zarr"
    with Acquiring(root, n_t=3, n_c=1, n_z=2, z_chunk=2) as stream:
        stream.append(2)
        viewer, state = viewer_on(root)

        viewer.dims.current_step = (0, 0, 1, 0, 0)  # the user moves z: still following
        stream.append(2)
        state.tick()
        assert viewer.dims.current_step[1] == 1

        viewer.dims.current_step = (0, 0, 1, 0, 0)  # the user moves t: pause
        stream.append(2)
        state.tick()
        assert viewer.dims.current_step[1] == 0

        state._on_home_clicked()
        assert viewer.dims.current_step[1] == 2


# -- growth --------------------------------------------------------------------


def test_layers_grow_as_timepoints_are_appended(tmp_path):
    root = tmp_path / "grow.ome.zarr"
    with Acquiring(root, n_t=3, n_c=1, n_z=2, z_chunk=2) as stream:
        stream.append(2)
        viewer, state = viewer_on(root)
        assert viewer.layers[0].data.shape[1] == 1

        stream.append(2)
        state.tick()
        assert viewer.layers[0].data.shape[1] == 2
        assert viewer.layers[0].data[0, 1, 0][0, 0] == 3


# -- deskew --------------------------------------------------------------------


def test_deskewed_view_is_the_default_when_available(store_path):
    viewer, state = viewer_on(store_path, deskew=True)
    # Deskew permutes the volume: depth comes from the tilt axis, width from the scan.
    assert viewer.layers[0].data.shape[2] == FRAME[0]
    assert viewer.layers[0].data.shape[3] == FRAME[1]

    state._display_raw()
    assert viewer.layers[0].data.shape == (1, 2, 4, *FRAME)
    state._display_deskewed()
    assert viewer.layers[0].data.shape[2] == FRAME[0]


def test_deskew_is_offered_only_for_a_real_z_stack(tmp_path):
    root = write_store(tmp_path / "flat.ome.zarr", n_t=1, n_c=1, n_z=1)
    _, state = viewer_on(root, deskew=True)
    assert not state._deskew_available


def test_deskew_scan_step_comes_from_the_store(store_path):
    """The scan step varies per acquisition and is recorded; the rest are defaults."""
    from napari_deskew_preview import LS_ANGLE_DEG, PIXEL_SIZE_UM

    _, state = viewer_on(store_path, deskew=True)
    assert state._geometry() == (
        LS_ANGLE_DEG,
        PIXEL_SIZE_UM,
        AcquisitionStore(store_path).scan_step_um,
    )


def test_live_deskew_matches_the_offline_widget(store_path):
    """The same data must deskew to the same pixels live and offline.

    Both go through the same source-agnostic core, so only their geometry inputs can
    make them disagree -- which is exactly the way a live deskew has gone wrong before.
    """
    from napari_deskew_preview import (
        LS_ANGLE_DEG,
        PIXEL_SIZE_UM,
        array_gather,
        deskewed_layer,
    )

    store = AcquisitionStore(store_path)
    geometry = {"ls_angle_deg": LS_ANGLE_DEG, "pixel_size_um": PIXEL_SIZE_UM}
    live, projector = deskewed_layer(
        store_gather(store, 0),
        raw_zyx_shape=(store.n_z, *store.frame_shape),
        scan_step_um=store.scan_step_um,
        batch_sizes=(store.n_positions, store.n_timepoints),
        **geometry,
    )
    # What the offline widget sees: the raw array straight off the store.
    raw = np.asarray(store._arrays[0][0, 0])
    offline, _ = deskewed_layer(
        array_gather(raw),
        raw_zyx_shape=raw.shape,
        scan_step_um=store.scan_step_um,
        **geometry,
    )
    for z_out in (0, projector.z_out // 2, projector.z_out - 1):
        assert np.array_equal(live[0, 0, z_out], offline[z_out])
