"""A bounded, shared-memory ring buffer for live image frames.

The ring holds only pixel data for the most-recent ``n_slots`` frames in a single
:class:`multiprocessing.shared_memory.SharedMemory` block. Frame *coordinates*
(position / time / channel / z and the slot they landed in) travel separately over a
:class:`multiprocessing.Queue`; this module is deliberately ignorant of them.

The acquisition process *creates* the ring and writes to it; the viewer process
*attaches* to it by name and reads from it lazily at draw time. Because the ring wraps
around, a reader may occasionally observe a slot mid-overwrite -- this is acceptable for
a best-effort live preview and never blocks the writer.
"""

from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np


class RingBuffer:
    """A fixed-size ring of identically shaped frames in shared memory.

    Parameters
    ----------
    shm : shared_memory.SharedMemory
        The backing shared-memory block.
    n_slots : int
        Number of frame slots.
    frame_shape : tuple[int, ...]
        Shape of a single frame (e.g. ``(height, width)``).
    dtype : np.dtype
        Frame data type.
    owner : bool
        Whether this instance owns ``shm`` (i.e. should ``unlink`` it on close).
    """

    def __init__(
        self,
        shm: shared_memory.SharedMemory,
        n_slots: int,
        frame_shape: tuple[int, ...],
        dtype: np.dtype,
        owner: bool,
    ) -> None:
        self._shm = shm
        self.n_slots = n_slots
        self.frame_shape = tuple(frame_shape)
        self.dtype = np.dtype(dtype)
        self._owner = owner
        self._array = np.ndarray(
            (n_slots, *self.frame_shape), dtype=self.dtype, buffer=shm.buf
        )

    @classmethod
    def create(cls, n_slots: int, frame_shape: tuple[int, ...], dtype: np.dtype) -> RingBuffer:
        """Allocate a new ring (acquisition-process side)."""
        dtype = np.dtype(dtype)
        nbytes = int(n_slots * np.prod(frame_shape) * dtype.itemsize)
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        return cls(shm, n_slots, frame_shape, dtype, owner=True)

    @classmethod
    def attach(
        cls, name: str, n_slots: int, frame_shape: tuple[int, ...], dtype: np.dtype
    ) -> RingBuffer:
        """Attach to an existing ring by name (viewer-process side)."""
        shm = shared_memory.SharedMemory(name=name)
        # On 'spawn' (macOS default), a non-owner process otherwise registers the segment
        # with its resource_tracker and may unlink it on exit -- destroying the parent's
        # buffer and emitting spurious "leaked shared_memory" warnings. The owner (the
        # acquisition process) is solely responsible for unlinking, so unregister here.
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - best-effort; private API may change
            pass
        return cls(shm, n_slots, frame_shape, dtype, owner=False)

    @property
    def name(self) -> str:
        """Name of the backing shared-memory block (used to attach from elsewhere)."""
        return self._shm.name

    def write(self, slot: int, frame: np.ndarray) -> None:
        """Copy ``frame`` into ``slot`` (writer side)."""
        self._array[slot] = frame

    def read(self, slot: int) -> np.ndarray:
        """Return a private copy of the frame in ``slot`` (reader side).

        A copy -- not a view -- so that a subsequent overwrite by the writer cannot
        mutate data already handed to the viewer.
        """
        return np.array(self._array[slot], copy=True)

    def read_rows(self, slots: list[int | None], row: int) -> np.ndarray:
        """Read one row (``array[slot, row, :]``) from many slots at once.

        Missing slots (``None``) yield a zero row. Used by the deskew projector to
        gather a single tilt row across the whole scan stack without copying full
        frames -- ~3.5 MB instead of ~1 GB.

        Returns an array of shape ``(len(slots), frame_width)``.
        """
        n_cols = self.frame_shape[1]
        out = np.zeros((len(slots), n_cols), dtype=self.dtype)
        present = [i for i, s in enumerate(slots) if s is not None]
        if present:
            out[present] = self._array[[slots[i] for i in present], row, :]
        return out

    def close(self) -> None:
        """Detach from shared memory; unlink it too if this instance is the owner."""
        try:
            self._shm.close()
        finally:
            if self._owner:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
