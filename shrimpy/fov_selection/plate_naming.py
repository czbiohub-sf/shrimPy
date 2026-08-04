"""Plate coordinates and position names -> the labels and path parts sinks require.

FOV selection writes a position's name and plate coordinates to several sinks (the
timelapse ``stage_positions``, the pre-scan OME-Zarr, the debug PNGs), and each
imposes a different rule. Collected here so there is one definition of each rule
and the call site says which sink it targets:

* :func:`plate_labels` / :func:`row_label` / :func:`col_label` -- useq expresses
  plate coordinates either as zero-based integers (what ``WellPlatePlan`` expands
  to, and what a config writes) or as the human labels they render to. These
  normalize both forms to the labels, so callers never branch on the type.
* :func:`zarr_path_name` -- an OME-Zarr path part, which iohub requires to be
  alphanumeric. Drops everything else (``"B4_0000" -> "B40000"``).
* :func:`file_stem_name` -- a filesystem stem for debug images, where ``_`` and
  ``-`` are legal and worth keeping (``"B4_0000" -> "B4_0000"``).
* :func:`well_field_name` -- the per-well field name for a plate position, i.e. the
  useq FOV name with its well prefix removed, then run through
  :func:`zarr_path_name`.

The two sanitizers differ deliberately, and the same FOV name legitimately renders
differently under each. Call them by name rather than inlining a ``str.isalnum``
comprehension, so which sink is meant is visible at the call site.

Scoped to ``fov_selection`` on purpose. ``shrimpy.replay_camera`` and
``shrimpy.dynatrack.tracking`` carry their own equivalents of :func:`row_label` /
:func:`col_label` and :func:`zarr_path_name`; unifying those is a separate change
to those modules, not something to reach across from here.
"""

from __future__ import annotations

try:  # useq's own row naming, so labels always agree with WellPlatePlan
    from useq._position import _index_to_row_name as _useq_index_to_row_name
except ImportError:  # pragma: no cover - private name; fall back to a local copy
    _useq_index_to_row_name = None


def row_label(value: int | str) -> str:
    """Plate row label: zero-based int -> ``A, B, ..., Z, AA, ...``; str -> as-is.

    Matches useq's own row naming (and so keeps rolling over past ``Z`` correctly).
    """
    if isinstance(value, str):
        return value
    index = int(value)
    if _useq_index_to_row_name is not None:
        return _useq_index_to_row_name(index)
    name = ""
    while index >= 0:
        name = chr(index % 26 + 65) + name
        index = index // 26 - 1
    return name


def col_label(value: int | str) -> str:
    """Plate column label: zero-based int -> one-based ``"1", "2", ...``; str -> as-is."""
    return value if isinstance(value, str) else str(int(value) + 1)


def plate_labels(pos) -> tuple[str, str] | None:
    """``(row_label, col_label)`` for a plate position, or ``None`` if not on a plate.

    ``pos`` is any object with ``plate_row``/``plate_col`` attributes (a useq
    ``Position``); a position missing either is not on a plate.
    """
    row = getattr(pos, "plate_row", None)
    col = getattr(pos, "plate_col", None)
    if row is None or col is None:
        return None
    return row_label(row), col_label(col)


def zarr_path_name(name, fallback: str) -> str:
    """``name`` as an OME-Zarr path part: non-alphanumerics dropped.

    iohub rejects a path part that is not alphanumeric, so ``"B4_0000"`` becomes
    ``"B40000"``. ``fallback`` is used when nothing survives (or ``name`` is empty).
    """
    return "".join(c for c in str(name) if c.isalnum()) or fallback


def file_stem_name(name, fallback: str = "fov") -> str:
    """``name`` as a filesystem stem for debug images: ``_`` and ``-`` kept.

    Unlike :func:`zarr_path_name` this preserves the readable form
    (``"B4_0000"`` stays ``"B4_0000"``); anything else becomes ``_``.
    """
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name)) or fallback


def well_field_name(pos, fallback: str) -> str:
    """The per-well field name of a plate position, as an OME-Zarr path part.

    useq names a plate FOV ``"{well}{field}"`` joined by ``_`` (e.g. ``"B4_0000"``).
    The well is already encoded in the zarr path by ``plate_row``/``plate_col``, so
    the position itself only needs the field (``"0000"``) -- which is what
    ome-writers' ``WellPlatePlan`` builder uses too.
    """
    well = plate_labels(pos)
    field = getattr(pos, "name", None) or ""
    if well is not None:
        prefix = f"{well[0]}{well[1]}_"
        if field.startswith(prefix):
            field = field[len(prefix) :]
    return zarr_path_name(field, fallback)
