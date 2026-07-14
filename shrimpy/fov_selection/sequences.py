"""Sequence builders for two-run FOV selection.

FOV selection is a two-run adaptive acquisition (see
``docs/fov_selection_integration_plan.md``): the engine first runs a *pre-scan*
sequence over all candidate FOVs to decide which are "good", then runs a
*timelapse* sequence over the good FOVs only. These helpers build those two
:class:`useq.MDASequence` objects from a single user config. They live in the
``fov_selection`` package (not the microscope engine) so the selection logic
stays self-contained and reusable across microscopes.
"""

from __future__ import annotations

import copy

from useq import MDASequence, Position, WellPlatePlan


def fov_selection_config(sequence: MDASequence) -> dict:
    """Return the ``metadata.mantis.fov_selection`` block (``{}`` if absent).

    The caller decides whether it is active by reading its ``enabled`` flag,
    e.g. ``fov_selection_config(sequence).get("enabled")``.
    """
    meta = sequence.metadata.get("mantis", {}) if sequence.metadata else {}
    return meta.get("fov_selection") or {}


# Mantis hardware-setup metadata keys that the pre-scan run must share with the
# timelapse run so the scope is configured identically for both.
_SHARED_MANTIS_KEYS = (
    "autofocus",
    "roi",
    "initialization_settings",
    "reset_hardware_sequencing_settings",
    "setup_hardware_sequencing_settings",
)


def build_prescan_sequence(sequence: MDASequence, fov_cfg: dict) -> MDASequence:
    """Build the pre-scan ``MDASequence`` from ``fov_selection.prescan_mda``.

    The pre-scan is configured as its own complete, valid ``MDASequence`` nested
    under ``metadata.mantis.fov_selection.prescan_mda``. It carries its own
    ``stage_positions`` (the candidate FOVs to search) and ``z_plan`` -- which
    may be a single 2D slice for fluorescence-based selection, independent of the
    timelapse z-plan. The parent ``fov_selection`` config (minus ``prescan_mda``)
    and the shared mantis hardware settings are injected into the pre-scan
    metadata so ``setup_sequence`` builds the ``FovSelection`` coordinator and
    configures the scope the same way as the timelapse run.

    Candidate FOVs may be defined in either useq style: a ``WellPlatePlan``
    (select wells + a per-well FOV grid; each FOV expands to its own position with
    ``plate_row``/``plate_col``), or explicit ``stage_positions`` plus a top-level
    ``grid_plan`` (a grid around each free XY center, no plate layout). The grid
    style is normalized to one position per FOV via :func:`expand_candidate_fovs`
    so both styles feed the same per-FOV decision path.

    Raises
    ------
    ValueError
        If ``prescan_mda`` is missing or defines no ``stage_positions``; if the
        pre-scan has more than one timepoint (a looping pre-scan is not yet
        supported); or if ``fov_selection_channel`` is not one of the pre-scan channels.
    """
    prescan_mda = fov_cfg.get("prescan_mda")
    if not prescan_mda:
        raise ValueError(
            "FOV selection requires metadata.mantis.fov_selection.prescan_mda "
            "(a valid MDASequence defining the candidate stage_positions and z_plan)."
        )

    prescan_seq = MDASequence(**prescan_mda)

    if not prescan_seq.stage_positions:
        raise ValueError(
            "FOV selection prescan_mda must define stage_positions (the candidate FOVs)."
        )
    if prescan_seq.sizes.get("t", 1) > 1:
        raise ValueError(
            "FOV selection prescan_mda must have a single timepoint (got "
            f"t={prescan_seq.sizes['t']}); a looping pre-scan is not yet supported."
        )

    fov_selection_channel = fov_cfg.get("fov_selection_channel", "BF - Oblique")
    channel_configs = [c.config for c in prescan_seq.channels]
    if fov_selection_channel not in channel_configs:
        raise ValueError(
            f"FOV selection fov_selection_channel {fov_selection_channel!r} is not one of the "
            f"pre-scan channels {channel_configs}."
        )

    # A top-level grid_plan keeps its FOVs on a separate `g` axis while the
    # selection pipeline scores one candidate FOV per `stage_position`. Flatten
    # the grid into explicit per-FOV positions (WellPlatePlan already expands
    # per-FOV, so it is left untouched).
    if (
        not isinstance(prescan_seq.stage_positions, WellPlatePlan)
        and prescan_seq.grid_plan is not None
    ):
        prescan_seq = prescan_seq.replace(
            stage_positions=expand_candidate_fovs(prescan_seq), grid_plan=None
        )

    # Inject the fov_selection config + shared mantis hardware settings into the
    # pre-scan metadata. Drop prescan_mda from the injected block so the coordinator
    # doesn't carry a redundant nested copy of itself.
    parent_mantis = (sequence.metadata or {}).get("mantis", {})
    fov_block = copy.deepcopy(fov_cfg)
    fov_block.pop("prescan_mda", None)

    prescan_meta = copy.deepcopy(prescan_seq.metadata) if prescan_seq.metadata else {}
    prescan_mantis = prescan_meta.setdefault("mantis", {})
    prescan_mantis["fov_selection"] = fov_block
    for key in _SHARED_MANTIS_KEYS:
        if key in parent_mantis and key not in prescan_mantis:
            prescan_mantis[key] = copy.deepcopy(parent_mantis[key])

    return prescan_seq.replace(metadata=prescan_meta)


def expand_candidate_fovs(prescan_seq: MDASequence) -> list[Position]:
    """Flatten explicit ``stage_positions`` + a top-level ``grid_plan`` into one
    :class:`useq.Position` per candidate FOV.

    useq keeps ``grid_plan`` FOVs on a separate ``g`` axis while the FOV-selection
    pipeline treats each candidate FOV as its own ``stage_position`` (its own
    ``p_idx`` + name -- see ``fov_selection.manager``). This expands each center's
    grid into distinct positions -- absolute XY from useq's event iterator, ``z``
    inherited from the center, names ``"{center}_{g:04d}"`` -- so the free-XY grid
    style is scored and filtered exactly like the ``WellPlatePlan`` style. The
    expanded positions carry no ``plate_row``/``plate_col`` (there is no plate),
    so their good FOVs fall through the non-plate branch of
    :func:`_filter_good_positions` and produce a flat (non-HCS) OME-Zarr.
    """
    centers = list(prescan_seq.stage_positions)
    grid = MDASequence(stage_positions=centers, grid_plan=prescan_seq.grid_plan)
    out: list[Position] = []
    for event in grid.iter_events():
        p_idx = event.index.get("p", 0)
        g_idx = event.index.get("g", 0)
        base = centers[p_idx].name or f"p{p_idx}"
        out.append(
            Position(
                x=event.x_pos,
                y=event.y_pos,
                z=centers[p_idx].z,
                name=f"{base}_{g_idx:04d}",
            )
        )
    return out


def build_timelapse_sequence(
    sequence: MDASequence, prescan_seq: MDASequence, good_names: list[str]
) -> MDASequence:
    """Timelapse sequence: good FOVs only, ``fov_selection`` disabled.

    The main ``sequence`` describes the timelapse (channels, z_plan, time_plan)
    and its ``stage_positions`` are empty; the good FOVs are taken from the
    pre-scan candidates (``prescan_seq.stage_positions``). Uses the original
    ``time_plan`` as-is (``loops`` is the timelapse point count; the pre-scan is
    its own run, so there is no ``+1``). Disabling ``fov_selection`` in the
    metadata makes ``setup_sequence`` build no coordinator for this run.
    """
    good_positions = _filter_good_positions(prescan_seq, good_names)
    meta = copy.deepcopy(sequence.metadata) if sequence.metadata else {}
    fov_cfg = meta.get("mantis", {}).get("fov_selection")
    if fov_cfg is not None:
        fov_cfg["enabled"] = False
    return sequence.replace(stage_positions=good_positions, metadata=meta)


def _row_index_to_letter(index: int) -> str:
    """Zero-based row index -> name (A, B, ..., Z, AA, ...), matching useq."""
    name = ""
    while index >= 0:
        name = chr(index % 26 + 65) + name
        index = index // 26 - 1
    return name


def _filter_good_positions(sequence: MDASequence, good_names: list[str]) -> list:
    """Candidate positions whose name is in ``good_names`` (order preserved).

    Iterating ``sequence.stage_positions`` yields expanded ``AbsolutePosition``
    objects that carry ``plate_row``/``plate_col``, so the filtered list still
    produces a proper HCS OME-Zarr for the good FOVs. Two adjustments are made so
    the rebuilt explicit list matches what the ``WellPlatePlan`` path would emit:

    * integer plate coordinates are converted to strings (``1 -> "B"``,
      ``3 -> "4"``) -- useq forbids a field-suffixed name on a standalone position
      with *integer* plate coords, but accepts an explicit name with *string* ones;
    * the useq FOV name (e.g. ``"B4_0000"``) is reduced to the per-well field name
      (``"0000"``) -- the well's image path must be alphanumeric (iohub rejects the
      underscore), and this is exactly what ome-writers' WellPlatePlan builder uses.

    The ReplayCamera maps positions by ``plate_row``/``plate_col`` and per-well
    order, not by name, so the rename does not affect offline replay.
    """
    good = set(good_names)
    out = []
    for idx, pos in enumerate(sequence.stage_positions):
        if (pos.name or f"p{idx}") not in good:
            continue
        if isinstance(pos.plate_row, int) and isinstance(pos.plate_col, int):
            row_letter = _row_index_to_letter(pos.plate_row)
            col_label = str(pos.plate_col + 1)
            well_name = f"{row_letter}{col_label}"
            field_name = pos.name or ""
            if field_name.startswith(f"{well_name}_"):
                field_name = field_name[len(well_name) + 1 :]
            field_name = "".join(c for c in field_name if c.isalnum()) or f"{idx:04d}"
            pos = pos.model_copy(
                update={
                    "plate_row": row_letter,
                    "plate_col": col_label,
                    "name": field_name,
                }
            )
        out.append(pos)
    return out
