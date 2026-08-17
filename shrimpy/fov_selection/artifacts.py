"""FOV-selection acquisition artifacts written around the two-run adaptive flow.

These helpers are called by :meth:`shrimpy.engines.base_engine.BaseEngine.acquire`
between the pre-scan and the timelapse, but the logic is pure FOV-selection domain (no
engine state), so it lives here beside the rest of the package rather than on the engine:

- :func:`save_selected_config` records the acquisition config with the SELECTED FOVs
  filled into ``stage_positions`` (a descriptive record; nothing reads it back).
- :func:`launch_feature_viewer` opens the feature viewer on a calibration pre-scan.
- :func:`write_rank_profile` dumps the config's ``model`` block beside the feature CSV so
  the viewer's Rank tab can seed from it.

Every function is best-effort: a failure to write/launch an artifact is logged, never
raised, so it cannot take the acquisition down between the two runs.
"""

from __future__ import annotations

import logging

from pathlib import Path

from useq import MDASequence

logger = logging.getLogger(__name__)


def save_selected_config(
    timelapse_seq: MDASequence, output_dir: Path, run_index: int | None
) -> None:
    """Record the acquisition config with the SELECTED FOVs in ``stage_positions``.

    The config an FOV-selection experiment starts from leaves ``stage_positions``
    empty -- the candidates live under ``fov_selection.prescan_mda`` and the real
    positions are only known after the pre-scan. This writes the same sequence with
    that gap filled: one entry per selected FOV carrying its absolute ``x``/``y``,
    the well's ``ZDrive`` coarse focus, and its ``plate_row``/``plate_col``.

    Saved as ``config_<experiment folder>.yaml`` in the experiment folder, beside
    the hand-written ``config.yaml`` it mirrors. A purely descriptive record of what
    the run chose -- nothing reads it back. Because the name follows the FOLDER
    rather than the acquisition, a second acquisition in the same folder would land
    on it; ``run_index`` is appended (``config_<folder>_1.yaml``) so an existing
    record is not silently replaced.

    ``exclude_defaults`` keeps the file close to the hand-written config rather than
    expanding every useq default. The ``setup.action`` type discriminator is restored
    by hand -- pydantic drops it as a default, and without it the emitted YAML would
    not be valid against the ``Action`` union even for inspection.

    Never raises -- this is a record written next to the data, and a failure to write
    it must not take the acquisition down between the pre-scan and the timelapse.
    """
    import yaml

    stem = f"config_{output_dir.name}"
    if run_index is not None:
        stem = f"{stem}_{run_index}"
    path = output_dir / f"{stem}.yaml"
    try:
        data = timelapse_seq.model_dump(mode="json", exclude_defaults=True)
        setup = data.get("setup")
        if isinstance(setup, dict) and isinstance(setup.get("action"), dict):
            setup["action"].setdefault("type", timelapse_seq.setup.action.type)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    except Exception:
        logger.exception(
            "FOV selection: could not write the selected-FOV config to %s; the "
            "acquisition is unaffected",
            path,
        )
        return
    logger.info(
        "FOV selection: wrote the acquisition config with %d selected FOVs to %s",
        len(timelapse_seq.stage_positions),
        path,
    )


def launch_feature_viewer(csv_path: Path | None, model_cfg: dict | None = None) -> None:
    """Open the FOV feature viewer on a calibration pre-scan's feature matrix.

    Launched as a detached subprocess (``python -m
    shrimpy.fov_selection.feature_viewer <csv>``) so its Qt event loop stays clear
    of the acquisition process. When ``model_cfg`` carries a ``features`` block, it
    is written beside the CSV and passed via ``--rank-profile`` so the Rank tab opens
    pre-populated with the config's ``fov_selection.model`` curves (merged over the
    data-seeded defaults) rather than bare defaults. Never raises: the calibration
    data is already on disk, so a failure to launch is logged with the manual command
    rather than taking the run down.
    """
    if csv_path is None or not Path(csv_path).exists():
        logger.warning(
            "FOV-selection calibration: feature matrix %s was not written; open the "
            "viewer manually once the CSV exists: "
            "`python -m shrimpy.fov_selection.feature_viewer <csv>`.",
            csv_path,
        )
        return
    import subprocess
    import sys

    csv_path = Path(csv_path)
    profile_path = write_rank_profile(csv_path, model_cfg)
    logger.info("FOV-selection calibration: launching the feature viewer on %s", csv_path)
    cmd = [
        sys.executable,
        "-m",
        "shrimpy.fov_selection.feature_viewer",
        "--start-tab",
        "rank",
    ]
    if profile_path is not None:
        cmd += ["--rank-profile", str(profile_path)]
    cmd.append(str(csv_path))
    try:
        subprocess.Popen(cmd)
    except Exception:
        logger.exception(
            "FOV-selection calibration: could not launch the feature viewer; open it "
            "manually: `python -m shrimpy.fov_selection.feature_viewer %s`.",
            csv_path,
        )


def write_rank_profile(csv_path: Path, model_cfg: dict | None) -> Path | None:
    """Write the config's ``model`` block beside the CSV so the viewer can seed the
    Rank tab from it (``--rank-profile``). Returns the file path, or ``None`` when
    there is nothing to seed (no model, or a model with no ``features`` mapping, e.g.
    a trained-tree model loaded from a ``.joblib``). Never raises: seeding is a
    convenience, so a write failure just falls back to the data-seeded defaults.
    """
    if not model_cfg or not model_cfg.get("features"):
        return None
    import yaml

    profile_path = csv_path.with_name(csv_path.stem + "_config_ranking_profile.yaml")
    try:
        profile_path.write_text(
            yaml.safe_dump(model_cfg, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    except Exception:
        logger.exception(
            "FOV-selection calibration: could not write the config ranking profile to "
            "%s; the viewer will open with data-seeded defaults",
            profile_path,
        )
        return None
    return profile_path
