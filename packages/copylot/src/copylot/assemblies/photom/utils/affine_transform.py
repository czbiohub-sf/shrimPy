"""Affine transform between camera (image) and mirror coordinates.

Vendored from coPylot ``copylot/assemblies/photom/utils/affine_transform.py``,
``utils/settings.py`` and ``utils/io.py`` (main @ 0bcc3a3), BSD-3-Clause;
see the package LICENSE.

Changes from upstream:

- the settings model is pydantic v2 (``ConfigDict(extra="forbid")``, ``model_dump``)
  and validates that the matrix is 3x3; the YAML helpers are inlined
- constructing without ``config_file`` no longer writes ``./affine_transform.yml``
  into the working directory (upstream's ``make_config`` tried to ``mkdir`` the
  YAML path and failed); the identity matrix stays in memory until
  :meth:`AffineTransform.save_matrix` is called
- scikit-image is imported only by :meth:`AffineTransform.compute_affine_matrix`
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from warnings import warn

import numpy as np
import yaml

from pydantic import BaseModel, ConfigDict, field_validator


class AffineTransformationSettings(BaseModel):
    """On-disk format of a mirror's affine calibration."""

    model_config = ConfigDict(extra="forbid")

    affine_transform_yx: list[list[float]]

    @field_validator("affine_transform_yx")
    @classmethod
    def _check_3x3(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("affine_transform_yx must be a 3x3 matrix")
        return value


def _load_settings(config_file: Path) -> AffineTransformationSettings:
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"The YAML file '{config_file}' does not exist.")
    with open(config_file) as f:
        return AffineTransformationSettings(**yaml.safe_load(f))


def _save_settings(settings: AffineTransformationSettings, config_file: Path) -> None:
    with open(config_file, "w") as f:
        yaml.dump(settings.model_dump(), f, default_flow_style=False, sort_keys=False)


class AffineTransform:
    """A 3x3 affine transformation, optionally backed by a YAML file."""

    def __init__(self, tx_matrix: np.ndarray | None = None, config_file: Path | None = None):
        """
        Parameters
        ----------
        tx_matrix : array-like, optional
            3x3 affine matrix. Defaults to identity.
        config_file : Path, optional
            YAML file to load the matrix from (overrides ``tx_matrix``); also
            the default destination of :meth:`save_matrix`.
        """
        self.config_file = Path(config_file) if config_file is not None else None

        if tx_matrix is None:
            self.reset_T_affine()
        else:
            tx_matrix = np.array(tx_matrix)
            assert tx_matrix.shape == (3, 3)
            self.T_affine = tx_matrix

        if self.config_file is not None:
            self.load_matrix()

    def reset_T_affine(self):
        """Reset the affine matrix to identity."""
        self.T_affine = np.eye(3)

    def compute_affine_matrix(self, origin, dest) -> np.ndarray:
        """Estimate the affine matrix mapping ``origin`` to ``dest`` points.

        Parameters
        ----------
        origin : sequence of (x, y)
            At least 3 origin coordinates, e.g. ``[(x1, y1), (x2, y2), (x3, y3)]``.
        dest : sequence of (x, y)
            The matching destination coordinates.

        Returns
        -------
        np.ndarray
            The 3x3 affine matrix (also stored on this object).
        """
        from skimage import transform

        if not (isinstance(origin, Iterable) and len(origin) >= 3):
            raise ValueError("origin needs 3 coordinates.")
        if not (isinstance(dest, Iterable) and len(dest) >= 3):
            raise ValueError("dest needs 3 coordinates.")
        self.T_affine = transform.estimate_transform(
            "affine", np.float32(origin), np.float32(dest)
        ).params
        return self.T_affine

    def get_affine_matrix(self) -> np.ndarray:
        return self.T_affine

    def set_affine_matrix(self, matrix: np.ndarray):
        assert matrix.shape == (3, 3)
        self.T_affine = matrix

    def apply_affine(self, coord_list: list) -> list:
        """Apply the affine transform.

        Parameters
        ----------
        coord_list : list
            Origin coordinates, either point-wise (``[[x, y], ...]``) or
            channel-wise (``[[x0, x1, ...], [y0, y1, ...]]``). Lists of two or
            fewer entries are read channel-wise.

        Returns
        -------
        list
            Destination coordinates, channel-wise: ``[[x...], [y...]]``.
        """
        coord_array = np.stack(coord_list, axis=0)
        if len(coord_list) <= 2:
            coord_array = coord_array.reshape(2, -1)
        else:
            coord_array = coord_array.T
        if self.T_affine is None:
            warn(
                "Affine matrix has not been determined yet. \n"
                "coord_list is returned without transformation.",
                stacklevel=2,
            )
            dest_list = coord_array
        else:
            coord_array = np.vstack((coord_array, np.ones((1, coord_array.shape[1]))))
            dest_list = self.T_affine @ coord_array
        return [list(i) for i in dest_list]

    def trans_pointwise(self, coord_list):
        """Channel-wise to point-wise: ``[[x...], [y...]]`` -> ``[[x, y], ...]``."""
        return [[x, y] for x, y in zip(coord_list[0], coord_list[1], strict=True)]

    def trans_chwise(self, coord_list):
        """Point-wise to channel-wise: ``[[x, y], ...]`` -> ``[[x...], [y...]]``."""
        return [[c[0] for c in coord_list], [c[1] for c in coord_list]]

    def save_matrix(self, matrix: np.ndarray | None = None, config_file: Path | None = None):
        """Save an affine matrix to YAML.

        Parameters
        ----------
        matrix : np.ndarray, optional
            3x3 matrix to save; defaults to the current matrix.
        config_file : Path, optional
            Destination; defaults to (and updates) :attr:`config_file`.
        """
        if matrix is None:
            if self.T_affine is None:
                raise ValueError("provided matrix is not defined")
            matrix = self.T_affine
        else:
            matrix = np.array(matrix)
            assert matrix.shape == (3, 3)

        if config_file is not None:
            self.config_file = Path(config_file)
        if self.config_file is None:
            raise ValueError("No config_file given to save the affine matrix to")

        _save_settings(
            AffineTransformationSettings(affine_transform_yx=matrix.tolist()),
            self.config_file,
        )

    def load_matrix(self, config_file: Path | None = None) -> None:
        """Load the affine matrix from YAML (defaults to :attr:`config_file`)."""
        if config_file is not None:
            self.config_file = Path(config_file)
        if self.config_file is None:
            raise ValueError("No config_file given to load the affine matrix from")

        settings = _load_settings(self.config_file)
        self.T_affine = np.array(settings.affine_transform_yx)
