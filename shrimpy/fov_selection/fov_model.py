"""Pluggable FOV-goodness models.

A :class:`FovModel` maps a per-FOV feature table (named columns) to a decision
``(proba, good)``. Models depend ONLY on feature *names* -- never on which channel /
projection produced a feature -- so feature extraction (preprocessing -> project ->
segment -> named features, in :mod:`pipeline`) is fully decoupled from the model. Any
model type can therefore pair with any preprocessing that yields the feature names the
model asks for (``feature_names``).

Built from the ``fov_selection.model`` config block by :func:`build_fov_model`; every
model has a ``type``:

    type: ranking_by_defined_range       -> DesirabilityModel  (weighted-desirability score)
    type: classification_by_thresholding -> ThresholdingModel  (hard [lo, hi] box)
    type: classification_tree            -> TrainedTreeModel   (trained .joblib: imputer + tree)

Adding a model type = add a :class:`FovModel` subclass + a branch in
:func:`build_fov_model`; nothing in the feature-extraction or acquisition path changes.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# Interpretable-parameter conversions, so every shape is defined by "where it's best" + "how
# forgiving it is" rather than raw internal bounds / steepness constants:
#   gaussian  : sigma = fwhm * _FWHM_TO_SIGMA        (FWHM = 2*sqrt(2 ln2)*sigma ~= 2.3548 sigma)
#   lognormal : log-space sigma = ln(fold) * _HWHM_TO_SIGMA  (d=0.5 at center*fold & center/fold)
#   sigmoid   : logistic rate k/span = _SIGMOID_10_90 / width (10%->90% rise spans `width`)
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
_HWHM_TO_SIGMA = 1.0 / math.sqrt(2.0 * math.log(2.0))
_SIGMOID_10_90 = math.log(81.0)

# All selectable FOV model types (every model config carries one of these as "type").
MODEL_TYPES = frozenset(
    {"ranking_by_defined_range", "classification_by_thresholding", "classification_tree"}
)


class FovModel:
    """Interface: a FOV-goodness model over a named feature table.

    Subclasses set :attr:`feature_names` (the columns the model reads) and implement
    :meth:`predict`. ``predict`` returns ``(proba, good)`` where ``proba`` is a per-row
    array in ``[0, 1]`` and ``good`` a list of bools.
    """

    feature_names: list[str] = []

    def predict(self, matrix_df, threshold: float = 0.5):  # pragma: no cover - interface
        raise NotImplementedError


class ThresholdingModel(FovModel):
    """Hard QC box: a FOV is good iff every feature is inside its ``[lo, hi]`` range.

    ``features`` maps feature name -> ``{range: [lo, hi]}`` (or a bare ``[lo, hi]`` list).
    ``proba`` is the fraction of features in range (1.0 == every feature in range);
    ``threshold`` is unused (the box is a hard AND).
    """

    def __init__(self, features: dict) -> None:
        self._features = features or {}
        self.feature_names = list(self._features)

    def predict(self, matrix_df, threshold: float = 0.5):
        n = len(matrix_df)
        in_box = np.ones(n, bool)
        in_count = np.zeros(n, float)
        for f, spec in self._features.items():
            rng = spec.get("range") if isinstance(spec, dict) else spec
            lo, hi = float(rng[0]), float(rng[1])
            col = (
                matrix_df[f].to_numpy(float) if f in matrix_df.columns else np.full(n, np.nan)
            )
            ok = (col >= lo) & (col <= hi)
            in_box &= ok
            in_count += ok.astype(float)
        proba = in_count / (len(self._features) or 1)
        return proba, [bool(x) for x in in_box]


class DesirabilityModel(FovModel):
    """User-defined desirable ranges -> weighted-desirability score (no training).

    Scores each FOV as ``sum_j weight_j * desirability_j(x_j)`` (see :meth:`_desirability`).
    ``proba`` is that score divided by the total weight (a weighted-mean desirability in
    ``[0, 1]``) and is used purely as a RANKING score. ``shape`` (linear|sigmoid|gaussian|
    lognormal -- see :attr:`SHAPES`) is REQUIRED per feature. Every shape is defined by
    interpretable "where it's best" + "how forgiving" parameters (no raw sigma / steepness):

    - ``gaussian`` (symmetric bell): ``center`` (peak, desirability 1) + ``fwhm`` (full width
      at half maximum -- desirability 0.5 at ``center +- fwhm/2``).
    - ``lognormal`` (right-skewed bell, x>0): ``center`` (peak) + ``fold`` (multiplicative
      tolerance -- desirability 0.5 at ``center*fold`` and ``center/fold``).
    - ``sigmoid`` (logistic): monotonic -- ``midpoint`` (desirability 0.5) + ``width`` (the
      10%->90% transition span) + ``direction`` (higher|lower); OR a soft band --
      ``half_band`` [lo, hi] (desirability 0.5 at each edge) + ``width``.
    - ``linear`` (bounded, reaches exactly 0/1): monotonic ramp -- ``onset`` (desirability 0)
      + ``ideal`` (desirability 1), direction implied by their order; OR a target band --
      ``range`` [lo, hi] (plateau of 1) + optional ``soft`` (linear shoulder width to 0; a
      scalar or ``[left, right]`` pair, ``soft_left``/``soft_right`` override either side).

    Every feature also takes an optional ``weight`` (default 1.0).

    A missing feature contributes 0 desirability regardless of direction -- it adds nothing to
    the weighted score while still counting in the total weight, so missing features pull the
    score down (a fully empty FOV scores 0). It is NOT imputed to a raw value of 0 (which for
    a "lower"-is-better feature would wrongly read as ideal). See :attr:`MISSING_DESIRABILITY`.

    Selection is pure ranking (no threshold): the manager keeps the ``model.top_fov``
    highest-scoring FOVs across the whole pre-scan (see
    :meth:`shrimpy.fov_selection.manager.FovSelection.passed_position_names`). The ``good``
    flag returned by :meth:`predict` (``proba >= threshold``) is therefore unused by the
    ranking selection and left only for standalone / test use.
    """

    DIRECTIONS = ("target", "higher", "lower")
    # Curve family for the transition. 'linear' is bounded (finite shoulders / clipped ramp,
    # reaches 0); the others have tails that never reach exactly 0. 'gaussian' is a normal bell
    # set by center/fwhm (see _gaussian_bounds). 'curve_k' tunes the remaining two: sigmoid =
    # logistic sharpness (def 6); lognormal = tail exponent beta (def 2) in log(x), a
    # right-skewed bell with a long RIGHT tail (needs x>0). 'linear'/'gaussian' ignore curve_k.
    SHAPES = ("linear", "sigmoid", "gaussian", "lognormal")

    # Desirability of a missing (NaN) feature: 0 for every direction. A NaN means the feature
    # could not be measured (e.g. too few / no objects segmented) -- evidence of a degenerate
    # FOV, not of an ideal value -- so it contributes 0 to the weighted score. It is NOT
    # imputed to a raw value of 0 (which for a "lower"-is-better feature would read as ideal).
    # The feature's weight still counts in the denominator, so missing features pull the score
    # down (penalizing sparse/empty FOVs) rather than being ignored.
    MISSING_DESIRABILITY = 0.0

    @staticmethod
    def _gaussian_bounds(center: float, fwhm: float, name: str = "") -> tuple[float, float]:
        """Convert an interpretable gaussian (``center`` peak + ``fwhm`` width at half max) to
        the internal ``(lo, hi)`` = ``center +- sigma`` (+-1 sigma) that :meth:`_desirability`
        consumes, where ``sigma = fwhm * _FWHM_TO_SIGMA``. Raises if ``fwhm <= 0``."""
        if fwhm <= 0:
            where = f"feature {name!r} " if name else ""
            raise ValueError(f"{where}gaussian 'fwhm' must be > 0; got {fwhm}")
        sigma = fwhm * _FWHM_TO_SIGMA
        return center - sigma, center + sigma

    @staticmethod
    def _lognormal_bounds(center: float, fold: float, name: str = "") -> tuple[float, float]:
        """Convert an interpretable lognormal (``center`` peak + ``fold`` multiplicative
        tolerance -- desirability 0.5 at ``center*fold`` and ``center/fold``) to the internal
        ``(lo, hi)`` = the +-1 sigma points in log-space that :meth:`_desirability` consumes
        (log-space ``sigma = ln(fold) * _HWHM_TO_SIGMA``). Raises if ``center<=0`` or ``fold<=1``."""
        where = f"feature {name!r} " if name else ""
        if center <= 0:
            raise ValueError(f"{where}lognormal 'center' must be > 0; got {center}")
        if fold <= 1.0:
            raise ValueError(f"{where}lognormal 'fold' must be > 1; got {fold}")
        s = math.log(fold) * _HWHM_TO_SIGMA
        return center * math.exp(-s), center * math.exp(s)

    # How per-feature desirabilities combine into the FOV score (see :meth:`predict`):
    #   'sum'      weighted arithmetic mean  (COMPENSATORY: a high feature offsets a low one)
    #   'product'  weighted geometric mean   (NON-compensatory: one near-0 feature tanks it)
    #   'gaussian' joint N-D gaussian density (product of per-feature gaussians; strongest veto)
    AGGREGATIONS = ("sum", "product", "gaussian")
    # Floor on a per-feature desirability before a log (product / gaussian modes), so a single
    # 0 (or a missing feature) drives the score very low without producing -inf / exactly 0.
    _DESIRABILITY_FLOOR = 1e-6

    def __init__(self, model_cfg: dict) -> None:
        feats = model_cfg.get("features") or {}
        if not feats:
            raise ValueError("ranking_by_defined_range model has no 'features'")
        self._aggregation = str(model_cfg.get("aggregation", "sum")).lower()
        if self._aggregation not in self.AGGREGATIONS:
            raise ValueError(
                f"ranking_by_defined_range 'aggregation' must be one of {self.AGGREGATIONS}; "
                f"got {self._aggregation!r}"
            )
        # (name, lo, hi, direction, soft_left, soft_right, weight)
        self._specs: list[tuple] = []
        self.feature_names: list[str] = []
        total_weight = 0.0
        for name, spec in feats.items():
            cfg = spec if isinstance(spec, dict) else {"range": spec}
            # `shape` is required: the desirability curve must be stated explicitly per feature
            # rather than silently defaulting, so a profile always records its intent.
            shape = cfg.get("shape")
            if shape is None:
                raise ValueError(
                    f"feature {name!r} needs a 'shape' (one of {self.SHAPES}); "
                    "it must be specified explicitly for ranking_by_defined_range"
                )
            if shape not in self.SHAPES:
                raise ValueError(
                    f"feature {name!r} shape must be one of {self.SHAPES}; got {shape!r}"
                )
            weight = float(cfg.get("weight", 1.0))
            # Defaults; each shape overrides what it uses. soft_left/right apply only to a
            # linear target band, curve_k only to sigmoid; both stay 0 otherwise. Every shape
            # is stored as the internal (lo, hi, direction, ...) tuple :meth:`_desirability` reads.
            direction, soft_left, soft_right, curve_k = "target", 0.0, 0.0, 0.0

            if shape == "gaussian":  # center (peak) + fwhm (width at half max)
                center, fwhm = cfg.get("center"), cfg.get("fwhm")
                if center is None or fwhm is None:
                    raise ValueError(
                        f"feature {name!r} gaussian shape needs 'center' and 'fwhm'"
                    )
                lo, hi = self._gaussian_bounds(float(center), float(fwhm), name)

            elif shape == "lognormal":  # center (peak) + fold (multiplicative tolerance)
                center, fold = cfg.get("center"), cfg.get("fold")
                if center is None or fold is None:
                    raise ValueError(
                        f"feature {name!r} lognormal shape needs 'center' and 'fold'"
                    )
                lo, hi = self._lognormal_bounds(float(center), float(fold), name)

            elif shape == "sigmoid":  # midpoint+width (monotonic) OR half_band+width (band)
                width = cfg.get("width")
                if width is None or float(width) <= 0:
                    raise ValueError(f"feature {name!r} sigmoid shape needs a 'width' > 0")
                width = float(width)
                if "half_band" in cfg:
                    band = cfg["half_band"]
                    if len(band) != 2 or float(band[1]) < float(band[0]):
                        raise ValueError(
                            f"feature {name!r} sigmoid 'half_band' must be [lo, hi] with hi>=lo"
                        )
                    lo, hi, direction = float(band[0]), float(band[1]), "target"
                    curve_k = _SIGMOID_10_90 * (hi - lo) / width
                elif "midpoint" in cfg:
                    direction = cfg.get("direction")
                    if direction not in ("higher", "lower"):
                        raise ValueError(
                            f"feature {name!r} sigmoid with 'midpoint' needs direction "
                            "'higher' or 'lower'"
                        )
                    mid = float(cfg["midpoint"])
                    lo, hi, curve_k = mid - 0.5 * width, mid + 0.5 * width, _SIGMOID_10_90
                else:
                    raise ValueError(
                        f"feature {name!r} sigmoid shape needs 'midpoint' (+direction) for a "
                        "monotonic curve or 'half_band' for a soft band"
                    )

            else:  # linear: onset/ideal ramp (monotonic) OR range+soft band (target)
                if "onset" in cfg or "ideal" in cfg:
                    onset, ideal = cfg.get("onset"), cfg.get("ideal")
                    if onset is None or ideal is None:
                        raise ValueError(
                            f"feature {name!r} linear ramp needs both 'onset' and 'ideal'"
                        )
                    onset, ideal = float(onset), float(ideal)
                    if onset == ideal:
                        raise ValueError(
                            f"feature {name!r} linear 'onset' and 'ideal' must differ"
                        )
                    if ideal > onset:  # higher-is-better: 0 at onset, 1 at ideal
                        direction, lo, hi = "higher", onset, ideal
                    else:  # lower-is-better: 1 at ideal, 0 at onset
                        direction, lo, hi = "lower", ideal, onset
                else:
                    rng = cfg.get("range")
                    if rng is None or len(rng) != 2:
                        raise ValueError(
                            f"feature {name!r} linear shape needs 'range' [lo, hi] (target "
                            "band) or 'onset'/'ideal' (monotonic ramp)"
                        )
                    lo, hi = float(rng[0]), float(rng[1])
                    if hi < lo:
                        raise ValueError(f"feature {name!r} range has hi < lo: {rng}")
                    # `soft` (shoulder) may be a scalar (both sides) or a [left, right] pair;
                    # `soft_left`/`soft_right` override either side. Default: the band width.
                    soft = cfg.get("soft")
                    if isinstance(soft, (list, tuple)):
                        soft_left, soft_right = float(soft[0]), float(soft[1])
                    else:
                        base = float(soft) if soft else max(hi - lo, 1e-9)
                        soft_left = float(cfg.get("soft_left", base))
                        soft_right = float(cfg.get("soft_right", base))

            self._specs.append(
                (name, lo, hi, direction, soft_left, soft_right, weight, shape, curve_k)
            )
            self.feature_names.append(name)
            total_weight += weight
        self._total_weight = total_weight or 1.0

    @classmethod
    def _desirability(
        cls,
        values,
        lo: float,
        hi: float,
        direction: str,
        soft_left: float,
        soft_right: float,
        shape: str = "linear",
        curve_k: float = 0.0,
    ) -> np.ndarray:
        """Per-value desirability in ``[0, 1]`` for one feature, with a selectable curve shape.

        linear      : BOUNDED. target = plateau 1 in ``[lo, hi]`` with finite linear shoulders
                      of width ``soft_left`` / ``soft_right`` reaching exactly 0; higher/lower
                      ramp linearly across ``[lo, hi]`` and clip to 0/1 beyond. Only this shape
                      uses the shoulder widths.
        gaussian    : generalized-gaussian bell (``lo``/``hi`` are the +-1 sigma points);
                      ``curve_k`` = tail exponent (2 normal, 1 Laplace/long, <1 longer). Tails
                      approach 0 but never reach it. Direction ignored.
        lognormal   : gaussian in ``log(x)`` -- a right-skewed bell with a long RIGHT tail
                      (needs ``x>0``; peak at the geometric mean ``sqrt(lo*hi)``, ``lo``/``hi``
                      the +-1 sigma points in log-space, ``curve_k`` the tail exponent). x<=0 -> 0.
        sigmoid     : logistic across the range (crossover at the midpoint, width ``hi - lo``,
                      sharpness ``curve_k`` [default 6]); asymptotic tails -- never exactly 0/1.

        A missing (NaN) value contributes 0 desirability (see :attr:`MISSING_DESIRABILITY`).
        """
        v = np.asarray(values, float)
        out = np.full(v.shape, cls.MISSING_DESIRABILITY)  # missing (NaN) -> 0 desirability
        m = ~np.isnan(v)
        x = v[m]
        span = (hi - lo) or 1e-9
        if shape == "gaussian":  # generalized gaussian bell; lo/hi are the +-1 sigma points.
            # curve_k = tail exponent beta (<=0 -> 2 = normal). beta=1 is Laplace (long
            # exponential tails), beta<1 even longer; beta only stretches the tail, the mean
            # (midpoint) and slope (spread) stay set by lo/hi. Never reaches exactly 0.
            c, s = 0.5 * (lo + hi), max(0.5 * span, 1e-9)
            beta = curve_k if curve_k > 0 else 2.0
            d = np.exp(-0.5 * np.abs((x - c) / s) ** beta)
        elif shape == "lognormal":  # generalized gaussian in log(x): right-skewed bell, long
            # RIGHT tail. lo/hi are the +-1 sigma points in log-space (needs lo, hi > 0); the
            # peak sits at the geometric mean sqrt(lo*hi). curve_k = tail exponent beta (as
            # gaussian). x <= 0 is outside the support -> 0.
            lo_p, hi_p = max(lo, 1e-12), max(hi, 1e-12)
            if hi_p <= lo_p:
                hi_p = lo_p * (1.0 + 1e-6)
            mu, s = 0.5 * (np.log(lo_p) + np.log(hi_p)), max(0.5 * np.log(hi_p / lo_p), 1e-9)
            beta = curve_k if curve_k > 0 else 2.0
            d = np.zeros_like(x)
            pos = x > 0
            d[pos] = np.exp(-0.5 * np.abs((np.log(x[pos]) - mu) / s) ** beta)
        elif shape == "sigmoid":  # logistic; asymptotic tails, never exactly 0/1
            k = curve_k or 6.0
            if direction == "target":
                up = 1.0 / (1.0 + np.exp(-k * (x - lo) / span))
                down = 1.0 / (1.0 + np.exp(k * (x - hi) / span))
                d = up * down
            else:
                s = 1.0 / (1.0 + np.exp(-k * (x - 0.5 * (lo + hi)) / span))
                d = s if direction == "higher" else 1.0 - s
        elif direction == "higher":  # linear, bounded
            d = np.clip((x - lo) / span, 0.0, 1.0)
        elif direction == "lower":
            d = np.clip(1.0 - (x - lo) / span, 0.0, 1.0)
        else:  # linear target: finite linear shoulders (the only shape that uses `soft`)
            rising = np.clip(1.0 - (lo - x) / (soft_left or 1e-9), 0.0, 1.0)
            falling = np.clip(1.0 - (x - hi) / (soft_right or 1e-9), 0.0, 1.0)
            d = np.minimum(rising, falling)
        out[m] = np.clip(d, 0.0, 1.0)
        return out

    def predict(self, matrix_df, threshold: float = 0.5):
        n = len(matrix_df)
        # Per-feature desirability array (n,) and weight, stacked into D (n_features, n).
        rows, weights = [], []
        for (
            name,
            lo,
            hi,
            direction,
            soft_left,
            soft_right,
            weight,
            shape,
            curve_k,
        ) in self._specs:
            col = (
                matrix_df[name].to_numpy(float)
                if name in matrix_df.columns
                else np.full(n, np.nan)
            )
            rows.append(
                self._desirability(col, lo, hi, direction, soft_left, soft_right, shape, curve_k)
            )
            weights.append(weight)
        proba = self._aggregate(np.vstack(rows), np.asarray(weights, float))
        return proba, [bool(p >= threshold) for p in proba]

    def _aggregate(self, d: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Combine the per-feature desirabilities ``d`` (n_features, n) with weights ``w`` into
        one score per FOV, per :attr:`AGGREGATIONS`.

        'sum'      weighted arithmetic mean ``Σ w·d / Σ w`` in [0, 1]. Compensatory: a strong
                   feature can offset a weak one (a good coverage can mask a starved center).
        'product'  weighted geometric mean ``(Π d^w)^(1/Σw)`` in [0, 1]. Non-compensatory: any
                   feature near 0 pulls the whole score toward 0, so gate features act as vetoes.
        'gaussian' joint N-D gaussian density ``exp(-0.5 Σ w·z²)`` with ``z² = -2 ln d`` (so a
                   gaussian feature contributes exactly ``((x-center)/sigma)²``). This is the
                   unnormalized product of the per-feature gaussians; it drops fastest as more
                   features fall short, the strongest veto of the three.

        Missing features already map to desirability 0 (:attr:`MISSING_DESIRABILITY`); under
        'sum' they keep their weight in the denominator (proportional penalty), under
        'product'/'gaussian' they are floored to :attr:`_DESIRABILITY_FLOOR` so one missing
        feature drives the score very low without making it exactly 0 for every FOV.
        """
        total = float(w.sum()) or 1.0
        if self._aggregation == "sum":
            return (w[:, None] * d).sum(0) / total
        dz = np.clip(d, self._DESIRABILITY_FLOOR, 1.0)
        if self._aggregation == "product":
            return np.exp((w[:, None] * np.log(dz)).sum(0) / total)  # weighted geometric mean
        # gaussian: exp(-0.5 Σ w z²), z² = -2 ln d  ->  exp(Σ w ln d) = Π d^w (joint density)
        z2 = -2.0 * np.log(dz)
        return np.exp(-0.5 * (w[:, None] * z2).sum(0))


class TrainedTreeModel(FovModel):
    """A trained decision-tree model dict ``{imputer, tree, features}`` (from a .joblib).

    Missing feature columns are added as NaN and filled by the model's median imputer,
    matching the offline predictor. ``proba`` is ``P(good)``; ``good`` is ``proba >= threshold``.
    """

    def __init__(self, model_dict: dict) -> None:
        self._model = model_dict
        self.feature_names = list(model_dict.get("features", []))

    def predict(self, matrix_df, threshold: float = 0.5):
        x = matrix_df.reindex(columns=self._model["features"])
        x_imputed = self._model["imputer"].transform(x)
        proba = self._model["tree"].predict_proba(x_imputed)[:, 1]
        return np.asarray(proba, float), [bool(p >= threshold) for p in proba]


def build_fov_model(model_cfg: dict) -> FovModel:
    """Construct the :class:`FovModel` for the ``fov_selection.model`` config block.

    ``type`` selects the model (:data:`MODEL_TYPES`); ``classification_tree`` additionally
    needs a ``path`` to the trained .joblib. Raises on an unknown/missing ``type``.
    """
    kind = model_cfg.get("type")
    if kind == "ranking_by_defined_range":
        return DesirabilityModel(model_cfg)
    if kind == "classification_by_thresholding":
        return ThresholdingModel(model_cfg.get("features") or {})
    if kind == "classification_tree":
        import joblib

        path = model_cfg.get("path")
        if not path:
            raise ValueError("model type 'classification_tree' requires a 'path' to a .joblib")
        logger.info("FOV selection: loading trained model %s", path)
        return TrainedTreeModel(joblib.load(path))
    raise ValueError(
        f"fov_selection.model.type must be one of {sorted(MODEL_TYPES)}; got {kind!r}."
    )


# --- interpretable <-> internal parameter conversions (for editors like the feature viewer) --
# The DesirabilityModel stores every shape as internal (lo, hi, direction, soft_left,
# soft_right, curve_k) bounds. These two functions map that to/from the interpretable per-shape
# parameters used in configs (center/fwhm, center/fold, midpoint/width, onset/ideal, ...), so a
# GUI can show and edit the SAME knobs the config uses. They are exact inverses (round-trip).
def curve_params(
    shape: str,
    direction: str,
    lo: float,
    hi: float,
    soft_left: float,
    soft_right: float,
    curve_k: float,
) -> dict:
    """Internal bounds -> ordered dict of interpretable params for ``shape`` / ``direction``."""
    if shape == "gaussian":
        return {"center": 0.5 * (lo + hi), "fwhm": (0.5 * (hi - lo)) / _FWHM_TO_SIGMA}
    if shape == "lognormal":
        s = 0.5 * math.log(max(hi, 1e-12) / max(lo, 1e-12))
        return {
            "center": math.sqrt(max(lo, 1e-12) * max(hi, 1e-12)),
            "fold": math.exp(s / _HWHM_TO_SIGMA),
        }
    if shape == "sigmoid":
        width = _SIGMOID_10_90 * (hi - lo) / (curve_k or 6.0)
        if direction == "target":
            return {"half_band_lo": lo, "half_band_hi": hi, "width": width}
        return {"midpoint": 0.5 * (lo + hi), "width": width}
    if direction == "target":  # linear target band
        return {
            "range_lo": lo,
            "range_hi": hi,
            "soft_left": soft_left,
            "soft_right": soft_right,
        }
    onset, ideal = (lo, hi) if direction == "higher" else (hi, lo)  # linear monotonic ramp
    return {"onset": onset, "ideal": ideal}


def curve_bounds(shape: str, direction: str, params: dict) -> tuple:
    """Inverse of :func:`curve_params`: interpretable params -> internal
    ``(lo, hi, soft_left, soft_right, curve_k)``. Raises on invalid values (e.g. fold<=1)."""
    if shape == "gaussian":
        lo, hi = DesirabilityModel._gaussian_bounds(params["center"], params["fwhm"])
        return lo, hi, 0.0, 0.0, 0.0
    if shape == "lognormal":
        lo, hi = DesirabilityModel._lognormal_bounds(params["center"], params["fold"])
        return lo, hi, 0.0, 0.0, 0.0
    if shape == "sigmoid":
        if direction == "target":
            lo, hi = float(params["half_band_lo"]), float(params["half_band_hi"])
            return (
                lo,
                hi,
                0.0,
                0.0,
                _SIGMOID_10_90 * (hi - lo) / (float(params["width"]) or 1e-9),
            )
        mid, w = float(params["midpoint"]), float(params["width"])
        return mid - 0.5 * w, mid + 0.5 * w, 0.0, 0.0, _SIGMOID_10_90
    if direction == "target":  # linear target band
        return (
            float(params["range_lo"]),
            float(params["range_hi"]),
            float(params["soft_left"]),
            float(params["soft_right"]),
            0.0,
        )
    onset, ideal = float(params["onset"]), float(params["ideal"])  # linear monotonic ramp
    return (
        (onset, ideal, 0.0, 0.0, 0.0)
        if direction == "higher"
        else (ideal, onset, 0.0, 0.0, 0.0)
    )
