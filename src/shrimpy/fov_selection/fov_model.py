"""Pluggable FOV-goodness models.

A :class:`FOVModel` maps a per-FOV feature table (named columns) to a decision
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

Adding a model type = add a :class:`FOVModel` subclass + a branch in
:func:`build_fov_model`; nothing in the feature-extraction or acquisition path changes.

:class:`DesirabilityScorer` exposes the ranking model to the feature viewer, which tunes
a profile through it without importing this module.
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


class FOVModel:
    """Interface: a FOV-goodness model over a named feature table.

    Subclasses set :attr:`feature_names` (the columns the model reads) and implement
    :meth:`predict`. ``predict`` returns ``(proba, good)`` where ``proba`` is a per-row array
    in ``[0, 1]`` and ``good`` is either a list of per-FOV bools (the CLASSIFICATION verdict,
    selected on directly) or ``None`` for a pure RANKING model, whose selection is a top-K over
    ``proba`` and which therefore has no per-FOV good/bad notion. ``threshold`` is a
    classification-only knob (see :class:`TrainedTreeModel`); ranking ignores it.
    """

    feature_names: list[str] = []

    def predict(self, matrix_df, threshold: float = 0.5):  # pragma: no cover - interface
        raise NotImplementedError


class ThresholdingModel(FOVModel):
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


class DesirabilityModel(FOVModel):
    """User-defined desirable ranges -> weighted-desirability score (no training).

    Produces one ``proba`` in ``[0, 1]`` per FOV, used purely as a RANKING score. By default
    (:attr:`DEFAULT_AGGREGATION`) that score is a single N-dimensional gaussian evaluated over
    all N features at once -- each feature contributing one axis, with its own centre and sigma
    -- so a single weak feature vetoes the FOV. The other two ``aggregation`` modes instead
    reduce each feature to a scalar desirability (:meth:`_desirability`) and combine those:
    ``sum`` (compensatory weighted mean) or ``product`` (weighted geometric mean). See
    :meth:`_aggregate`. ``shape`` (sigmoid|gaussian|lognormal -- see :attr:`SHAPES`) is
    REQUIRED per feature. Every shape is defined by interpretable "where it's best" + "how
    forgiving" parameters (no raw sigma / steepness):

    - ``gaussian`` (symmetric bell): ``center`` (peak, desirability 1) + ``fwhm`` (full width
      at half maximum -- desirability 0.5 at ``center +- fwhm/2``).
    - ``lognormal`` (right-skewed bell, x>0): ``center`` (peak) + ``fold`` (multiplicative
      tolerance -- desirability 0.5 at ``center*fold`` and ``center/fold``).
    - ``sigmoid`` (monotonic logistic): ``midpoint`` (desirability 0.5) + ``width`` (the
      10%->90% transition span) + ``direction`` (higher|lower).

    Every feature also takes an optional ``weight`` (default 1.0).

    A missing feature contributes 0 desirability regardless of direction, so missing features
    pull the score down (under ``sum`` it adds nothing while still counting in the total
    weight; under ``product``/``gaussian`` it is floored, driving the score very low). It is NOT
    imputed to a raw value of 0 (which for a "lower"-is-better feature would wrongly read as
    ideal). See :attr:`MISSING_DESIRABILITY`.

    Selection is pure ranking: the manager keeps the ``model.top_fov`` highest-scoring FOVs
    per position across the whole pre-scan (see
    :meth:`shrimpy.fov_selection.manager.FOVSelection.passed_position_names`). ``top_fov`` is
    therefore REQUIRED (validated here and, at config-load time, by
    :class:`shrimpy.fov_selection.config.RankingModelSettings`), and there is no per-FOV
    good/bad notion: :meth:`predict` returns ``good=None`` and ignores ``threshold`` (a
    classification-only knob).
    """

    DIRECTIONS = ("target", "higher", "lower")
    # Curve family for the transition; every shape has tails that never reach exactly 0.
    # 'gaussian' is a normal bell set by center/fwhm (see _gaussian_bounds). 'curve_k' tunes
    # the other two: sigmoid = logistic sharpness (def 6); lognormal = tail exponent beta
    # (def 2) in log(x), a right-skewed bell with a long RIGHT tail (needs x>0). 'gaussian'
    # ignores curve_k.
    SHAPES = ("sigmoid", "gaussian", "lognormal")

    # Desirability of a missing (NaN) feature: 0 for every direction. A NaN means the feature
    # could not be measured (e.g. too few / no objects segmented) -- evidence of a degenerate
    # FOV, not of an ideal value -- so it contributes 0 to the weighted score. It is NOT
    # imputed to a raw value of 0 (which for a "lower"-is-better feature would read as ideal).
    # The feature's weight still counts in the denominator, so missing features pull the score
    # down (penalizing sparse/empty FOVs) rather than being ignored.
    MISSING_DESIRABILITY = 0.0

    # Squared normalized distance assigned to a feature that could not be measured (NaN), under
    # the 'gaussian' aggregation. Real distances there are exact and unbounded, so an
    # unmeasurable feature needs its own explicit penalty rather than sharing a saturation cap
    # with genuinely-terrible values: 10 sigma. A feature measured FURTHER off-target than this
    # therefore counts as worse than one that could not be measured -- deliberate, since a
    # 15-sigma coverage is positive evidence of a bad FOV while a NaN is only absent evidence.
    MISSING_Z2 = 100.0

    @classmethod
    def _squared_distance(
        cls,
        values,
        lo: float,
        hi: float,
        direction: str,
        shape: str = "gaussian",
        curve_k: float = 0.0,
    ) -> np.ndarray:
        """Per-value squared normalized distance from the ideal -- the ``z²`` of one axis of the
        joint gaussian score (see :meth:`_aggregate`).

        Computed from the curve's own parameters, NOT recovered from the scalar desirability.
        Inverting ``d`` (``z² = -2 ln d``) saturates as soon as ``d`` hits its floor, which caps
        every distance at ~5.26 sigma and makes all badly-off-target FOVs score identically;
        going direct keeps the score discriminating at any distance.

        ``gaussian``  ``|(x - center)/sigma|^beta`` -- the Mahalanobis distance of the bell
                      (``beta = 2`` gives the textbook ``z²``).
        ``lognormal`` the same in ``log(x)``; ``x <= 0`` is outside the support -> missing.
        ``sigmoid``   no scale parameter exists, so the equivalent distance is derived from the
                      desirability and capped at :attr:`MISSING_Z2`.

        A missing (NaN) value gets :attr:`MISSING_Z2`.
        """
        v = np.asarray(values, float)
        out = np.full(v.shape, cls.MISSING_Z2)
        m = ~np.isnan(v)
        x = v[m]
        span = (hi - lo) or 1e-9
        beta = curve_k if curve_k > 0 else 2.0
        if shape == "gaussian":
            c, s = 0.5 * (lo + hi), max(0.5 * span, 1e-9)
            out[m] = np.abs((x - c) / s) ** beta
        elif shape == "lognormal":
            lo_p, hi_p = max(lo, 1e-12), max(hi, 1e-12)
            if hi_p <= lo_p:
                hi_p = lo_p * (1.0 + 1e-6)
            mu = 0.5 * (np.log(lo_p) + np.log(hi_p))
            s = max(0.5 * np.log(hi_p / lo_p), 1e-9)
            z2 = np.full(x.shape, cls.MISSING_Z2)  # x <= 0 is off the support
            pos = x > 0
            z2[pos] = np.abs((np.log(x[pos]) - mu) / s) ** beta
            out[m] = z2
        else:
            d = cls._desirability(x, lo, hi, direction, shape, curve_k)
            with np.errstate(divide="ignore"):
                out[m] = np.minimum(-2.0 * np.log(np.clip(d, 0.0, 1.0)), cls.MISSING_Z2)
        return out

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

    # How the features become the one FOV score (see :meth:`_aggregate`):
    #   'sum'      weighted arithmetic mean of the per-feature desirabilities
    #              (COMPENSATORY: a high feature offsets a low one)
    #   'product'  weighted geometric mean of them
    #              (NON-compensatory: one near-0 feature tanks it)
    #   'gaussian' NOT a combination of per-feature scores: ONE N-dimensional gaussian over all
    #              N features at once, from each feature's exact distance (strongest veto; the
    #              default -- see DEFAULT_AGGREGATION)
    AGGREGATIONS = ("sum", "product", "gaussian")
    # Aggregation used when a profile does not name one. 'gaussian' by default: a FOV is only
    # good if EVERY feature is acceptable, so one feature falling short should veto the FOV
    # rather than be averaged away by the others (which is what 'sum' does -- under it a FOV
    # can rank first on one strong feature while another sits near 0). Set `aggregation`
    # explicitly in the model config to choose a different combination rule.
    DEFAULT_AGGREGATION = "gaussian"
    # Floor on a per-feature desirability before a log (product / gaussian modes), so a single
    # 0 (or a missing feature) drives the score very low without producing -inf / exactly 0.
    _DESIRABILITY_FLOOR = 1e-6

    def __init__(self, model_cfg: dict) -> None:
        feats = model_cfg.get("features") or {}
        if not feats:
            raise ValueError("ranking_by_defined_range model has no 'features'")
        # top_fov is REQUIRED: this model selects purely by ranking (top-K per position), so
        # without a quota there is nothing to select on. An acquisition config is already
        # rejected for it by RankingModelSettings; duplicated here because this constructor
        # also takes hand-built dicts (tests, offline use, the feature viewer), which would
        # otherwise produce a ranking model that silently has no selection rule.
        top_fov = model_cfg.get("top_fov")
        if top_fov is None or int(top_fov) < 1:
            raise ValueError(
                "ranking_by_defined_range requires 'top_fov' (a positive int): the N "
                f"highest-ranked FOVs per position pass. Got {top_fov!r}."
            )
        self._aggregation = str(
            model_cfg.get("aggregation") or self.DEFAULT_AGGREGATION
        ).lower()
        if self._aggregation not in self.AGGREGATIONS:
            raise ValueError(
                f"ranking_by_defined_range 'aggregation' must be one of {self.AGGREGATIONS}; "
                f"got {self._aggregation!r}"
            )
        # (name, lo, hi, direction, weight, shape, curve_k)
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
            # Defaults; each shape overrides what it uses. curve_k applies to sigmoid. Every
            # shape is stored as the internal (lo, hi, direction, curve_k) tuple that
            # :meth:`_desirability` reads.
            direction, curve_k = "target", 0.0

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

            elif shape == "sigmoid":  # monotonic: midpoint + width + direction (higher|lower)
                width = cfg.get("width")
                if width is None or float(width) <= 0:
                    raise ValueError(f"feature {name!r} sigmoid shape needs a 'width' > 0")
                width = float(width)
                if "midpoint" not in cfg:
                    raise ValueError(f"feature {name!r} sigmoid shape needs a 'midpoint'")
                direction = cfg.get("direction")
                if direction not in ("higher", "lower"):
                    raise ValueError(
                        f"feature {name!r} sigmoid needs direction 'higher' or 'lower'"
                    )
                mid = float(cfg["midpoint"])
                lo, hi, curve_k = mid - 0.5 * width, mid + 0.5 * width, _SIGMOID_10_90

            self._specs.append((name, lo, hi, direction, weight, shape, curve_k))
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
        shape: str = "gaussian",
        curve_k: float = 0.0,
    ) -> np.ndarray:
        """Per-value desirability in ``[0, 1]`` for one feature, with a selectable curve shape.

        gaussian    : generalized-gaussian bell (``lo``/``hi`` are the +-1 sigma points);
                      ``curve_k`` = tail exponent (2 normal, 1 Laplace/long, <1 longer). Tails
                      approach 0 but never reach it. Direction ignored.
        lognormal   : gaussian in ``log(x)`` -- a right-skewed bell with a long RIGHT tail
                      (needs ``x>0``; peak at the geometric mean ``sqrt(lo*hi)``, ``lo``/``hi``
                      the +-1 sigma points in log-space, ``curve_k`` the tail exponent). x<=0 -> 0.
        sigmoid     : monotonic logistic (``direction`` higher|lower) with the crossover at the
                      midpoint, width ``hi - lo``, sharpness ``curve_k`` [default 6]; asymptotic
                      tails -- never exactly 0/1.

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
        else:  # sigmoid: monotonic logistic; asymptotic tails, never exactly 0/1
            k = curve_k or 6.0
            s = 1.0 / (1.0 + np.exp(-k * (x - 0.5 * (lo + hi)) / span))
            d = s if direction == "higher" else 1.0 - s
        out[m] = np.clip(d, 0.0, 1.0)
        return out

    def predict(self, matrix_df, threshold: float = 0.5):
        n = len(matrix_df)
        # Per-feature desirability array (n,) and weight, stacked into D (n_features, n). The
        # 'gaussian' aggregation is a single joint gaussian rather than a combination of the
        # per-feature scores, so it needs each feature's exact squared distance instead.
        joint_gaussian = self._aggregation == "gaussian"
        rows, dists, weights = [], [], []
        for name, lo, hi, direction, weight, shape, curve_k in self._specs:
            col = (
                matrix_df[name].to_numpy(float)
                if name in matrix_df.columns
                else np.full(n, np.nan)
            )
            args = (col, lo, hi, direction, shape, curve_k)
            rows.append(self._desirability(*args))
            if joint_gaussian:
                dists.append(self._squared_distance(*args))
            weights.append(weight)
        proba = self._aggregate(
            np.vstack(rows),
            np.asarray(weights, float),
            np.vstack(dists) if joint_gaussian else None,
        )
        # Ranking model: proba IS the ranking key; there is no per-FOV good/bad verdict
        # (selection is top_fov per position, in the manager). `threshold` is ignored.
        return proba, None

    def _aggregate(
        self, d: np.ndarray, w: np.ndarray, z2: np.ndarray | None = None
    ) -> np.ndarray:
        """One score per FOV, per :attr:`AGGREGATIONS`.

        ``d`` and ``z2`` are both ``(n_features, n)``. The first two modes COMBINE the
        per-feature desirabilities ``d``; the third does not -- it evaluates a single joint
        distribution over the raw distances ``z2``.

        'sum'      weighted arithmetic mean ``Σ w·d / Σ w`` in [0, 1]. Compensatory: a strong
                   feature can offset a weak one (a good coverage can mask a starved center).
        'product'  weighted geometric mean ``(Π d^w)^(1/Σw)`` in [0, 1]. Non-compensatory: any
                   feature near 0 pulls the whole score toward 0, so gate features act as vetoes.
        'gaussian' ONE N-dimensional gaussian over all N features at once, not a combination of
                   N per-feature scores: ``exp(-0.5 · Σ w·z² / Σw)`` with each ``z²`` the exact
                   squared distance of that feature's own curve (:meth:`_squared_distance`), i.e.
                   a gaussian with diagonal covariance whose axes are the features' sigmas. It
                   drops fastest as more features fall short -- the strongest veto of the three.

                   Two consequences of evaluating the joint density directly rather than
                   inverting ``d``: distances stay exact instead of saturating at the
                   desirability floor (~5.26 sigma), so badly-off-target FOVs are still ranked
                   against each other; and a missing feature gets the explicit
                   :attr:`MISSING_Z2` penalty. Dividing by ``Σw`` makes it the geometric mean of
                   the per-axis gaussians rather than their raw product -- a monotonic rescale,
                   so ranking is untouched, but scores stay comparable when the feature count
                   changes and cannot underflow to a tie.

        Missing features map to desirability 0 (:attr:`MISSING_DESIRABILITY`); under 'sum' they
        keep their weight in the denominator (proportional penalty), under 'product' they are
        floored to :attr:`_DESIRABILITY_FLOOR` so one missing feature drives the score very low
        without making it exactly 0 for every FOV.
        """
        total = float(w.sum()) or 1.0
        if self._aggregation == "sum":
            return (w[:, None] * d).sum(0) / total
        if self._aggregation == "product":
            dz = np.clip(d, self._DESIRABILITY_FLOOR, 1.0)
            return np.exp((w[:, None] * np.log(dz)).sum(0) / total)  # weighted geometric mean
        if z2 is None:  # pragma: no cover - predict always supplies it for this mode
            raise ValueError("the 'gaussian' aggregation requires per-feature z² distances")
        return np.exp(-0.5 * (w[:, None] * z2).sum(0) / total)


class TrainedTreeModel(FOVModel):
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


def build_fov_model(model_cfg: dict) -> FOVModel:
    """Construct the :class:`FOVModel` for the ``fov_selection.model`` config block.

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
# The DesirabilityModel stores every shape as internal (lo, hi, direction, curve_k) bounds.
# These two functions map that to/from the interpretable per-shape parameters used in configs
# (center/fwhm, center/fold, midpoint/width), so a GUI can show and edit the
# SAME knobs the config uses. They are exact inverses (round-trip).
def curve_params(shape: str, lo: float, hi: float, curve_k: float) -> dict:
    """Internal bounds -> ordered dict of interpretable params for ``shape``."""
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
        return {"midpoint": 0.5 * (lo + hi), "width": width}
    raise ValueError(f"unknown shape {shape!r}; expected one of {DesirabilityModel.SHAPES}")


def curve_bounds(shape: str, params: dict) -> tuple:
    """Inverse of :func:`curve_params`: interpretable params -> internal
    ``(lo, hi, curve_k)``. Raises on invalid values (e.g. fold<=1)."""
    if shape == "gaussian":
        lo, hi = DesirabilityModel._gaussian_bounds(params["center"], params["fwhm"])
        return lo, hi, 0.0
    if shape == "lognormal":
        lo, hi = DesirabilityModel._lognormal_bounds(params["center"], params["fold"])
        return lo, hi, 0.0
    if shape == "sigmoid":
        mid, w = float(params["midpoint"]), float(params["width"])
        if w <= 0:
            raise ValueError(f"sigmoid 'width' must be > 0; got {w}")
        return mid - 0.5 * w, mid + 0.5 * w, _SIGMOID_10_90
    raise ValueError(f"unknown shape {shape!r}; expected one of {DesirabilityModel.SHAPES}")


def _feature_from_bounds(shape, direction, lo, hi, curve_k, weight) -> dict:
    """Internal ``(lo, hi, curve_k)`` bounds -> a config feature dict.

    Carries only the interpretable params the shape uses (center/fwhm, center/fold, or
    midpoint/width), plus ``direction`` for the monotonic sigmoid and ``weight``.
    """
    feat = {"shape": shape, **curve_params(shape, lo, hi, curve_k)}
    if shape == "sigmoid":
        feat["direction"] = direction
    feat["weight"] = weight
    return feat


def _bounds_from_feature(feat: dict) -> tuple[str, str, float, float, float]:
    """A config feature dict -> internal ``(shape, direction, lo, hi, curve_k)``.

    Reads the interpretable schema via :func:`curve_bounds`. Legacy ``range``-style profiles
    (``range``/``lo``/``hi``, with ``curve_k`` for sigmoids) are read as internal bounds
    directly, so old profile files still open. Raises ``ValueError`` on an unusable spec.
    """
    shape = feat.get("shape", "gaussian")
    if shape == "linear":
        raise ValueError("the 'linear' shape was removed; use gaussian, lognormal, or sigmoid")
    if shape not in DesirabilityModel.SHAPES:
        raise ValueError(
            f"unknown shape {shape!r}; expected one of {DesirabilityModel.SHAPES}"
        )
    names = [name for name, _ in DesirabilityScorer._PARAMS[shape]]
    if all(name in feat for name in names):
        lo, hi, curve_k = curve_bounds(shape, {name: float(feat[name]) for name in names})
        direction = feat.get("direction", "higher") if shape == "sigmoid" else "target"
    else:
        rng = feat.get("range", [feat.get("lo"), feat.get("hi")])
        if rng[0] is None or rng[1] is None:
            raise ValueError(
                f"feature spec {feat!r} lacks the params for a {shape!r} curve "
                "(gaussian: center/fwhm; lognormal: center/fold; "
                "sigmoid: midpoint/width/direction)"
            )
        lo, hi = float(rng[0]), float(rng[1])
        direction = feat.get("direction", "target") if shape == "sigmoid" else "target"
        curve_k = float(feat.get("curve_k", 0.0)) if shape == "sigmoid" else 0.0
    if shape == "sigmoid" and direction not in ("higher", "lower"):
        direction = "higher"
    return shape, direction, lo, hi, curve_k


class DesirabilityScorer:
    """The ranking model (``ranking_by_defined_range``) as an editor sees it.

    Implements the feature viewer's ``Scorer`` interface: every method takes or returns
    one feature's *spec* -- the exact dict that sits under ``fov_selection.model.features``
    in the config (e.g. ``{shape: gaussian, center: 0.4, fwhm: 0.2, weight: 1.0}``) -- so an
    editor never sees the model's internal bounds, and a profile it saves is a valid config.
    Scores are computed by :class:`DesirabilityModel` itself, so what the viewer shows is
    what the acquisition selects on.

    Each spec also has two draggable *handles*, ``lo`` and ``hi``: the band edges of its
    curve (the +-1 sigma points of a bell, the ends of a sigmoid's 10-90 % rise).
    """

    shapes = DesirabilityModel.SHAPES
    aggregations = DesirabilityModel.AGGREGATIONS
    default_aggregation = DesirabilityModel.DEFAULT_AGGREGATION
    default_shape = "gaussian"

    # Interpretable params per shape, in display order, with each one's smallest valid
    # value (None = unbounded): widths must be positive, a lognormal fold must exceed 1.
    _PARAMS = {
        "gaussian": (("center", None), ("fwhm", 1e-9)),
        "lognormal": (("center", 1e-12), ("fold", 1.0 + 1e-6)),
        "sigmoid": (("midpoint", None), ("width", 1e-9)),
    }
    HANDLES = ("lo", "hi")

    def directions(self, shape: str) -> tuple[str, ...]:
        """Directions ``shape`` allows: bells are always 'target', a sigmoid higher|lower."""
        return ("higher", "lower") if shape == "sigmoid" else ("target",)

    def params(self, shape: str) -> tuple[tuple[str, float | None], ...]:
        """``(name, minimum)`` of each interpretable param of ``shape``, in display order."""
        return self._PARAMS[shape]

    def seed(self, values) -> dict:
        """A default spec fitted to one feature's measured ``values``.

        A gaussian whose +-1 sigma band is the values' interquartile range, so it starts
        centered on the data. NaNs are ignored; a constant or empty feature gets a band of
        10 % of its value (or 1.0 around 0), so the spec is always valid.
        """
        v = np.asarray(values, float)
        v = v[~np.isnan(v)]
        lo, hi = (
            (float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75))) if v.size else (0, 0)
        )
        if hi <= lo:
            half = 0.05 * abs(lo) or 0.5
            lo, hi = lo - half, lo + half
        return _feature_from_bounds(self.default_shape, "target", lo, hi, 0.0, 1.0)

    def reshape(
        self, spec: dict, shape: str | None = None, direction: str | None = None
    ) -> dict:
        """``spec`` as a valid spec of ``shape`` / ``direction``, keeping its curve's band.

        ``None`` keeps the current value; a direction ``shape`` does not allow becomes its
        first allowed one. Switching shape keeps the band edges, so a gaussian's center/fwhm
        becomes a sigmoid's midpoint/width over the same range. Also normalizes a spec
        (e.g. one just read back from edited widgets). Raises ``ValueError`` if ``spec``'s
        params are invalid (e.g. ``fold <= 1``).
        """
        cur_shape, cur_direction, lo, hi, curve_k = self._bounds(spec)
        shape = shape or cur_shape
        if shape == "sigmoid" and cur_shape != "sigmoid":
            curve_k = _SIGMOID_10_90  # a bell's +-1 sigma band becomes the 10-90 % rise
        allowed = self.directions(shape)
        direction = direction or cur_direction
        if direction not in allowed:
            direction = allowed[0]
        return _feature_from_bounds(
            shape, direction, lo, hi, curve_k, float(spec.get("weight", 1.0))
        )

    def handles(self, spec: dict) -> list[tuple[str, float]]:
        """``(name, x)`` of each draggable handle: the curve's band edges ``lo`` and ``hi``."""
        _shape, _direction, lo, hi, _curve_k = self._bounds(spec)
        return [("lo", lo), ("hi", hi)]

    def drag(self, spec: dict, handle: str, x: float) -> dict:
        """``spec`` with handle ``lo`` or ``hi`` moved to ``x`` (never past the other one)."""
        shape, direction, lo, hi, curve_k = self._bounds(spec)
        # Keep a sliver between the edges so the band never collapses to an invalid width.
        gap = 1e-9 * max(1.0, abs(lo), abs(hi))
        if handle == "lo":
            lo = min(float(x), hi - gap)
        elif handle == "hi":
            hi = max(float(x), lo + gap)
        else:
            raise ValueError(f"unknown handle {handle!r}; expected one of {self.HANDLES}")
        return _feature_from_bounds(
            shape, direction, lo, hi, curve_k, float(spec.get("weight", 1.0))
        )

    def curve(self, spec: dict, x) -> np.ndarray:
        """Desirability in ``[0, 1]`` of ``spec`` at each value of ``x`` (NaN -> 0)."""
        shape, direction, lo, hi, curve_k = self._bounds(spec)
        return DesirabilityModel._desirability(x, lo, hi, direction, shape, curve_k)

    def score(self, df, features: dict[str, dict], aggregation: str) -> np.ndarray:
        """One score per row of ``df`` from ``features`` (name -> spec), as the acquisition
        computes it. A feature missing from ``df`` counts as unmeasured (NaN)."""
        model = DesirabilityModel(
            {
                "type": "ranking_by_defined_range",
                # A selection quota the model requires but scoring never reads.
                "top_fov": 1,
                "aggregation": aggregation,
                "features": features,
            }
        )
        return np.asarray(model.predict(df)[0], float)

    def read_profile(self, cfg) -> dict[str, dict]:
        """Feature specs (name -> spec) from a saved profile or a ``fov_selection.model``.

        Accepts a bare ``features`` mapping or a full model dict, and upgrades legacy
        ``range``-style specs. Raises ``ValueError`` on a malformed profile.
        """
        feats = cfg.get("features", cfg) if isinstance(cfg, dict) else None
        if not isinstance(feats, dict):
            raise ValueError("a profile must be a mapping of feature name -> spec")
        out = {}
        for name, feat in feats.items():
            if not isinstance(feat, dict):
                raise ValueError(f"feature {name!r}: spec must be a mapping, got {feat!r}")
            shape, direction, lo, hi, curve_k = _bounds_from_feature(feat)
            out[name] = _feature_from_bounds(
                shape, direction, lo, hi, curve_k, float(feat.get("weight", 1.0))
            )
        return out

    def _bounds(self, spec: dict) -> tuple[str, str, float, float, float]:
        """``spec`` -> internal ``(shape, direction, lo, hi, curve_k)``; ValueError if invalid."""
        try:
            return _bounds_from_feature(spec)
        except (KeyError, TypeError) as e:
            raise ValueError(f"invalid feature spec {spec!r}: {e}") from e
