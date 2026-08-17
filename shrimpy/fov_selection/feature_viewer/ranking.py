"""
Tunable FOV ranking --- five interchangeable models over one shared scaffold.

A fixed feature matrix (built by scripts/build_fov_matrix.py) is scored by a *swappable*
RankingProfile, so "good" can be redefined per biological question without touching feature
extraction. All five models below read the same features and `goodness` labels, write a
`score` column via `apply`, and are compared by the same `evaluate`; only the head that
turns features into a score differs.

    Model 1  "desirability"  score = sum_j w_j * d_j(x_j)      (hand-set shapes, learned/knob weights)
    Model 2  "prototype"     score = -(x-mu)^T A (x-mu)        (distance to a learned ideal point)
    Model 3  "linear"        score = sum_j w_j * z_j           (weighted standardized raw features)
    Model 4  "ebm"           score = sum_j g_j(x_j) + ...      (learned per-feature curves; EBM)
    Model 5  "gbm"           score = sum_t f_t(x)              (gradient-boosted trees; LambdaMART)

Two ideas unify the code:

1. A per-feature *representation*. Models 1--3 are all linear in some representation of the
   features -- desirability d(x) in [0,1] (Model 1), a robust z-score z(x) (Model 3), or the
   negative squared deviation -(z-mu)^2 from an ideal point (Model 2). So all three score as
   `M . w` and share ONE convex pairwise-logistic weight solver (`_fit_pairwise_weights`).
   Models 4--5 are non-additive and dispatch to their own libraries.

2. Supervision from label *order*. Labels are ordinal (bad<neutral<good), so training uses
   ordered PAIRS (good>neutral, good>bad, neutral>bad): the learner picks parameters so the
   better FOV of each pair scores higher. `evaluate` reports held-out pairwise accuracy / NDCG.

A RankingProfile bundles the feature list, frozen normalization, per-feature desirability
specs, knob (prior) weights, and the fitted model; it serializes to JSON (with sidecar files
for the GBM booster / EBM object). Qt-free and unit-tested headless (see
shrimpy/tests/test_ranking.py).
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# The per-feature desirability CURVES (sigmoid / gaussian / lognormal)
# are shared with the online model, so Model 1 here and the acquisition-time DesirabilityModel
# score every shape identically -- one implementation, no drift. (fov_model is numpy-only, so
# this keeps ranking.py headless / Qt-free.)
from shrimpy.fov_selection.fov_model import DesirabilityModel

# Optional heavy models are import-guarded so the core stays usable without them.
try:
    import lightgbm as lgb  # noqa: F401

    HAS_LGBM = True
except Exception:  # noqa: BLE001
    HAS_LGBM = False

try:
    from interpret.glassbox import ExplainableBoostingRegressor  # noqa: F401

    HAS_EBM = True
except Exception:  # noqa: BLE001
    HAS_EBM = False

# The five model kinds, in increasing flexibility. Used as `profile.model["kind"]`.
MODELS = ("desirability", "prototype", "linear", "ebm", "gbm")

# Desirability directions a feature can take (Model 1). "target" is the non-monotone one:
# a peak of full desirability on the band [lo, hi] that fades on each side.
DIRECTIONS = ("higher", "lower", "target")

# Default desirability direction per feature-name suffix (the token after "__", or the whole
# name when there is no prefix). Seeds a fresh profile; see feature_extraction.py.
DEFAULT_DIRECTION: dict[str, str] = {
    "coverage_frac": "target",  # area covered: target band
    "nn_um_mean": "target",  # spacing depends on desired density -> target
    "nn_cv": "lower",  # even spacing (low variability) is better
    "empty_grid_frac": "lower",  # few empty regions -> even coverage
    "occupancy_entropy": "higher",  # spread-out occupancy is better
    "max_radius_corner_to_edge": "lower",  # small largest object-free region
    "max_radius_between_cells_norm": "lower",  # small largest empty circle -> no big void
    "nn_mask_um_mean": "target",  # cell-to-cell edge spacing depends on desired density
}

# Robust quantiles frozen into a profile's normalization (feature scale + target defaults).
_NORM_Q = (0.05, 0.25, 0.5, 0.75, 0.95)
_NORM_KEYS = ("q05", "q25", "q50", "q75", "q95")

# goodness label -> ordinal relevance for ranking (higher = better FOV).
GOODNESS_RELEVANCE = {-1.0: 0.0, 0.0: 1.0, 1.0: 2.0}
_GOOD = GOODNESS_RELEVANCE[1.0]  # relevance value of a "good" FOV


# =============================================================== profile container
@dataclass
class RankingProfile:
    """A named, serializable definition of "good" for one biological question.

    features       : feature columns the profile ranks on (order == parameter order).
    normalization  : feature -> {q05,q25,q50,q75,q95}, frozen on a reference pool so scores
                     are comparable across acquisitions.
    desirability   : feature -> {"type": higher|lower|target, "lo","hi", and optional
                     "shape","curve_k" for the transition curve} (Model 1; see `desirability`).
    prior_weights  : feature -> knob weight; the L2 prior for Model 1's learned weights.
    model          : the fitted head, e.g. {"kind": "desirability", "weights": [...]},
                     {"kind": "prototype", "mu": [...], "a": [...], "mode": ...},
                     {"kind": "gbm", "booster": <lgb.Booster>}, etc. Empty -> score from knobs.
    metrics        : last `evaluate` output (held-out pairwise acc / NDCG / leak flags).
    """

    name: str
    features: list[str]
    normalization: dict[str, dict[str, float]]
    desirability: dict[str, dict]
    prior_weights: dict[str, float]
    model: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    # ---- serialization -----------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        """Write JSON. A GBM booster / EBM object cannot live in JSON, so they are saved
        beside the JSON as `<path>.lgb` / `<path>.ebm.joblib` and reloaded by `from_json`."""
        plain = {k: v for k, v in self.model.items() if k not in ("booster", "model")}
        Path(path).write_text(
            json.dumps(
                {
                    "name": self.name,
                    "features": self.features,
                    "normalization": self.normalization,
                    "desirability": self.desirability,
                    "prior_weights": self.prior_weights,
                    "model": plain,
                    "metrics": self.metrics,
                },
                indent=2,
            )
        )
        if self.model.get("booster") is not None:
            Path(str(path) + ".lgb").write_text(self.model["booster"].model_to_string())
        if self.model.get("model") is not None:  # EBM
            import joblib

            joblib.dump(self.model["model"], str(path) + ".ebm.joblib")

    @classmethod
    def from_json(cls, path: str | Path) -> RankingProfile:
        d = json.loads(Path(path).read_text())
        prof = cls(
            name=d["name"],
            features=d["features"],
            normalization=d["normalization"],
            desirability=d["desirability"],
            prior_weights=d["prior_weights"],
            model=d.get("model", {}),
            metrics=d.get("metrics", {}),
        )
        kind = prof.model.get("kind")
        if kind == "gbm" and Path(str(path) + ".lgb").exists() and HAS_LGBM:
            prof.model["booster"] = lgb.Booster(model_str=Path(str(path) + ".lgb").read_text())
        if kind == "ebm" and Path(str(path) + ".ebm.joblib").exists() and HAS_EBM:
            import joblib

            prof.model["model"] = joblib.load(str(path) + ".ebm.joblib")
        return prof

    # ---- weight helpers (Model 1) ------------------------------------------------
    def prior_vector(self) -> np.ndarray:
        """The knob weights as a vector, aligned to `features`."""
        return np.array([float(self.prior_weights[f]) for f in self.features])

    def desirability_weights(self) -> np.ndarray:
        """Effective Model-1 weights: the learned ones if fit, else the knob (prior) ones."""
        if self.model.get("kind") == "desirability" and "weights" in self.model:
            return np.array(self.model["weights"], float)
        return self.prior_vector()


# ========================================================== profile construction
def fit_normalization(df: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    """Freeze robust per-feature quantiles from a reference pool (NaNs ignored)."""
    norm: dict[str, dict[str, float]] = {}
    for f in features:
        v = df[f].to_numpy(float) if f in df.columns else np.array([])
        v = v[~np.isnan(v)]
        qs = np.quantile(v, _NORM_Q) if v.size else np.zeros(len(_NORM_Q))
        norm[f] = {k: float(q) for k, q in zip(_NORM_KEYS, qs, strict=True)}
    return norm


def _feature_suffix(col: str) -> str:
    """'nuclei_vs_max__objects_per_10um2' -> 'objects_per_10um2' (key for direction defaults)."""
    return col.split("__")[-1]


def default_profile(
    df: pd.DataFrame, features: list[str], name: str = "default"
) -> RankingProfile:
    """A fresh profile: robust normalization + suffix-seeded desirability directions + equal
    knob weights. `target` bands default to the feature's [q25, q75]."""
    norm = fit_normalization(df, features)
    desirability: dict[str, dict] = {}
    for f in features:
        direction = DEFAULT_DIRECTION.get(_feature_suffix(f), "higher")
        spec: dict = {"type": direction}
        if direction == "target":
            n = norm[f]
            spec.update(lo=n["q25"], hi=n["q75"])
        desirability[f] = spec
    prior = {f: 1.0 for f in features}
    return RankingProfile(name, list(features), norm, desirability, prior)


# ============================================================ feature representations
def desirability(value: float, spec: dict, norm: dict[str, float]) -> float:
    """How ideal ONE raw feature value is, in [0, 1] (1 == ideal); see module docstring.

    Two independent knobs, both shared with the online DesirabilityModel (one curve impl):

    DIRECTION (``type``; alias ``direction``) -- which way is good:
        higher : desirability rises across the robust range [q05, q95];
        lower  : desirability falls across that range;
        target : peaks on the band [lo, hi] (alias ``range``: [lo, hi]), fading on each side.
    SHAPE (``shape``, default ``gaussian``) -- the transition CURVE across that range:
        sigmoid|gaussian|lognormal (see :attr:`DesirabilityModel.SHAPES`). ``gaussian``
        is an interpretable bell defined by ``center`` (peak) + ``fwhm`` (width at half max),
        ignoring direction/range; ``sigmoid``/``lognormal`` use ``curve_k`` for steepness/tail.

    A missing value (NaN) returns NaN; callers neutralize it to 0.5."""
    if value is None or value != value:  # value != value is True only for NaN
        return float("nan")
    kind = spec.get("type", spec.get("direction", "higher"))
    if kind not in DIRECTIONS:
        raise ValueError(f"unknown desirability type {kind!r}")
    shape = spec.get("shape", "gaussian")
    if shape not in DesirabilityModel.SHAPES:
        raise ValueError(f"unknown desirability shape {shape!r}")
    if shape == "gaussian":
        # Interpretable bell: center (peak) + fwhm (width at half max) -> internal +-1 sigma.
        center, fwhm = spec.get("center"), spec.get("fwhm")
        if center is None or fwhm is None:
            raise ValueError("gaussian desirability needs 'center' and 'fwhm'")
        lo, hi = DesirabilityModel._gaussian_bounds(float(center), float(fwhm))
        d = DesirabilityModel._desirability(
            np.asarray([value], float), lo, hi, "target", "gaussian", 0.0
        )
        return float(d[0])
    curve_k = float(spec.get("curve_k", 0.0))
    rng = spec.get("range")
    if kind == "target":
        # target band [lo, hi].
        if rng is not None and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
        else:
            lo, hi = float(spec.get("lo", norm["q50"])), float(spec.get("hi", norm["q50"]))
    else:
        # higher / lower: the curve spans the robust range [q05, q95] (or an explicit `range`).
        if rng is not None and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
        else:
            lo, hi = float(norm["q05"]), float(norm["q95"])
    # Delegate the actual curve to the shared implementation (a 1-element array in / out).
    d = DesirabilityModel._desirability(
        np.asarray([value], float), lo, hi, kind, shape, curve_k
    )
    return float(d[0])


def desirability_matrix(
    df: pd.DataFrame, profile: RankingProfile, missing: float = 0.5
) -> np.ndarray:
    """(n_fov, n_feature) desirabilities in [0, 1]; a value missing for a FOV -> `missing`."""
    n = len(df)
    des = np.full((n, len(profile.features)), float(missing))
    for j, f in enumerate(profile.features):
        raw = df[f].to_numpy(float) if f in df.columns else np.full(n, np.nan)
        col = np.array(
            [desirability(v, profile.desirability[f], profile.normalization[f]) for v in raw]
        )
        measured = ~np.isnan(raw)
        des[measured, j] = col[measured]
    return des


def standardize_matrix(df: pd.DataFrame, profile: RankingProfile) -> np.ndarray:
    """(n_fov, n_feature) robust z-scores (x - median) / IQR; missing -> 0 (the median)."""
    n = len(df)
    z = np.zeros((n, len(profile.features)))
    for j, f in enumerate(profile.features):
        raw = df[f].to_numpy(float) if f in df.columns else np.full(n, np.nan)
        nrm = profile.normalization[f]
        iqr = (nrm["q75"] - nrm["q25"]) or 1.0
        col = (raw - nrm["q50"]) / iqr
        col[np.isnan(col)] = 0.0
        z[:, j] = col
    return z


def _raw_matrix(df: pd.DataFrame, profile: RankingProfile) -> np.ndarray:
    """(n_fov, n_feature) raw feature values, NaN imputed to the median (for EBM input)."""
    n = len(df)
    r = np.zeros((n, len(profile.features)))
    for j, f in enumerate(profile.features):
        raw = df[f].to_numpy(float) if f in df.columns else np.full(n, np.nan)
        raw = np.where(np.isnan(raw), profile.normalization[f]["q50"], raw)
        r[:, j] = raw
    return r


# =================================================================== labels & pairs
def _labeled_relevance(df: pd.DataFrame, label_col: str) -> np.ndarray:
    """Ordinal relevance per row from `goodness` (bad=0, neutral=1, good=2); NaN if unlabeled."""
    if label_col not in df.columns:
        return np.full(len(df), np.nan)
    g = df[label_col].to_numpy(float)
    rel = np.full(len(df), np.nan)
    for k, v in GOODNESS_RELEVANCE.items():
        rel[g == k] = v
    return rel


def build_pairs(rel: np.ndarray, max_pairs: int = 20000, seed: int = 0) -> np.ndarray:
    """All (winner, loser) index pairs where rel[i] > rel[j], subsampled to `max_pairs`.

    good>neutral, good>bad, neutral>bad all become training pairs. Returns an (m, 2) array
    of positional indices into `rel`; empty if there is no contrast (e.g. one class only)."""
    idx = np.where(~np.isnan(rel))[0]
    pairs = [(i, j) for i in idx for j in idx if rel[i] > rel[j]]
    if not pairs:
        return np.empty((0, 2), int)
    pairs = np.array(pairs, int)
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = pairs[rng.choice(len(pairs), max_pairs, replace=False)]
    return pairs


# ============================================= shared convex pairwise weight solver
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _fit_pairwise_weights(
    m: np.ndarray,
    rel: np.ndarray,
    *,
    prior: np.ndarray | None = None,
    lam: float = 1.0,
    nonneg: bool = False,
    max_pairs: int = 20000,
    seed: int = 0,
) -> np.ndarray:
    """Learn weights `w` so that, for every labeled pair, the winner scores higher under
    `score = row . w`. Shared by Models 1--3 (they differ only in the representation `m`).

    Minimizes the pairwise-logistic loss with an L2 pull toward `prior`:

        (1/|P|) * sum_pairs softplus( -(m_winner - m_loser) . w )  +  lam * ||w - prior||^2

    This is CONVEX (one global minimum). `prior` is the knob weights (Model 1) or zeros;
    `lam` is the knobs<->labels strength; `nonneg=True` constrains w >= 0 (Model 2's
    importances). With no labeled pairs, returns `prior` unchanged."""
    from scipy.optimize import minimize

    ncol = m.shape[1]
    w0 = np.zeros(ncol) if prior is None else np.asarray(prior, float)
    pairs = build_pairs(rel, max_pairs=max_pairs, seed=seed)
    if len(pairs) == 0:
        return w0
    diff = m[pairs[:, 0]] - m[pairs[:, 1]]  # winner - loser (want diff . w > 0)

    def loss_and_grad(w):
        z = diff @ w
        loss = np.logaddexp(0.0, -z).mean() + lam * float(np.sum((w - w0) ** 2))
        grad = -(diff * _sigmoid(-z)[:, None]).mean(axis=0) + 2.0 * lam * (w - w0)
        return loss, grad

    bounds = [(0.0, None)] * ncol if nonneg else None
    return minimize(loss_and_grad, w0, jac=True, method="L-BFGS-B", bounds=bounds).x


# ======================================================================= the 5 models
# Each fit_* returns a `model` dict to store in profile.model. `train` (below) is the
# dispatcher that also runs `evaluate`.


def fit_desirability(
    df: pd.DataFrame, profile: RankingProfile, *, lam: float = 1.0, seed: int = 0
) -> dict:
    """Model 1. Learn the weights of the weighted-desirability sum by pairwise ranking, with
    the knob weights as the L2 prior (lam = knobs<->labels blend)."""
    w = _fit_pairwise_weights(
        desirability_matrix(df, profile),
        _labeled_relevance(df, "goodness"),
        prior=profile.prior_vector(),
        lam=lam,
        seed=seed,
    )
    return {"kind": "desirability", "weights": [float(x) for x in w], "trained": True}


def fit_prototype(
    df: pd.DataFrame,
    profile: RankingProfile,
    *,
    mode: str = "supervised",
    shrinkage: float = 0.25,
    lam: float = 1.0,
    seed: int = 0,
) -> dict:
    """Model 2. Score = -distance to an ideal point mu, on standardized features. Two modes:

    "exemplar"   (closed form): mu = mean of the good FOVs; per-feature importance
                 a_j = 1 / var_j of the good FOVs, shrunk toward 1 (uses ONLY good labels).
    "supervised" (convex fit) : mu = mean of the good FOVs (fixed); learn the importances a>=0
                 by pairwise ranking so good FOVs sit closer to mu than bad ones."""
    z = standardize_matrix(df, profile)
    rel = _labeled_relevance(df, "goodness")
    good = z[rel == _GOOD]
    if len(good) < 2:
        raise ValueError("need >= 2 good FOVs to fit a prototype")
    mu = good.mean(axis=0)
    if mode == "exemplar":
        var = (1.0 - shrinkage) * good.var(axis=0) + shrinkage * 1.0  # shrink toward unit var
        a = 1.0 / (var + 1e-9)
        a = a / a.mean()  # normalize scale so scores are comparable across profiles
    elif mode == "supervised":
        # Representation -(z - mu)^2: then score = a . rep = -sum a_j (z_j - mu_j)^2, and the
        # weight solver (a >= 0) makes losers land farther from mu than winners.
        a = _fit_pairwise_weights(
            -((z - mu) ** 2), rel, prior=np.ones(z.shape[1]), lam=lam, nonneg=True, seed=seed
        )
    else:
        raise ValueError(f"unknown prototype mode {mode!r} (use 'exemplar' or 'supervised')")
    return {
        "kind": "prototype",
        "mode": mode,
        "mu": [float(x) for x in mu],
        "a": [float(x) for x in a],
    }


def fit_linear(
    df: pd.DataFrame, profile: RankingProfile, *, lam: float = 1.0, seed: int = 0
) -> dict:
    """Model 3. Pairwise learning-to-rank on standardized RAW features (no desirability
    shaping): score = w . z. Monotone by construction; a useful data-driven baseline."""
    w = _fit_pairwise_weights(
        standardize_matrix(df, profile), _labeled_relevance(df, "goodness"), lam=lam, seed=seed
    )
    return {"kind": "linear", "weights": [float(x) for x in w]}


def fit_ebm(
    df: pd.DataFrame, profile: RankingProfile, *, interactions: int = 3, seed: int = 0
) -> dict:
    """Model 4. Explainable Boosting Machine: learns an additive per-feature shape g_j(x_j)
    (plus a few pairwise interactions) by cyclic gradient boosting on the ordinal goodness
    target. Interpretable (each g_j is a plottable curve) and non-monotone-capable."""
    if not HAS_EBM:
        raise RuntimeError("interpret is not installed (pip install interpret)")
    rel = _labeled_relevance(df, "goodness")
    keep = ~np.isnan(rel)
    if keep.sum() < 4:
        raise ValueError("need >= 4 labeled FOVs to fit an EBM")
    ebm = ExplainableBoostingRegressor(
        feature_names=list(profile.features), interactions=interactions, random_state=seed
    )
    ebm.fit(_raw_matrix(df, profile)[keep], rel[keep])
    return {"kind": "ebm", "model": ebm}


def fit_gbm(
    df: pd.DataFrame,
    profile: RankingProfile,
    *,
    num_leaves: int = 15,
    n_estimators: int = 200,
    seed: int = 0,
) -> dict:
    """Model 5. LightGBM LambdaMART over the desirability features, with monotone constraints
    from the desirability direction (desirability is already higher-is-better, so every
    constraint is +1). Captures interactions; optimizes NDCG. Black box (explain via SHAP)."""
    if not HAS_LGBM:
        raise RuntimeError("lightgbm is not installed (pip install lightgbm)")
    des = desirability_matrix(df, profile)
    rel = _labeled_relevance(df, "goodness")
    keep = ~np.isnan(rel)
    if keep.sum() < 2:
        raise ValueError("need >= 2 labeled FOVs to fit a GBM ranker")
    ds = lgb.Dataset(des[keep], label=rel[keep].astype(int), group=[int(keep.sum())])
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "num_leaves": num_leaves,
        "monotone_constraints": [1] * des.shape[1],  # desirability is higher-is-better
        "seed": seed,
        "verbosity": -1,
    }
    return {"kind": "gbm", "booster": lgb.train(params, ds, num_boost_round=n_estimators)}


_FITTERS = {
    "desirability": fit_desirability,
    "prototype": fit_prototype,
    "linear": fit_linear,
    "ebm": fit_ebm,
    "gbm": fit_gbm,
}


def train(kind: str, df: pd.DataFrame, profile: RankingProfile, **opts) -> RankingProfile:
    """Fit model `kind` on `df` (in place: sets profile.model and profile.metrics). `opts`
    are forwarded to the chosen fit_* (e.g. lam=..., mode=..., interactions=...)."""
    if kind not in _FITTERS:
        raise ValueError(f"unknown model {kind!r}; choose from {MODELS}")
    profile.model = _FITTERS[kind](df, profile, **opts)
    profile.metrics = evaluate(profile, df)
    return profile


# ========================================================================= scoring
def score_fov(
    profile: RankingProfile, feature_values: dict, missing: float = 0.5
) -> tuple[float, dict]:
    """Score ONE FOV with the Model-1 (desirability) head and return (score, breakdown).

    The readable single-FOV reference: for each feature, desirability (how ideal the value
    is, in [0,1]) times weight (how much it matters) gives a contribution, and the score is
    their sum. `apply` does the same for a whole table; other heads are scored only by `apply`."""
    weights = dict(zip(profile.features, profile.desirability_weights(), strict=True))
    breakdown, total = {}, 0.0
    for f in profile.features:
        d = desirability(
            feature_values.get(f, float("nan")),
            profile.desirability[f],
            profile.normalization[f],
        )
        if d != d:  # NaN -> not measured for this FOV
            d = missing
        w = float(weights[f])
        breakdown[f] = {"desirability": d, "weight": w, "contribution": w * d}
        total += w * d
    return total, breakdown


def _linear_head(profile: RankingProfile, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (representation matrix M, weights w) for the three additive heads, so that
    score = M . w. Desirability/linear/prototype differ ONLY in M and w."""
    kind = profile.model.get("kind", "desirability")
    if kind == "desirability":
        return desirability_matrix(df, profile), profile.desirability_weights()
    if kind == "linear":
        return standardize_matrix(df, profile), np.array(profile.model["weights"], float)
    if kind == "prototype":
        z = standardize_matrix(df, profile)
        mu = np.array(profile.model["mu"], float)
        return -((z - mu) ** 2), np.array(profile.model["a"], float)
    raise ValueError(f"{kind!r} is not an additive head")


def apply(profile: RankingProfile, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Score every FOV in `df`. Returns (scores, contributions), where `contributions` has one
    column per feature giving that feature's additive push on the score.

    Additive heads (desirability / linear / prototype): score = sum_j w_j * M_j, and the
    contribution of feature j is w_j * M_j (exact, explainable).
    Non-additive heads (ebm / gbm): the library model is evaluated directly; contributions are
    left 0 here (explain those models with their own tools, e.g. EBM curves / SHAP)."""
    kind = profile.model.get("kind", "desirability")
    if kind in ("desirability", "linear", "prototype"):
        m, w = _linear_head(profile, df)
        contrib = m * w[None, :]
        scores = contrib.sum(axis=1)
        return np.asarray(scores, float), pd.DataFrame(
            contrib, index=df.index, columns=profile.features
        )
    if kind == "gbm":
        scores = profile.model["booster"].predict(desirability_matrix(df, profile))
    elif kind == "ebm":
        scores = profile.model["model"].predict(_raw_matrix(df, profile))
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    zeros = pd.DataFrame(0.0, index=df.index, columns=profile.features)
    return np.asarray(scores, float), zeros


# ====================================================================== evaluation
def ndcg(scores: np.ndarray, rel: np.ndarray, k: int | None = None) -> float:
    """NDCG@k of `scores` against relevance `rel` (NaN rows ignored)."""
    keep = ~np.isnan(rel)
    s, r = scores[keep], rel[keep]
    if len(s) == 0 or r.max() <= 0:
        return float("nan")
    k = len(s) if k is None else min(k, len(s))

    def dcg(order):
        gains = (2.0 ** r[order] - 1.0)[:k]
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        return float(np.sum(gains * discounts))

    return dcg(np.argsort(-s)) / (dcg(np.argsort(-r)) or 1.0)


def pairwise_accuracy(scores: np.ndarray, rel: np.ndarray, pairs: np.ndarray) -> float:
    """Fraction of (winner, loser) pairs whose scores are in the correct order."""
    if len(pairs) == 0:
        return float("nan")
    return float((scores[pairs[:, 0]] > scores[pairs[:, 1]]).mean())


def _dataset_leak_flags(
    profile: RankingProfile, df: pd.DataFrame, thresh: float = 0.85
) -> list:
    """Flag features whose desirability tracks the source dataset (a batch artifact rather
    than real quality): a high |correlation| with any dataset-membership indicator."""
    col = "dataset" if "dataset" in df.columns else "__dataset"
    if col not in df.columns or df[col].nunique(dropna=False) < 2:
        return []
    des = desirability_matrix(df, profile)
    dummies = pd.get_dummies(df[col].astype(str)).to_numpy(float)
    flags = []
    for j, f in enumerate(profile.features):
        if des[:, j].std() == 0:
            continue
        corr = max(
            abs(np.corrcoef(des[:, j], dummies[:, c])[0, 1]) for c in range(dummies.shape[1])
        )
        if corr >= thresh:
            flags.append(f"{f} (|corr dataset|={corr:.2f})")
    return flags


def evaluate(
    profile: RankingProfile,
    df: pd.DataFrame,
    *,
    label_col: str = "goodness",
    test_frac: float = 0.25,
    seed: int = 0,
) -> dict:
    """Held-out ranking quality: split labeled FOVs, score with the profile as given, and
    report test pairwise accuracy + NDCG plus dataset-leak flags. (For a clean fit-on-train /
    test-on-held-out benchmark, split the frame yourself and call `apply` + the metrics.)"""
    rel = _labeled_relevance(df, label_col)
    labeled = np.where(~np.isnan(rel))[0]
    if len(labeled) < 4:
        return {"note": "need >= 4 labeled FOVs to evaluate", "n_labeled": int(len(labeled))}
    rng = np.random.default_rng(seed)
    test = rng.permutation(labeled)[: max(1, int(round(len(labeled) * test_frac)))]
    scores, _ = apply(profile, df)
    test_rel = np.where(np.isin(np.arange(len(df)), test), rel, np.nan)
    return {
        "n_labeled": int(len(labeled)),
        "n_test": int(len(test)),
        "pairwise_accuracy": pairwise_accuracy(scores, rel, build_pairs(test_rel, seed=seed)),
        "ndcg": ndcg(scores[test], rel[test]),
        "leak_flags": _dataset_leak_flags(profile, df),
    }
