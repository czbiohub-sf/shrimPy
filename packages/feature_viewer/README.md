# fov-feature-viewer

A Qt GUI to explore per-FOV features, label FOVs, and tune a ranking model, with the FOVs
shown as image thumbnails. It reads the `fov_summary.csv` + `prescan_*/` image folders
that shrimpy's FOV-selection pre-scan writes, but works on any CSV with a `filename`
column.

| Tab | What it does | Needs a scorer |
|---|---|---|
| Analysis | Scatter / PCA / t-SNE (UMAP with `[umap]`) of the features, filters, lasso selection | no |
| Label | Drag FOV thumbnails between good / neutral / bad panels; writes `goodness` to the CSV | no |
| Rank | Tune each feature's desirability curve and rank the FOVs by the resulting score | yes |
| Score map | The score of a pair of features as a 2D contour + 3D surface | yes |

## Install and run

```bash
pip install "fov-feature-viewer @ git+https://github.com/czbiohub-sf/shrimPy.git#subdirectory=packages/feature_viewer"
fov-feature-viewer path/to/fov_summary.csv
```

In the shrimPy repo it is a uv workspace member, installed with shrimPy's `fov` extra.

## Scorers

The viewer does no model math. The Rank and Score-map tabs run on a *scorer*, an object
implementing the `fov_feature_viewer.scorer.Scorer` protocol: it supplies the curve
shapes and their parameters, fits a default curve to the data, converts between shapes,
moves drag handles, evaluates curves and scores, and parses saved profiles. Each
feature's settings are the scorer's own dict (a *spec*); a saved profile is a mapping of
feature name to spec.

The viewer finds a scorer through the `fov_feature_viewer.scorers` entry-point group, or
takes one explicitly:

```bash
fov-feature-viewer --scorer shrimpy data.csv            # a registered name
fov-feature-viewer --scorer mypkg.models:MyScorer data.csv  # module:attr
```

shrimPy registers its ranking model (`ranking_by_defined_range`) as `shrimpy`, so a
profile tuned here can be pasted under `fov_selection.model.features` in an acquisition
config. To provide another, register a class, factory, or instance:

```toml
[project.entry-points."fov_feature_viewer.scorers"]
myscorer = "mypkg.models:MyScorer"
```

With no scorer installed the viewer still explores and labels data; the Rank and
Score-map tabs are not shown.

## Environment variables

- `FOV_VIEWER_DIR`: where the Load dialog starts (default: the current directory).
- `FOV_RANK_PROFILE_DIR`: where profile dialogs start (default: `$FOV_VIEWER_DIR/ranking_profiles`).
