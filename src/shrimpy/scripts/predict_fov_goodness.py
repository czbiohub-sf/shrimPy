"""
Final FOV-goodness model: train ONCE on all three annotated mantis A549 datasets
(no leave-one-out), then predict good-vs-bad on an unannotated FOV feature matrix and
write the predictions back as new columns.

Features (both virtual-staining channels):
  nuclei_vs_sum__coverage_frac, membrane_vs_sum__objects_per_10um2
Label: good+neutral (goodness >= 0) = 1 ("keep"), bad (goodness < 0) = 0.
Model: depth-3 balanced DecisionTree, median-imputer fit on the pooled training data
(same estimator family as the LOOV comparison).

Because the label is binary (good+neutral vs bad), the model predicts "keep vs bad";
it does not separate good from neutral. Two columns are added to the target matrix:
  predicted_good_proba  P(keep) from the tree, in [0, 1]
  predicted_goodness    "good" if predicted keep, else "bad" (threshold 0.5)

The fitted model (imputer + tree + metadata) is saved so it can be re-applied without
retraining, plus a rules text file and two figures (tree diagram + diagnostics).

Running this regenerates the ENTIRE final_model/ folder so all artifacts stay consistent
with the current labels/matrices (re-run it whenever labels change).

ARTIFACTS (under .../split_analysis/final_model/)
  final_model_nuclei-vs_membrane-vs.joblib   {imputer, tree, features, train_datasets}
  final_model_rules.txt                       tree thresholds + training stats
  final_model_tree.png                        the fitted decision tree
  final_model_diagnostics.png                 feature histograms + importance + boundary (4 panels)
  confusion_matrices_train.png                confusion per dataset + combined (final model vs truth)
  feature_histograms_<channel>.png            per-dataset good/bad histograms of ALL base
                                              features (one per model channel) + decision lines

    python -m shrimpy.scripts.predict_fov_goodness
    python -m shrimpy.scripts.predict_fov_goodness --target <matrix.csv> --out <matrix.csv>
"""

from __future__ import annotations

import argparse

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(
    "/hpc/projects/comp.micro/microscope_dev/smart_fov_selection/"
    "fov_selection_output/fov_features"
)
MODEL_DIR = ROOT / "split_analysis" / "final_model"

# short name -> training FOV feature matrix (all three have both VS channels + labels)
TRAIN_SETS = {
    "organelles": "2026_03_25_A549_organelles_DENV_ZIKV_fov_feature_matrix.csv",
    "caax_h2b": "2026_03_26_A549_CAAX_H2B_DENV_ZIKV_fov_feature_matrix.csv",
    "h2bc21": "2026_06_24_A549_H2BC21_fov_feature_matrix.csv",
}
TRAIN_MATRICES = list(TRAIN_SETS.values())
# full dataset name (matrix stem) for each short key -- used in all figure text
FULL_NAME = {s: m.replace("_fov_feature_matrix.csv", "") for s, m in TRAIN_SETS.items()}
DEFAULT_TARGET = "2026_05_27_A549_SEC61B_TOMM20_G3BP1_ZIKV_fov_feature_matrix.csv"

FEATURES = [
    "nuclei_vs_sum__coverage_frac",
    "membrane_vs_sum__objects_per_10um2",
]
SHORT = [f.split("__", 1)[1] for f in FEATURES]
MODEL_PATH = "final_model_nuclei-vs_membrane-vs.joblib"
MAX_DEPTH = 3
SEED = 0


def load_train_by_dataset() -> dict[str, pd.DataFrame]:
    """{short name -> labeled FOVs (FEATURES + goodness + binary y)} for each dataset."""
    out = {}
    for short, m in TRAIN_SETS.items():
        df = pd.read_csv(ROOT / m).dropna(subset=["goodness"]).reset_index(drop=True)
        df = df[FEATURES + ["goodness"]].copy()
        df["y"] = (df["goodness"] >= 0).astype(int)  # good+neutral=1 (keep), bad=0
        out[short] = df
    return out


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    data = load_train_by_dataset()
    pooled = pd.concat(data.values(), ignore_index=True)
    return pooled[FEATURES], pooled["y"]


def _cm_panel(ax, cm, title):
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            str(v),
            ha="center",
            va="center",
            fontsize=13,
            color="white" if v > cm.max() / 2 else "black",
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred bad", "pred good"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["true bad", "true good"])
    ax.set_title(title, fontsize=10)


# the base per-variant features (feature_extraction.FEATURE_NAMES order)
BASE_FEATURES = [
    "coverage_frac",
    "nn_um_mean",
    "nn_cv",
    "empty_grid_frac",
    "occupancy_entropy",
    "max_empty_radius",
]


def figure_feature_histograms() -> None:
    """One figure per model channel: rows = datasets, cols = ALL 10 base features,
    good vs bad histograms. Each feature column shares the SAME x-range across datasets.
    A yellow frame marks the column of each feature used by the final model."""
    channels = list(dict.fromkeys(f.rsplit("__", 1)[0] + "__" for f in FEATURES))
    dfs = {}
    for short, m in TRAIN_SETS.items():
        d = pd.read_csv(ROOT / m).dropna(subset=["goodness"])
        d["y"] = (d["goodness"] >= 0).astype(int)
        dfs[short] = d

    for chan in channels:
        # shared x-range per base feature (pooled 1-99 pct across all datasets)
        xr = {}
        for b in BASE_FEATURES:
            col = chan + b
            vals = np.concatenate(
                [dfs[s][col].dropna().to_numpy() for s in dfs if col in dfs[s].columns]
                or [np.array([])]
            )
            xr[b] = np.nanpercentile(vals, [1, 99]) if vals.size else (0.0, 1.0)
        nrow, ncol = len(dfs), len(BASE_FEATURES)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow), squeeze=False)
        for r, (short, d) in enumerate(dfs.items()):
            good = d[d.y == 1]
            bad = d[d.y == 0]
            for c, b in enumerate(BASE_FEATURES):
                col = chan + b
                ax = axes[r][c]
                if col not in d.columns:
                    ax.set_visible(False)
                    continue
                lo, hi = xr[b]
                bins = np.linspace(lo, hi, 30) if hi > lo else 30
                for sub, cl, lab in [
                    (good, "tab:blue", "good+neutral"),
                    (bad, "tab:red", "bad"),
                ]:
                    v = sub[col].dropna().to_numpy()
                    if v.size:
                        ax.hist(
                            v,
                            bins=bins,
                            weights=np.full(v.size, 1 / v.size),
                            alpha=0.55,
                            color=cl,
                            label=lab,
                        )
                if hi > lo:
                    ax.set_xlim(lo, hi)
                sel = col in FEATURES  # used by the final model
                if sel:  # yellow frame on that column
                    for spine in ax.spines.values():
                        spine.set_edgecolor("gold")
                        spine.set_linewidth(3.5)
                if r == 0:
                    ax.set_title(b, fontsize=10, fontweight="bold" if sel else "normal")
                if c == 0:
                    ax.set_ylabel(
                        f"{FULL_NAME[short]}\n(fraction)", fontsize=8, fontweight="bold"
                    )
                ax.tick_params(labelsize=7)
        axes[0][0].legend(fontsize=7, loc="upper right")
        fig.suptitle(
            f"{chan.rstrip('_')}: all base features per dataset (good vs bad); "
            "yellow frame = feature used by the final model",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(
            MODEL_DIR / f"feature_histograms_{chan.rstrip('_')}.png",
            dpi=140,
            bbox_inches="tight",
        )
        plt.close(fig)


def figure_confusion_train(data: dict, imp, clf) -> None:
    """Confusion of the FINAL model vs ground truth, per dataset AND combined (all three
    datasets pooled)."""
    pooled = pd.concat(data.values(), ignore_index=True)
    panels = [(FULL_NAME[s], d) for s, d in data.items()] + [
        ("combined (all 3 datasets)", pooled)
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.8))
    for ax, (name, df) in zip(np.atleast_1d(axes), panels):
        y = df["y"]
        pred = clf.predict(imp.transform(df[FEATURES]))
        proba = clf.predict_proba(imp.transform(df[FEATURES]))[:, 1]
        cm = confusion_matrix(y, pred, labels=[0, 1])
        auc = roc_auc_score(y, proba) if y.nunique() > 1 else float("nan")
        bacc = balanced_accuracy_score(y, pred)
        tn, fp = cm[0]
        brec = tn / (tn + fp) if (tn + fp) else float("nan")
        _cm_panel(
            ax,
            cm,
            f"{name}\n(n={len(df)}) acc={(y.values == pred).mean():.2f} "
            f"bal-acc={bacc:.2f} AUC={auc:.2f} bad-recall={brec:.2f}",
        )
    fig.suptitle("Final model vs human annotation: per dataset and combined", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(MODEL_DIR / "confusion_matrices_train.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_tree(clf) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=SHORT,
        class_names=["bad", "good"],
        filled=True,
        rounded=True,
        proportion=True,
        impurity=False,
        fontsize=10,
        ax=ax,
    )
    ax.set_title(
        "Final FOV-goodness tree\ntrained on: "
        + ", ".join(FULL_NAME.values())
        + "\nfeatures: "
        + " | ".join(FEATURES),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "final_model_tree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_diagnostics(data, clf, imp) -> None:
    """Deployed-model diagnostics (4 panels): good/bad histograms of the two features, a
    feature-importance panel, and the training datasets in the plane of the two features
    with the model's decision boundary."""
    from matplotlib.lines import Line2D

    pooled = pd.concat(data.values(), ignore_index=True)
    Xtr, ytr = pooled[FEATURES], pooled["y"]
    good = Xtr[ytr == 1]
    bad = Xtr[ytr == 0]
    imps = dict(zip(FEATURES, clf.feature_importances_))
    kept = [f for f in FEATURES if imps[f] > 0]  # the features actually used
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # decision-boundary tree on the two features (drawn in the scatter); its per-feature
    # split thresholds are also drawn as dashed lines on the histograms below.
    fx, fy = kept[0], kept[1]
    imp2 = SimpleImputer(strategy="median").fit(Xtr[[fx, fy]])
    clf2 = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, class_weight="balanced", random_state=SEED
    ).fit(imp2.transform(Xtr[[fx, fy]]), ytr)
    t2 = clf2.tree_
    thr2 = {fx: [], fy: []}
    for i in range(t2.node_count):
        if t2.feature[i] >= 0:
            thr2[[fx, fy][t2.feature[i]]].append(float(t2.threshold[i]))
    thr2 = {f: sorted(set(v)) for f, v in thr2.items()}
    # one boundary colour per feature, shared between its histogram and the scatter
    bcol = {fx: "tab:purple", fy: "tab:orange"}

    # (0,0) & (0,1): histograms of the two features, with the decision thresholds (dashed,
    # coloured per feature to match the scatter)
    for ax, f in zip([axes[0, 0], axes[0, 1]], kept[:2]):
        s = f.split("__", 1)[1]
        both = np.concatenate([good[f].dropna(), bad[f].dropna()])
        lo, hi = np.nanpercentile(both, [1, 99])
        bins = np.linspace(lo, hi, 25)
        for sub, col, lab in [(good, "tab:blue", "good+neutral"), (bad, "tab:red", "bad")]:
            v = sub[f].dropna().to_numpy()
            ax.hist(
                v,
                bins=bins,
                weights=np.full(v.size, 1 / max(v.size, 1)),
                alpha=0.55,
                color=col,
                label=lab,
            )
        for k, x in enumerate(thr2.get(f, [])):
            ax.axvline(
                x,
                ls="--",
                color=bcol[f],
                lw=1.5,
                label="decision boundary" if k == 0 else None,
            )
        ax.set_title(f"train separation: {s}", fontsize=11)
        ax.set_xlabel(s)
        ax.legend(fontsize=8)

    # (1,0): feature-importance panel (kept features highlighted green)
    ax = axes[1, 0]
    names = [f.split("__", 1)[1] for f in FEATURES]
    vals = [imps[f] for f in FEATURES]
    colors = ["tab:green" if imps[f] > 0 else "0.7" for f in FEATURES]
    ax.barh(range(len(FEATURES)), vals, color=colors)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("importance")
    ax.set_title("feature importance", fontsize=11)

    # (1,1): the three TRAINING datasets in the plane of the two features, with the
    # model's decision boundary (clf2, fit above)
    ax = axes[1, 1]
    x0, x1 = np.nanpercentile(Xtr[fx], [1, 99])
    y0, y1 = np.nanpercentile(Xtr[fy], [1, 99])
    xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    zz = clf2.predict_proba(
        imp2.transform(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[fx, fy]))
    )[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.45)
    marks = {s: m for s, m in zip(data, ["o", "s", "^", "D", "v"])}
    for s, d in data.items():
        for lab, col in [(1, "tab:blue"), (0, "tab:red")]:
            sub = d[d["y"] == lab]
            ax.scatter(
                sub[fx],
                sub[fy],
                s=16,
                c=col,
                marker=marks[s],
                edgecolor="k",
                linewidth=0.2,
                alpha=0.6,
            )
    # decision thresholds, coloured to match the histograms (fx vertical, fy horizontal)
    for x in thr2[fx]:
        ax.axvline(x, ls="--", color=bcol[fx], lw=1.3)
    for yv in thr2[fy]:
        ax.axhline(yv, ls="--", color=bcol[fy], lw=1.3)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(fx.split("__", 1)[1])
    ax.set_ylabel(fy.split("__", 1)[1])
    ax.set_title(
        "training datasets + decision boundary\n(blue=P(good) high, red=P(bad) high)",
        fontsize=11,
    )
    leg = [
        Line2D(
            [],
            [],
            marker=marks[s],
            color="w",
            markerfacecolor="grey",
            markeredgecolor="k",
            markersize=8,
            label=FULL_NAME[s],
        )
        for s in marks
    ] + [
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor="tab:blue",
            markersize=8,
            label="good+neutral",
        ),
        Line2D(
            [], [], marker="o", color="w", markerfacecolor="tab:red", markersize=8, label="bad"
        ),
        Line2D([], [], ls="--", color=bcol[fx], label=f"{fx.split('__', 1)[1]} threshold"),
        Line2D([], [], ls="--", color=bcol[fy], label=f"{fy.split('__', 1)[1]} threshold"),
    ]
    ax.legend(handles=leg, fontsize=7, loc="upper right")

    fig.suptitle("Final FOV-goodness model diagnostics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(MODEL_DIR / "final_model_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="FOV feature matrix (filename under ROOT, or absolute path) to predict on.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Where to write the matrix + prediction columns (default: "
        "overwrite the target in place).",
    )
    cli = ap.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = load_train_by_dataset()
    pooled = pd.concat(data.values(), ignore_index=True)
    Xtr, ytr = pooled[FEATURES], pooled["y"]
    imp = SimpleImputer(strategy="median").fit(Xtr)
    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, class_weight="balanced", random_state=SEED
    ).fit(imp.transform(Xtr), ytr)
    rules = export_text(clf, feature_names=FEATURES)
    header = (
        f"trained on {len(ytr)} labeled FOVs "
        f"(keep={int((ytr == 1).sum())}, bad={int((ytr == 0).sum())}) from "
        f"{len(TRAIN_MATRICES)} datasets:\n  " + "\n  ".join(TRAIN_MATRICES)
    )
    print(header + f"\ntree rules (thresholds on the {len(FEATURES)} features):\n" + rules)

    # --- persist the model (imputer + tree travel together) + rules ---
    joblib.dump(
        {
            "imputer": imp,
            "tree": clf,
            "features": FEATURES,
            "train_datasets": TRAIN_MATRICES,
            "label": "good+neutral(1) vs bad(0)",
            "max_depth": MAX_DEPTH,
            "seed": SEED,
        },
        MODEL_DIR / MODEL_PATH,
    )
    (MODEL_DIR / "final_model_rules.txt").write_text(
        header + "\n\nfeatures = " + str(FEATURES) + "\n\n" + rules + "\n"
    )

    target = Path(cli.target) if Path(cli.target).is_absolute() else ROOT / cli.target
    out = Path(cli.out) if cli.out else target
    df = pd.read_csv(target)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"target is missing required feature columns: {missing}")

    proba = clf.predict_proba(imp.transform(df[FEATURES]))[:, 1]
    df["predicted_good_proba"] = proba.round(4)
    df["predicted_goodness"] = ["good" if p >= 0.5 else "bad" for p in proba]
    df.to_csv(out, index=False)

    # --- figures (regenerate the whole final_model/ folder) ---
    figure_tree(clf)
    figure_diagnostics(data, clf, imp)
    figure_confusion_train(data, imp, clf)
    figure_feature_histograms()

    n_good = int((df["predicted_goodness"] == "good").sum())
    print(
        f"\npredicted {len(df)} FOVs in {target.name}: "
        f"good={n_good} ({n_good / len(df):.1%}), bad={len(df) - n_good}"
    )
    print(f"wrote predictions (predicted_good_proba, predicted_goodness) -> {out}")
    print(f"saved model + rules + figures under {MODEL_DIR}")


if __name__ == "__main__":
    main()
