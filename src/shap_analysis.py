"""
Phase 6 -- SHAP interpretation.

    python -m src.shap_analysis

Ranks features by contribution on the best pooled model, then asks whether that ranking
shifts across life cycle stages (RQ1 and RQ2).

Why stage heterogeneity is measured on the pooled model
-------------------------------------------------------
Phase 5 found that stage-specific models predict *worse* than one pooled model in every stage:
splitting the sample costs more in observations than it gains in specificity. Reading feature
importance off five weaker models would therefore describe five noisier fits, not the life
cycle.

The pooled model is instead explained separately over each stage's observations. SHAP is
additive and computed per row, so partitioning its output by stage is exact -- it asks "within
firm-years in this stage, what drove the prediction?" while keeping one well-estimated model.
That answers RQ2 without paying the small-sample penalty.

Rankings are bootstrapped by firm, because a top-10 list from 213 Decline firm-years looks
authoritative on the page and is not.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C, splits, viz
from src.models import RANDOM_STATE
from src.stage_models import XGB_PARAMS
from xgboost import XGBRegressor

FIGURES = C.OUTPUTS / "figures"
TABLES = C.OUTPUTS / "tables"
TOP_N = 15
N_BOOT = 500


def fit_best_model(df: pd.DataFrame, features: list[str]):
    design = splits.build_design(df, features)
    model = XGBRegressor(**XGB_PARAMS).fit(design.X_train, design.y_train)
    return model, design


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Per-row, per-feature SHAP contributions.

    Computed through XGBoost's own `pred_contribs`, which runs the same exact TreeSHAP
    algorithm as `shap.TreeExplainer` but reads the model in-process. The shap package's
    XGBoost loader cannot parse the `base_score` format written by xgboost >= 3.x
    (it arrives as '[1.9E-1]' rather than a scalar), so going direct avoids a version
    dependency in the middle of the analysis. `test_shap_matches_additivity` checks the
    output satisfies the SHAP additivity property against the model's own predictions.

    The final column returned by pred_contribs is the bias (expected value), dropped here so
    the frame lines up with the feature columns.
    """
    import xgboost as xgb
    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index)


def rank_features(sv: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature -- the standard global importance measure."""
    return sv.abs().mean().sort_values(ascending=False)


def bootstrap_rank_stability(sv: pd.DataFrame, tickers: pd.Series, top_n: int = 10,
                             n_boot: int = N_BOOT, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    How often each feature lands in the top-N when firms are resampled.

    A ranking computed once on a few hundred firm-years is a point estimate with no error bar.
    Resampling whole firms (not rows) preserves within-firm correlation and shows which
    positions are stable and which are noise.
    """
    rng = np.random.default_rng(seed)
    firms = tickers.unique()
    idx_by_firm = {f: np.flatnonzero(tickers.to_numpy() == f) for f in firms}

    counts = pd.Series(0, index=sv.columns, dtype=float)
    for _ in range(n_boot):
        drawn = rng.choice(firms, size=len(firms), replace=True)
        idx = np.concatenate([idx_by_firm[f] for f in drawn])
        top = sv.iloc[idx].abs().mean().nlargest(top_n).index
        counts[top] += 1
    return (counts / n_boot).sort_values(ascending=False).rename("top_n_frequency")


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

def fig_global_importance(ranking: pd.Series, stability: pd.Series):
    top = ranking.head(TOP_N)[::-1]
    freq = stability.reindex(top.index)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.barh(range(len(top)), top.values, color=viz.SERIES[0], height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    for i, (v, f) in enumerate(zip(top.values, freq.values)):
        note = f"{v:.4f}" + (f"   ({f:.0%} stable)" if pd.notna(f) else "")
        ax.annotate(note, (v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=viz.TEXT_SECONDARY)
    ax.set_xlim(0, top.max() * 1.5)
    ax.set_xlabel("Mean |SHAP| (contribution to predicted leverage)")
    ax.set_title(f"Top {TOP_N} features — pooled XGBoost, no-lag specification")
    viz.strip_spines(ax)
    viz.caption(fig, "Bars are average absolute contribution per firm-year. The percentage is "
                     "how often the feature stays in the top 10 across 500 firm-level "
                     "bootstrap resamples — a low figure means the position is not reliable.")
    return fig


def fig_stage_ranking_shift(stage_ranks: pd.DataFrame, top_n: int = 8, n_test: int = 0):
    """
    Slope chart: how each feature's rank moves across the life cycle.

    Capped at eight features because the categorical palette has eight fixed slots and cycling
    hues past that would put two indistinguishable blues on the same chart. Every line is
    directly labelled, so identity never rests on colour alone.
    """
    features = stage_ranks.min(axis=1).nsmallest(top_n).index
    sub = stage_ranks.loc[features]
    stages = [s for s in viz.STAGE_ORDER if s in sub.columns]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for slot, feat in enumerate(features):
        color = viz.SERIES[slot]
        y = sub.loc[feat, stages].values
        flat = len(set(y)) == 1
        ax.plot(range(len(stages)), y, color=color, marker="o", markersize=5,
                linewidth=2.4 if flat else 1.6)
        ax.annotate(feat, (len(stages) - 1, y[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ranks = sorted(set(sub.values.ravel()))
    ax.set_yticks(ranks)
    ax.set_yticklabels([str(int(r)) for r in ranks])
    ax.invert_yaxis()
    ax.set_ylabel("Rank by mean |SHAP| (1 = most important)")
    ax.set_xlim(-0.3, len(stages) - 0.3 + 1.7)
    ax.set_title("Feature importance rank across life cycle stages")
    viz.strip_spines(ax)
    viz.caption(fig, "One pooled model, explained separately over each stage's firm-years. "
                     "The four leading features hold the same rank in every stage (drawn "
                     "heavier); only the middle of the ranking reshuffles. Decline rests on "
                     f"{n_test:,} firm-years, so its column is the least reliable.")
    return fig


def fig_dependence(sv: pd.DataFrame, X: pd.DataFrame, features: list[str]):
    """How each top feature's contribution varies with its own value."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), sharey=True)
    for ax, feat in zip(np.atleast_1d(axes), features):
        x, y = X[feat].to_numpy(), sv[feat].to_numpy()
        lo, hi = np.nanpercentile(x, [1, 99])
        m = (x >= lo) & (x <= hi)
        ax.scatter(x[m], y[m], s=4, alpha=0.18, color=viz.SERIES[0], linewidths=0)
        if m.sum() > 50:  # running median makes the shape legible through the cloud
            q = pd.qcut(pd.Series(x[m]), 20, duplicates="drop")
            med = pd.Series(y[m]).groupby(q, observed=True).median()
            centres = [iv.mid for iv in med.index]
            ax.plot(centres, med.values, color=viz.SERIES[1], linewidth=2)
        ax.axhline(0, color=viz.TEXT_MUTED, linewidth=0.8)
        ax.set_xlabel(feat)
        viz.strip_spines(ax)
    np.atleast_1d(axes)[0].set_ylabel("SHAP contribution")
    fig.suptitle("Dependence: how each driver shifts predicted leverage", x=0.005,
                 ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    viz.caption(fig, "Each point is a firm-year; the orange line is the running median. "
                     "A non-monotone or kinked line is exactly what a linear model cannot "
                     "represent, and explains the gap over OLS. Axes trimmed to the 1st-99th "
                     "percentile.", y=-0.06)
    return fig


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def write_report(path, ranking: pd.Series, stability: pd.Series, stage_ranks: pd.DataFrame,
                 stage_means: pd.DataFrame, stage_n: dict) -> None:
    L = ["# Phase 6 -- SHAP Interpretation", "",
         "Generated by `python -m src.shap_analysis`.", "",
         "Model: pooled XGBoost, no-lag specification (out-of-sample R² = 0.453). The no-lag "
         "specification is used because RQ1 asks which *firm characteristics* predict "
         "leverage; with `leverage_lag1` included, the ranking mostly reports that leverage is "
         "persistent.", "",
         "Stage heterogeneity is measured by explaining the **pooled** model separately over "
         "each stage's firm-years, not by fitting five stage-specific models. Phase 5 showed "
         "stage-specific models predict worse in every stage, so their SHAP values would "
         "describe five noisier fits rather than the life cycle.", "",
         "## RQ1 — which features predict leverage?", "",
         f"| Rank | Feature | Mean \\|SHAP\\| | Top-10 stability |", "|---:|---|---:|---:|"]
    for i, (feat, val) in enumerate(ranking.head(TOP_N).items(), 1):
        s = stability.get(feat, float("nan"))
        L.append(f"| {i} | `{feat}` | {val:.4f} | {s:.0%} |" if pd.notna(s)
                 else f"| {i} | `{feat}` | {val:.4f} | — |")
    L += ["", "Stability is the share of 500 firm-level bootstrap resamples in which the "
          "feature stays in the top 10. Anything below ~70% should be described as "
          "'among the leading features' rather than given a precise rank.", ""]

    L += ["## RQ2 — does the ranking shift across the life cycle?", "",
          "Rank by mean |SHAP| within each stage (1 = most important):", "",
          "| Feature | " + " | ".join(f"{s} (n={stage_n[s]:,})" for s in stage_ranks.columns)
          + " |",
          "|---|" + "---:|" * len(stage_ranks.columns)]
    top_overall = ranking.head(TOP_N).index
    for feat in top_overall:
        if feat in stage_ranks.index:
            L.append(f"| `{feat}` | " +
                     " | ".join(f"{int(stage_ranks.loc[feat, s])}" for s in stage_ranks.columns)
                     + " |")
    L.append("")

    spread = (stage_ranks.loc[top_overall].max(axis=1)
              - stage_ranks.loc[top_overall].min(axis=1)).sort_values(ascending=False)
    movers = spread.head(4)
    steady = spread.tail(4)
    L += ["**Most stage-dependent:** " +
          ", ".join(f"`{f}` (rank span {int(v)})" for f, v in movers.items()) + ".", "",
          "**Most stable across stages:** " +
          ", ".join(f"`{f}` (span {int(v)})" for f, v in steady.items()) + ".", "",
          "A large span means the feature matters in some stages and not others; a small span "
          "means it drives leverage throughout the life cycle. Read Decline's column with "
          f"care — it rests on {stage_n.get('Decline', 0):,} firm-years.", ""]

    L += ["## Figures", "",
          "- `fig10_shap_global.png` — top-15 features with bootstrap stability",
          "- `fig11_shap_stage_shift.png` — how ranks move across stages",
          "- `fig12_shap_dependence.png` — dependence plots for the top 5", "",
          "## Caveats", "",
          "- **SHAP measures contribution to *this model's* predictions**, not causal effect. "
          "A feature can rank highly because it proxies for something unobserved.",
          "- **Correlated features share credit.** `profitability` and `ocf_to_assets` "
          "correlate at 0.82, so SHAP splits their joint contribution and each looks weaker "
          "than it is.",
          "- **`efwamb` is absent** — Baker–Wurgler weights need equity and debt issuance "
          "separately, which the extract does not contain, so the market-timing channel is "
          "represented only by market-to-book and returns.",
          "- **Stage labels are firm-year states**, persistent only ~52% of the time, so RQ2's "
          "answer is about cash-flow patterns rather than firms settled in a life stage.", ""]

    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    src = C.OUTPUTS / "panel_features.csv"
    if not src.exists():
        print(f"ERROR: {src} not found -- run `python -m src.features` first", file=sys.stderr)
        return 1

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    viz.apply_theme()

    df = pd.read_csv(src)
    features = [f for f in pd.read_csv(C.OUTPUTS / "feature_list.csv")["feature"]
                if f in df.columns and df[f].notna().any() and f != "leverage_lag1"]

    model, design = fit_best_model(df, features)
    print(f"  model fitted on {len(design.y_train):,} rows, explaining {len(design.y_test):,}")

    sv = shap_values(model, design.X_test)
    ranking = rank_features(sv)
    tickers = design.test_rows["ticker"]
    stability = bootstrap_rank_stability(sv, tickers)
    print(f"  top 5: {', '.join(ranking.head(5).index)}")

    stages = design.test_rows["lifecycle_stage"]
    stage_means, stage_n = {}, {}
    for stage in viz.STAGE_ORDER:
        mask = (stages == stage).to_numpy()
        if mask.sum() < 50:
            continue
        stage_means[stage] = rank_features(sv[mask])
        stage_n[stage] = int(mask.sum())
    stage_means = pd.DataFrame(stage_means)
    stage_ranks = stage_means.rank(ascending=False, method="min").astype(int)

    ranking.rename("mean_abs_shap").to_frame().join(
        stability.rename("top10_stability")).to_csv(TABLES / "shap_global_ranking.csv")
    stage_ranks.to_csv(TABLES / "shap_stage_ranks.csv")
    stage_means.to_csv(TABLES / "shap_stage_means.csv")

    viz.save(fig_global_importance(ranking, stability), FIGURES / "fig10_shap_global.png")
    viz.save(fig_stage_ranking_shift(stage_ranks, n_test=stage_n.get('Decline', 0)),
             FIGURES / "fig11_shap_stage_shift.png")
    viz.save(fig_dependence(sv, design.X_test, list(ranking.head(5).index)),
             FIGURES / "fig12_shap_dependence.png")
    print(f"  figures: 3 written")

    write_report(C.OUTPUTS / "shap_report.md", ranking, stability, stage_ranks,
                 stage_means, stage_n)
    print(f"  report : outputs/shap_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
