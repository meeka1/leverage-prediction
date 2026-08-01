"""
Phase 3 -- exploratory data analysis.

    python -m src.eda

Reads outputs/panel_features.csv and writes figures to outputs/figures/, tables to
outputs/tables/, and a narrative summary to outputs/eda_report.md.

The logic lives here rather than in the notebook so it is version-controlled, re-runnable and
reviewable; notebooks/03_eda.ipynb is a thin wrapper that calls these functions and displays
the results.

Covers dissertation steps 3.1-3.5, plus the size non-linearity result, which belongs in EDA
because it is the empirical motivation for preferring tree models to OLS.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C, viz
from src.features import THEORY_SIGNS

FIGURES = C.OUTPUTS / "figures"
TABLES = C.OUTPUTS / "tables"


# --------------------------------------------------------------------------------------
# 3.1 descriptive statistics
# --------------------------------------------------------------------------------------

def descriptives(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cols = [C.TARGET] + features
    d = df[cols]
    out = pd.DataFrame({
        "n": d.notna().sum(),
        "coverage": d.notna().mean(),
        "mean": d.mean(),
        "sd": d.std(),
        "p1": d.quantile(.01),
        "p25": d.quantile(.25),
        "median": d.median(),
        "p75": d.quantile(.75),
        "p99": d.quantile(.99),
    })
    return out.round(4)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

def fig_leverage_distribution(df: pd.DataFrame):
    """The zero-debt spike is the defining feature of this sample, so it is drawn apart."""
    lev = df[C.TARGET].dropna()
    zero_share = (lev == 0).mean()
    positive = lev[lev > 0]

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(positive, bins=60, range=(0, 1.5), color=viz.SERIES[0], edgecolor=viz.SURFACE,
            linewidth=0.4)
    ax.axvline(positive.median(), color=viz.TEXT_SECONDARY, linewidth=1.2, linestyle="--")
    ax.annotate(f"median of levered firms {positive.median():.2f}",
                xy=(positive.median(), ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=viz.TEXT_SECONDARY, va="top")
    ax.set_xlabel("Book leverage (total debt / total assets)")
    ax.set_ylabel("Firm-years")
    ax.set_title("Book leverage is concentrated near zero")
    viz.strip_spines(ax)
    viz.caption(fig, f"Levered firm-years only (n={len(positive):,}). A further "
                     f"{zero_share:.0%} of firm-years carry exactly zero debt and are excluded "
                     "from this axis — that mass is why OLS is misspecified at the boundary.")
    return fig


def fig_leverage_over_time(df: pd.DataFrame):
    """Two measures on different scales -> two panels, never a second y-axis."""
    g = df.groupby("year")[C.TARGET]
    med, zero = g.median(), g.apply(lambda s: (s == 0).mean())
    firms = df.groupby("year")["ticker"].nunique()

    years, last = med.index, int(df["year"].max())
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.2), sharex=True)
    for i, (ax, series, label, title) in enumerate([
        (axes[0], med, "Median book leverage", "Median leverage"),
        (axes[1], zero, "Share with zero debt", "Zero-debt share"),
        (axes[2], firms, "Firms", "Firms in sample"),
    ]):
        ax.plot(series.index, series.values, color=viz.SERIES[0])
        ax.set_ylabel(label)
        ax.set_title(title)
        # label on the middle panel: its line sits low inside the shaded window, so the
        # annotation has clear space there
        viz.shade_test_period(ax, C.TEST_START_YEAR, last, label=(i == 1))
        viz.year_axis(ax, years)
        viz.strip_spines(ax)
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[2].set_xlabel("Year")
    fig.tight_layout()
    viz.caption(fig, "The 2019-20 break is an accounting change, not economics: IFRS 16 moved "
                     "operating leases onto the balance sheet, adding lease liabilities to debt. "
                     "It is not composition — a balanced panel of firms present throughout shows "
                     "the same jump (see fig09).")
    return fig


def fig_ifrs16_break(df: pd.DataFrame, raw: pd.DataFrame):
    """
    Evidence that the 2019-20 level shift is IFRS 16 lease capitalisation.

    IFRS 16 (effective for periods beginning on or after 1 January 2019) requires operating
    leases on the balance sheet: a right-of-use asset in PP&E and a lease liability in debt.
    Bloomberg's SHORT_AND_LONG_TERM_DEBT includes the liability, so book leverage jumps
    without any borrowing. Adoption is staggered across 2019 and 2020 because firms whose
    fiscal year began before 1 January 2019 were caught a year later.

    Three panels, because a single one would not distinguish the explanations:
      (a) a balanced panel rules out survivorship composition;
      (b) debt and PP&E inflating together while revenue does not rules out real growth;
      (c) debt appearing without a financing cash inflow rules out actual borrowing.
    """
    both = set(df.loc[df.year == 2018, "ticker"]) & set(df.loc[df.year == 2022, "ticker"])
    bal = df[df.ticker.isin(both)]
    years = sorted(df.loc[df.year.between(2014, 2023), "year"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

    ax = axes[0]
    for data, label, color in [(df, "All firms", viz.SERIES[1]),
                               (bal, f"Balanced panel (n={len(both)})", viz.SERIES[0])]:
        s = data[data.year.isin(years)].groupby("year")[C.TARGET].apply(lambda x: (x == 0).mean())
        ax.plot(s.index, s.values, color=color, label=label)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Share with zero debt")
    ax.set_title("(a) Not composition")
    ax.legend(loc="lower left")

    ax = axes[1]
    r = raw[raw.year.isin(years)].sort_values(["ticker", "year"])
    for col, label, slot in [("total_debt", "Total debt", 0), ("net_ppe", "Net PP&E", 1),
                             ("revenue", "Revenue", 2)]:
        g = r.groupby("ticker")[col].pct_change(fill_method=None)
        med = g.groupby(r["year"]).median()
        ax.plot(med.index, med.values, color=viz.SERIES[slot], label=label)
    ax.axhline(0, color=viz.TEXT_MUTED, linewidth=0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Median year-on-year growth")
    ax.set_title("(b) Debt and PP&E move together")
    ax.legend()

    ax = axes[2]
    r["prev_debt"] = r.groupby("ticker")["total_debt"].shift(1)
    r["gapok"] = r.groupby("ticker")["year"].diff().eq(1)
    newly = r[r.gapok & (r.prev_debt == 0) & (r.total_debt > 0)]
    share = newly.groupby("year")["cf_financing"].apply(lambda s: (s <= 0).mean())
    counts = newly.groupby("year").size()
    ax.bar(share.index, share.values, color=viz.SERIES[0], width=0.68)
    for x, v, n in zip(share.index, share.values, counts.values):
        ax.annotate(f"n={n}", (x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color=viz.TEXT_SECONDARY)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Share with no cash inflow")
    ax.set_title("(c) Debt without borrowing")

    for ax in axes:
        ax.set_xlabel("Year")
        viz.year_axis(ax, years, step=2)
        viz.strip_spines(ax)
    fig.tight_layout()
    viz.caption(fig, "(a) Firms present throughout show the same collapse in the zero-debt "
                     "share, so it is not survivorship. (b) Debt and PP&E inflate together "
                     "while revenue does not — the signature of a right-of-use asset and a "
                     "matching lease liability. (c) Among firms moving from zero to positive "
                     "debt, the share raising no cash rises to ~50% in 2019-20, against ~25-33% "
                     "before. Example: Greggs plc, debt 0 to 361m USD and PP&E 420m to 820m USD "
                     "in one year on +8% revenue.", y=-0.08)
    return fig


def fig_stage_distribution(df: pd.DataFrame):
    d = df.dropna(subset=["lifecycle_stage"])
    counts = d["lifecycle_stage"].value_counts().reindex(viz.STAGE_ORDER)
    share = (pd.crosstab(d["year"], d["lifecycle_stage"], normalize="index")
             .reindex(columns=viz.STAGE_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    ax.bar(range(len(counts)), counts.values, width=0.68,
           color=[viz.STAGE_COLORS[s] for s in counts.index])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    for i, v in enumerate(counts.values):
        ax.annotate(f"{v:,}", (i, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=viz.TEXT_SECONDARY)
    ax.set_ylabel("Firm-years")
    ax.set_title("Firm-years per stage")
    viz.strip_spines(ax)

    ax = axes[1]
    bottom = np.zeros(len(share))
    for stage in viz.STAGE_ORDER:
        ax.fill_between(share.index, bottom, bottom + share[stage].values,
                        color=viz.STAGE_COLORS[stage], label=stage,
                        linewidth=0.8, edgecolor=viz.SURFACE)
        bottom = bottom + share[stage].values
    viz.year_axis(ax, share.index)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Year")
    ax.set_title("Stage composition over time")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    viz.strip_spines(ax)
    fig.tight_layout()
    viz.caption(fig, "Dickinson (2011) stages from the signs of operating, investing and "
                     "financing cash flow. Introduction is far larger than in US samples, "
                     "reflecting the weight of loss-making AIM microcaps on the LSE.")
    return fig


def fig_leverage_by_stage(df: pd.DataFrame):
    d = df.dropna(subset=["lifecycle_stage"])
    med = (d.groupby(["year", "lifecycle_stage"], observed=True)[C.TARGET]
           .median().unstack().reindex(columns=viz.STAGE_ORDER))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for stage in viz.STAGE_ORDER:
        ax.plot(med.index, med[stage].values, color=viz.STAGE_COLORS[stage], label=stage)
    viz.year_axis(ax, med.index)
    ax.set_xlabel("Year")
    ax.set_ylabel("Median book leverage")
    ax.set_title("Leverage differs persistently across life cycle stages")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    viz.strip_spines(ax)
    fig.tight_layout()
    # caption must clear the legend, which sits below the axes
    viz.caption(fig, "Stable ordering across two decades is what makes stage-specific models "
                     "worth estimating: the gap is structural, not a single-period artefact.",
                y=-0.10)
    return fig


def fig_leverage_by_sector(df: pd.DataFrame):
    g = (df.groupby("gics_sector")[C.TARGET]
         .agg(["median", "size"]).sort_values("median"))
    colors = [viz.SERIES[0] if s != C.UNCLASSIFIED_SECTOR else viz.TEXT_MUTED
              for s in g.index]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.barh(range(len(g)), g["median"].values, color=colors, height=0.68)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index)
    for i, (v, n) in enumerate(zip(g["median"], g["size"])):
        ax.annotate(f"{v:.3f}  (n={n:,})", (v, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=viz.TEXT_SECONDARY)
    ax.set_xlim(0, g["median"].max() * 1.45)
    ax.set_xlabel("Median book leverage")
    ax.set_title("Median leverage by sector")
    viz.strip_spines(ax)
    viz.caption(fig, "Unclassified (grey) pools firms for which Bloomberg does not resolve "
                     "GICS — mostly delisted. It is kept to preserve the survivorship-bias-"
                     "free sample, but its median is a median over a heterogeneous bucket.")
    return fig


def fig_size_nonlinearity(df: pd.DataFrame):
    """The empirical case for tree models over OLS."""
    d = df.dropna(subset=["size", C.TARGET]).copy()
    d["decile"] = pd.qcut(d["size"], 10, labels=False, duplicates="drop")
    g = d.groupby("decile")[C.TARGET]
    med, zero = g.median(), g.apply(lambda s: (s == 0).mean())

    pearson = d["size"].corr(d[C.TARGET])
    spearman = d["size"].rank().corr(d[C.TARGET].rank())

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    axes[0].plot(med.index + 1, med.values, color=viz.SERIES[0], marker="o")
    axes[0].set_ylabel("Median book leverage")
    axes[0].set_title("Median leverage by size decile")
    axes[1].plot(zero.index + 1, zero.values, color=viz.SERIES[0], marker="o")
    axes[1].set_ylabel("Share with zero debt")
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].set_title("Zero-debt share by size decile")
    for ax in axes:
        ax.set_xlabel("Size decile (1 = smallest)")
        ax.set_xticks(range(1, 11))
        viz.strip_spines(ax)
    fig.tight_layout()
    viz.caption(fig, f"Monotone in rank, invisible to a linear fit: Pearson correlation "
                     f"between size and leverage is {pearson:+.3f}, Spearman {spearman:+.3f}. "
                     "A linear model sees almost nothing here; a tree model sees the whole "
                     "gradient.")
    return fig


def fig_correlation(df: pd.DataFrame, features: list[str]):
    cols = [C.TARGET] + [f for f in features
                         if df[f].notna().any() and df[f].nunique() > 2]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(corr.values, cmap=viz.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7.5)
    ax.set_yticklabels(cols, fontsize=7.5)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if abs(v) >= 0.5 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color=viz.SURFACE if abs(v) > 0.75 else viz.TEXT_PRIMARY)
    cbar = fig.colorbar(im, ax=ax, shrink=0.65, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Pearson correlation", color=viz.TEXT_SECONDARY, fontsize=8.5)
    cbar.outline.set_visible(False)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    viz.caption(fig, "Diverging scale: neutral grey at zero, opposed hues for sign. "
                     "Coefficients shown only where |r| >= 0.5. Pairs above 0.8 are "
                     "candidates for removal before the Lasso and OLS specifications.",
                y=-0.03)
    return fig


def fig_missingness(df: pd.DataFrame, features: list[str]):
    cols = [f for f in features if df[f].isna().any()]
    miss = (df.assign(**{f: df[f].isna() for f in cols})
            .groupby("year")[cols].mean().T)
    order = miss.mean(axis=1).sort_values(ascending=False).index
    miss = miss.loc[order]

    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.28 * len(miss))))
    im = ax.imshow(miss.values, cmap=viz.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(miss.columns)))
    ax.set_xticklabels(miss.columns, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(miss)))
    ax.set_yticklabels(miss.index, fontsize=7.5)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0%", "50%", "100%"])
    cbar.set_label("Share missing", color=viz.TEXT_SECONDARY, fontsize=8.5)
    cbar.outline.set_visible(False)
    ax.set_title("Missing data by feature and year")
    viz.caption(fig, "Sequential single-hue scale: light is complete, dark is missing. "
                     "Features are ordered by overall missingness. Rows dark on the left only "
                     "are history-dependent (lags, rolling windows), not data failures.")
    return fig


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def write_report(path, df: pd.DataFrame, features: list[str], desc: pd.DataFrame,
                 figures: list[tuple[str, str]]) -> None:
    d = df.dropna(subset=["lifecycle_stage"])
    lev = df[C.TARGET]

    L = ["# Phase 3 -- Exploratory Data Analysis", "",
         "Generated by `python -m src.eda`. Figures in `outputs/figures/`, tables in "
         "`outputs/tables/`.", "",
         f"- Firm-years: **{len(df):,}** · firms: **{df['ticker'].nunique():,}** · "
         f"{int(df['year'].min())}–{int(df['year'].max())}",
         f"- Train (<{C.TEST_START_YEAR}): **{(df['year'] < C.TEST_START_YEAR).sum():,}** · "
         f"test (≥{C.TEST_START_YEAR}): **{(df['year'] >= C.TEST_START_YEAR).sum():,}**", "",
         "## Figures", ""]
    for fname, cap in figures:
        L += [f"**`{fname}`** — {cap}", ""]

    L += ["## What the data says", "",
          f"**Leverage is concentrated at zero.** {(lev == 0).mean():.0%} of firm-years carry "
          f"no debt at all; the median across all firm-years is {lev.median():.3f}, but "
          f"{lev[lev > 0].median():.3f} among levered firms. Book leverage is therefore "
          "censored, not continuous — OLS is misspecified at the boundary and Tobit belongs "
          "in the baseline set.", "",
          "**The size–leverage relationship is monotone but not linear.** Median leverage "
          "rises across every size decile while the zero-debt share falls from "
          f"{_decile_edge(df, 0):.0%} to {_decile_edge(df, 9):.0%}. Pearson correlation is "
          f"{df['size'].corr(lev):+.3f}; Spearman is "
          f"{df['size'].rank().corr(lev.rank()):+.3f}. This gap is the empirical case for "
          "tree-based models over linear ones, and it is worth stating in the Discussion "
          "rather than leaving implicit.", ""]

    L += ["**A structural break in 2019–20 shifts the level of leverage, but not the "
          "relationships.** IFRS 16 (effective for periods beginning on or after 1 January "
          "2019) put operating leases on the balance sheet; Bloomberg's total-debt field "
          "includes the resulting lease liability, so leverage jumps without any borrowing. "
          "Three checks separate this from the alternatives (`fig09`): a balanced panel of "
          "firms present throughout shows the same jump, so it is not survivorship; debt and "
          "PP&E inflate together while revenue does not, matching a right-of-use asset and its "
          "matching liability; and among firms moving from zero to positive debt, the share "
          "raising no cash at all rises from ~25–33% before to ~50% in 2019–20. Adoption is "
          "spread over two years because firms whose fiscal year began before 1 January 2019 "
          "were caught a year later.", "",
          "This matters for interpretation more than for prediction. Out-of-sample R² is "
          "stable whichever side of the break the split falls on, so the specified "
          f"train/test boundary at {C.TEST_START_YEAR} stands. But **any statement about "
          "leverage levels across 2019 is an accounting artefact, not economics** — including "
          "the collapse in the zero-debt share — and the Discussion must say so rather than "
          "read it as COVID-era borrowing.", ""]

    stage_med = d.groupby("lifecycle_stage", observed=True)[C.TARGET].median()
    L += ["**Stages separate on leverage and hold that ordering over two decades.** Median "
          "book leverage by stage: " +
          ", ".join(f"{s} {stage_med[s]:.3f}" for s in viz.STAGE_ORDER if s in stage_med) +
          ". A persistent gap is what justifies estimating stage-specific models rather than "
          "adding a stage dummy.", ""]

    counts = d["lifecycle_stage"].value_counts()
    test_counts = (d[d["year"] >= C.TEST_START_YEAR]["lifecycle_stage"].value_counts())
    smallest = test_counts.idxmin()
    L += [f"**Stage sample sizes are unbalanced.** The smallest test cell is `{smallest}` at "
          f"**{test_counts.min():,}** firm-years (train {counts[smallest] - test_counts[smallest]:,}). "
          "Stage-specific metrics and SHAP rankings there need bootstrapped intervals, not "
          "point estimates.", ""]

    persist, never_moves, tm = stage_persistence(df)
    L += [f"**Dickinson stages are firm-year states, not firm attributes.** A firm stays in "
          f"the same stage from one year to the next only **{persist:.0%}** of the time, and "
          f"just **{never_moves:.0%}** of firms never change stage at all. Shakeout and "
          "Decline are the least persistent, and Decline→Introduction is common because both "
          "are negative-operating-cash-flow states separated only by the sign of investing "
          "cash flow.", "",
          "Phase 5 therefore classifies *observations*, not firms, and the claim it supports "
          "is \"feature importance within this cash-flow pattern\" rather than \"within firms "
          "at this stage of life\". A robustness check on firms holding a stage for two or "
          "more consecutive years would show whether the rankings are driven by the stable "
          "core or by the churn.", "",
          "| From \\ To | " + " | ".join(viz.STAGE_ORDER) + " |",
          "|---|" + "---:|" * len(viz.STAGE_ORDER)]
    for r in viz.STAGE_ORDER:
        L.append(f"| **{r}** | " +
                 " | ".join(f"{tm.loc[r, c]:.0%}" for c in viz.STAGE_ORDER) + " |")
    L.append("")

    hi = _high_correlations(df, features)
    if len(hi):
        L += ["**Collinear pairs.** Above |r| = 0.8:", "",
              "| Feature A | Feature B | r |", "|---|---|---:|"]
        L += [f"| {a} | {b} | {v:+.2f} |" for a, b, v in hi]
        L += ["", "Harmless for trees; for OLS and Lasso, drop one of each pair or rely on "
              "the L1 penalty and say so.", ""]

    L += ["## Descriptive statistics (Step 3.1)", "",
          "Full table: `outputs/tables/descriptives.csv`.", "",
          "| Variable | n | mean | sd | p25 | median | p75 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in desc.iterrows():
        L.append(f"| {name} | {int(r['n']):,} | {r['mean']:.3f} | {r['sd']:.3f} | "
                 f"{r['p25']:.3f} | {r['median']:.3f} | {r['p75']:.3f} |")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


def stage_persistence(df: pd.DataFrame):
    """
    How often a firm stays in the same Dickinson stage from one year to the next.

    Dickinson assigns a stage from a single year's cash-flow signs, so a firm can move every
    year. If persistence is low, stage-specific models are fitted to a noisy label and the
    resulting claim is about cash-flow patterns rather than about firms.

    Returns (year-on-year persistence, share of firms never changing stage, transition matrix).
    """
    d = df.sort_values(["ticker", "year"]).copy()
    d["prev"] = d.groupby("ticker")["lifecycle_stage"].shift(1)
    ok = (d.groupby("ticker")["year"].diff().eq(1)
          & d["prev"].notna() & d["lifecycle_stage"].notna())
    sub = d[ok]
    persistence = float((sub["lifecycle_stage"] == sub["prev"]).mean())
    never = float((df.groupby("ticker")["lifecycle_stage"].nunique() == 1).mean())
    tm = (pd.crosstab(sub["prev"], sub["lifecycle_stage"], normalize="index")
          .reindex(index=viz.STAGE_ORDER, columns=viz.STAGE_ORDER))
    return persistence, never, tm


def _decile_edge(df: pd.DataFrame, decile: int) -> float:
    d = df.dropna(subset=["size", C.TARGET]).copy()
    d["decile"] = pd.qcut(d["size"], 10, labels=False, duplicates="drop")
    return (d[d["decile"] == decile][C.TARGET] == 0).mean()


def _high_correlations(df: pd.DataFrame, features: list[str], threshold: float = 0.8):
    cols = [f for f in features if df[f].notna().any() and df[f].nunique() > 2]
    corr = df[cols].corr().abs()
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = df[a].corr(df[b])
            if abs(v) >= threshold:
                out.append((a, b, v))
    return sorted(out, key=lambda t: -abs(t[2]))


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------

def main() -> int:
    src = C.OUTPUTS / "panel_features.csv"
    if not src.exists():
        print(f"ERROR: {src} not found -- run `python -m src.features` first", file=sys.stderr)
        return 1

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    viz.apply_theme()

    df = pd.read_csv(src)
    raw = pd.read_csv(C.OUTPUTS / "panel_raw.csv")
    features = pd.read_csv(C.OUTPUTS / "feature_list.csv")["feature"].tolist()
    features = [f for f in features if f in df.columns]
    print(f"  panel_features : {len(df):,} firm-years, {len(features)} features")

    desc = descriptives(df, features)
    desc.to_csv(TABLES / "descriptives.csv")
    (df.groupby("lifecycle_stage", observed=True)[C.TARGET]
       .describe().to_csv(TABLES / "leverage_by_stage.csv"))
    (df.groupby("gics_sector")[C.TARGET]
       .describe().to_csv(TABLES / "leverage_by_sector.csv"))
    df[[C.TARGET] + features].corr().to_csv(TABLES / "correlations.csv")

    specs = [
        ("fig01_leverage_distribution.png", fig_leverage_distribution(df),
         "Distribution of book leverage among levered firms, with the zero-debt mass quantified."),
        ("fig02_leverage_over_time.png", fig_leverage_over_time(df),
         "Median leverage, zero-debt share and sample size by year; test period shaded."),
        ("fig03_stage_distribution.png", fig_stage_distribution(df),
         "Firm-years per Dickinson stage, and stage composition over time."),
        ("fig04_leverage_by_stage.png", fig_leverage_by_stage(df),
         "Median leverage by stage and year."),
        ("fig05_leverage_by_sector.png", fig_leverage_by_sector(df),
         "Median leverage by GICS sector, Unclassified shown in grey."),
        ("fig06_size_nonlinearity.png", fig_size_nonlinearity(df),
         "Median leverage and zero-debt share across size deciles — the non-linearity that "
         "motivates tree models."),
        ("fig07_correlation.png", fig_correlation(df, features),
         "Feature correlation matrix on a diverging scale."),
        ("fig08_missingness.png", fig_missingness(df, features),
         "Share missing by feature and year."),
        ("fig09_ifrs16_break.png", fig_ifrs16_break(df, raw),
         "Evidence that the 2019-20 leverage jump is IFRS 16 lease capitalisation, not "
         "borrowing, growth or survivorship."),
    ]
    for fname, fig, _ in specs:
        viz.save(fig, FIGURES / fname)
    print(f"  figures        : {len(specs)} written to outputs/figures/")

    write_report(C.OUTPUTS / "eda_report.md", df, features, desc,
                 [(f, cap) for f, _, cap in specs])
    print(f"  tables         : 4 written to outputs/tables/")
    print(f"  report         : outputs/eda_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
