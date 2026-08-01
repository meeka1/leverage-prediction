"""
Phase 2 -- construct the target and the ~20 firm-level features.

    python -m src.features

Reads outputs/panel_raw.csv, builds the feature set from the dissertation specification,
applies the Dickinson life cycle classification, winsorises ratios within year, and writes
outputs/panel_features.csv plus a feature report.

Design notes that matter for the results
----------------------------------------
* Industry median leverage is **leave-one-out**: a firm never contributes to the median it is
  compared against. Including it induces a mechanical correlation with the target, which would
  show up as spurious feature importance.
* Ratios are winsorised **within year** (1st/99th), not pooled, so crisis years are not clipped
  disproportionately.
* Every ratio uses total assets as the denominator unless the literature uses another scale,
  so coefficients stay comparable.
* Features whose inputs are missing from the extract are emitted as all-NaN with a flag rather
  than silently dropped, so the feature table always has the same shape.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from src import cleaning, config as C, lifecycle


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide, returning NaN where the denominator is zero or missing."""
    den = den.replace(0, np.nan)
    return num / den


def _by_firm(df: pd.DataFrame, col: str):
    return df.groupby("ticker", sort=False)[col]


# Features derived from exactly one prior period. `groupby.shift(1)` returns the previous
# *observation*, not the previous *year*, so where a firm's series skips a year these carry a
# stale value silently labelled as last year's. They are blanked at those rows.
GAP_SENSITIVE = ["leverage_lag1", "asset_growth", "stock_return", "financing_deficit"]


def mask_year_gaps(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Blank one-period features where the preceding observation is not the preceding year."""
    out = df.sort_values(["ticker", "year"]).copy()
    gap = _by_firm(out, "year").diff()
    broken = gap.notna() & (gap != 1)
    for col in columns:
        if col in out.columns:
            out.loc[broken, col] = np.nan
    return out


# --------------------------------------------------------------------------------------
# target
# --------------------------------------------------------------------------------------

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Book leverage = total debt / total assets (dissertation Step 2.1)."""
    out = df.copy()
    out[C.TARGET] = _safe_div(out["total_debt"], out["total_assets"])
    # Debt cannot be negative; the handful of negatives are reporting artefacts.
    out.loc[out[C.TARGET] < 0, C.TARGET] = np.nan
    return out


# --------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------

def add_trade_off_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out, built = df.copy(), []
    ta = out["total_assets"]

    out["tangibility"] = _safe_div(out["net_ppe"], ta)
    out["size"] = np.log(ta.where(ta > 0))
    out["ndts"] = _safe_div(out["depreciation"], ta)
    built += ["tangibility", "size", "ndts"]

    # Effective tax rate is undefined once pretax income is non-positive. A loss-making firm
    # gets no benefit from the interest tax shield, so zero is the economically right value
    # rather than a missing one -- the indicator preserves the distinction for the model.
    # Left as NaN only where pretax income itself was never reported.
    positive_pretax = out["pretax_income"] > 0
    etr = _safe_div(out["tax_expense"], out["pretax_income"].where(positive_pretax))
    lo, hi = C.EFFECTIVE_TAX_BOUNDS
    etr = etr.clip(lo, hi)
    out["effective_tax_rate_missing"] = etr.isna().astype(int)
    out["effective_tax_rate"] = etr.mask(out["pretax_income"].notna() & ~positive_pretax, 0.0)
    built += ["effective_tax_rate", "effective_tax_rate_missing"]

    # Altman (1968) Z-score. X4 needs total liabilities; the extract has none, so it falls
    # back to assets - common equity, which folds minority interest and preferred stock into
    # "liabilities". Flagged so the fallback is visible in the write-up.
    if "total_liabilities" in out.columns and out["total_liabilities"].notna().any():
        liabilities, out["altman_z_proxy_liabilities"] = out["total_liabilities"], 0
    else:
        liabilities, out["altman_z_proxy_liabilities"] = ta - out["common_equity"], 1

    out["altman_z"] = (
        1.2 * _safe_div(out["working_capital"], ta)
        + 1.4 * _safe_div(out["retained_earnings"], ta)
        + 3.3 * _safe_div(out["ebit"], ta)
        + 0.6 * _safe_div(out["market_cap"], liabilities.where(liabilities > 0))
        + 1.0 * _safe_div(out["revenue"], ta)
    )
    built += ["altman_z", "altman_z_proxy_liabilities"]

    # Earnings volatility: rolling sd of ROA within firm.
    roa = _safe_div(out["ebitda"], ta)
    out["_roa"] = roa
    out["earnings_volatility"] = (
        _by_firm(out.sort_values(["ticker", "year"]), "_roa")
        .rolling(C.VOLATILITY_WINDOW, min_periods=C.VOLATILITY_MIN_PERIODS)
        .std().reset_index(level=0, drop=True).reindex(out.index)
    )
    out = out.drop(columns="_roa")
    built.append("earnings_volatility")
    return out, built


def add_pecking_order_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out, built = df.copy(), []
    ta = out["total_assets"]

    out["profitability"] = _safe_div(out["ebitda"], ta)
    out["cash_holdings"] = _safe_div(out["cash"], ta)
    out["ocf_to_assets"] = _safe_div(out["cf_operating"], ta)
    built += ["profitability", "cash_holdings", "ocf_to_assets"]

    # R&D is reported by a minority of firms; absence usually means "no R&D", so it is set to
    # zero with a companion indicator rather than dropped.
    rd = _safe_div(out["rd_expense"].abs(), out["revenue"])
    out["rd_intensity_missing"] = rd.isna().astype(int)
    out["rd_intensity"] = rd.fillna(0.0)
    built += ["rd_intensity", "rd_intensity_missing"]

    out["dividend_dummy"] = (out["dividends_paid"].abs() > 0).astype(int)
    built.append("dividend_dummy")

    out = out.sort_values(["ticker", "year"])
    out["asset_growth"] = _by_firm(out, "total_assets").pct_change(fill_method=None)
    built.append("asset_growth")

    # Frank & Goyal (2003) financing deficit:
    #   DEF = dividends + investment + change in working capital - operating cash flow
    # scaled by total assets. Capex stands in for net investment.
    d_wc = _by_firm(out, "working_capital").diff()
    deficit = (out["dividends_paid"].abs() + out["capex"].abs()
               + d_wc - out["cf_operating"])
    out["financing_deficit"] = _safe_div(deficit, ta)
    built.append("financing_deficit")
    return out, built


def add_market_timing_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out, built = df.copy(), []
    ta = out["total_assets"]

    # Market-to-book as in Rajan & Zingales (1995): (assets - book equity + market equity)/assets
    out["market_to_book"] = _safe_div(ta - out["common_equity"] + out["market_cap"], ta)
    built.append("market_to_book")

    # The RZ ratio expands to 1 - BE/TA + ME/TA, and its first term is mechanically increasing
    # in leverage (more debt => less book equity per unit of assets). Plain price-to-book has
    # no such component, so it is carried alongside for robustness. Undefined at negative book
    # equity, which is why it cannot simply replace the RZ measure.
    out["price_to_book"] = _safe_div(out["market_cap"],
                                     out["common_equity"].where(out["common_equity"] > 0))
    built.append("price_to_book")

    # Negative book equity, stated explicitly rather than left to leak in through
    # price_to_book's missingness pattern. Without it the model recovers the same signal as an
    # uninterpretable `price_to_book_isna` indicator, which ranked first on mean |SHAP| and
    # told a reader nothing. These firms are financially distressed and carry mean leverage of
    # 0.82 against 0.15 elsewhere. The association is strong but not tautological -- negative
    # equity can come from accumulated losses without much debt -- so it belongs in the
    # ranking, flagged as a distress marker rather than read as a financing choice.
    out["negative_book_equity"] = (out["common_equity"] <= 0).astype(int)
    built.append("negative_book_equity")

    out = out.sort_values(["ticker", "year"])
    ret = _by_firm(out, "price_last").pct_change(fill_method=None)
    out["stock_return"] = ret
    out["cum_return_3y"] = (
        _by_firm(out.assign(_g=np.log1p(ret)), "_g")
        .rolling(3, min_periods=2).sum().reset_index(level=0, drop=True).reindex(out.index)
    )
    out["cum_return_3y"] = np.expm1(out["cum_return_3y"])
    built += ["stock_return", "cum_return_3y"]

    # Volatility of ANNUAL returns: a 5-year window gives at most 5 observations, so this is a
    # coarse proxy. A monthly price pull would replace it; flagged in the report.
    out["stock_return_volatility"] = (
        _by_firm(out, "stock_return")
        .rolling(C.VOLATILITY_WINDOW, min_periods=C.VOLATILITY_MIN_PERIODS)
        .std().reset_index(level=0, drop=True).reindex(out.index)
    )
    built.append("stock_return_volatility")

    out, efwamb_cols = _add_efwamb(out)
    return out, built + efwamb_cols


def _add_efwamb(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Baker & Wurgler (2002) external-finance-weighted average market-to-book.

        efwamb_t = sum_{s<t} [ (e_s + d_s) / sum_{r<t}(e_r + d_r) ] * (M/B)_s

    The weights need equity and debt issuance separately. The extract has only aggregate
    `cf_financing`, which cannot be decomposed, so the true measure is not computable.

    Emitted instead:
      * `efwamb`            -- all NaN unless the issuance fields are present,
      * `mtb_hist_avg`      -- the equal-weighted historical mean of M/B, which is what efwamb
                               collapses to when issuance is constant. A documented fallback,
                               NOT Baker-Wurgler, and it must be labelled as such in the text.
    """
    out = df.sort_values(["ticker", "year"]).copy()
    have = [c for c in ("equity_issued", "equity_repurchased", "lt_debt_net")
            if c in out.columns and out[c].notna().any()]

    # expanding mean of past M/B, excluding the current year
    prior = _by_firm(out, "market_to_book").shift(1)
    with warnings.catch_warnings():  # all-NaN windows for firms with no market data
        warnings.simplefilter("ignore", RuntimeWarning)
        out["mtb_hist_avg"] = (
            prior.groupby(out["ticker"], sort=False).expanding().mean()
            .reset_index(level=0, drop=True).reindex(out.index)
        )

    if len(have) == 3:
        ext = (out["equity_issued"].abs() - out["equity_repurchased"].abs()
               + out["lt_debt_net"])
        w = ext.shift(1)
        cum_w = w.groupby(out["ticker"], sort=False).expanding().sum() \
                 .reset_index(level=0, drop=True).reindex(out.index)
        weighted = (w * prior).groupby(out["ticker"], sort=False).expanding().sum() \
                    .reset_index(level=0, drop=True).reindex(out.index)
        out["efwamb"] = _safe_div(weighted, cum_w)
        out["efwamb_available"] = 1
    else:
        out["efwamb"] = np.nan
        out["efwamb_available"] = 0
    return out, ["efwamb", "efwamb_available", "mtb_hist_avg"]


def add_industry_median_leverage(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Leave-one-out median leverage of the firm's sector in that year (Step 2.3).

    Leave-one-out matters: with the firm included, the feature partly contains the target,
    and gradient-boosted models will happily exploit that.
    """
    out = df.copy()
    grp = out.groupby(["gics_sector", "year"], observed=True)[C.TARGET]

    def loo_median(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float)
        res = np.full(len(vals), np.nan)
        for i in range(len(vals)):
            others = np.delete(vals, i)
            others = others[~np.isnan(others)]
            if len(others) >= C.MIN_INDUSTRY_PEERS:
                res[i] = np.median(others)
        return pd.Series(res, index=s.index)

    out["industry_median_leverage"] = grp.transform(loo_median)
    out["industry_unclassified"] = (out["gics_sector"] == C.UNCLASSIFIED_SECTOR).astype(int)
    return out, ["industry_median_leverage", "industry_unclassified"]


def add_lag_and_age(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values(["ticker", "year"]).copy()
    built = []

    out["leverage_lag1"] = _by_firm(out, C.TARGET).shift(1)
    built.append("leverage_lag1")

    if "listing_year" in out.columns:
        age = out["year"] - out["listing_year"]
        out["firm_age"] = age.where(age >= 0)
        out["firm_age_missing"] = out["firm_age"].isna().astype(int)
        built += ["firm_age", "firm_age_missing"]
    return out, built


# --------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------

WINSOR_EXCLUDE = {
    "dividend_dummy", "rd_intensity_missing", "effective_tax_rate_missing",
    "firm_age_missing", "industry_unclassified", "efwamb_available",
    "altman_z_proxy_liabilities", "firm_age", "negative_book_equity",
}


def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = panel.sort_values(["ticker", "year"]).reset_index(drop=True)
    df = add_target(df)

    # The target is winsorised BEFORE anything is derived from it. Industry median leverage
    # and leverage_lag1 are both functions of the target, and letting a raw leverage of 5,932
    # propagate into them would contaminate two predictors at once.
    df = cleaning.winsorize_by_year(df, [C.TARGET])

    features: list[str] = []
    for step in (add_trade_off_features, add_pecking_order_features,
                 add_market_timing_features, add_industry_median_leverage,
                 add_lag_and_age):
        df, built = step(df)
        features += built

    df = lifecycle.classify(df)
    df = mask_year_gaps(df, GAP_SENSITIVE)
    df = df.sort_values(["ticker", "year"]).reset_index(drop=True)

    to_winsorize = [f for f in features
                    if f not in WINSOR_EXCLUDE and f != "leverage_lag1"]
    df = cleaning.winsorize_by_year(df, to_winsorize)

    if C.LAG_PREDICTORS:
        df = lag_predictors(df, features)
    return df, features


def lag_predictors(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Shift every predictor back one year within firm (see config.LAG_PREDICTORS)."""
    out = df.sort_values(["ticker", "year"]).copy()
    out[features] = out.groupby("ticker", sort=False)[features].shift(1)
    return out


# Signs predicted by the capital-structure literature, used as a build-time sanity check.
# `ndts` and `effective_tax_rate` routinely come out against prediction in empirical work
# (Frank & Goyal 2009 report the same positive NDTS sign), so a mismatch there is expected.
THEORY_SIGNS = {
    "tangibility": "+", "size": "+", "industry_median_leverage": "+", "leverage_lag1": "+",
    "effective_tax_rate": "+", "ndts": "-", "profitability": "-", "market_to_book": "-",
    "price_to_book": "-", "cash_holdings": "-", "ocf_to_assets": "-", "altman_z": "-",
}


def feature_sets(df: pd.DataFrame, features: list[str]) -> dict[str, list[str]]:
    """
    Nested feature sets ordered by how much of the sample each one costs.

    Built from measured coverage rather than a hand-written list, so it stays correct when a
    later Bloomberg pull fills in the sparse fields. Features that are entirely missing
    (currently `efwamb`) are excluded from every set.
    """
    usable = [f for f in features if df[f].notna().any()]
    coverage = {f: df[f].notna().mean() for f in usable}
    out: dict[str, list[str]] = {}
    for label, threshold in [("core (≥95% coverage)", 0.95),
                             ("broad (≥90%)", 0.90),
                             ("wide (≥75%)", 0.75),
                             ("all available", 0.0)]:
        out[label] = [f for f in usable if coverage[f] >= threshold]
    return out


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def write_report(path, df: pd.DataFrame, features: list[str]) -> None:
    L = ["# Phase 2 -- Feature Report", "",
         "Generated by `python -m src.features`.", "",
         f"- Firm-years: **{len(df):,}** across **{df['ticker'].nunique():,}** firms, "
         f"{int(df['year'].min())}–{int(df['year'].max())}",
         f"- Target: `{C.TARGET}` = total debt / total assets",
         f"- Predictors: **{len(features)}**",
         f"- Predictors lagged one year: **{C.LAG_PREDICTORS}**", ""]

    L += ["## Usable sample by feature set", "",
          "List-wise deletion cost is driven by the sparsest feature, so choosing a feature "
          "set is really choosing a sample size. Tree models (RF, XGBoost) take NaN natively "
          "and can use the widest set; OLS, Tobit and Lasso need complete rows or imputation.",
          "", "| Feature set | Predictors | Complete cases | Share | Firms |",
          "|---|---:|---:|---:|---:|"]
    sets = feature_sets(df, features)
    for label, feats in sets.items():
        sub = df.dropna(subset=[C.TARGET] + feats)
        L.append(f"| {label} | {len(feats)} | {len(sub):,} | "
                 f"{len(sub) / len(df):.0%} | {sub['ticker'].nunique():,} |")
    L.append("")
    if "efwamb" in features and df["efwamb"].isna().all():
        L += ["> `efwamb` is all-NaN, so any set containing it has zero complete cases. It is "
              "excluded from every set above and must be dropped in Phase 4.", ""]
    L += [f"Recommended default: **{max(sets, key=lambda k: len(sets[k]) if len(df.dropna(subset=[C.TARGET] + sets[k])) > 0.4 * len(df) else -1)}** "
          "— the widest set that still retains over 40% of firm-years.", ""]

    L += ["## Target", "",
          "| Statistic | Value |", "|---|---:|"]
    t = df[C.TARGET]
    for label, val in [("mean", t.mean()), ("median", t.median()), ("sd", t.std()),
                       ("p1", t.quantile(.01)), ("p99", t.quantile(.99)),
                       ("zero-debt share", (t == 0).mean()), ("missing", t.isna().mean())]:
        L.append(f"| {label} | {val:.3f} |")
    L.append("")

    L += ["## Life cycle stages (Dickinson 2011)", "",
          "| Stage | Train <2020 | Test ≥2020 | Total | Share |", "|---|---:|---:|---:|---:|"]
    s = lifecycle.stage_summary(df, split_year=C.TEST_START_YEAR)
    for stage, row in s.iterrows():
        cols = [c for c in s.columns if c not in ("n", "share")]
        a = int(row[cols[0]]) if len(cols) > 0 else 0
        b = int(row[cols[1]]) if len(cols) > 1 else 0
        L.append(f"| {stage} | {a:,} | {b:,} | {int(row['n']):,} | {row['share']:.1f}% |")
    L.append("")
    zero = df["cf_sign_zero"].mean()
    L.append(f"Firm-years with an exact zero in a cash-flow sign: **{zero:.1%}** "
             "(treated as non-positive, per Dickinson).")
    L.append("")

    L += ["## Feature coverage and distribution", "",
          "| Feature | Coverage | Mean | Median | SD |", "|---|---:|---:|---:|---:|"]
    for f in features:
        c = df[f]
        if not c.notna().any():
            L.append(f"| {f} | 0.0% | — | — | — |")
            continue
        L.append(f"| {f} | {c.notna().mean():.1%} | {c.mean():.3f} | "
                 f"{c.median():.3f} | {c.std():.3f} |")
    L.append("")

    L += ["## Sign check against the literature", "",
          "A build-time sanity check, not a result. Pearson is linear; Spearman is rank-based, "
          "so a large gap between them signals a non-linear relationship — which is exactly "
          "what the ML models are meant to capture and OLS is not.", "",
          "| Feature | Expected | Pearson | Spearman | Pearson sign |",
          "|---|:--:|---:|---:|:--:|"]
    for f, sign in THEORY_SIGNS.items():
        if f not in df.columns or not df[f].notna().any():
            continue
        p = df[f].corr(df[C.TARGET])
        # Pearson on ranks == Spearman, and avoids a scipy dependency at this stage.
        s = df[f].rank().corr(df[C.TARGET].rank())
        ok = "✓" if (p > 0) == (sign == "+") else "✗"
        L.append(f"| {f} | {sign} | {p:+.3f} | {s:+.3f} | {ok} |")
    L.append("")
    L.append("`size` is the clearest case: near-zero Pearson but strongly positive Spearman. "
             "Median leverage rises monotonically across size deciles (0.000 → 0.259) while "
             "the zero-debt share falls (60% → 3%); the relationship is real but not linear.")
    L.append("")

    L += ["## Known limitations carried into Phase 4", ""]
    if df.get("efwamb", pd.Series(dtype=float)).isna().all():
        L.append("- **`efwamb` is not computable.** Baker–Wurgler weights need equity and debt "
                 "issuance separately; the extract has only aggregate `cf_financing`. "
                 "`mtb_hist_avg` (equal-weighted historical mean M/B) is provided as a "
                 "documented fallback — it is what efwamb reduces to under constant issuance, "
                 "and must not be reported as Baker–Wurgler.")
    if df.get("altman_z_proxy_liabilities", pd.Series([0])).max() == 1:
        L.append("- **Altman Z uses a proxy for total liabilities** (`total_assets − "
                 "common_equity`), which absorbs minority interest and preferred stock. "
                 "`BS_TOT_LIAB2` would remove the approximation.")
    L.append("- **`stock_return_volatility` is built from annual returns**, so a 5-year window "
             "rests on at most 5 observations. A monthly price pull would make it a real "
             "volatility measure.")
    unc = df["industry_unclassified"].mean()
    L.append(f"- **{unc:.0%} of firm-years sit in the `Unclassified` sector**, so their "
             "industry median leverage is a median over a heterogeneous bucket. Re-run the "
             "ranking with `industry_unclassified == 0` as a robustness check.")
    L.append(f"- **Leverage is censored at zero** "
             f"({(df[C.TARGET] == 0).mean():.0%} of firm-years), so OLS is misspecified at the "
             "boundary. Tobit is the natural second baseline.")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    src = C.OUTPUTS / "panel_raw.csv"
    if not src.exists():
        print(f"ERROR: {src} not found -- run `python -m src.build_panel` first",
              file=sys.stderr)
        return 1

    panel = pd.read_csv(src)
    print(f"  panel_raw     : {len(panel):,} firm-years, {panel['ticker'].nunique():,} firms")

    df, features = build_features(panel)
    print(f"  features      : {len(features)} built")

    stages = df["lifecycle_stage"].value_counts()
    print("  stages        : " + ", ".join(f"{k} {v:,}" for k, v in stages.items()))

    df.to_csv(C.OUTPUTS / "panel_features.csv", index=False)
    write_report(C.OUTPUTS / "feature_report.md", df, features)
    pd.Series(features, name="feature").to_csv(C.OUTPUTS / "feature_list.csv", index=False)

    print(f"\n  panel_features.csv : {len(df):,} firm-years x {df.shape[1]} cols")
    print(f"  report             : outputs/feature_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
