# ML in Capital Structure

*A Machine Learning Approach to Predicting Firm Leverage in the UK Market: Which Features
Matter Most, and How Does This Vary Across Firm Life Cycle Stages?*

UK non-financial, non-utility listed firms, 2005–2023, Bloomberg Terminal (UCL SoM).

## Status

| Phase | State |
|---|---|
| 1 — Data preparation | ✅ **Done** — `outputs/panel_raw.csv` |
| 2 — Feature engineering | ✅ **Done** — `outputs/panel_features.csv` |
| 3 — EDA | ✅ **Done** — `notebooks/03_eda.ipynb`, 9 figures, 4 tables |
| 4 — Modelling | ✅ **Done** — `outputs/model_report.md` |
| 5 — Life cycle heterogeneity | ✅ **Done** — `outputs/stage_report.md` |
| 6 — SHAP interpretation | ✅ **Done** — `outputs/shap_report.md` |
| 7 — Writing | ⏭ Next |

## Quick start

```bash
pip install -r requirements.txt
python -m src.build_panel   # Phase 1 -> panel_raw.csv
python -m src.features      # Phase 2 -> panel_features.csv
python -m src.eda           # Phase 3 -> figures, tables, eda_report.md
python -m src.models        # Phase 4 -> model_report.md
python -m src.stage_models  # Phase 5 -> stage_report.md
python -m src.shap_analysis # Phase 6 -> shap_report.md
python -m tests.test_models # correctness checks
```

`notebooks/03_eda.ipynb` and `notebooks/04_modelling.ipynb` present the results; the analysis
itself lives in `src/` so it stays version-controlled and re-runnable.

| Output | Contents |
|---|---|
| `outputs/panel_raw.csv` | **17,842 firm-years × 1,388 firms, 2005–2023.** Raw Bloomberg fields + security master |
| `outputs/panel_features.csv` | Target, 29 predictors, Dickinson life cycle stage |
| `outputs/feature_list.csv` | Predictor names, for Phase 4 |
| `outputs/security_master.csv` | Sector, ISIN, currency, listing date, per ticker |
| `outputs/data_quality_report.md` | **Read first** — sample attrition, coverage, currency test |
| `outputs/feature_report.md` | Coverage, usable sample per feature set, theory sign check |
| `outputs/eda_report.md` | Narrative EDA findings |
| `outputs/figures/` | 9 figures (`fig01`–`fig09`) |
| `outputs/tables/` | Descriptives, correlations, model metrics, SHAP rankings |
| `outputs/models/` | Fitted models (joblib) |

## The panel

Built from five Bloomberg extracts in `data/raw/` (balance sheet, cash flow, income statement,
market data, static). Layout differs between files, so it is detected rather than hard-coded.

Sample construction, as reported by every build:

| Step | Firm-years | Firms |
|---|---:|---:|
| parsed | 22,111 | 2,347 |
| has total assets | 19,364 | 1,949 |
| total assets > 0 | 19,304 | 1,948 |
| ≥5 years per firm | **17,842** | **1,388** |

Train 2005–2019: 14,561 · Test 2020–2023: 3,281.

## Three things the parser gets right

**Fiscal-year indexing.** Fundamentals are indexed by the calendar year the fiscal year *ends*
in — Diageo's June-2023 year-end and Greggs' December-2023 year-end both land in 2023, the
standard Compustat `fyear` convention. Market data is indexed by calendar year end. Merging on
that year pairs each fiscal year with market equity at year end. No offset is needed.

**`#N/A` vs empty.** Bloomberg writes `#N/A` when no data exists and leaves a cell empty when
the request never completed. The parser keeps them apart, and the report checks that every
empty cell falls *after* a firm's last observation. It currently does — 100% — so there is no
truncated download.

**Currency.** Verified by a two-group test that needs no outside data: if fundamentals were in
local currency while market cap was USD, GBP reporters would show a market-to-book ~30–90%
above native-USD reporters. The measured gap is 0.7% (1.726 vs 1.714), so both blocks are USD.
Market leverage and market-to-book are internally consistent. The test re-runs on every build.

## Phase 2 — features

`src/features.py` builds the target and 29 predictors; `src/lifecycle.py` applies Dickinson
(2011). Four choices affect the results:

**Leave-one-out industry median.** A firm never contributes to the industry median it is
compared against — otherwise the feature partly contains the target and gradient boosting will
exploit it.

**The target is winsorised before anything is derived from it.** Industry median leverage and
`leverage_lag1` are both functions of the target; letting a raw leverage of 5,932 through would
contaminate two predictors at once.

**Year gaps are masked.** `groupby.shift(1)` returns the previous *observation*, not the
previous *year*. Where a firm's series skips a year, the one-period features (`leverage_lag1`,
`asset_growth`, `stock_return`, `financing_deficit`) are blanked rather than carrying a stale
value labelled as last year's — 92 firm-years affected.

**Effective tax rate is zero, not missing, for loss-makers.** A firm with no taxable income
gets no interest tax shield, so zero is the economically correct value; the companion indicator
preserves the distinction. This lifts coverage from 37% to 68%.

Usable sample depends on which features you take, since list-wise deletion is driven by the
sparsest one:

| Feature set | Predictors | Complete cases | Firms |
|---|---:|---:|---:|
| core (≥95% coverage) | 12 | 16,680 (93%) | 1,361 |
| broad (≥90%) | 16 | 14,286 (80%) | 1,303 |
| wide (≥75%) | 23 | 11,336 (64%) | 1,212 |
| all available | 28 | 4,799 (27%) | 679 |

Tree models take NaN natively and can use the widest set; OLS, Tobit and Lasso need complete
rows or imputation.

## Phase 3 — what the EDA found

**Leverage is censored at zero.** 26% of firm-years carry no debt; the median is 0.097 overall
but 0.182 among levered firms. OLS is misspecified at that boundary, so Tobit joins the
baseline set.

**Size predicts leverage, but not linearly.** Median leverage rises across every size decile
while the zero-debt share falls from 63% to 3% — yet the Pearson correlation is −0.009 against
a Spearman of +0.374. A linear model sees nothing here. This is the empirical case for the
tree models and belongs in the Discussion explicitly (`fig06`).

**Stages separate and stay separated.** Median leverage by stage — Growth 0.200, Mature 0.130,
Shakeout 0.040, Introduction 0.039, Decline 0.007 — with the ordering stable across two
decades. A persistent gap is what justifies stage-specific models rather than a stage dummy.

**The smallest test cell is Decline at 216 firm-years.** Stage-specific metrics and SHAP
rankings there need bootstrapped intervals, not point estimates.

**A structural break in 2019–20 (IFRS 16).** Median leverage jumps and the zero-debt share
collapses from 31% to 12%. This is lease capitalisation, not borrowing: a balanced panel shows
the same jump (not survivorship), debt and PP&E inflate together while revenue does not (a
right-of-use asset plus its matching liability), and among firms moving from zero to positive
debt the share raising *no cash at all* rises from ~25–33% to ~50%. Greggs plc is the clean
example — debt 0 → 361m USD, PP&E 420m → 820m USD, on +8% revenue. Out-of-sample R² is stable
either side of the break, so the specified split stands; but **statements about leverage levels
across 2019 are accounting artefacts, not economics** (`fig09`).

**Dickinson stages are firm-year states, not firm attributes.** Year-on-year persistence is
52%, and only 5% of firms never change stage. Phase 5 results are therefore about cash-flow
patterns rather than firms, and warrant a robustness check on firms holding a stage for two or
more consecutive years.

**Only one collinear pair above |r| = 0.8**: `profitability` ↔ `ocf_to_assets` (+0.82).
Harmless for trees; drop one for OLS or lean on the L1 penalty and say so.

**Expected R² for Phase 4.** A pooled OLS on core features gives out-of-sample R² ≈ 0.14
without lagged leverage and ≈ 0.59 with it. That gap is why both specifications are needed:
the second is the honest prediction benchmark, the first is the feature-ranking exercise.

Figures use the data-viz reference palette unmodified, in its documented slot order, on a light
surface (the output is a printed dissertation, so no dark variant is generated).

## Phases 4–6 — results

**ML beats OLS, and the margin is significant.** Out-of-sample R² on 2020–2023:

| Model | with_lag | no_lag |
|---|---:|---:|
| Random Forest | **0.709** | **0.469** |
| XGBoost | 0.689 | 0.452 |
| OLS | 0.632 | 0.379 |
| Lasso | 0.627 | 0.378 |
| Tobit | 0.624 | 0.366 |
| Naive (train mean) | −0.011 | −0.011 |

Bootstrapped by firm (2,000 draws), XGBoost − OLS is +0.073 on `no_lag`, 95% CI [+0.012, +0.145],
better in 99% of resamples. The gap holds in every individual test year.

**Stage-specific models are worse than pooled in every stage** (`no_lag`, XGBoost):

| Stage | n test | Stage-specific R² | Pooled R² | Gain |
|---|---:|---:|---:|---:|
| Introduction | 741 | 0.474 | 0.494 | −0.020 |
| Growth | 441 | 0.281 | 0.361 | −0.080 |
| Mature | 1,351 | 0.573 | 0.630 | −0.057 |
| Shakeout | 431 | 0.330 | 0.566 | −0.236 |
| Decline | 213 | 0.278 | 0.407 | −0.129 |

Splitting the sample costs more in observations than it gains in specificity. That is a
finding, not a failure: the *relationships* are common across the life cycle even where the
*levels* differ. It is also why Phase 6 reads stage heterogeneity off the pooled model.

**RQ1 — top drivers** (mean |SHAP|, all 100% bootstrap-stable): `negative_book_equity`,
`altman_z`, `cash_holdings`, `size`, `price_to_book`. Stability drops to 0% at rank 11, so the
top 10 is an unambiguous cutoff.

**RQ2 — the answer is qualified.** The four leading features hold the *same rank in every
stage*; only the middle of the ranking reshuffles (`market_to_book` moves 5↔9, `tangibility`
6↔8). The dominant drivers of leverage are universal across the life cycle.

## Design notes

**Unclassified sector.** Bloomberg resolves GICS for only 726 of 3,000 tickers — mostly the
still-listed ones. Those firms are kept and pooled into an explicit `Unclassified` class rather
than dropped, because dropping them would reintroduce the survivorship bias the design exists
to avoid. The GICS 40/55 screen was applied at source (no Financials or Utilities appear
anywhere), so the unclassified bucket is screened too. Industry median leverage over that
bucket is not economically meaningful — Phase 2 flags it, and the feature ranking should be
re-checked with it excluded as a robustness test.

**Winsorisation** is defined in `src/cleaning.py` but deliberately *not* applied to raw levels —
clipping total assets at the 1st/99th percentile would erase the genuine scale difference
between a microcap and Diageo. It is applied per year to constructed ratios in Phase 2.

**Fields absent from the current extract:** `interest_expense`, `total_liabilities`,
`equity_issued`, `equity_repurchased`, `lt_debt_net`. They are parsed automatically if a later
pull adds them. Without the three cash-flow items, Baker–Wurgler EFWAMB cannot be built;
without `total_liabilities`, Altman Z falls back to `total_assets − common_equity`.

## Layout

```
data/raw/            Bloomberg extracts (+ optional fx_rates.csv)
docs/                Dissertation specification
src/config.py        Paths, field map, sample filters, currency notes
src/cleaning.py      Winsorisation, minimum-years filter
src/build_panel.py   Phase 1: extracts -> panel_raw.csv + quality report
outputs/             Generated
```

`outputs/panel_annual*.csv` and `repull_ticker_list.csv` are leftovers from the first Bloomberg
extract, which has since been replaced. They are not used by anything and can be deleted.
