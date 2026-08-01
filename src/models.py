"""
Phase 4 -- model estimation and out-of-sample evaluation.

    python -m src.models

Fits every model on the same rows and the same columns, evaluates on the held-out 2020-2023
window, and writes metrics, predictions and fitted models to outputs/.

Two specifications are run, because they answer different questions:

  * **with_lag** includes `leverage_lag1`. Leverage is ~0.67 autocorrelated, so this is the
    honest forecasting benchmark -- but the lag dominates, and a feature ranking built on it
    mostly says "last year's leverage predicts this year's".
  * **no_lag** drops it. This is the specification RQ1 actually asks about: which firm
    characteristics predict leverage.

Both are reported. The gap between them is itself a result.
"""

from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src import config as C, splits

RANDOM_STATE = 42
MODELS_DIR = C.OUTPUTS / "models"


# --------------------------------------------------------------------------------------
# Tobit
# --------------------------------------------------------------------------------------

class TobitRegressor:
    """
    Left-censored (at zero) linear regression, estimated by maximum likelihood.

    26% of firm-years carry exactly zero debt. That is a real corner solution, not a missing
    value: OLS treats it as an ordinary draw and is misspecified at the boundary, which shows
    up as systematically negative fitted values. Tobit models the censoring directly.

        L = prod_{y>0} (1/s) phi((y - xb)/s)  *  prod_{y=0} Phi(-xb/s)

    Predictions are the expected *observed* value E[y|x], not the latent xb, so they are
    directly comparable with the other models' predictions and are never negative:

        E[y|x] = Phi(xb/s) * xb + s * phi(xb/s)

    statsmodels has no Tobit, so this is a direct implementation. It is validated against OLS
    on uncensored data in tests/test_models.py.
    """

    def __init__(self, max_iter: int = 500):
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None
        self.converged_ = False

    @staticmethod
    def _neg_loglik(params, X, y, uncensored):
        beta, log_sigma = params[:-1], params[-1]
        sigma = np.exp(log_sigma)          # keeps sigma positive without a constraint
        resid = y - X @ beta
        ll = np.empty(len(y))
        ll[uncensored] = (stats.norm.logpdf(resid[uncensored] / sigma) - log_sigma)
        ll[~uncensored] = stats.norm.logcdf(-(X[~uncensored] @ beta) / sigma)
        return -np.sum(ll)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xc = np.c_[np.ones(len(X)), X]
        uncensored = y > 0

        ols, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        resid_sd = np.std(y - Xc @ ols) or 1.0
        start = np.append(ols, np.log(resid_sd))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(self._neg_loglik, start, args=(Xc, y, uncensored),
                           method="L-BFGS-B", options={"maxiter": self.max_iter})

        self.converged_ = bool(res.success)
        self.intercept_, self.coef_ = res.x[0], res.x[1:-1]
        self.sigma_ = float(np.exp(res.x[-1]))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        xb = self.intercept_ + X @ self.coef_
        z = xb / self.sigma_
        return stats.norm.cdf(z) * xb + self.sigma_ * stats.norm.pdf(z)


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------

def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pred_negative_share": float((y_pred < 0).mean()),
    }


# --------------------------------------------------------------------------------------
# model definitions
# --------------------------------------------------------------------------------------

def linear_pipeline(estimator):
    """Scaling matters for Lasso's penalty; harmless for OLS. Kept identical for both."""
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def fit_models(design: splits.Design, cv, tune: bool = True) -> dict:
    """Fit every model on the same design matrix. Returns {name: fitted estimator}."""
    X, y = design.X_train, design.y_train
    out: dict[str, object] = {}

    out["Naive (train mean)"] = DummyRegressor(strategy="mean").fit(X, y)
    out["Naive (train median)"] = DummyRegressor(strategy="median").fit(X, y)
    out["OLS"] = linear_pipeline(LinearRegression()).fit(X, y)

    scaler = StandardScaler().fit(X)
    tobit = TobitRegressor().fit(scaler.transform(X), y)
    out["Tobit"] = Pipeline([("scale", scaler), ("model", tobit)])
    if not tobit.converged_:
        print("    WARNING: Tobit did not converge")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["Lasso"] = linear_pipeline(
            LassoCV(cv=list(cv), random_state=RANDOM_STATE, max_iter=5000, n_jobs=-1)
        ).fit(X, y)

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
                       objective="reg:squarederror")

    if tune:
        rf_grid = {"n_estimators": [300, 500], "max_depth": [8, 14, None],
                   "min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", 0.5]}
        xgb_grid = {"n_estimators": [300, 600], "max_depth": [3, 5, 8],
                    "learning_rate": [0.03, 0.08], "subsample": [0.7, 1.0],
                    "colsample_bytree": [0.7, 1.0], "min_child_weight": [1, 10],
                    "reg_lambda": [1.0, 5.0]}
        for name, est, grid, n_iter in [("Random Forest", rf, rf_grid, 12),
                                        ("XGBoost", xgb, xgb_grid, 20)]:
            search = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=list(cv),
                                        scoring="neg_root_mean_squared_error",
                                        random_state=RANDOM_STATE, n_jobs=-1)
            search.fit(X, y)
            out[name] = search.best_estimator_
            print(f"    {name} best: {search.best_params_}")
    else:
        out["Random Forest"] = rf.fit(X, y)
        out["XGBoost"] = xgb.fit(X, y)

    return out


def run_specification(df: pd.DataFrame, features: list[str], label: str,
                      tune: bool = True) -> tuple[pd.DataFrame, dict, splits.Design]:
    print(f"\n  [{label}] {len(features)} features")
    design = splits.build_design(df, features)
    print(f"    {splits.describe_split(design)}")

    cv = list(splits.expanding_window_folds(design.train_rows))
    print(f"    CV: {len(cv)} expanding-window folds")

    t0 = time.time()
    models = fit_models(design, cv, tune=tune)
    print(f"    fitted {len(models)} models in {time.time() - t0:.0f}s")

    rows, preds = [], {}
    for name, model in models.items():
        p_test = model.predict(design.X_test)
        p_train = model.predict(design.X_train)
        rows.append({"specification": label, "model": name,
                     **evaluate(design.y_test, p_test),
                     "train_r2": float(r2_score(design.y_train, p_train))})
        preds[name] = p_test

    metrics = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    predictions = pd.DataFrame(preds, index=design.test_rows.index)
    predictions.insert(0, "y_true", design.y_test)
    predictions.insert(0, "year", design.test_rows["year"])
    predictions.insert(0, "ticker", design.test_rows["ticker"])
    return metrics, {"models": models, "predictions": predictions}, design


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------

def check_tree_imputation_sensitivity(df: pd.DataFrame, features: list[str],
                                      design: splits.Design) -> dict:
    """
    Refit the trees on raw NaN input to confirm imputation is not driving their advantage.

    Trees split on missingness natively. If the imputed and raw fits differ materially, the
    comparison against the linear models is contaminated by the imputation choice rather than
    by model flexibility.
    """
    out = {}
    for name, est in [("Random Forest", RandomForestRegressor(
                           n_estimators=300, min_samples_leaf=5,
                           random_state=RANDOM_STATE, n_jobs=-1)),
                      ("XGBoost", XGBRegressor(
                           n_estimators=300, max_depth=5, learning_rate=0.08,
                           random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"))]:
        try:
            est.fit(design.X_train_raw, design.y_train)
            out[name] = float(r2_score(design.y_test, est.predict(design.X_test_raw)))
        except ValueError:
            out[name] = float("nan")  # RandomForest rejects NaN in older sklearn
    return out


def bootstrap_r2_gap(predictions: pd.DataFrame, model_a: str, model_b: str,
                     n_boot: int = 2000, seed: int = RANDOM_STATE) -> dict:
    """
    Confidence interval for the R2 gap between two models, bootstrapped by firm.

    Resampling rows would treat a firm's 4 test-period observations as 4 independent draws.
    They are not -- leverage is highly autocorrelated within firm -- so row resampling
    understates the standard error and overstates significance. Resampling whole firms keeps
    each firm's block of years together.
    """
    rng = np.random.default_rng(seed)
    firms = predictions["ticker"].unique()
    ticker = predictions["ticker"].to_numpy()
    index_by_firm = {f: np.flatnonzero(ticker == f) for f in firms}

    gaps = np.empty(n_boot)
    for i in range(n_boot):
        drawn = rng.choice(firms, size=len(firms), replace=True)
        idx = np.concatenate([index_by_firm[f] for f in drawn])
        g = predictions.iloc[idx]
        gaps[i] = r2_score(g["y_true"], g[model_a]) - r2_score(g["y_true"], g[model_b])

    observed = (r2_score(predictions["y_true"], predictions[model_a])
                - r2_score(predictions["y_true"], predictions[model_b]))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"gap": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "p_a_better": float((gaps > 0).mean())}


def yearly_r2(predictions: pd.DataFrame, model: str) -> pd.Series:
    """Test-period R2 year by year -- a single pooled figure can hide a bad year."""
    return (predictions.groupby("year")
            .apply(lambda g: r2_score(g["y_true"], g[model]), include_groups=False))


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def write_report(path, all_metrics: pd.DataFrame, extras: dict) -> None:
    L = ["# Phase 4 -- Modelling Results", "",
         "Generated by `python -m src.models`.", "",
         f"- Split: train {C.TEST_START_YEAR - 1} and earlier, test "
         f"{C.TEST_START_YEAR}–2023 (chronological, never random)",
         "- Every model sees identical rows and columns; imputation is fitted on train only",
         "- Hyperparameters tuned by expanding-window CV inside the training period", ""]

    for spec in all_metrics["specification"].unique():
        m = all_metrics[all_metrics.specification == spec]
        L += [f"## Specification: `{spec}`", "",
              extras["descriptions"][spec], "",
              "| Model | OOS R² | RMSE | MAE | Train R² | Negative predictions |",
              "|---|---:|---:|---:|---:|---:|"]
        for _, r in m.iterrows():
            L.append(f"| {r['model']} | **{r['r2']:.3f}** | {r['rmse']:.3f} | {r['mae']:.3f} | "
                     f"{r['train_r2']:.3f} | {r['pred_negative_share']:.1%} |")
        L.append("")

    L += ["## Reading these numbers", ""]
    best = all_metrics.loc[all_metrics.r2.idxmax()]
    L.append(f"Best overall: **{best['model']}** on `{best['specification']}` "
             f"(R² = {best['r2']:.3f}).")
    L.append("")

    for spec in all_metrics["specification"].unique():
        m = all_metrics[all_metrics.specification == spec].set_index("model")
        if "OLS" in m.index:
            ml = m.drop(index=[i for i in ("Naive (train mean)", "Naive (train median)")
                               if i in m.index])
            top = ml.r2.idxmax()
            gain = ml.loc[top, "r2"] - m.loc["OLS", "r2"]
            L.append(f"- `{spec}`: {top} beats OLS by {gain:+.3f} R². "
                     f"OLS itself beats predicting the training mean by "
                     f"{m.loc['OLS', 'r2'] - m.loc['Naive (train mean)', 'r2']:+.3f}.")
    L.append("")

    if extras.get("gaps"):
        L += ["### Is the gap over OLS real?", "",
              "R² difference between XGBoost and OLS, bootstrapped by **firm** (2,000 draws). "
              "Resampling rows would treat one firm's four test-period years as four "
              "independent observations; they are not, and doing so would overstate "
              "significance.", "",
              "| Specification | XGBoost − OLS | 95% CI | P(XGBoost better) |",
              "|---|---:|---|---:|"]
        for spec, g in extras["gaps"].items():
            L.append(f"| `{spec}` | {g['gap']:+.3f} | "
                     f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] | "
                     f"{g['p_a_better']:.0%} |")
        L.append("")

    if extras.get("imputation_check"):
        L += ["### Are the trees just exploiting imputation?", "",
              "Refitting the tree models on raw input with missing values left in place, "
              "instead of median-imputed:", "",
              "| Model | R² (imputed) | R² (raw NaN) |", "|---|---:|---:|"]
        for spec, vals in extras["imputation_check"].items():
            m = all_metrics[all_metrics.specification == spec].set_index("model")
            for name, raw_r2 in vals.items():
                if name in m.index:
                    L.append(f"| {name} (`{spec}`) | {m.loc[name, 'r2']:.3f} | {raw_r2:.3f} |")
        L += ["", "Close values mean the tree advantage is model flexibility, not the "
              "imputation choice.", ""]

    if extras.get("yearly"):
        L += ["### Test-period R² by year", "",
              "A pooled figure can hide a single bad year, and 2020 is the COVID shock.", "",
              "| Specification · model | " +
              " | ".join(str(y) for y in extras["yearly_years"]) + " |",
              "|---|" + "---:|" * len(extras["yearly_years"])]
        for key, series in extras["yearly"].items():
            L.append(f"| {key} | " +
                     " | ".join(f"{series.get(y, float('nan')):.3f}"
                               for y in extras["yearly_years"]) + " |")
        L.append("")

    L += ["## Caveats carried into Phase 5–6", "",
          "- **The 2019–20 IFRS 16 break sits at the train/test boundary.** Lease "
          "capitalisation raised measured leverage without any borrowing, so the test period "
          "is on a slightly different definition of the target than most of the training "
          "period. EDA showed out-of-sample R² is stable either side of the break, so the "
          "split stands, but level comparisons across 2019 are accounting artefacts.",
          "- **`leverage_lag1` dominates wherever it is included.** Treat `with_lag` as the "
          "forecasting benchmark and `no_lag` as the answer to RQ1.",
          "- **Tobit predictions are expected observed values** E[y|x], not the latent index, "
          "so they are comparable with the other models and never negative. The negative-"
          "prediction column shows how often OLS and Lasso violate the y ≥ 0 boundary.", ""]

    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------

def main() -> int:
    src = C.OUTPUTS / "panel_features.csv"
    if not src.exists():
        print(f"ERROR: {src} not found -- run `python -m src.features` first", file=sys.stderr)
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    features = pd.read_csv(C.OUTPUTS / "feature_list.csv")["feature"].tolist()
    features = [f for f in features if f in df.columns and df[f].notna().any()]
    print(f"  panel: {len(df):,} firm-years · {len(features)} usable features")

    specs = {
        "with_lag": features,
        "no_lag": [f for f in features if f != "leverage_lag1"],
    }
    descriptions = {
        "with_lag": "Includes `leverage_lag1`. Leverage is ~0.67 autocorrelated, so this is "
                    "the honest forecasting benchmark — but the lag dominates any feature "
                    "ranking built on it.",
        "no_lag": "Drops `leverage_lag1`. This is the specification RQ1 asks about: which "
                  "firm characteristics predict leverage, rather than how persistent leverage "
                  "is.",
    }

    all_metrics, artefacts, imputation_check, yearly, gaps = [], {}, {}, {}, {}
    for label, feats in specs.items():
        metrics, art, design = run_specification(df, feats, label)
        all_metrics.append(metrics)
        artefacts[label] = art
        imputation_check[label] = check_tree_imputation_sensitivity(df, feats, design)
        for model in ("XGBoost", "Random Forest", "OLS"):
            if model in art["predictions"].columns:
                yearly[f"`{label}` · {model}"] = yearly_r2(art["predictions"], model)
        art["predictions"].to_csv(C.OUTPUTS / f"predictions_{label}.csv", index=False)
        gaps[label] = bootstrap_r2_gap(art["predictions"], "XGBoost", "OLS")
        print(metrics[["model", "r2", "rmse", "mae"]].to_string(index=False))
        g = gaps[label]
        print(f"    XGBoost - OLS = {g['gap']:+.3f}  95% CI "
              f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]  "
              f"P(better) = {g['p_a_better']:.0%}")

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(C.OUTPUTS / "tables" / "model_metrics.csv", index=False)

    years = sorted(artefacts["no_lag"]["predictions"]["year"].unique())
    write_report(C.OUTPUTS / "model_report.md", metrics,
                 {"descriptions": descriptions, "imputation_check": imputation_check,
                  "yearly": yearly, "yearly_years": years, "gaps": gaps})

    try:
        import joblib
        for label, art in artefacts.items():
            for name, model in art["models"].items():
                if name.startswith("Naive"):
                    continue
                joblib.dump(model, MODELS_DIR / f"{label}__{name.replace(' ', '_')}.joblib")
        print(f"\n  models saved to outputs/models/")
    except ImportError:
        print("\n  (joblib unavailable — models not persisted)")

    print(f"  metrics: outputs/tables/model_metrics.csv")
    print(f"  report : outputs/model_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
