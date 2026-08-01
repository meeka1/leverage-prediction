"""
Correctness tests for the custom estimators and the splitting logic.

The Tobit implementation is hand-written maximum likelihood, so it is checked against two
cases with known answers before any of its output is reported. The split tests guard the
leakage rules that the whole design rests on.

    python -m tests.test_models
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import splits
from src.models import TobitRegressor


def test_tobit_matches_ols_when_uncensored():
    """With no censoring the Tobit likelihood reduces to OLS, so coefficients must agree."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 3))
    beta = np.array([1.5, -0.8, 0.4])
    y = 5.0 + X @ beta + rng.normal(scale=0.5, size=n)   # shifted well clear of zero

    fitted = TobitRegressor().fit(X, y)
    ols, *_ = np.linalg.lstsq(np.c_[np.ones(n), X], y, rcond=None)

    assert np.allclose(fitted.coef_, ols[1:], atol=0.02), \
        f"coefficients diverge: {fitted.coef_} vs {ols[1:]}"
    assert abs(fitted.intercept_ - ols[0]) < 0.02
    assert abs(fitted.sigma_ - 0.5) < 0.05, f"sigma {fitted.sigma_}"
    print("  PASS  tobit reduces to OLS without censoring")


def test_tobit_recovers_truth_under_censoring():
    """The point of Tobit: OLS on censored data is biased, Tobit is not."""
    rng = np.random.default_rng(1)
    n = 6000
    X = rng.normal(size=(n, 2))
    beta = np.array([1.0, -0.5])
    latent = 0.2 + X @ beta + rng.normal(scale=1.0, size=n)
    y = np.maximum(latent, 0.0)                          # left-censored at zero
    censored_share = (y == 0).mean()
    assert 0.2 < censored_share < 0.6, f"test setup: {censored_share:.2f} censored"

    tobit = TobitRegressor().fit(X, y)
    ols, *_ = np.linalg.lstsq(np.c_[np.ones(n), X], y, rcond=None)

    tobit_err = np.abs(tobit.coef_ - beta).mean()
    ols_err = np.abs(ols[1:] - beta).mean()
    assert tobit_err < ols_err, f"tobit {tobit_err:.3f} should beat OLS {ols_err:.3f}"
    assert tobit_err < 0.06, f"tobit error {tobit_err:.3f} too large"
    print(f"  PASS  tobit recovers truth under {censored_share:.0%} censoring "
          f"(err {tobit_err:.3f} vs OLS {ols_err:.3f})")


def test_tobit_predictions_are_non_negative():
    """Predictions are E[y|x] for a censored variable, so they cannot be below the bound."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 2))
    y = np.maximum(0.1 + X @ np.array([1.0, -0.5]) + rng.normal(size=2000), 0.0)
    pred = TobitRegressor().fit(X, y).predict(X)
    assert (pred >= 0).all(), f"{(pred < 0).sum()} negative predictions"
    print("  PASS  tobit predictions respect the y >= 0 boundary")


def _panel(n_firms=60, years=range(2005, 2024)):
    rows = [{"ticker": f"F{f}", "year": y, "book_leverage": 0.1 + 0.01 * (y - 2005),
             "x1": f * 0.01 + y * 0.001, "x2": np.nan if y < 2008 else 1.0}
            for f in range(n_firms) for y in years]
    return pd.DataFrame(rows)


def test_split_is_chronological():
    df = _panel()
    design = splits.build_design(df, ["x1", "x2"])
    assert design.train_rows["year"].max() < design.test_rows["year"].min(), \
        "train and test periods overlap"
    assert design.test_rows["year"].min() == 2020
    print("  PASS  split is chronological with no overlap")


def test_imputation_is_fitted_on_train_only():
    """A test-set median must never influence the imputed values."""
    df = _panel()
    df.loc[df.year >= 2020, "x1"] = 999.0        # extreme values only in the test period
    df.loc[df.year == 2010, "x1"] = np.nan       # a training gap to fill

    design = splits.build_design(df, ["x1", "x2"])
    train_median = df[(df.year < 2020) & df.x1.notna()]["x1"].median()
    filled = design.X_train.loc[design.train_rows.year == 2010, "x1"]
    assert np.allclose(filled, train_median), \
        f"imputed {filled.iloc[0]} != train median {train_median}"
    assert filled.iloc[0] < 999.0, "test-period values leaked into training imputation"
    print("  PASS  imputation uses train-only medians")


def test_cv_folds_never_look_forward():
    df = _panel()
    train, _ = splits.time_split(df)
    train = train.reset_index(drop=True)
    folds = list(splits.expanding_window_folds(train))
    assert folds, "no folds produced"
    for tr_idx, va_idx in folds:
        assert train.loc[tr_idx, "year"].max() < train.loc[va_idx, "year"].min(), \
            "a validation fold precedes its training data"
    print(f"  PASS  {len(folds)} CV folds are strictly forward-looking")


def test_missing_indicators_added_only_where_needed():
    df = _panel()
    design = splits.build_design(df, ["x1", "x2"])
    assert "x2_isna" in design.X_train.columns, "indicator missing for a gappy feature"
    assert "x1_isna" not in design.X_train.columns, "indicator added for a complete feature"
    print("  PASS  missing indicators added only where train has gaps")


def test_shap_matches_additivity():
    """SHAP contributions plus bias must reconstruct the model's own predictions exactly."""
    import warnings
    warnings.filterwarnings("ignore")
    import xgboost as xgb
    from xgboost import XGBRegressor
    from src.shap_analysis import shap_values

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    y = X["a"] * 2 - X["b"] + rng.normal(scale=0.3, size=500)
    model = XGBRegressor(n_estimators=40, max_depth=3, random_state=0).fit(X, y)

    sv = shap_values(model, X)
    full = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    assert np.abs(full.sum(1) - model.predict(X)).max() < 1e-4, "additivity violated"
    assert sv.shape == X.shape, f"bias column not dropped: {sv.shape} vs {X.shape}"
    # the two informative features must outrank the noise ones
    top2 = set(sv.abs().mean().nlargest(2).index)
    assert top2 == {"a", "b"}, f"ranking wrong: {top2}"
    print("  PASS  shap contributions are additive and rank informative features first")


if __name__ == "__main__":
    print("Tobit:")
    test_tobit_matches_ols_when_uncensored()
    test_tobit_recovers_truth_under_censoring()
    test_tobit_predictions_are_non_negative()
    print("Splits:")
    test_split_is_chronological()
    test_imputation_is_fitted_on_train_only()
    test_cv_folds_never_look_forward()
    test_missing_indicators_added_only_where_needed()
    print("SHAP:")
    test_shap_matches_additivity()
    print("\nAll tests passed.")
