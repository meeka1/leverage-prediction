"""
Panel-aware train/test splitting and design-matrix construction.

Isolated in its own module so leakage is structurally impossible rather than something to
remember: nothing here ever looks forward in time, and every transformation is fitted on the
training rows alone and only then applied to validation and test.

Three rules this enforces:

1. **Splits are chronological, never random.** A random split over a panel puts the same firm
   in train and test in adjacent years, and leverage is ~0.67 autocorrelated, so a random split
   inflates out-of-sample R2 by leaking the answer.

2. **Imputation is fitted on train only.** Median-filling with the full-sample median leaks the
   test distribution into training.

3. **Every model sees the same rows and the same columns.** Tree models tolerate NaN and the
   linear models do not; if each used the sample it could handle, their R2 values would be
   computed over different denominators and would not be comparable. One imputed design matrix
   is built for all of them, and the trees are additionally checked on raw NaN input to confirm
   imputation is not doing the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config as C


@dataclass
class Design:
    """A model-ready dataset: aligned matrices plus the metadata needed to interpret them."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame
    features: list[str]
    X_train_raw: pd.DataFrame = field(repr=False, default=None)
    X_test_raw: pd.DataFrame = field(repr=False, default=None)

    def __repr__(self) -> str:
        return (f"Design(train={len(self.y_train):,} x {len(self.features)}, "
                f"test={len(self.y_test):,})")


def time_split(df: pd.DataFrame, test_start: int = C.TEST_START_YEAR):
    """Split chronologically. Everything before `test_start` trains; the rest is held out."""
    return df[df["year"] < test_start].copy(), df[df["year"] >= test_start].copy()


def expanding_window_folds(train: pd.DataFrame, n_folds: int = 4, val_years: int = 2):
    """
    Expanding-window cross-validation for hyperparameter tuning.

    Each fold trains on everything up to a cutoff and validates on the following `val_years`,
    mirroring how the model will actually be used. `sklearn.TimeSeriesSplit` splits on row
    position, which is wrong for a panel -- rows are firm-years, so position does not order
    time. Splitting on the year column is what makes this panel-aware.

    Yields (train_idx, val_idx) as positional arrays for sklearn's `cv` parameter.
    """
    years = np.sort(train["year"].unique())
    last_cut = years[-1] - val_years
    cuts = np.linspace(years[0] + 4, last_cut, n_folds).round().astype(int)
    pos = np.arange(len(train))
    yr = train["year"].to_numpy()

    seen = set()
    for cut in cuts:
        if cut in seen:
            continue
        seen.add(cut)
        tr = pos[yr <= cut]
        va = pos[(yr > cut) & (yr <= cut + val_years)]
        if len(tr) and len(va):
            yield tr, va


def build_design(df: pd.DataFrame, features: list[str], target: str = C.TARGET,
                 test_start: int = C.TEST_START_YEAR,
                 add_missing_indicators: bool = True) -> Design:
    """
    Assemble aligned train/test matrices with train-fitted median imputation.

    Rows are kept when the target is present; a row missing some predictors is imputed rather
    than dropped, because dropping would silently change the sample between feature sets and
    make R2 incomparable across models.
    """
    keep = [f for f in features if f in df.columns and df[f].notna().any()]
    dropped = [f for f in features if f not in keep]
    if dropped:
        print(f"    dropping {len(dropped)} all-missing feature(s): {', '.join(dropped)}")

    data = df.dropna(subset=[target]).copy()
    train, test = time_split(data, test_start)

    X_train_raw, X_test_raw = train[keep].copy(), test[keep].copy()

    # Indicators are built before imputation, and only where the training data is actually
    # missing values -- an indicator that is constant in train carries no information and
    # would just add a collinear column.
    indicator_cols = []
    if add_missing_indicators:
        for f in keep:
            if X_train_raw[f].isna().any():
                indicator_cols.append(f"{f}_isna")

    medians = X_train_raw.median()
    X_train = X_train_raw.fillna(medians)
    X_test = X_test_raw.fillna(medians)

    # A feature entirely absent from train cannot be imputed from train; fall back to zero and
    # let the indicator carry the signal.
    X_train = X_train.fillna(0.0)
    X_test = X_test.fillna(0.0)

    for f in keep:
        col = f"{f}_isna"
        if col in indicator_cols:
            X_train[col] = X_train_raw[f].isna().astype(int)
            X_test[col] = X_test_raw[f].isna().astype(int)

    return Design(
        X_train=X_train, y_train=train[target],
        X_test=X_test, y_test=test[target],
        train_rows=train, test_rows=test,
        features=list(X_train.columns),
        X_train_raw=X_train_raw, X_test_raw=X_test_raw,
    )


def describe_split(design: Design) -> str:
    tr, te = design.train_rows, design.test_rows
    return (f"train {len(tr):,} firm-years / {tr['ticker'].nunique():,} firms "
            f"({tr['year'].min()}-{tr['year'].max()})  |  "
            f"test {len(te):,} / {te['ticker'].nunique():,} firms "
            f"({te['year'].min()}-{te['year'].max()})")
