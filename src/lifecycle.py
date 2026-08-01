"""
Dickinson (2011) firm life cycle classification.

Each firm-year is assigned to one of five stages from the signs of the three cash flows.
There are eight sign combinations, not five -- three of them map to Shakeout and two to
Decline. Collapsing that to a five-way mapping by hand is the usual place this goes wrong.

    Ref: Dickinson, V. (2011). Cash Flow Patterns as a Proxy for Firm Life Cycle.
         The Accounting Review, 86(6), 1969-1994.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STAGES = ["Introduction", "Growth", "Mature", "Shakeout", "Decline"]

# (operating > 0, investing > 0, financing > 0) -> stage
SIGN_MAP = {
    (False, False, True):  "Introduction",
    (True,  False, True):  "Growth",
    (True,  False, False): "Mature",
    (True,  True,  True):  "Shakeout",
    (True,  True,  False): "Shakeout",
    (False, False, False): "Shakeout",
    (False, True,  True):  "Decline",
    (False, True,  False): "Decline",
}


def classify(df: pd.DataFrame, oper: str = "cf_operating", inv: str = "cf_investing",
             fin: str = "cf_financing") -> pd.DataFrame:
    """
    Add `lifecycle_stage` and `cf_sign_zero` columns.

    Dickinson's partition is on strict signs. Exact zeros are treated as non-positive, which
    follows the paper's own convention, but they are flagged so their weight can be checked --
    a large share would mean the mapping is doing more work than the data supports.
    """
    out = df.copy()
    missing = out[[oper, inv, fin]].isna().any(axis=1)

    signs = list(zip(out[oper] > 0, out[inv] > 0, out[fin] > 0))
    stage = pd.Series([SIGN_MAP[s] for s in signs], index=out.index, dtype="object")
    stage[missing] = np.nan

    out["lifecycle_stage"] = pd.Categorical(stage, categories=STAGES)
    out["cf_sign_zero"] = (out[[oper, inv, fin]] == 0).any(axis=1) & ~missing
    return out


def stage_summary(df: pd.DataFrame, year_col: str = "year",
                  split_year: int | None = None) -> pd.DataFrame:
    """Counts per stage, optionally split into train/test at `split_year`."""
    d = df.dropna(subset=["lifecycle_stage"])
    if split_year is None:
        out = d["lifecycle_stage"].value_counts().to_frame("n")
    else:
        out = pd.crosstab(d["lifecycle_stage"], d[year_col] >= split_year)
        out.columns = [f"train_<{split_year}", f"test_>={split_year}"][:out.shape[1]]
        out["n"] = out.sum(axis=1)
    out["share"] = (out["n"] / out["n"].sum() * 100).round(1)
    return out.sort_values("n", ascending=False)
