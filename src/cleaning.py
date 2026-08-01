"""Cleaning utilities shared by Phase 1 (panel construction) and Phase 2 (features)."""

from __future__ import annotations

import pandas as pd

from src import config as C


def winsorize_by_year(df: pd.DataFrame, columns: list[str], year_col: str = "year",
                      limits: tuple[float, float] = C.WINSOR_LIMITS) -> pd.DataFrame:
    """
    Clip each column to its within-year percentile bounds (dissertation Step 1.4).

    Winsorising *per year* rather than pooled matters: leverage and profitability distributions
    shift over the cycle, and a pooled cut would clip disproportionately from crisis years.

    Applied to constructed ratios, not to raw levels -- see the note in config.WINSOR_LIMITS.
    """
    out = df.copy()
    lo, hi = limits
    for col in columns:
        if col not in out.columns:
            continue
        bounds = out.groupby(year_col)[col].quantile([lo, hi]).unstack()
        low = out[year_col].map(bounds[lo])
        high = out[year_col].map(bounds[hi])
        out[col] = out[col].clip(lower=low, upper=high)
    return out


def apply_min_years(df: pd.DataFrame, min_years: int = C.MIN_YEARS_PER_FIRM,
                    id_col: str = "ticker", required: str = "total_assets") -> pd.DataFrame:
    """Keep firms with at least `min_years` observations of `required` (Step 1.3)."""
    counts = df.dropna(subset=[required]).groupby(id_col).size()
    keep = counts[counts >= min_years].index
    return df[df[id_col].isin(keep)].copy()
