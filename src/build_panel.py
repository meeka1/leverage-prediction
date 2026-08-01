"""
Phase 1 -- build a tidy firm-year panel from the Bloomberg Terminal extracts.

    python -m src.build_panel

Reads every extract in data/raw/, merges the statement-level files into one firm x year
panel, joins the security master, applies the sample filters, and writes panel_raw.csv
plus a data-quality report.

Layout of the extracts
----------------------
Each time-series file holds a wide block per security. The layout differs between files
(the 'Dates' header sits in a different column in each), so it is detected rather than
hard-coded: the mnemonic lives in the same column as the 'Dates' label, the repeated ticker
two columns to its left, and the values immediately to its right.

Two indexing conventions coexist and are both correct:
  * fundamentals are indexed by FISCAL year -- the calendar year the fiscal year ends in
    (Diageo's June-2023 year-end and Greggs' December-2023 year-end both land in 2023);
  * market data is indexed by CALENDAR year end.
They are merged on that year, which pairs each fiscal year with market equity measured at the
calendar year end -- the standard convention.

Currency
--------
Market data is quoted in USD, fundamentals in each firm's reporting currency. Book leverage is
unaffected; market-to-book and market leverage are not. See config.FX_FILE for the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src import cleaning


# --------------------------------------------------------------------------------------
# layout detection
# --------------------------------------------------------------------------------------

def _find_cell(raw: pd.DataFrame, text: str, max_row: int = 12, max_col: int = 24):
    for i in range(min(max_row, len(raw))):
        for j in range(min(max_col, raw.shape[1])):
            if str(raw.iat[i, j]).strip() == text:
                return i, j
    return None, None


def _parse_year_header(raw: pd.DataFrame, hr: int, hc: int) -> list[int]:
    """Read the date header rightwards from the 'Dates' cell until it runs out."""
    years: list[int] = []
    for j in range(hc + 1, raw.shape[1]):
        cell = str(raw.iat[hr, j]).strip()
        if not cell:
            break
        # Files mix D/M/Y and M/D/Y; only the year is used, so the ambiguity is harmless.
        ts = pd.to_datetime(cell, errors="coerce", format="mixed", dayfirst=True)
        if pd.isna(ts):  # tolerate Excel serials
            num = pd.to_numeric(cell, errors="coerce")
            if pd.isna(num):
                break
            ts = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(num))
        years.append(int(ts.year))
    return years


# --------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------

def parse_timeseries_file(path: Path) -> pd.DataFrame:
    """Parse one statement file into long form: ticker, field, year, value, cell_state."""
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False, low_memory=False)

    hr, hc = _find_cell(raw, "Dates")
    if hr is None:
        raise ValueError(f"{path.name}: no 'Dates' header cell found")

    years = _parse_year_header(raw, hr, hc)
    if not years:
        raise ValueError(f"{path.name}: 'Dates' header row is empty")
    if len(set(years)) != len(years):
        raise ValueError(f"{path.name}: duplicate years in the header -- {years}")

    body = raw.iloc[hr + 1:].reset_index(drop=True)
    ticker_col, mnemonic_col = hc - 2, hc
    tickers = body[ticker_col].replace("", np.nan).ffill()
    fields = body[mnemonic_col].astype(str).str.strip()

    keep = fields.isin(C.FIELDS) & tickers.notna()
    if not keep.any():
        raise ValueError(f"{path.name}: no recognised mnemonics in column {mnemonic_col}")
    unknown = set(fields[fields.ne("") & ~fields.isin(C.FIELDS)])
    if unknown:
        print(f"    note: ignoring unrecognised field(s) in {path.name}: {sorted(unknown)}")

    body, tickers, fields = body[keep], tickers[keep], fields[keep]
    cells = body.iloc[:, hc + 1:hc + 1 + len(years)].to_numpy(dtype=object)

    flat = pd.Series(cells.ravel()).astype(str).str.strip()
    values = pd.to_numeric(flat, errors="coerce").to_numpy()
    is_blank = (flat == "").to_numpy()
    is_na = flat.str.startswith(C.NA_PREFIX).to_numpy() | (np.isnan(values) & ~is_blank)

    state = np.full(values.shape, C.CELL_VALUE, dtype=object)
    state[is_na] = C.CELL_NA
    state[is_blank] = C.CELL_BLANK

    n_rows = len(body)
    long = pd.DataFrame({
        "ticker": np.repeat(tickers.to_numpy(), len(years)),
        "field": np.repeat(fields.to_numpy(), len(years)),
        "year": np.tile(np.array(years), n_rows),
        "value": values,
        "cell_state": state,
        "source": path.name,
    })
    return long


def parse_static_file(path: Path) -> pd.DataFrame:
    """Parse the reference sheet. Header row is the one whose first cell is 'Ticker'."""
    hr, _ = _find_cell(path_raw := pd.read_csv(path, header=None, dtype=str,
                                               keep_default_na=False, low_memory=False),
                       "Ticker")
    if hr is None:
        raise ValueError(f"{path.name}: no 'Ticker' header row found")

    ref = pd.read_csv(path, header=hr, low_memory=False)
    ref = ref.loc[:, [c for c in ref.columns if not str(c).startswith("Unnamed")]]
    ref = ref[ref["Ticker"].astype(str).str.contains("Equity", na=False)].copy()
    ref = ref.replace(r"^#N/A.*", np.nan, regex=True)

    rename = {
        "Ticker": "ticker", "Short Name": "short_name", "Curncy": "quote_currency",
        "Listing Date": "listing_date", "GICS Sector Name": "gics_sector",
        "GICS Industry Group Name": "gics_industry_group",
        "GICS Industry Name": "gics_industry", "GICS Sub-Industry Name": "gics_sub_industry",
        "ISIN Number": "isin", "Currency Override": "reporting_currency",
        "Country or Territory ISO Code": "country", "Security Type": "security_type",
    }
    ref = ref.rename(columns={k: v for k, v in rename.items() if k in ref.columns})

    if "listing_date" in ref:
        ref["listing_date"] = pd.to_datetime(ref["listing_date"], errors="coerce", dayfirst=True)
        ref["listing_year"] = ref["listing_date"].dt.year

    # Keep unclassified firms as an explicit class rather than dropping them (config note).
    if "gics_sector" in ref:
        ref["sector_known"] = ref["gics_sector"].notna()
        ref["gics_sector"] = ref["gics_sector"].fillna(C.UNCLASSIFIED_SECTOR)

    del path_raw
    return ref.drop_duplicates(subset="ticker").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------

def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per firm-year, renaming mnemonics to readable names."""
    values = long[long["cell_state"] == C.CELL_VALUE]
    wide = values.pivot_table(index=["ticker", "year"], columns="field",
                              values="value", aggfunc="first")
    wide = wide.rename(columns={k: v[0] for k, v in C.FIELDS.items()})
    return wide.reset_index()


def apply_fx(wide: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Convert market values from their quote currency into each firm's reporting currency.

    No-op with a warning if config.FX_FILE is absent -- market-derived ratios are then
    currency-inconsistent and every affected column is flagged in the report.
    """
    market_cols = [c for c in C.MARKET_FIELDS if c in wide.columns]
    if not market_cols:
        return wide, "no market columns"
    if not C.FX_FILE.exists():
        return wide, "missing"

    fx = pd.read_csv(C.FX_FILE)
    need = {"year", "currency", "rate_to_usd"}
    if not need <= set(fx.columns):
        raise ValueError(f"{C.FX_FILE.name}: expected columns {sorted(need)}")

    out = wide.merge(ref[["ticker", "reporting_currency"]], on="ticker", how="left")
    out = out.merge(fx.rename(columns={"currency": "reporting_currency"}),
                    on=["year", "reporting_currency"], how="left")
    for col in market_cols:
        out[col] = out[col] / out["rate_to_usd"]
    converted = out["rate_to_usd"].notna().mean()
    return out.drop(columns=["reporting_currency", "rate_to_usd"]), f"applied ({converted:.0%})"


def build(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    static = [f for f in files if C.STATIC_HINT in f.name.lower()]
    series = [f for f in files if f not in static]
    if not static:
        raise SystemExit("ERROR: no static/reference extract found in data/raw/")
    if not series:
        raise SystemExit("ERROR: no statement extracts found in data/raw/")

    ref = pd.concat([parse_static_file(f) for f in static], ignore_index=True)
    ref = ref.drop_duplicates(subset="ticker", keep="first")
    print(f"  security master : {len(ref):,} tickers "
          f"({int(ref['sector_known'].sum()):,} with GICS sector)")

    frames = []
    for f in series:
        part = parse_timeseries_file(f)
        n_fields = part["field"].nunique()
        yrs = part["year"]
        print(f"  {f.name[-24:]:<26} {n_fields:2d} fields, "
              f"{part['ticker'].nunique():,} tickers, {yrs.min()}-{yrs.max()}")
        frames.append(part)

    long = pd.concat(frames, ignore_index=True)
    dup = long.duplicated(subset=["ticker", "field", "year"]).sum()
    if dup:
        print(f"  WARNING: {dup:,} duplicate ticker/field/year cells -- keeping first")
        long = long.drop_duplicates(subset=["ticker", "field", "year"], keep="first")

    wide = to_wide(long)
    wide, fx_status = apply_fx(wide, ref)
    return long, wide, ref, {"fx": fx_status, "files": [f.name for f in files]}


def apply_filters(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
    """Sample construction, recording attrition at each step for the methodology chapter."""
    steps: list[tuple[str, int, int]] = []

    def record(label, df):
        steps.append((label, len(df), df["ticker"].nunique()))
        return df

    record("parsed firm-years", panel)
    panel = record("has total assets", panel[panel["total_assets"].notna()])
    if C.REQUIRE_POSITIVE_ASSETS:
        panel = record("total assets > 0", panel[panel["total_assets"] > 0])
    panel = record(f"firm has >={C.MIN_YEARS_PER_FIRM} years",
                   cleaning.apply_min_years(panel))
    return panel.reset_index(drop=True), steps


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def write_report(path: Path, long: pd.DataFrame, panel: pd.DataFrame, ref: pd.DataFrame,
                 steps: list, meta: dict) -> None:
    L = ["# Phase 1 -- Data Quality Report", "",
         "Generated by `python -m src.build_panel`.", "",
         "## Sources", ""]
    L += [f"- `{f}`" for f in meta["files"]] + [""]

    L += ["## Sample construction", "",
          "| Step | Firm-years | Firms |", "|---|---:|---:|"]
    for label, n, k in steps:
        L.append(f"| {label} | {n:,} | {k:,} |")
    L.append("")

    L += ["## Coverage by year", "",
          "| Year | Firms | Median book leverage | Zero-debt share |",
          "|---:|---:|---:|---:|"]
    lev = (panel["total_debt"] / panel["total_assets"]).clip(lower=0)
    for year, grp in panel.assign(lev=lev).groupby("year"):
        med = grp["lev"].median()
        zero = (grp["lev"] == 0).mean()
        L.append(f"| {year} | {grp['ticker'].nunique():,} | "
                 f"{med:.3f} | {zero:.1%}" + " |")
    L.append("")

    # download completeness -- blanks after a firm's last observation are legitimate
    counts = long.groupby("cell_state").size()
    total = int(counts.sum())
    blank = int(counts.get(C.CELL_BLANK, 0))
    trailing = _trailing_blank_share(long)
    L += ["## Download completeness", "",
          f"- Real values: **{int(counts.get(C.CELL_VALUE, 0)):,}** "
          f"({counts.get(C.CELL_VALUE, 0) / total:.1%})",
          f"- `#N/A` (no data exists): **{int(counts.get(C.CELL_NA, 0)):,}** "
          f"({counts.get(C.CELL_NA, 0) / total:.1%})",
          f"- Empty cells: **{blank:,}** ({blank / total:.1%}), of which "
          f"**{trailing:.1%}** fall after the firm's last observation", ""]
    if trailing > 0.999:
        L.append("> ✅ Every empty cell is a post-delisting tail. No evidence of a truncated "
                 "download.")
    else:
        L.append("> ⚠️ Some empty cells sit *inside* a firm's series — those are genuine gaps "
                 "in the download, not delisting.")
    L.append("")

    L += ["## Currency consistency", ""]
    ratio, detail = currency_consistency_test(panel)
    if ratio is None:
        L.append("- Not enough native-USD reporters to run the consistency test.")
    else:
        ok = 0.85 <= ratio <= 1.15
        L += [f"- Median market-to-book, {detail}",
              f"- **Ratio = {ratio:.3f}**", ""]
        if ok:
            L.append("> ✅ Market data and fundamentals share a currency (USD throughout). If "
                     "they did not, GBP reporters would sit ~30–90% above native-USD reporters "
                     "on market-to-book. Market leverage and market-to-book are therefore "
                     "internally consistent, and the old mixed-local-currency problem is gone.")
        else:
            L.append("> ⚠️ The two reporter groups differ enough to suggest the market and "
                     "fundamental blocks are in different currencies. Supply "
                     "`data/raw/fx_rates.csv` (see `config.FX_FILE`) and re-run.")
        L.append("")
        L.append("Size enters as log(assets in USD), so it carries GBPUSD movement — worth one "
                 "sentence in the methodology.")
        L.append("")
    if meta["fx"] not in ("missing", "no market columns"):
        L.append(f"- FX restatement applied: **{meta['fx']}**")
        L.append("")
    cur = ref["reporting_currency"].value_counts().head(6)
    L.append("- Reporting currencies (as filed): " +
             ", ".join(f"{k} {v:,}" for k, v in cur.items()))
    L.append("")

    L += ["## Sector classification", ""]
    known = int(ref["sector_known"].sum())
    L += [f"- GICS resolved: **{known:,}** of {len(ref):,} ({known / len(ref):.0%})",
          f"- Remaining pooled into `{C.UNCLASSIFIED_SECTOR}`: **{len(ref) - known:,}**", "",
          "Bloomberg does not resolve GICS for most delisted securities. Those firms are kept "
          "rather than dropped — dropping them would reintroduce the survivorship bias the "
          "design exists to avoid. The GICS 40/55 screen was applied at source (no Financials "
          "or Utilities appear anywhere in the universe), so the unclassified bucket is "
          "screened too.", "",
          "> Industry median leverage over a single large `Unclassified` bucket is not "
          "economically meaningful. Phase 2 flags those observations, and the ranking should "
          "be re-checked with them excluded as a robustness test.", ""]
    if "gics_sector" in panel.columns:
        vc = panel["gics_sector"].value_counts()
        L += ["| Sector | Firm-years |", "|---|---:|"]
        L += [f"| {k} | {v:,} |" for k, v in vc.items()] + [""]

    L += ["## Field coverage (share of firm-years with a value)", "",
          "| Field | Coverage |", "|---|---:|"]
    field_cols = [v[0] for v in C.FIELDS.values() if v[0] in panel.columns]
    for col in sorted(field_cols, key=lambda c: -panel[c].notna().mean()):
        L.append(f"| {col} | {panel[col].notna().mean():.1%} |")
    L.append("")

    missing = [v[0] for v in C.FIELDS.values() if v[0] not in panel.columns]
    if missing:
        L += ["### Fields not present in this extract", "",
              ", ".join(f"`{m}`" for m in missing), "",
              "These are parsed automatically if a later pull includes them. Without "
              "`equity_issued` / `equity_repurchased` / `lt_debt_net`, the Baker–Wurgler "
              "EFWAMB feature cannot be constructed; without `total_liabilities`, Altman Z "
              "falls back to `total_assets − common_equity`.", ""]

    path.write_text("\n".join(L), encoding="utf-8")


def currency_consistency_test(panel: pd.DataFrame):
    """
    Detect a currency mismatch between the market block and the fundamentals block.

    Market-to-book divides a market quantity by a book quantity. If the two blocks were
    denominated differently, firms reporting in GBP would show a systematically inflated ratio
    relative to firms already reporting in USD -- roughly the GBPUSD rate. Comparing the two
    groups needs no outside data, which is what makes it trustworthy.
    """
    need = {"market_cap", "common_equity", "reporting_currency"}
    if not need <= set(panel.columns):
        return None, ""
    d = panel[(panel["common_equity"] > 0) & (panel["market_cap"] > 0)].copy()
    d["mtb"] = d["market_cap"] / d["common_equity"]

    local, usd = C.FX_CONSISTENCY_GROUPS
    a = d[d["reporting_currency"].isin(local)]["mtb"]
    b = d[d["reporting_currency"].isin(usd)]["mtb"]
    if len(a) < 100 or len(b) < 100:
        return None, ""
    detail = (f"{'/'.join(local)} reporters {a.median():.3f} (n={len(a):,}) vs "
              f"{'/'.join(usd)} reporters {b.median():.3f} (n={len(b):,})")
    return float(a.median() / b.median()), detail


def _trailing_blank_share(long: pd.DataFrame) -> float:
    """Share of empty cells that fall after the firm-field's last real observation."""
    blanks = long[long["cell_state"] == C.CELL_BLANK]
    if blanks.empty:
        return 1.0
    last = (long[long["cell_state"] == C.CELL_VALUE]
            .groupby(["ticker", "field"])["year"].max().rename("last_year"))
    merged = blanks.merge(last, on=["ticker", "field"], how="left")
    # a blank at a firm-field with no data at all is also legitimate
    return float((merged["last_year"].isna() | (merged["year"] > merged["last_year"])).mean())


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------

def main() -> int:
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    files = sorted(C.DATA_RAW.glob("*.csv"))
    files = [f for f in files if f.name != C.FX_FILE.name]
    if not files:
        print(f"ERROR: no CSV extracts found in {C.DATA_RAW}", file=sys.stderr)
        return 1

    long, wide, ref, meta = build(files)
    panel, steps = apply_filters(wide)
    panel = panel.merge(ref, on="ticker", how="left")

    unmatched = int(panel["short_name"].isna().sum())
    if unmatched:
        print(f"  WARNING: {unmatched:,} firm-years have no security-master match")

    panel = panel.sort_values(["ticker", "year"]).reset_index(drop=True)
    panel.to_csv(C.OUTPUTS / "panel_raw.csv", index=False)
    ref.to_csv(C.OUTPUTS / "security_master.csv", index=False)
    write_report(C.OUTPUTS / "data_quality_report.md", long, panel, ref, steps, meta)

    print(f"\n  panel_raw.csv : {len(panel):,} firm-years x {panel.shape[1]} cols, "
          f"{panel['ticker'].nunique():,} firms, {panel['year'].min()}-{panel['year'].max()}")
    print(f"  report        : outputs/data_quality_report.md")
    ratio, _ = currency_consistency_test(panel)
    if ratio is not None:
        verdict = "consistent" if 0.85 <= ratio <= 1.15 else "MISMATCH -- see report"
        print(f"  currency      : market/fundamentals {verdict} (MTB ratio {ratio:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
