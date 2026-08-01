"""Paths, field metadata and cleaning thresholds for the capital-structure panel."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"

# --------------------------------------------------------------------------------------
# raw extracts
# --------------------------------------------------------------------------------------
# Bloomberg writes '#N/A ...' when it answers "no data exists", and leaves a cell genuinely
# empty when the request never completed. Keeping these apart is how the defects in the first
# extract were detected; the current extract's blanks are all legitimate post-delisting tails.
NA_PREFIX = "#N/A"
CELL_VALUE, CELL_NA, CELL_BLANK = "value", "na", "blank"

# Files whose name matches this are parsed as time-series blocks; the rest as static reference.
STATIC_HINT = "static"

# --------------------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------------------
# mnemonic -> (short name, statement)
FIELDS = {
    # balance sheet
    "BS_TOT_ASSET":             ("total_assets",        "balance_sheet"),
    "SHORT_AND_LONG_TERM_DEBT": ("total_debt",          "balance_sheet"),
    "TOT_COMMON_EQY":           ("common_equity",       "balance_sheet"),
    "BS_NET_FIX_ASSET":         ("net_ppe",             "balance_sheet"),
    "BS_CASH_NEAR_CASH_ITEM":   ("cash",                "balance_sheet"),
    "BS_RETAIN_EARN":           ("retained_earnings",   "balance_sheet"),
    "WORKING_CAPITAL":          ("working_capital",     "balance_sheet"),
    # income statement
    "SALES_REV_TURN":           ("revenue",             "income"),
    "EBITDA":                   ("ebitda",              "income"),
    "EBIT":                     ("ebit",                "income"),
    "IS_PRETAX_INCOME":         ("pretax_income",       "income"),
    "IS_INC_TAX_EXP":           ("tax_expense",         "income"),
    "IS_OPERATING_EXPENSES_RD": ("rd_expense",          "income"),
    # cash flow
    "CF_CASH_FROM_OPER":        ("cf_operating",        "cash_flow"),
    "CF_CASH_FROM_INV_ACT":     ("cf_investing",        "cash_flow"),
    "CF_CASH_FROM_FNC_ACT":     ("cf_financing",        "cash_flow"),
    "CF_CAP_EXPEND_PRPTY_ADD":  ("capex",               "cash_flow"),
    "CF_DEPR_AMORT":            ("depreciation",        "cash_flow"),
    "CF_DVD_PAID":              ("dividends_paid",      "cash_flow"),
    # market (NOTE: quoted in USD -- see CURRENCY notes below)
    "CUR_MKT_CAP":              ("market_cap",          "market"),
    "PX_LAST":                  ("price_last",          "market"),
    # --- not in the current extract; parsed automatically if a later pull adds them ---
    "IS_INT_EXPENSE":           ("interest_expense",    "income"),
    "BS_TOT_LIAB2":             ("total_liabilities",   "balance_sheet"),
    "CF_INCR_CAP_STOCK":        ("equity_issued",       "cash_flow"),
    "CF_DECR_CAP_STOCK":        ("equity_repurchased",  "cash_flow"),
    "CF_LT_DEBT_CASH_FLOW":     ("lt_debt_net",         "cash_flow"),
}

MARKET_FIELDS = {v[0] for v in FIELDS.values() if v[1] == "market"}

# --------------------------------------------------------------------------------------
# currency
# --------------------------------------------------------------------------------------
# The extract is denominated in USD throughout -- both market data and fundamentals.
#
#   * Market data: Diageo's 2023 close of 36.45 = GBP 28.56 x 1.2731 (GBPUSD on 29-Dec-2023).
#   * Fundamentals: verified by a two-group test rather than against outside figures. If
#     fundamentals were in local currency while market cap was USD, GBP reporters would show a
#     market-to-book systematically ~30-90% above native-USD reporters. Measured gap is 0.7%
#     (median 1.726 vs 1.714 over 13,791 firm-years), so both sides share a currency.
#     build_panel re-runs this test on every build and reports the result.
#
# Consequence: market-to-book, market leverage and size are all internally consistent, and the
# old mixed-LCL problem is gone. Size in log(USD assets) does carry GBPUSD movement, which is
# worth one sentence in the methodology.
#
# Optional: to restate everything in GBP, drop a CSV at data/raw/fx_rates.csv with columns
# year,currency,rate_to_usd (USD per 1 unit, e.g. GBP 2023 -> 1.2731). Conversion is applied
# to market columns automatically. Not required for the analysis.
FX_FILE = DATA_RAW / "fx_rates.csv"
MARKET_QUOTE_CURRENCY = "USD"

# Firms whose market-to-book is compared to detect a currency mismatch between the market and
# fundamental blocks. Ratio near 1.0 => consistent.
FX_CONSISTENCY_GROUPS = (("GBP", "GBp"), ("USD",))

# --------------------------------------------------------------------------------------
# sample construction (dissertation Phase 1, Step 1.3)
# --------------------------------------------------------------------------------------
MIN_YEARS_PER_FIRM = 5
REQUIRE_POSITIVE_ASSETS = True

# Sector is missing for ~76% of tickers because Bloomberg does not resolve GICS for delisted
# securities. Those firms are kept and pooled into one explicit class rather than dropped --
# dropping them would reintroduce exactly the survivorship bias the design avoids.
# The GICS 40/55 screen was applied at source (no Financials or Utilities appear anywhere),
# so the unclassified bucket is screened too.
UNCLASSIFIED_SECTOR = "Unclassified"

# Winsorisation is defined here but deliberately NOT applied to raw levels -- clipping total
# assets at the 1st/99th percentile would distort genuine scale differences between a microcap
# and Diageo. It is applied to constructed ratios in Phase 2 (see src/cleaning.py).
WINSOR_LIMITS = (0.01, 0.99)

EXCEL_EPOCH_OFFSET = 25569  # days between 1899-12-30 and 1970-01-01

# --------------------------------------------------------------------------------------
# Phase 2 -- features
# --------------------------------------------------------------------------------------
TARGET = "book_leverage"

# Industry median leverage is computed leave-one-out (a firm never contributes to its own
# industry median) and only where the sector-year cell has at least this many other firms.
MIN_INDUSTRY_PEERS = 5

# Rolling window for earnings and stock-return volatility, and the minimum periods required.
VOLATILITY_WINDOW = 5
VOLATILITY_MIN_PERIODS = 3

# Ratios bounded on economic grounds before winsorising. Effective tax rate is meaningless
# outside [0, 1] once pretax income turns negative.
EFFECTIVE_TAX_BOUNDS = (0.0, 1.0)

# Train/test boundary from the dissertation spec (train 2005-2019, test 2020-2023).
TEST_START_YEAR = 2020

# Predictors are built contemporaneously with the target. Set True to shift every predictor
# back one year within firm, turning the exercise into genuine one-step-ahead prediction.
LAG_PREDICTORS = False

