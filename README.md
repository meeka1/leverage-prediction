# ML in Capital Structure — MSc Dissertation

Machine learning applied to capital structure decisions of UK-listed non-financial firms,
2005–2023, using Bloomberg Terminal data.

## Status

| Stage | State |
|---|---|
| Data collection | ⛔ **Incomplete — re-pull required.** See `docs/01_bloomberg_data_requirements.md` |
| Data pipeline (raw → tidy annual panel) | ✅ Built and runnable |
| Variable construction | ⏸ Waiting on the dissertation thesis |
| Modelling | ⏸ Waiting on the dissertation thesis |

## Quick start

```bash
pip install -r requirements.txt
python -m src.build_panel
```

Reads every Bloomberg extract in `data/raw/` (falling back to the project root, where the
original two files sit), and writes to `outputs/`:

| File | Contents |
|---|---|
| `panel_annual.csv` | The panel: one row per firm-year, raw Bloomberg fields plus quality flags |
| `panel_annual_long.csv.gz` | Same data in long form, with per-cell download status |
| `security_master.csv` | Cleaned reference data (sector, currency, listing dates, trading status) |
| `data_quality_report.md` | **Read this first.** Coverage, defects, and the go/no-go checklist |
| `repull_ticker_list.csv` | The 1,949 tickers worth re-requesting, priority-ordered |

## Two defects in the current extract

Both are documented with evidence in `docs/01_bloomberg_data_requirements.md` §1.

1. **~40% of cells were never downloaded.** The original request asked for 2.39m data points,
   far above a normal monthly Bloomberg quota. Everything from 2014 onward is mostly empty.
2. **The date axis is wrong for 2,827 of 3,000 firms.** The pull hid dates (`Dts=H`), so one
   shared header row is applied to securities with different history lengths. Seraphim Space
   (IPO July 2021) has its real 2021–23 price path sitting in columns labelled 2013–2015.

Only **173 tickers** currently have a trustworthy time axis. The pipeline exposes this as
`date_axis_reliable`; filter on it before doing anything time-indexed.

## Working on partial data

The pipeline is deliberately split so the incomplete extract does not block progress:

- **`src/build_panel.py` makes no thesis-specific choices.** It parses, splices, collapses the
  pseudo-semi-annual snapshots to annual, and flags problems. It does not compute leverage,
  drop outliers, or impute — those are research decisions.
- **Cleaning thresholds are reported, not applied.** The quality report lists negative-equity
  firm-years, shell companies, stale filings and extreme ratios so those calls can be made
  deliberately and defended in the methodology.
- **Re-running is the only step needed after a re-pull.** Drop new CSVs into `data/raw/` and run
  the command again. Extracts are spliced with real values preferred over `#N/A`, and `#N/A`
  preferred over blanks, so overlapping tranches are safe and the 173 good tickers from the
  original pull are retained.

## Layout

```
data/raw/        Bloomberg extracts (tidy or legacy block layout, both parsed)
docs/            Data requirements and re-pull specification
src/config.py    Paths, field metadata, cleaning thresholds
src/build_panel.py   Raw extracts -> tidy annual panel + quality report
outputs/         Generated; safe to delete and rebuild
```
