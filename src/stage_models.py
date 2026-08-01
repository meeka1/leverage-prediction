"""
Phase 5 -- life cycle heterogeneity.

    python -m src.stage_models

Trains a separate model within each Dickinson stage and asks whether that buys anything over
one pooled model. Writes metrics and per-stage models to outputs/.

The comparison that matters
---------------------------
Reporting "R2 within Growth = 0.4" alone says nothing: stages differ in how predictable they
are, so a stage-specific model can look good simply because its stage is easy. The benchmark
is therefore **the pooled model evaluated on exactly the same rows**. Only the difference
between the two answers the question RQ2 poses -- whether leverage is generated differently
across the life cycle, or whether one model covers every stage.

Two constraints from the EDA shape this
---------------------------------------
* Stage cells are small (Decline: 942 train / 216 test), so hyperparameters are **not** re-tuned
  per stage -- that would overfit the cell. The pooled-tuned settings from Phase 4 are reused,
  which also keeps the stage-specific and pooled models directly comparable.
* Stage persistence is only ~52%, so a stage label is a firm-*year* state. A `stable` variant
  restricts to firm-years whose stage matches the previous year, testing whether results are
  driven by firms genuinely settled in a stage rather than by churn.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src import config as C, splits, viz
from src.models import RANDOM_STATE, MODELS_DIR, bootstrap_r2_gap, evaluate

# Tuned on the pooled sample in Phase 4; reused unchanged so stage models are not
# advantaged by per-cell tuning on a few hundred observations.
XGB_PARAMS = dict(n_estimators=300, max_depth=8, learning_rate=0.03, subsample=0.7,
                  colsample_bytree=1.0, min_child_weight=10, reg_lambda=1.0,
                  random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
RF_PARAMS = dict(n_estimators=500, max_depth=None, min_samples_leaf=1, max_features=0.5,
                 random_state=RANDOM_STATE, n_jobs=-1)

MIN_TRAIN, MIN_TEST = 200, 50


def add_stable_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Mark firm-years whose stage equals the previous year's (see module docstring)."""
    out = df.sort_values(["ticker", "year"]).copy()
    prev = out.groupby("ticker")["lifecycle_stage"].shift(1)
    contiguous = out.groupby("ticker")["year"].diff().eq(1)
    out["stage_stable"] = (out["lifecycle_stage"] == prev) & contiguous
    return out


def build_models() -> dict:
    return {
        "OLS": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": RandomForestRegressor(**RF_PARAMS),
        "XGBoost": XGBRegressor(**XGB_PARAMS),
    }


def run_stages(df: pd.DataFrame, features: list[str], label: str,
               stable_only: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Fit per-stage models and compare each against the pooled model on identical rows.

    Returns (metrics, fitted stage models, per-row predictions).
    """
    data = df[df["lifecycle_stage"].notna()].copy()
    if stable_only:
        data = data[data["stage_stable"]]

    # The pooled benchmark is trained once on every stage, then scored stage by stage.
    pooled_design = splits.build_design(data, features)
    pooled = build_models()
    for m in pooled.values():
        m.fit(pooled_design.X_train, pooled_design.y_train)

    rows, fitted, preds = [], {}, []
    for stage in viz.STAGE_ORDER:
        mask_tr = pooled_design.train_rows["lifecycle_stage"] == stage
        mask_te = pooled_design.test_rows["lifecycle_stage"] == stage
        n_tr, n_te = int(mask_tr.sum()), int(mask_te.sum())
        if n_tr < MIN_TRAIN or n_te < MIN_TEST:
            print(f"    {stage:13s} SKIPPED (train {n_tr}, test {n_te} below minimum)")
            continue

        Xtr, ytr = pooled_design.X_train[mask_tr.values], pooled_design.y_train[mask_tr.values]
        Xte, yte = pooled_design.X_test[mask_te.values], pooled_design.y_test[mask_te.values]

        stage_pred = {}
        for name, proto in build_models().items():
            model = proto.__class__(**proto.get_params()) if not isinstance(proto, Pipeline) \
                else Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
            model.fit(Xtr, ytr)
            p_stage = model.predict(Xte)
            p_pooled = pooled[name].predict(Xte)

            rows.append({
                "specification": label, "stage": stage, "model": name,
                "n_train": n_tr, "n_test": n_te,
                **{f"stage_{k}": v for k, v in evaluate(yte, p_stage).items()},
                "pooled_r2": float(r2_score(yte, p_pooled)),
                "gain": float(r2_score(yte, p_stage) - r2_score(yte, p_pooled)),
            })
            fitted[f"{stage}__{name}"] = model
            stage_pred[f"{name}_stage"] = p_stage
            stage_pred[f"{name}_pooled"] = p_pooled

        block = pd.DataFrame(stage_pred, index=Xte.index)
        block.insert(0, "y_true", yte.values)
        block.insert(0, "stage", stage)
        block.insert(0, "year", pooled_design.test_rows.loc[mask_te.values, "year"].values)
        block.insert(0, "ticker", pooled_design.test_rows.loc[mask_te.values, "ticker"].values)
        preds.append(block)
        print(f"    {stage:13s} train {n_tr:5,d}  test {n_te:4,d}  "
              f"XGB stage {rows[-1]['stage_r2']:.3f} vs pooled {rows[-1]['pooled_r2']:.3f}")

    return pd.DataFrame(rows), fitted, pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def write_report(path, metrics: pd.DataFrame, stable: pd.DataFrame, gaps: dict) -> None:
    L = ["# Phase 5 -- Life Cycle Heterogeneity", "",
         "Generated by `python -m src.stage_models`.", "",
         "Each stage gets its own model, trained only on that stage's firm-years. The "
         "benchmark is the **pooled model scored on exactly the same rows**, because stages "
         "differ in how predictable they are and a raw within-stage R² would confound the two.",
         "", "Hyperparameters are the pooled-tuned values from Phase 4, not re-tuned per "
         "stage: with as few as 942 training rows in Decline, per-cell tuning would fit noise.",
         ""]

    for spec in metrics["specification"].unique():
        m = metrics[metrics.specification == spec]
        L += [f"## Specification: `{spec}`", "",
              "| Stage | n train | n test | Model | Stage-specific R² | Pooled R² | Gain |",
              "|---|---:|---:|---|---:|---:|---:|"]
        for _, r in m.iterrows():
            flag = " ✅" if r["gain"] > 0.02 else (" ⚠️" if r["gain"] < -0.02 else "")
            L.append(f"| {r['stage']} | {r['n_train']:,} | {r['n_test']:,} | {r['model']} | "
                     f"{r['stage_r2']:.3f} | {r['pooled_r2']:.3f} | "
                     f"**{r['gain']:+.3f}**{flag} |")
        L.append("")

    xgb = metrics[metrics.model == "XGBoost"]
    helped = xgb[xgb.gain > 0.02]["stage"].unique()
    hurt = xgb[xgb.gain < -0.02]["stage"].unique()
    L += ["## Does splitting by stage help?", ""]
    if len(helped):
        L.append(f"- Stage-specific training **helps** in: {', '.join(sorted(set(helped)))}")
    if len(hurt):
        L.append(f"- Stage-specific training **hurts** in: {', '.join(sorted(set(hurt)))} "
                 "— the cell is too small to outweigh the information lost by discarding the "
                 "other stages.")
    if not len(helped) and not len(hurt):
        L.append("- No stage shows a material gain either way: one pooled model covers the "
                 "life cycle as well as five separate ones.")
    L += ["", "A negative gain is a real finding, not a failure. It says the *relationships* "
          "between firm characteristics and leverage are common across the life cycle even "
          "where the *levels* differ, which is what the EDA's persistent level gaps already "
          "hinted at.", ""]

    if gaps:
        L += ["### Are the gains real?", "",
              "Stage-specific minus pooled R², bootstrapped by firm (2,000 draws).", "",
              "| Specification | Stage | Gain | 95% CI | P(stage-specific better) |",
              "|---|---|---:|---|---:|"]
        for (spec, stage), g in gaps.items():
            L.append(f"| `{spec}` | {stage} | {g['gap']:+.3f} | "
                     f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] | {g['p_a_better']:.0%} |")
        L.append("")

    if len(stable):
        L += ["## Robustness: firms settled in a stage", "",
              "Stage persistence is only ~52%, so a label is a firm-year state rather than a "
              "firm attribute. Restricting to firm-years whose stage matches the previous "
              "year tests whether the results come from firms genuinely settled in a stage.",
              "", "| Stage | n test | Stage-specific R² | Pooled R² | Gain |",
              "|---|---:|---:|---:|---:|"]
        for _, r in stable[stable.model == "XGBoost"].iterrows():
            L.append(f"| {r['stage']} | {r['n_test']:,} | {r['stage_r2']:.3f} | "
                     f"{r['pooled_r2']:.3f} | {r['gain']:+.3f} |")
        L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    src = C.OUTPUTS / "panel_features.csv"
    if not src.exists():
        print(f"ERROR: {src} not found -- run `python -m src.features` first", file=sys.stderr)
        return 1

    df = add_stable_flag(pd.read_csv(src))
    features = [f for f in pd.read_csv(C.OUTPUTS / "feature_list.csv")["feature"]
                if f in df.columns and df[f].notna().any()]
    specs = {"with_lag": features, "no_lag": [f for f in features if f != "leverage_lag1"]}

    all_metrics, gaps, fitted_all = [], {}, {}
    for label, feats in specs.items():
        print(f"\n  [{label}]")
        metrics, fitted, preds = run_stages(df, feats, label)
        all_metrics.append(metrics)
        fitted_all[label] = fitted
        if len(preds):
            preds.to_csv(C.OUTPUTS / f"predictions_stage_{label}.csv", index=False)
            for stage in preds["stage"].unique():
                sub = preds[preds.stage == stage].rename(
                    columns={"XGBoost_stage": "A", "XGBoost_pooled": "B"})
                gaps[(label, stage)] = bootstrap_r2_gap(sub, "A", "B", n_boot=2000)

    print("\n  [no_lag · stable stages only]")
    stable_metrics, _, _ = run_stages(df, specs["no_lag"], "no_lag_stable", stable_only=True)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(C.OUTPUTS / "tables" / "stage_metrics.csv", index=False)
    write_report(C.OUTPUTS / "stage_report.md", metrics, stable_metrics, gaps)

    try:
        import joblib
        (MODELS_DIR / "stages").mkdir(parents=True, exist_ok=True)
        for label, models in fitted_all.items():
            for key, model in models.items():
                joblib.dump(model, MODELS_DIR / "stages" /
                            f"{label}__{key.replace(' ', '_')}.joblib")
    except ImportError:
        pass

    print(f"\n  metrics: outputs/tables/stage_metrics.csv")
    print(f"  report : outputs/stage_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
