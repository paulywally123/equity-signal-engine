"""Fifth timing-overlay attempt, and the first using a signal genuinely
independent of this strategy's own price/return history: VIX level and a
credit-spread proxy (Baa corporate yield minus 10-year Treasury), pulled
from FRED. RUN ON YOUR MACHINE.

Same honest methodology as the last two attempts (drawdown_cap_holdout.py,
tail_trigger_holdout.py) -- select the trigger threshold using ONLY periods
before 2020, then evaluate that one config on the true 2020-2026 holdout it
never saw, against the no-filter baseline over the same window.

    python -m src.backtest.macro_trigger_holdout --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, level_trigger_exposure, run_backtest
from src.backtest.macro_data import fetch_fred_series
from src.report.attribution import annual_attribution

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_START = "2020-01-01"

INDICATORS = {
    "VIX":     {"series_id": "VIXCLS", "grid": [20, 25, 30, 35, 40]},
    "BAA10Y":  {"series_id": "BAA10Y", "grid": [2.5, 2.75, 3.0, 3.5, 4.0]},
}
GRID_FLOOR = [0.0, 0.5]


def _calmar(cagr: float, max_dd: float) -> float:
    return round(cagr / abs(max_dd), 3) if max_dd != 0 else float("nan")


def _metrics_on_slice(port_returns: pd.DataFrame, ppy: int, start=None, end=None) -> dict:
    sl = port_returns
    if start is not None:
        sl = sl[sl.index >= start]
    if end is not None:
        sl = sl[sl.index < end]
    m = compute_metrics(sl, periods_per_year=ppy)
    m["calmar"] = _calmar(m["cagr"], m["max_drawdown"])
    m["n_periods"] = len(sl)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--force-refresh", action="store_true")
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    raw_dir       = Path(cfg["data"]["raw_dir"])
    costs_bps     = cfg["costs"]["per_side_bps"]
    horizon       = cfg["labels"]["horizon_days"]
    ppy           = int(round(252 / horizon))

    predictions = pd.read_parquet(processed_dir / "predictions.parquet")
    labels      = pd.read_parquet(processed_dir / "labels.parquet")

    baseline_bt = run_backtest(predictions, labels, costs_bps=costs_bps,
                                top_q=0.20, long_only=True)
    baseline_sel  = _metrics_on_slice(baseline_bt, ppy, end=HOLDOUT_START)
    baseline_hold = _metrics_on_slice(baseline_bt, ppy, start=HOLDOUT_START)

    print()
    print("-" * 95)
    print(f"BASELINE  selection(n={baseline_sel['n_periods']}): Sharpe {baseline_sel['sharpe']:.3f} "
          f"CAGR {100*baseline_sel['cagr']:.2f}% maxDD {100*baseline_sel['max_drawdown']:.2f}% Calmar {baseline_sel['calmar']:.3f}")
    print(f"BASELINE  holdout(n={baseline_hold['n_periods']}):   Sharpe {baseline_hold['sharpe']:.3f} "
          f"CAGR {100*baseline_hold['cagr']:.2f}% maxDD {100*baseline_hold['max_drawdown']:.2f}% Calmar {baseline_hold['calmar']:.3f}")

    holdout_rows = [{"config": "baseline (no filter)",
                     **{k: baseline_hold[k] for k in ["sharpe", "cagr", "max_drawdown", "calmar"]}}]

    for name, spec in INDICATORS.items():
        logger.info("Fetching %s (FRED series %s) ...", name, spec["series_id"])
        series = fetch_fred_series(spec["series_id"], raw_dir / "macro", force_refresh=args.force_refresh)
        logger.info("  %s: %s .. %s (%d obs)", name, series.index.min().date(), series.index.max().date(), len(series))

        logger.info("Selecting %s threshold using ONLY periods before %s ...", name, HOLDOUT_START)
        rows = []
        bts = {}
        for th in spec["grid"]:
            for fl in GRID_FLOOR:
                exp = level_trigger_exposure(series, threshold=th, floor=fl)
                bt  = run_backtest(predictions, labels, costs_bps=costs_bps,
                                   top_q=0.20, long_only=True, exposure=exp)
                bts[(th, fl)] = bt
                sel_m = _metrics_on_slice(bt, ppy, end=HOLDOUT_START)
                rows.append({"threshold": th, "floor": fl,
                             "sel_sharpe": sel_m["sharpe"], "sel_cagr": sel_m["cagr"],
                             "sel_max_dd": sel_m["max_drawdown"], "sel_calmar": sel_m["calmar"]})

        sel_results = pd.DataFrame(rows)
        print()
        print("-" * 95)
        print(f"{name} SELECTION (periods before {HOLDOUT_START})")
        print("-" * 95)
        print(sel_results.sort_values("sel_calmar", ascending=False).to_string(index=False))

        best_row = sel_results.sort_values("sel_calmar", ascending=False).iloc[0]
        best_th, best_fl = float(best_row["threshold"]), float(best_row["floor"])
        logger.info("Honestly-selected %s config: threshold=%.2f floor=%.2f", name, best_th, best_fl)

        best_bt = bts[(best_th, best_fl)]
        best_hold = _metrics_on_slice(best_bt, ppy, start=HOLDOUT_START)
        holdout_rows.append({"config": f"{name}: threshold={best_th} floor={best_fl}",
                             **{k: best_hold[k] for k in ["sharpe", "cagr", "max_drawdown", "calmar"]}})

        print()
        print(f"{name} BAD-YEAR DETAIL IN HOLDOUT (2020, 2022):")
        ann_base = annual_attribution(baseline_bt, periods_per_year=ppy)
        ann_best = annual_attribution(best_bt, periods_per_year=ppy)
        cmp = pd.DataFrame({
            "baseline_return": ann_base["ann_return"],
            "trigger_return":  ann_best["ann_return"],
            "baseline_maxdd":  ann_base["max_dd"],
            "trigger_maxdd":   ann_best["max_dd"],
        }).loc[[2020, 2022]]
        print(cmp.to_string())

    print()
    print("-" * 95)
    print(f"HOLDOUT SUMMARY ({HOLDOUT_START} to end)")
    print("-" * 95)
    print(pd.DataFrame(holdout_rows).set_index("config").to_string())

    out = processed_dir / "macro_trigger_holdout.parquet"
    pd.DataFrame(holdout_rows).to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
