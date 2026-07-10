"""Fourth and final timing-overlay attempt: a fast tail-only drawdown
trigger, validated honestly out-of-sample from the start (not swept on the
full sample and read off after the fact -- see drawdown_cap_holdout.py for
why that order of operations gave a misleading answer last time). RUN ON
YOUR MACHINE.

Genuinely different design from the previous three (binary 200d SMA regime
filter, continuous vol-target scaling, 250d partial trend filter): reacts to
the equal-weight index's drawdown from its own trailing peak over a SHORT
window (10-40 days), rather than a slow long-window trend/vol level. Meant
to fire only on sharp, tail-scale drawdowns and -- critically -- re-arm
quickly once the market actually recovers, since the short window cuts both
ways instead of being structurally slow to turn back on.

    python -m src.backtest.tail_trigger_holdout --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, drawdown_trigger_exposure, run_backtest
from src.report.attribution import annual_attribution

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_START = "2020-01-01"


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
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    costs_bps     = cfg["costs"]["per_side_bps"]
    horizon       = cfg["labels"]["horizon_days"]
    ppy           = int(round(252 / horizon))

    predictions   = pd.read_parquet(processed_dir / "predictions.parquet")
    labels        = pd.read_parquet(processed_dir / "labels.parquet")
    returns_panel = pd.read_parquet(processed_dir / "returns.parquet")
    returns_panel.index = pd.to_datetime(returns_panel.index)

    baseline_bt = run_backtest(predictions, labels, costs_bps=costs_bps,
                                top_q=0.20, long_only=True)

    grid_lookback  = [10, 20, 40]
    grid_threshold = [0.08, 0.10, 0.15, 0.20]
    grid_floor     = [0.0, 0.5]

    logger.info("Selecting (lookback, threshold, floor) using ONLY periods before %s ...", HOLDOUT_START)
    rows = []
    bts = {}
    for lb in grid_lookback:
        for th in grid_threshold:
            for fl in grid_floor:
                exp = drawdown_trigger_exposure(returns_panel, lookback=lb, threshold=th, floor=fl)
                bt  = run_backtest(predictions, labels, costs_bps=costs_bps,
                                   top_q=0.20, long_only=True, exposure=exp)
                bts[(lb, th, fl)] = bt
                sel_m = _metrics_on_slice(bt, ppy, end=HOLDOUT_START)
                rows.append({"lookback": lb, "threshold": th, "floor": fl,
                             "sel_sharpe": sel_m["sharpe"], "sel_cagr": sel_m["cagr"],
                             "sel_max_dd": sel_m["max_drawdown"], "sel_calmar": sel_m["calmar"]})

    sel_results = pd.DataFrame(rows)
    baseline_sel = _metrics_on_slice(baseline_bt, ppy, end=HOLDOUT_START)

    print()
    print("-" * 95)
    print(f"SELECTION WINDOW (periods before {HOLDOUT_START}, n={baseline_sel['n_periods']})")
    print("-" * 95)
    print(f"Baseline (no filter):  Sharpe {baseline_sel['sharpe']:.3f}  CAGR {100*baseline_sel['cagr']:.2f}%  "
          f"maxDD {100*baseline_sel['max_drawdown']:.2f}%  Calmar {baseline_sel['calmar']:.3f}")
    print(sel_results.sort_values("sel_calmar", ascending=False).head(10).to_string(index=False))

    best_row = sel_results.sort_values("sel_calmar", ascending=False).iloc[0]
    best_lb  = int(best_row["lookback"])
    best_th  = float(best_row["threshold"])
    best_fl  = float(best_row["floor"])
    logger.info("Honestly-selected config (best Calmar, pre-2020 only): lookback=%d threshold=%.2f floor=%.2f",
                best_lb, best_th, best_fl)

    print()
    print("-" * 95)
    holdout_n = _metrics_on_slice(baseline_bt, ppy, start=HOLDOUT_START)["n_periods"]
    print(f"HOLDOUT ({HOLDOUT_START} to end, n={holdout_n})")
    print("-" * 95)

    baseline_hold = _metrics_on_slice(baseline_bt, ppy, start=HOLDOUT_START)
    best_bt = bts[(best_lb, best_th, best_fl)]
    best_hold = _metrics_on_slice(best_bt, ppy, start=HOLDOUT_START)

    holdout_results = pd.DataFrame([
        {"config": "baseline (no filter)", **{k: baseline_hold[k] for k in ["sharpe", "cagr", "max_drawdown", "calmar"]}},
        {"config": f"honestly-selected: lookback={best_lb} threshold={best_th} floor={best_fl}",
         **{k: best_hold[k] for k in ["sharpe", "cagr", "max_drawdown", "calmar"]}},
    ]).set_index("config")
    print(holdout_results.to_string())

    print()
    print("-" * 95)
    print("BAD-YEAR DETAIL IN HOLDOUT (2020, 2022) -- baseline vs. honestly-selected config")
    print("-" * 95)
    ann_base = annual_attribution(baseline_bt, periods_per_year=ppy)
    ann_best = annual_attribution(best_bt, periods_per_year=ppy)
    cmp = pd.DataFrame({
        "baseline_return": ann_base["ann_return"],
        "trigger_return":  ann_best["ann_return"],
        "baseline_maxdd":  ann_base["max_dd"],
        "trigger_maxdd":   ann_best["max_dd"],
    }).loc[[2020, 2022]]
    print(cmp.to_string())

    out = processed_dir / "tail_trigger_holdout.parquet"
    sel_results.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
