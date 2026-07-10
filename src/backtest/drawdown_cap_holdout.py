"""Held-out validation of the drawdown-cap trend filter, same methodology as
the README's "Feature selection was in-sample" check. RUN ON YOUR MACHINE.

drawdown_cap.py picked window=250/floor=0.75 by sweeping the full
2013-2026 sample and reading off the best cell -- exactly the kind of
in-sample selection that check warned about. Here: select (window, floor)
using ONLY pre-2020 periods, then evaluate that single, honestly-selected
config on the true 2020-2026 holdout it never saw, against the no-filter
baseline over the same holdout window. If the holdout-selected config
doesn't hold up, that's the honest answer, not a reason to keep sweeping
until something does (see README's insider-trading feature section for why
that would defeat the point).

    python -m src.backtest.drawdown_cap_holdout --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, run_backtest, trend_exposure

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

    grid_window = [100, 150, 200, 250]
    grid_floor  = [0.0, 0.25, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    logger.info("Selecting (window, floor) using ONLY periods before %s ...", HOLDOUT_START)
    rows = []
    bts = {}
    for w in grid_window:
        for f in grid_floor:
            exp = trend_exposure(returns_panel, window=w, floor=f)
            bt  = run_backtest(predictions, labels, costs_bps=costs_bps,
                               top_q=0.20, long_only=True, exposure=exp)
            bts[(w, f)] = bt
            sel_m = _metrics_on_slice(bt, ppy, end=HOLDOUT_START)
            rows.append({"window": w, "floor": f,
                         "sel_sharpe": sel_m["sharpe"], "sel_cagr": sel_m["cagr"],
                         "sel_max_dd": sel_m["max_drawdown"], "sel_calmar": sel_m["calmar"],
                         "sel_n": sel_m["n_periods"]})

    sel_results = pd.DataFrame(rows)
    baseline_sel = _metrics_on_slice(baseline_bt, ppy, end=HOLDOUT_START)

    print()
    print("-" * 90)
    print(f"SELECTION WINDOW (periods before {HOLDOUT_START}, n={baseline_sel['n_periods']})")
    print("-" * 90)
    print(f"Baseline (no filter):  Sharpe {baseline_sel['sharpe']:.3f}  CAGR {100*baseline_sel['cagr']:.2f}%  "
          f"maxDD {100*baseline_sel['max_drawdown']:.2f}%  Calmar {baseline_sel['calmar']:.3f}")
    print(sel_results.sort_values("sel_calmar", ascending=False).head(10).to_string(index=False))

    best_row = sel_results.sort_values("sel_calmar", ascending=False).iloc[0]
    best_w, best_f = int(best_row["window"]), float(best_row["floor"])
    logger.info("Honestly-selected config (best Calmar, pre-2020 only): window=%d floor=%.2f",
                best_w, best_f)

    # Also carry forward the config picked by eyeballing the full-sample
    # sweep in drawdown_cap.py, purely for contrast -- this one WAS
    # selected in-sample and is shown only to see whether it agrees.
    insample_w, insample_f = 250, 0.75

    print()
    print("-" * 90)
    print(f"HOLDOUT ({HOLDOUT_START} to end, n={_metrics_on_slice(baseline_bt, ppy, start=HOLDOUT_START)['n_periods']})")
    print("-" * 90)

    holdout_rows = []
    baseline_hold = _metrics_on_slice(baseline_bt, ppy, start=HOLDOUT_START)
    holdout_rows.append({"config": "baseline (no filter)", "window": None, "floor": None,
                         **{k: baseline_hold[k] for k in ["sharpe", "cagr", "max_drawdown", "calmar"]}})

    for label, w, f in [
        ("honestly-selected (pre-2020 only)", best_w, best_f),
        ("in-sample pick from full sweep",    insample_w, insample_f),
    ]:
        bt = bts[(w, f)]
        m  = _metrics_on_slice(bt, ppy, start=HOLDOUT_START)
        holdout_rows.append({"config": f"{label}: window={w} floor={f}", "window": w, "floor": f,
                             **{k: m[k] for k in ["sharpe", "cagr", "max_drawdown", "calmar"]}})

    holdout_results = pd.DataFrame(holdout_rows).set_index("config")
    print(holdout_results.to_string())

    out = processed_dir / "drawdown_cap_holdout.parquet"
    sel_results.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
