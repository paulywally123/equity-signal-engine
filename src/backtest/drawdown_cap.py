"""Tune the trend-filter drawdown cap: how much CAGR to give up for how much
drawdown protection. RUN ON YOUR MACHINE.

Chosen direction after vol_scaling.py found no free lunch (every timing
overlay tested so far reduces Sharpe -- see README "Decomposition" and
vol_scaling.py's sweep): explicitly accept lower average return in exchange
for a capped worst-case drawdown, rather than continuing to look for a
design that avoids the cost. `trend_exposure()` generalises the original
binary 200d SMA filter (fully out below trend) into a tunable partial
version (`floor` = exposure kept below trend, e.g. 0.5 = half-sized rather
than flat), so the tradeoff is an explicit dial instead of one all-or-
nothing point.

    python -m src.backtest.drawdown_cap --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, run_backtest, trend_exposure
from src.report.attribution import annual_attribution

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BAD_YEARS = [2015, 2018, 2020, 2022]


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
    baseline_m  = compute_metrics(baseline_bt, periods_per_year=ppy)

    grid_window = [100, 150, 200, 250]
    grid_floor  = [0.0, 0.25, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    rows = []
    bts = {}

    def _calmar(cagr, max_dd):
        return round(cagr / abs(max_dd), 3) if max_dd != 0 else float("nan")

    rows.append({
        "label": "baseline (no filter)", "window": None, "floor": None,
        "sharpe": baseline_m["sharpe"], "cagr": baseline_m["cagr"],
        "max_dd": baseline_m["max_drawdown"],
        "calmar": _calmar(baseline_m["cagr"], baseline_m["max_drawdown"]),
    })

    for w in grid_window:
        for f in grid_floor:
            exp = trend_exposure(returns_panel, window=w, floor=f)
            bt  = run_backtest(predictions, labels, costs_bps=costs_bps,
                               top_q=0.20, long_only=True, exposure=exp)
            m   = compute_metrics(bt, periods_per_year=ppy)
            key = f"window={w} floor={f}"
            bts[(w, f)] = bt
            rows.append({
                "label": key, "window": w, "floor": f,
                "sharpe": m["sharpe"], "cagr": m["cagr"],
                "max_dd": m["max_drawdown"],
                "calmar": _calmar(m["cagr"], m["max_drawdown"]),
            })

    results = pd.DataFrame(rows).set_index("label")
    sep = "-" * 90
    print()
    print(sep)
    print("DRAWDOWN-CAP SWEEP (window x floor)")
    print(sep)
    print(results.to_string())
    print()
    print("calmar = CAGR / |max_drawdown| -- higher is more return per unit of worst-case loss.")
    print("floor=0.0 is the original binary 200d-style filter; floor=1.0 would be the baseline.")

    # Detail on two illustrative configs: the original binary filter (w=200,
    # floor=0.0) and a middle-ground partial version (w=200, floor=0.5), so
    # the actual dollar tradeoff at each end of the dial is visible.
    for w, f in [(200, 0.0), (200, 0.5), (250, 0.75), (250, 0.85)]:
        bt = bts[(w, f)]
        m  = compute_metrics(bt, periods_per_year=ppy)
        print()
        print(sep)
        print(f"BAD-YEAR DETAIL: window={w} floor={f}  (vs. no-filter baseline)")
        print(sep)
        ann_base = annual_attribution(baseline_bt, periods_per_year=ppy)
        ann_this = annual_attribution(bt, periods_per_year=ppy)
        cmp = pd.DataFrame({
            "baseline_return": ann_base["ann_return"],
            "filtered_return": ann_this["ann_return"],
            "baseline_maxdd": ann_base["max_dd"],
            "filtered_maxdd": ann_this["max_dd"],
        }).loc[BAD_YEARS]
        print(cmp.to_string())
        print(f"Full period: CAGR {100*baseline_m['cagr']:.2f}% -> {100*m['cagr']:.2f}%   "
              f"maxDD {100*baseline_m['max_drawdown']:.2f}% -> {100*m['max_drawdown']:.2f}%")

    out = processed_dir / "drawdown_cap.parquet"
    results.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
