"""Test whether vol-target exposure scaling beats no timing overlay at all. RUN ON YOUR MACHINE.

Same question the 200d-SMA regime filter was built to answer (see
build_backtest.py / README "Decomposition" and "Sensitivity summary" for why
that binary in/out filter was dropped -- it reduced Sharpe in every
configuration tested). This tries a continuous alternative instead of a
binary one: shrink position size as trailing realized vol rises, rather than
fully exiting, so a sharp post-selloff recovery (e.g. 2020-04, 2022-10 --
both among the best individual periods in the whole backtest) isn't missed
entirely.

    python -m src.backtest.vol_scaling --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, run_backtest, vol_target_exposure
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

    # Baseline (current default): model, top_q=0.2, no timing overlay at all.
    baseline_bt = run_backtest(predictions, labels, costs_bps=costs_bps,
                                top_q=0.20, long_only=True)
    baseline_m  = compute_metrics(baseline_bt, periods_per_year=ppy)

    # Full-universe equal-weight benchmark, no overlay -- same reference
    # point the README decomposition table uses.
    bmark_bt = run_backtest(predictions, labels, costs_bps=costs_bps,
                             top_q=1.00, long_only=True)
    bmark_m  = compute_metrics(bmark_bt, periods_per_year=ppy)

    logger.info("Baseline (model, no overlay):     Sharpe %.3f  CAGR %.2f%%  vol %.2f%%  maxDD %.2f%%",
                baseline_m["sharpe"], 100*baseline_m["cagr"], 100*baseline_m["ann_vol"], 100*baseline_m["max_drawdown"])
    logger.info("Benchmark (equal-weight, no overlay): Sharpe %.3f  CAGR %.2f%%  vol %.2f%%  maxDD %.2f%%",
                bmark_m["sharpe"], 100*bmark_m["cagr"], 100*bmark_m["ann_vol"], 100*bmark_m["max_drawdown"])

    # Grid over target_vol / window rather than reporting one cherry-picked
    # configuration -- same spirit as sensitivity.py's regime-window sweep,
    # which is what caught the SMA filter's monotonic degradation.
    grid_target_vol = [0.12, 0.165, 0.20]   # 0.165 = current baseline's realized ann. vol
    grid_window     = [10, 20, 40]

    rows = []

    def _run(label, top_q, target_vol, window):
        exp = vol_target_exposure(returns_panel, target_vol=target_vol, window=window)
        bt  = run_backtest(predictions, labels, costs_bps=costs_bps,
                           top_q=top_q, long_only=True, exposure=exp)
        m   = compute_metrics(bt, periods_per_year=ppy)
        rows.append({
            "label":        label,
            "target_vol":   target_vol,
            "window":       window,
            "sharpe":       m["sharpe"],
            "ann_return":   m["ann_return"],
            "ann_vol":      m["ann_vol"],
            "max_dd":       m["max_drawdown"],
            "cagr":         m["cagr"],
            "avg_exposure": round(bt["exposure"].mean(), 3),
        })
        return bt

    logger.info("Sweeping vol-scaled MODEL portfolio (top_q=0.20) ...")
    model_vol_bts = {}
    for tv in grid_target_vol:
        for w in grid_window:
            bt = _run(f"model target_vol={tv} window={w}", top_q=0.20, target_vol=tv, window=w)
            model_vol_bts[(tv, w)] = bt

    logger.info("Sweeping vol-scaled BENCHMARK (top_q=1.00, equal-weight) ...")
    for tv in grid_target_vol:
        for w in grid_window:
            _run(f"benchmark target_vol={tv} window={w}", top_q=1.00, target_vol=tv, window=w)

    results = pd.DataFrame(rows).set_index("label")
    sep = "-" * 90
    print()
    print(sep)
    print("VOL-SCALING SWEEP")
    print(sep)
    print(results.to_string())
    print()
    print(f"Baseline (model, no overlay):          Sharpe {baseline_m['sharpe']:.3f}  CAGR {100*baseline_m['cagr']:.2f}%  maxDD {100*baseline_m['max_drawdown']:.2f}%")
    print(f"Benchmark (equal-weight, no overlay):   Sharpe {bmark_m['sharpe']:.3f}  CAGR {100*bmark_m['cagr']:.2f}%  maxDD {100*bmark_m['max_drawdown']:.2f}%")

    # Annual attribution for the single best-Sharpe vol-scaled model config,
    # side by side with baseline, focused on the bad years the model has no
    # protection in today (see README annual attribution: 2015, 2018, 2020, 2022).
    best_key = results[results.index.str.startswith("model")]["sharpe"].idxmax()
    best_tv, best_w = [
        (tv, w) for tv in grid_target_vol for w in grid_window
        if f"model target_vol={tv} window={w}" == best_key
    ][0]
    best_bt = model_vol_bts[(best_tv, best_w)]

    print()
    print(sep)
    print(f"BAD-YEAR COMPARISON: baseline vs. best vol-scaled config (target_vol={best_tv}, window={best_w})")
    print(sep)
    ann_base = annual_attribution(baseline_bt, periods_per_year=ppy)
    ann_best = annual_attribution(best_bt, periods_per_year=ppy)
    cmp = pd.DataFrame({
        "baseline_return": ann_base["ann_return"],
        "vol_scaled_return": ann_best["ann_return"],
        "baseline_maxdd": ann_base["max_dd"],
        "vol_scaled_maxdd": ann_best["max_dd"],
    }).loc[BAD_YEARS]
    print(cmp.to_string())

    out = processed_dir / "vol_scaling.parquet"
    results.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
