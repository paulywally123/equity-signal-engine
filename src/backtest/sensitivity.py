"""Sensitivity analysis: vary top_q, cost assumptions, and regime window. RUN ON YOUR MACHINE.

    python -m src.backtest.sensitivity --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, equal_weight_index, run_backtest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _regime(returns: pd.DataFrame, window: int) -> pd.Series:
    idx = equal_weight_index(returns)
    r = idx > idx.rolling(window).mean()
    r.index = pd.to_datetime(r.index)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    horizon       = cfg["labels"]["horizon_days"]
    ppy           = int(round(252 / horizon))

    predictions   = pd.read_parquet(processed_dir / "predictions.parquet")
    labels        = pd.read_parquet(processed_dir / "labels.parquet")
    returns_panel = pd.read_parquet(processed_dir / "returns.parquet")
    returns_panel.index = pd.to_datetime(returns_panel.index)

    grids = {
        "top_q":         [0.10, 0.15, 0.20, 0.25, 0.30],
        "costs_bps":     [5, 10, 20],
        "regime_window": [100, 150, 200, 250, None],
    }

    # No regime filter by default -- see build_backtest.py for why it was
    # dropped. Kept here as an explicit, opt-in variation to explore.
    base = dict(top_q=0.20, costs_bps=cfg["costs"]["per_side_bps"], regime_window=None)

    rows = []

    def _run(top_q, costs_bps, regime_window, label):
        regime = _regime(returns_panel, regime_window) if regime_window else None
        bt = run_backtest(predictions, labels,
                          costs_bps=costs_bps, top_q=top_q,
                          long_only=True, regime=regime)
        m = compute_metrics(bt, periods_per_year=ppy)
        rows.append({
            "label":       label,
            "top_q":       top_q,
            "costs_bps":   costs_bps,
            "regime_win":  str(regime_window) if regime_window else "none",
            "sharpe":      m["sharpe"],
            "ann_return":  m["ann_return"],
            "ann_vol":     m["ann_vol"],
            "max_dd":      m["max_drawdown"],
            "cagr":        m["cagr"],
        })

    # Baseline
    _run(**base, label="baseline")

    # Vary top_q
    logger.info("Varying top_q ...")
    for v in grids["top_q"]:
        if v != base["top_q"]:
            _run(top_q=v, costs_bps=base["costs_bps"],
                 regime_window=base["regime_window"], label=f"top_q={v}")

    # Vary costs
    logger.info("Varying costs_bps ...")
    for v in grids["costs_bps"]:
        if v != base["costs_bps"]:
            _run(top_q=base["top_q"], costs_bps=v,
                 regime_window=base["regime_window"], label=f"costs={v}bps")

    # Vary regime window
    logger.info("Varying regime window ...")
    for v in grids["regime_window"]:
        if v != base["regime_window"]:
            _run(top_q=base["top_q"], costs_bps=base["costs_bps"],
                 regime_window=v, label=f"regime={v}")

    results = pd.DataFrame(rows).set_index("label")
    sep = "-" * 80
    print()
    print(sep)
    print("SENSITIVITY ANALYSIS")
    print(sep)
    print(results.to_string())
    print()

    # Highlight range across each dimension
    for dim, col in [("top_q", "top_q"), ("costs_bps", "costs_bps"), ("regime_win", "regime_win")]:
        sub = results[results.index.str.startswith(dim.split("_")[0]) | (results.index == "baseline")]
        sharpes = sub["sharpe"]
        print(f"{dim}: Sharpe range {sharpes.min():.3f} – {sharpes.max():.3f}  "
              f"(baseline {results.loc['baseline','sharpe']:.3f})")

    out = processed_dir / "sensitivity.parquet"
    results.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
