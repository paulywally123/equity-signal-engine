"""Generate today's ranked portfolio signal from the latest model run. RUN ON YOUR MACHINE.

    python -m src.signal.build_live_signal --config config/config.yaml

Outputs:
    data/processed/live_signal.csv   — full ranked universe
    data/processed/portfolio.csv     — long-leg holdings only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.signal.live_signal import latest_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--top-q", type=float, default=0.20,
                    help="Fraction of universe to go long (default 0.20)")
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    raw_dir       = Path(cfg["data"]["raw_dir"])

    predictions = pd.read_parquet(processed_dir / "live_predictions.parquet")
    features    = pd.read_parquet(processed_dir / "features.parquet")

    # Load cached sector map
    sector_cache = raw_dir / "sectors.parquet"
    if sector_cache.exists():
        sectors = pd.read_parquet(sector_cache)["sector"]
    else:
        logger.warning("No sector cache found — run build_backtest first")
        sectors = pd.Series(dtype=str)

    signal = latest_signal(predictions, features, sectors, top_q=args.top_q)
    signal_date = predictions.index.get_level_values("date").max().date()

    logger.info("Signal as of: %s", signal_date)
    logger.info("Universe size: %d  |  Long leg: %d stocks",
                len(signal), signal["long_leg"].sum())

    # --- Full ranked universe ---
    signal_path = processed_dir / "live_signal.csv"
    signal.to_csv(signal_path)
    logger.info("Wrote full ranking → %s", signal_path)

    # --- Long-leg portfolio ---
    portfolio = signal[signal["long_leg"]].drop(columns="long_leg")
    port_path = processed_dir / "portfolio.csv"
    portfolio.to_csv(port_path)

    logger.info("\nCurrent portfolio (%d holdings, equal-weight):", len(portfolio))
    logger.info("-" * 60)
    for ticker, row in portfolio.iterrows():
        logger.info("  %4d  %-7s  %-35s  score=%.3f",
                    int(row["rank"]), ticker, row["sector"], row["score"])
    logger.info("-" * 60)
    logger.info("Wrote portfolio → %s", port_path)


if __name__ == "__main__":
    main()
