"""Compute forward return labels. RUN ON YOUR MACHINE.

    python -m src.labels.build_labels --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.features.features import rebalance_dates
from src.labels.labels import build_label_panel, rank_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg     = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    horizon = cfg["labels"]["horizon_days"]

    prices = pd.read_parquet(processed_dir / "prices_clean.parquet")
    prices.index = pd.to_datetime(prices.index)

    dates      = rebalance_dates(prices.index, freq=horizon)
    raw_labels = build_label_panel(prices, dates, horizon=horizon)

    logger.info(
        "Labels: %d observations  dates: %s .. %s",
        len(raw_labels),
        raw_labels.index.get_level_values("date").min().date(),
        raw_labels.index.get_level_values("date").max().date(),
    )

    # Raw returns — used by the backtest to compute actual portfolio P&L
    raw_path = processed_dir / "labels.parquet"
    raw_labels.to_parquet(raw_path)
    logger.info("Wrote %s  (raw returns)", raw_path)

    # Cross-sectional ranks — used by the model as training target
    ranked_path = processed_dir / "labels_ranked.parquet"
    rank_labels(raw_labels).to_parquet(ranked_path)
    logger.info("Wrote %s  (cross-sectional ranks)", ranked_path)


if __name__ == "__main__":
    main()
