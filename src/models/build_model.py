"""Walk-forward model training and scoring. RUN ON YOUR MACHINE.

    python -m src.models.build_model --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.models.model import walk_forward_predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])

    features = pd.read_parquet(processed_dir / "features.parquet")
    labels   = pd.read_parquet(processed_dir / "labels_ranked.parquet")

    mode      = cfg["universe"]["mode"]
    dev_top_n = cfg["universe"]["dev_top_n"] if mode == "dev" else None
    if dev_top_n:
        logger.info("Dev mode: top-%d tickers by dollar volume", dev_top_n)

    start             = pd.to_datetime(cfg["dates"]["start"])
    initial_train_end = str((start + pd.DateOffset(years=3)).date())
    logger.info("Initial training window ends: %s", initial_train_end)

    predictions = walk_forward_predict(
        features, labels,
        initial_train_end=initial_train_end,
        dev_top_n=dev_top_n,
    )

    out = processed_dir / "predictions.parquet"
    predictions.to_parquet(out)
    logger.info(
        "Wrote %s  (%d predictions, %d dates)",
        out,
        len(predictions),
        predictions.index.get_level_values("date").nunique(),
    )


if __name__ == "__main__":
    main()
