"""Compute and cache the feature panel. RUN ON YOUR MACHINE.

    python -m src.features.build_features --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.prices import load_panel
from src.features.features import build_feature_panel, rank_normalize, rebalance_dates

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    price_cache   = Path(cfg["data"]["raw_dir"]) / "prices"

    prices  = pd.read_parquet(processed_dir / "prices_clean.parquet")
    returns = pd.read_parquet(processed_dir / "returns.parquet")
    prices.index  = pd.to_datetime(prices.index)
    returns.index = pd.to_datetime(returns.index)

    logger.info("Loading volume panel ...")
    tickers = list(prices.columns)
    volumes = load_panel(tickers, cache_dir=price_cache, price_col="Volume")
    volumes = volumes.reindex(index=prices.index, columns=prices.columns)

    freq  = cfg["labels"]["horizon_days"]
    dates = rebalance_dates(prices.index, freq=freq)
    logger.info("Rebalance dates: %d  (%s .. %s)",
                len(dates), dates[0].date(), dates[-1].date())

    logger.info("Computing price features ...")
    features = build_feature_panel(prices, returns, volumes, dates)

    # Join fundamental features if the panel has been built
    fund_path = processed_dir / "fundamentals_panel.parquet"
    if fund_path.exists():
        logger.info("Joining fundamental features ...")
        fund = pd.read_parquet(fund_path)
        # Ensure date level is datetime
        fund = fund.reset_index()
        fund["date"] = pd.to_datetime(fund["date"])
        fund = fund.set_index(["date", "ticker"])
        # Rank-normalize each fundamental feature cross-sectionally on each date
        # then join into the main panel
        fund_cols = list(fund.columns)
        fund_wide = fund.unstack(level="ticker")   # (date, feature-ticker wide)
        fund_ranked_frames = []
        for col in fund_cols:
            if col in fund_wide.columns.get_level_values(0):
                wide = fund_wide[col]              # date × ticker
                ranked = rank_normalize(wide)      # cross-sectional rank
                ranked.index.name = "date"
                s = ranked.stack(future_stack=True)
                s.name = col
                fund_ranked_frames.append(s)
        if fund_ranked_frames:
            fund_ranked = pd.concat(fund_ranked_frames, axis=1)
            fund_ranked.index.names = ["date", "ticker"]
            features = features.join(fund_ranked, how="left")
            logger.info("Added fundamental columns: %s", fund_cols)
    else:
        logger.info("No fundamentals_panel.parquet found — skipping fundamental features")

    logger.info(
        "Feature panel: %d observations  %d tickers  %d features",
        len(features),
        features.index.get_level_values("ticker").nunique(),
        features.shape[1],
    )

    out = processed_dir / "features.parquet"
    features.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
