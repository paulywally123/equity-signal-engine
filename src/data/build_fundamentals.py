"""Fetch SEC EDGAR fundamentals and build the point-in-time feature panel. RUN ON YOUR MACHINE.

    python -m src.data.build_fundamentals --config config/config.yaml

Uses the public EDGAR XBRL API — no API key required.
Rate-limited to 8 requests/second per SEC fair-use guidelines.
Estimated runtime: ~10 minutes for 600+ tickers.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.edgar import build_fundamental_panel, fetch_all, fetch_cik_map

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--force-refresh", action="store_true",
                    help="Re-download even if ticker cache exists")
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    raw_dir       = Path(cfg["data"]["raw_dir"])
    edgar_cache   = raw_dir / "edgar"

    prices = pd.read_parquet(processed_dir / "prices_clean.parquet")
    prices.index = pd.to_datetime(prices.index)
    tickers = list(prices.columns)
    logger.info("Universe: %d tickers", len(tickers))

    # 1. CIK mapping (ticker → SEC CIK number)
    cik_path = raw_dir / "sec_cik_map.parquet"
    logger.info("Loading SEC ticker→CIK mapping ...")
    cik_map  = fetch_cik_map(cik_path)
    matched  = sum(1 for t in tickers if t in cik_map)
    logger.info("CIK matches: %d / %d", matched, len(tickers))

    # 2. Fetch company facts from EDGAR (rate-limited)
    logger.info("Fetching EDGAR company facts (~8 req/s) ...")
    ok, fail = fetch_all(tickers, cik_map, edgar_cache,
                         force_refresh=args.force_refresh)
    logger.info("Fetch complete: %d ok  %d failed/skipped", ok, fail)

    # 3. Shares outstanding from yfinance info (current value, used only for ep_ratio)
    logger.info("Loading shares outstanding from yfinance ...")
    shares_map: dict[str, float] = {}
    try:
        import yfinance as yf
        for i, ticker in enumerate(tickers, 1):
            try:
                info = yf.Ticker(ticker).fast_info
                sh = getattr(info, "shares", None)
                if sh and sh > 0:
                    shares_map[ticker] = float(sh)
            except Exception:
                pass
            if i % 100 == 0:
                logger.info("  shares: %d / %d", i, len(tickers))
        logger.info("Got shares for %d tickers", len(shares_map))
    except Exception as exc:
        logger.warning("Could not load shares outstanding: %s", exc)

    # 4. Build point-in-time fundamental panel
    from src.features.features import rebalance_dates
    horizon = cfg["labels"]["horizon_days"]
    dates   = rebalance_dates(prices.index, freq=horizon)
    logger.info("Building point-in-time panel (%d rebalance dates) ...", len(dates))

    panel = build_fundamental_panel(tickers, edgar_cache, prices, dates, shares_map)

    if panel.empty:
        logger.error("Fundamental panel is empty — check EDGAR cache at %s", edgar_cache)
        return

    n_tickers = panel.index.get_level_values("ticker").nunique()
    n_dates   = panel.index.get_level_values("date").nunique()
    logger.info(
        "Panel: %d obs  %d tickers  %d dates  columns=%s",
        len(panel), n_tickers, n_dates, list(panel.columns),
    )

    # Coverage by year for diagnostics
    years = pd.Series(panel.index.get_level_values("date")).dt.year
    logger.info("Coverage by year:\n%s", years.value_counts().sort_index().to_string())

    out = processed_dir / "fundamentals_panel.parquet"
    panel.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
