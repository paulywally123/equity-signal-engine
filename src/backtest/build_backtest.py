"""Run backtest and generate performance report. RUN ON YOUR MACHINE.

    python -m src.backtest.build_backtest --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, feature_ic, information_coefficient, run_backtest
from src.report.attribution import print_attribution
from src.report.report import plot_report


def _load_sectors(tickers: list[str], cache_path: Path) -> pd.Series:
    """Return ticker → GICS sector mapping, fetching from yfinance if not cached."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)["sector"]

    import yfinance as yf
    records = []
    for i, t in enumerate(tickers, 1):
        try:
            sector = yf.Ticker(t).info.get("sector", "Unknown") or "Unknown"
        except Exception:
            sector = "Unknown"
        records.append({"ticker": t, "sector": sector})
        if i % 100 == 0:
            logger.info("  Sector fetch: %d / %d", i, len(tickers))

    df = pd.DataFrame(records).set_index("ticker")
    df.to_parquet(cache_path)
    return df["sector"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    costs_bps     = cfg["costs"]["per_side_bps"]

    predictions = pd.read_parquet(processed_dir / "predictions.parquet")
    labels      = pd.read_parquet(processed_dir / "labels.parquet")
    features    = pd.read_parquet(processed_dir / "features.parquet")
    prices      = pd.read_parquet(processed_dir / "prices_clean.parquet")

    # Sector map: ticker → GICS sector (cached so reruns are instant)
    sector_cache = Path(cfg["data"]["raw_dir"]) / "sectors.parquet"
    sectors = _load_sectors(list(prices.columns), sector_cache)
    n_sectors = sectors.nunique()
    logger.info("Sector map: %d tickers  %d sectors", len(sectors), n_sectors)
    for sec, cnt in sectors.value_counts().items():
        logger.info("  %-35s %d", sec, cnt)

    # Market regime: equal-weight universe price index vs 200-day SMA.
    # Using a strict prior-bar look-up in run_backtest prevents any look-ahead.
    mkt_index   = prices.mean(axis=1)
    regime      = mkt_index > mkt_index.rolling(200).mean()
    regime.index = pd.to_datetime(regime.index)
    invested_pct = regime.mean()
    logger.info("Market regime: in-market %.0f%% of days (200d SMA filter)", 100 * invested_pct)

    horizon = cfg["labels"]["horizon_days"]
    periods_per_year = int(round(252 / horizon))
    logger.info("Running backtest (costs: %d bps, long-only, %d-day hold, 200d regime) ...",
                costs_bps, horizon)
    # sectors kwarg available for sector-neutral construction; omitted here because
    # the model's quality signals (roe, gross_prof) carry genuine sector-timing
    # information that neutralisation would remove at cost of ~0.54 Sharpe
    # (re-verified after trimming to 5 features; was ~0.27 with the original 8).
    port_returns = run_backtest(predictions, labels, costs_bps=costs_bps,
                                long_only=True, regime=regime)

    pct_invested = port_returns["in_regime"].mean()
    metrics = compute_metrics(port_returns, periods_per_year=periods_per_year)
    logger.info("Performance summary:")
    for k, v in metrics.items():
        logger.info("  %-16s %s", k, v)
    logger.info("  pct_in_market    %.1f%%", 100 * pct_invested)

    ic = information_coefficient(predictions, labels)
    logger.info("IC  mean=%.4f  std=%.4f  pct>0=%.1f%%  t-stat=%.2f",
                ic.mean(), ic.std(), 100 * (ic > 0).mean(),
                ic.mean() / (ic.std() / len(ic) ** 0.5))

    logger.info("Per-feature IC (mean Spearman vs realized returns):")
    fic = feature_ic(features, labels)
    for col in fic.columns:
        s = fic[col].dropna()
        t = s.mean() / (s.std() / len(s) ** 0.5) if len(s) > 1 else float("nan")
        logger.info("  %-16s mean=%+.4f  pct>0=%.0f%%  t=%.2f", col, s.mean(), 100*(s>0).mean(), t)

    # Equal-weighted benchmark: average 20-day return across the whole universe
    raw_labels = pd.read_parquet(processed_dir / "labels.parquet")
    bmark = raw_labels.groupby(level="date")["fwd_return"].mean()
    bmark_aligned = bmark.reindex(port_returns.index)
    bmark_ann = bmark_aligned.mean() * periods_per_year
    excess_ann = metrics["ann_return"] - bmark_ann
    logger.info("Equal-weight benchmark ann return: %.4f", bmark_ann)
    logger.info("Ann excess return vs benchmark:    %.4f", excess_ann)

    print_attribution(port_returns, predictions, sectors,
                      periods_per_year=periods_per_year)

    bt_path = processed_dir / "backtest.parquet"
    port_returns.to_parquet(bt_path)
    logger.info("Wrote %s", bt_path)

    report_path = processed_dir / "report.png"
    plot_report(port_returns, ic, metrics, report_path)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
