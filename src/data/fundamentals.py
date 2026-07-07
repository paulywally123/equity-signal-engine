"""Quarterly fundamental data fetch, caching, and point-in-time panel construction.

Each ticker is cached at data/raw/fundamentals/{TICKER}.parquet with columns:
    period_end (index), revenue, gross_profit, net_income,
    total_assets, equity, shares_outstanding (constant scalar column)

The point-in-time panel applies a filing lag of 60 calendar days before
making any quarterly observation available on a rebalance date, preventing
look-ahead bias from earnings that had not yet been publicly reported.

Features produced (all TTM — trailing four quarters summed):
    gross_prof   = gross_profit_ttm / total_assets       (Novy-Marx 2013)
    roe          = net_income_ttm   / abs(equity)        (quality)
    ep_ratio     = net_income_ttm   / (price × shares)   (earnings yield)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

FILING_LAG_DAYS = 60

# Each tuple: (yfinance row label, our internal name).
# Multiple source names for the same target = fallback aliases tried in order.
_INCOME_FIELDS: list[tuple[str, str]] = [
    ("Total Revenue",   "revenue"),
    ("Gross Profit",    "gross_profit"),
    ("Net Income",      "net_income"),
    ("Net Income Common Stockholders", "net_income"),  # alternate label
]
_BALANCE_FIELDS: list[tuple[str, str]] = [
    ("Total Assets",             "total_assets"),
    ("Stockholders Equity",      "equity"),
    ("Total Stockholder Equity", "equity"),            # alternate label
]


# ---------------------------------------------------------------------------
# Fetch and cache
# ---------------------------------------------------------------------------

def _fetch_one(ticker: str, cache_dir: Path, force_refresh: bool) -> bool:
    out = cache_dir / f"{ticker}.parquet"
    if out.exists() and not force_refresh:
        return True
    try:
        t = yf.Ticker(ticker)

        inc = t.quarterly_financials
        bal = t.quarterly_balance_sheet
        if inc is None or inc.empty or bal is None or bal.empty:
            return False

        inc = inc.T
        inc.index = pd.to_datetime(inc.index)
        bal = bal.T
        bal.index = pd.to_datetime(bal.index)

        # For each target field, pick the first available source alias
        seen_targets: set[str] = set()
        frames: list[pd.DataFrame] = []
        for df, field_list in [(inc, _INCOME_FIELDS), (bal, _BALANCE_FIELDS)]:
            for src, dst in field_list:
                if dst in seen_targets:
                    continue     # already captured from an earlier alias
                if src in df.columns:
                    frames.append(df[[src]].rename(columns={src: dst}))
                    seen_targets.add(dst)

        if not frames:
            return False

        merged = pd.concat(frames, axis=1)
        merged = merged.loc[~merged.index.duplicated(keep="last")].sort_index()
        merged.index.name = "period_end"

        # Attach current shares outstanding as a constant column
        try:
            info = t.info
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        except Exception:
            shares = None
        merged["shares"] = float(shares) if shares else np.nan

        merged.to_parquet(out)
        return True
    except Exception as exc:
        logger.debug("Failed %s: %s", ticker, exc)
        return False


def fetch_all(
    tickers: list[str],
    cache_dir: Path,
    force_refresh: bool = False,
    workers: int = 8,
) -> tuple[int, int]:
    """Fetch quarterly fundamentals for all tickers, return (ok, fail) counts."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, t, cache_dir, force_refresh): t
            for t in tickers
        }
        for i, fut in enumerate(as_completed(futures), 1):
            success = False
            try:
                success = fut.result()
            except Exception:
                pass
            if success:
                ok += 1
            else:
                fail += 1
            if i % 100 == 0 or i == len(tickers):
                logger.info("  %d / %d  (ok=%d  fail=%d)", i, len(tickers), ok, fail)

    return ok, fail


# ---------------------------------------------------------------------------
# Load and point-in-time panel
# ---------------------------------------------------------------------------

def _load_one(ticker: str, cache_dir: Path) -> pd.DataFrame | None:
    p = cache_dir / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _ttm(series: pd.Series, n: int = 4) -> pd.Series:
    return series.rolling(n, min_periods=n).sum()


def build_fundamental_panel(
    tickers: list[str],
    cache_dir: Path,
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    filing_lag_days: int = FILING_LAG_DAYS,
) -> pd.DataFrame:
    """Point-in-time fundamental features indexed by (date, ticker).

    For each rebalance date, uses the most recent quarterly report whose
    period_end is at least `filing_lag_days` before that date.
    Requires at least 4 quarters of history to compute TTM values.
    """
    lag = pd.Timedelta(days=filing_lag_days)
    records: list[dict] = []

    for ticker in tickers:
        raw = _load_one(ticker, cache_dir)
        if raw is None or raw.empty:
            continue

        gp_series = raw["gross_profit"] if "gross_profit" in raw.columns else None
        ni_series = raw["net_income"]   if "net_income"   in raw.columns else None
        eq_series = raw["equity"]       if "equity"       in raw.columns else None
        at_series = raw["total_assets"] if "total_assets" in raw.columns else None
        shares     = float(raw["shares"].iloc[-1]) if "shares" in raw.columns else np.nan

        ttm_gp = _ttm(gp_series) if gp_series is not None else None
        ttm_ni = _ttm(ni_series) if ni_series is not None else None

        price_col = prices.get(ticker)

        for date in rebalance_dates:
            cutoff    = date - lag
            available = raw.index[raw.index <= cutoff]
            if len(available) < 4:
                continue
            q = available[-1]

            row: dict = {"date": date, "ticker": ticker}

            gp  = ttm_gp.loc[q]  if ttm_gp is not None  else np.nan
            ni  = ttm_ni.loc[q]  if ttm_ni is not None  else np.nan
            eq  = eq_series.loc[q] if eq_series is not None else np.nan
            ast = at_series.loc[q] if at_series is not None else np.nan

            if pd.notna(gp) and pd.notna(ast) and ast > 0:
                row["gross_prof"] = float(gp / ast)

            if pd.notna(ni) and pd.notna(eq) and abs(eq) > 0:
                row["roe"] = float(ni / abs(eq))

            if pd.notna(ni) and not np.isnan(shares) and shares > 0 and price_col is not None:
                px = price_col.get(date)
                if pd.notna(px) and px > 0:
                    mkt_cap = px * shares
                    row["ep_ratio"] = float(ni / mkt_cap)

            if len(row) > 2:   # at least one feature beyond date/ticker
                records.append(row)

    if not records:
        logger.warning("No fundamental records built — check cache at %s", cache_dir)
        return pd.DataFrame(
            columns=["gross_prof", "roe", "ep_ratio"],
            index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]),
        )

    out = (
        pd.DataFrame(records)
        .set_index(["date", "ticker"])
        .sort_index()
    )
    out.index.names = ["date", "ticker"]
    return out
