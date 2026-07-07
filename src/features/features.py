"""Cross-sectional price signals for the monthly equity strategy.

Features (all rank-normalized to [0,1] cross-sectionally on each date)
-----------------------------------------------------------------------
mom_12_1      : 12-month return skipping last month  (252d → 21d lookback)
mom_6_1       : 6-month return skipping last month   (126d → 21d lookback)
mom_3         : 3-month return                       (63d  lookback)
mom_1         : 1-month return (short-term reversal / residual signal)
high_52w      : price / 252-day rolling high (52-week-high proximity anomaly)
vol_21        : 21-day realized volatility, annualized (std × √252)
rsi_14        : 14-day RSI (simple rolling-mean variant)
dollar_vol_60 : 60-day avg of Close × Volume — size/liquidity proxy and
                the criterion for dev-mode top-N universe selection
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, freq: int = 5) -> pd.DatetimeIndex:
    """Every freq-th date in index — the weekly rebalance schedule."""
    return index[::freq]


# ---------------------------------------------------------------------------
# Raw feature helpers
# ---------------------------------------------------------------------------

def _mom(prices: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    return prices.shift(skip) / prices.shift(lookback) - 1


def _vol(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(252)


def _rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)   # NaN where no losses
    rsi = 100 - 100 / (1 + rs)
    # loss=0, gain>0 → pure uptrend → RSI=100
    return rsi.where(~((loss == 0) & (gain > 0)), 100.0)


def _high_52w(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Current price as a fraction of the 252-day rolling high."""
    return prices / prices.rolling(window).max()


def _dollar_vol(prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    return (prices * volumes).rolling(window).mean()


# ---------------------------------------------------------------------------
# Cross-sectional normalization
# ---------------------------------------------------------------------------

def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank on each row; NaN tickers excluded."""
    return df.rank(axis=1, pct=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def compute_all(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volumes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute all raw features on every date in the panel."""
    return {
        "mom_12_1":      _mom(prices, lookback=252, skip=21),
        "mom_6_1":       _mom(prices, lookback=126, skip=21),
        "mom_3":         _mom(prices, lookback=63,  skip=0),
        "mom_1":         _mom(prices, lookback=21,  skip=0),
        "high_52w":      _high_52w(prices, window=252),
        "vol_21":        _vol(returns, window=21),
        "rsi_14":        _rsi(prices, window=14),
        "dollar_vol_60": _dollar_vol(prices, volumes, window=60),
    }


def build_feature_panel(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volumes: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Rank-normalized features on rebalance dates as (date, ticker) MultiIndex.

    Rows with all-NaN features (out-of-membership or insufficient history) are
    dropped. Partial NaN rows are kept — LightGBM handles missing values natively.
    """
    raw = compute_all(prices, returns, volumes)
    normalized = {name: rank_normalize(df.loc[dates]) for name, df in raw.items()}

    series: dict[str, pd.Series] = {}
    for name, df in normalized.items():
        df.index.name = "date"
        s = df.stack(future_stack=True)
        s.name = name
        series[name] = s

    out = pd.concat(series, axis=1)
    out.index.names = ["date", "ticker"]
    return out.dropna(how="all")
