"""Forward return labels for cross-sectional prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """horizon-day log return at each date: log(price[T+horizon] / price[T]).

    Last ``horizon`` rows are NaN (no future prices available). NaN prices
    (non-membership periods) propagate naturally so the membership mask flows
    through to the label layer without extra masking logic.
    """
    return np.log(prices / prices.shift(horizon)).shift(-horizon)


def rank_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Replace fwd_return with its cross-sectional percentile rank on each date.

    Produces a stable [0, 1] target that is scale-invariant across market
    regimes — a volatile month and a quiet month both have the same target
    distribution, so the model learns relative ordering rather than absolute
    return magnitude.
    """
    ranked = labels.copy()
    ranked["fwd_return"] = (
        labels.groupby(level="date")["fwd_return"].rank(pct=True)
    )
    return ranked


def build_label_panel(
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    horizon: int = 5,
) -> pd.DataFrame:
    """Forward returns on rebalance dates as (date, ticker) MultiIndex.

    NaN labels are dropped; observations without a known realized return must
    never enter training or backtest evaluation.
    """
    fwd = forward_returns(prices, horizon=horizon).loc[dates]
    fwd.index.name = "date"
    s = fwd.stack(future_stack=True)
    s.name = "fwd_return"
    out = s.to_frame()
    out.index.names = ["date", "ticker"]
    return out.dropna()
