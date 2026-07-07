"""Tests for src/features/features.py — all synthetic, no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.features import (
    _dollar_vol,
    _high_52w,
    _mom,
    _rsi,
    _vol,
    build_feature_panel,
    rank_normalize,
    rebalance_dates,
)

DATES = pd.bdate_range("2020-01-02", periods=300)


def _prices(n=300, val=100.0, seed=0):
    rng = np.random.default_rng(seed)
    data = val + rng.standard_normal((n, 3)).cumsum(axis=0)
    return pd.DataFrame(data, index=DATES[:n], columns=["A", "B", "C"])


def _returns(prices):
    return np.log(prices / prices.shift(1))


def _volumes(n=300):
    return pd.DataFrame(1_000_000, index=DATES[:n], columns=["A", "B", "C"], dtype=float)


# ---------------------------------------------------------------------------
# rebalance_dates
# ---------------------------------------------------------------------------

def test_rebalance_dates_picks_every_nth_day():
    idx = pd.bdate_range("2020-01-02", periods=20)
    rb  = rebalance_dates(idx, freq=5)
    assert list(rb) == list(idx[::5])
    assert len(rb) == 4


# ---------------------------------------------------------------------------
# _mom
# ---------------------------------------------------------------------------

def test_mom_correct_lookback():
    prices = pd.DataFrame({"A": [100.0] * 252 + [110.0]},
                          index=pd.bdate_range("2020-01-02", periods=253))
    result = _mom(prices, lookback=252, skip=0)
    # At the last row: price[252]/price[0] - 1 = 110/100 - 1 = 0.10
    assert result["A"].iloc[-1] == pytest.approx(0.10, rel=1e-6)


# ---------------------------------------------------------------------------
# _high_52w
# ---------------------------------------------------------------------------

def test_high_52w_bounded_0_to_1():
    prices = _prices()
    h52 = _high_52w(prices, window=252)
    valid = h52.stack(future_stack=True).dropna()
    assert (valid > 0).all() and (valid <= 1.0).all()


def test_high_52w_at_new_high_equals_1():
    """When price is at a new 252-day high the ratio must equal 1."""
    vals = list(range(1, 260))   # monotonically increasing
    prices = pd.DataFrame({"A": [float(v) for v in vals]},
                          index=pd.bdate_range("2020-01-02", periods=259))
    h52 = _high_52w(prices, window=252)
    # After 252 bars the price is always the running max → ratio = 1
    assert h52["A"].iloc[252:].dropna().eq(1.0).all()


# ---------------------------------------------------------------------------
# _vol
# ---------------------------------------------------------------------------

def test_vol_annualized_by_sqrt252():
    prices  = _prices()
    returns = _returns(prices)
    vol     = _vol(returns, window=21)
    valid   = vol.dropna()
    # Manually compute for one cell
    end = valid.index[0]
    manual = returns["A"].loc[:end].iloc[-21:].std() * np.sqrt(252)
    assert valid["A"].iloc[0] == pytest.approx(manual, rel=1e-6)


# ---------------------------------------------------------------------------
# _rsi
# ---------------------------------------------------------------------------

def test_rsi_bounded_0_to_100():
    prices = _prices()
    rsi    = _rsi(prices, window=14)
    valid  = rsi.stack(future_stack=True).dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_pure_uptrend_gives_100():
    rising = pd.DataFrame(
        {"A": [100.0 + i for i in range(30)]},
        index=pd.bdate_range("2020-01-02", periods=30),
    )
    rsi = _rsi(rising, window=14)
    # After warm-up (index 14 onward) pure rising prices → RSI=100
    assert rsi["A"].iloc[14:].dropna().eq(100.0).all()


# ---------------------------------------------------------------------------
# rank_normalize
# ---------------------------------------------------------------------------

def test_rank_normalize_output_in_0_1():
    df     = _prices(n=10)
    ranked = rank_normalize(df)
    valid  = ranked.stack(future_stack=True).dropna()
    assert (valid > 0).all() and (valid <= 1.0).all()


def test_rank_normalize_nan_ticker_excluded_from_ranking():
    df = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [np.nan, 4.0], "C": [3.0, np.nan]},
        index=pd.bdate_range("2020-01-02", periods=2),
    )
    ranked = rank_normalize(df)
    # Row 0: A=1, C=3 → A ranks 1/2=0.5, C ranks 2/2=1.0; B is NaN
    assert pd.isna(ranked["B"].iloc[0])
    assert ranked["A"].iloc[0] == pytest.approx(0.5)
    assert ranked["C"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_feature_panel
# ---------------------------------------------------------------------------

def test_build_feature_panel_multiindex_and_columns():
    prices  = _prices()
    returns = _returns(prices)
    volumes = _volumes()
    dates   = rebalance_dates(DATES[:300], freq=5)

    panel = build_feature_panel(prices, returns, volumes, dates)

    assert panel.index.names == ["date", "ticker"]
    assert set(panel.columns) == {"mom_12_1", "mom_6_1", "mom_3", "mom_1", "high_52w", "vol_21", "rsi_14", "dollar_vol_60"}
    assert len(panel) > 0
    # No all-NaN rows
    assert not panel.isna().all(axis=1).any()
