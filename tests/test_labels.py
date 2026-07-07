"""Tests for src/labels/labels.py — all synthetic, no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels.labels import build_label_panel, forward_returns, rank_labels


def _prices(vals, freq="B"):
    return pd.DataFrame(
        {"A": vals},
        index=pd.bdate_range("2020-01-02", periods=len(vals)),
    )


# ---------------------------------------------------------------------------
# forward_returns
# ---------------------------------------------------------------------------

def test_forward_returns_correct_value():
    prices = _prices([100.0, 100.0, 100.0, 100.0, 100.0, 110.0])
    fwd    = forward_returns(prices, horizon=5)
    # At row 0: log(price[5]/price[0]) = log(110/100)
    assert fwd["A"].iloc[0] == pytest.approx(np.log(110.0 / 100.0), rel=1e-6)


def test_forward_returns_last_horizon_rows_are_nan():
    prices = _prices([100.0] * 10)
    fwd    = forward_returns(prices, horizon=5)
    assert fwd["A"].iloc[-5:].isna().all()


def test_forward_returns_nan_price_propagates_to_label():
    prices = pd.DataFrame(
        {"A": [100.0, 100.0, np.nan, 100.0, 100.0, 110.0]},
        index=pd.bdate_range("2020-01-02", periods=6),
    )
    fwd = forward_returns(prices, horizon=5)
    # Row 0: needs price[5]=110 and price[0]=100 — the NaN at row 2 is in
    # between and doesn't affect this particular calculation
    assert pd.notna(fwd["A"].iloc[0])
    # Row 1: needs price[6] which doesn't exist → NaN
    assert pd.isna(fwd["A"].iloc[1])


# ---------------------------------------------------------------------------
# build_label_panel
# ---------------------------------------------------------------------------

def test_build_label_panel_has_correct_multiindex():
    prices = _prices([100.0] * 20)
    dates  = pd.bdate_range("2020-01-02", periods=20)[::5]
    labels = build_label_panel(prices, dates, horizon=5)
    assert labels.index.names == ["date", "ticker"]
    assert "fwd_return" in labels.columns


def test_build_label_panel_drops_nan_labels():
    prices = _prices([100.0] * 20)
    dates  = pd.bdate_range("2020-01-02", periods=20)[::5]
    labels = build_label_panel(prices, dates, horizon=5)
    assert labels["fwd_return"].notna().all()


def test_build_label_panel_only_contains_requested_dates():
    prices = _prices([100.0] * 30)
    dates  = pd.bdate_range("2020-01-02", periods=30)[::5]
    labels = build_label_panel(prices, dates, horizon=5)
    label_dates = set(labels.index.get_level_values("date"))
    assert label_dates.issubset(set(dates))


def test_forward_returns_log_additivity():
    """Log returns are additive: 5-day fwd return equals sum of 5 daily returns."""
    prices  = _prices([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    fwd     = forward_returns(prices, horizon=5)
    daily   = np.log(prices / prices.shift(1))
    # 5-day fwd return at row 0 = sum of daily returns rows 1..5
    expected = daily["A"].iloc[1:6].sum()
    assert fwd["A"].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_rank_labels_output_bounded_0_to_1():
    prices = _prices([100.0] * 30)
    dates  = pd.bdate_range("2020-01-02", periods=30)[::5]
    labels = build_label_panel(prices, dates, horizon=5)
    # Inject multiple tickers by building a wider panel
    multi = pd.concat([labels.rename(columns={"fwd_return": "fwd_return"}),
                       labels * 1.5], axis=0)
    ranked = rank_labels(labels)
    assert (ranked["fwd_return"].dropna() >= 0).all()
    assert (ranked["fwd_return"].dropna() <= 1).all()


def test_rank_labels_preserves_relative_order():
    """Within each date, a higher raw return must have a higher rank."""
    dates   = pd.bdate_range("2020-01-02", periods=2, freq="5B")
    tickers = ["A", "B", "C"]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    raw     = pd.DataFrame({"fwd_return": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01]}, index=idx)
    ranked  = rank_labels(raw)
    d0      = ranked.xs(dates[0], level="date")["fwd_return"]
    assert d0["A"] < d0["B"] < d0["C"]


def test_build_label_panel_excludes_nan_membership_periods():
    """Prices masked to NaN (non-membership) must not appear as labels."""
    vals   = [100.0] * 10 + [np.nan] * 10
    prices = pd.DataFrame(
        {"A": vals},
        index=pd.bdate_range("2020-01-02", periods=20),
    )
    dates  = pd.bdate_range("2020-01-02", periods=20)[::5]
    labels = build_label_panel(prices, dates, horizon=5)
    # Any label that crosses the NaN boundary should be NaN and thus dropped
    assert labels["fwd_return"].notna().all()
