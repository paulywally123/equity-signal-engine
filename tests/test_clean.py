"""Tests for src/data/clean.py — all synthetic, no network or disk I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    apply_membership_mask,
    clean_panel,
    fill_gaps,
    filter_by_coverage,
    log_returns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prices(tickers, dates, val=100.0):
    return pd.DataFrame(
        val,
        index=pd.DatetimeIndex(dates),
        columns=list(tickers),
        dtype=float,
    )


def _coverage(statuses: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": t, "status": s} for t, s in statuses.items()
    ])


def _membership(ticker, start, end=None) -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker,
        "start": pd.Timestamp(start),
        "end": pd.NaT if end is None else pd.Timestamp(end),
    }])


# ---------------------------------------------------------------------------
# filter_by_coverage
# ---------------------------------------------------------------------------

def test_filter_keeps_full_and_partial_drops_empty():
    dates = pd.bdate_range("2020-01-02", periods=5)
    panel = _prices(["A", "B", "C"], dates)
    coverage = _coverage({"A": "full", "B": "partial", "C": "empty"})

    result = filter_by_coverage(panel, coverage)

    assert set(result.columns) == {"A", "B"}
    assert "C" not in result.columns


def test_filter_preserves_column_order_relative_to_panel():
    dates = pd.bdate_range("2020-01-02", periods=3)
    panel = _prices(["Z", "A", "M"], dates)
    coverage = _coverage({"Z": "full", "A": "empty", "M": "full"})

    result = filter_by_coverage(panel, coverage)

    # Order should follow the panel, not coverage
    assert list(result.columns) == ["Z", "M"]


# ---------------------------------------------------------------------------
# apply_membership_mask
# ---------------------------------------------------------------------------

def test_mask_nullifies_dates_outside_membership_window():
    dates = pd.bdate_range("2020-01-02", periods=10)
    prices = _prices(["AAPL"], dates)
    # Membership covers first 5 dates only; dates[5] is the first day OUT
    mem = _membership("AAPL", dates[0], dates[5])

    masked = apply_membership_mask(prices, mem)

    assert masked["AAPL"].iloc[:5].notna().all()
    assert masked["AAPL"].iloc[5:].isna().all()


def test_mask_preserves_prices_inside_membership_window():
    dates = pd.bdate_range("2020-01-02", periods=5)
    prices = _prices(["MSFT"], dates, val=250.0)
    mem = _membership("MSFT", dates[0])  # current member, no end date

    masked = apply_membership_mask(prices, mem)

    assert (masked["MSFT"] == 250.0).all()


# ---------------------------------------------------------------------------
# fill_gaps
# ---------------------------------------------------------------------------

def test_fill_gaps_fills_short_gaps_up_to_max():
    dates = pd.bdate_range("2020-01-02", periods=6)
    prices = pd.DataFrame(
        {"A": [100.0, np.nan, np.nan, 103.0, 104.0, 105.0]},
        index=dates,
    )
    filled = fill_gaps(prices, max_fill=5)

    assert filled["A"].iloc[1] == pytest.approx(100.0)
    assert filled["A"].iloc[2] == pytest.approx(100.0)


def test_fill_gaps_leaves_gaps_longer_than_max_as_nan():
    dates = pd.bdate_range("2020-01-02", periods=8)
    prices = pd.DataFrame(
        {"A": [100.0] + [np.nan] * 7},
        index=dates,
    )
    filled = fill_gaps(prices, max_fill=3)

    # Positions 1-3 filled, 4-7 still NaN
    assert filled["A"].iloc[1:4].notna().all()
    assert filled["A"].iloc[4:].isna().all()


# ---------------------------------------------------------------------------
# clean_panel (mask → fill → re-mask)
# ---------------------------------------------------------------------------

def test_clean_panel_does_not_spill_fill_past_membership_end():
    """Forward-fill must not carry a price into dates after the ticker left
    the index -- that would constitute lookahead through the universe."""
    dates = pd.bdate_range("2020-01-02", periods=10)
    # Price data for all 10 dates, but membership ends after the first 5
    prices = _prices(["A"], dates, val=100.0)
    mem = _membership("A", dates[0], dates[5])

    clean = clean_panel(prices, mem, max_fill=5)

    assert clean["A"].iloc[:5].notna().all()
    assert clean["A"].iloc[5:].isna().all()


# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------

def test_log_returns_first_row_is_nan_and_values_are_correct():
    dates = pd.bdate_range("2020-01-02", periods=3)
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=dates)

    rets = log_returns(prices)

    assert pd.isna(rets["A"].iloc[0])
    assert rets["A"].iloc[1] == pytest.approx(np.log(110.0 / 100.0))
    assert rets["A"].iloc[2] == pytest.approx(np.log(99.0 / 110.0))
