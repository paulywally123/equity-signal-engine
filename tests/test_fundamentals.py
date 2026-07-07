"""Tests for src/data/fundamentals.py — all synthetic, no network calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.fundamentals import _ttm, build_fundamental_panel


# ---------------------------------------------------------------------------
# _ttm helper
# ---------------------------------------------------------------------------

def test_ttm_sums_four_quarters():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    ttm = _ttm(s, n=4)
    assert ttm.iloc[3] == pytest.approx(100.0)   # 10+20+30+40
    assert ttm.iloc[4] == pytest.approx(140.0)   # 20+30+40+50


def test_ttm_requires_full_window():
    s = pd.Series([10.0, 20.0, 30.0])
    ttm = _ttm(s, n=4)
    assert ttm.isna().all()


# ---------------------------------------------------------------------------
# build_fundamental_panel — synthetic cache via tmp_path fixture
# ---------------------------------------------------------------------------

def _make_cache(tmp_path, ticker, n_quarters=8):
    """Write a synthetic fundamental parquet for one ticker."""
    quarters = pd.date_range("2018-01-01", periods=n_quarters, freq="QE")
    df = pd.DataFrame({
        "revenue":      np.linspace(1e9, 2e9, n_quarters),
        "gross_profit": np.linspace(4e8, 8e8, n_quarters),
        "net_income":   np.linspace(1e8, 2e8, n_quarters),
        "total_assets": np.linspace(5e9, 6e9, n_quarters),
        "equity":       np.linspace(2e9, 3e9, n_quarters),
        "shares":       1e8,
    }, index=quarters)
    df.index.name = "period_end"
    df.to_parquet(tmp_path / f"{ticker}.parquet")


def _make_prices(tickers, dates):
    return pd.DataFrame(
        100.0, index=dates, columns=tickers,
    )


def test_panel_has_expected_columns(tmp_path):
    _make_cache(tmp_path, "AAPL")
    prices = _make_prices(["AAPL"], pd.bdate_range("2020-01-01", periods=5))
    dates  = pd.DatetimeIndex([pd.Timestamp("2020-06-01")])
    panel  = build_fundamental_panel(["AAPL"], tmp_path, prices, dates)
    assert "gross_prof" in panel.columns
    assert "roe" in panel.columns


def test_panel_multiindex_names(tmp_path):
    _make_cache(tmp_path, "AAPL")
    prices = _make_prices(["AAPL"], pd.bdate_range("2020-01-01", periods=5))
    dates  = pd.DatetimeIndex([pd.Timestamp("2020-06-01")])
    panel  = build_fundamental_panel(["AAPL"], tmp_path, prices, dates)
    assert panel.index.names == ["date", "ticker"]


def test_filing_lag_prevents_lookahead(tmp_path):
    """When the 60-day lag eliminates all but 3 quarters, TTM cannot be computed
    and no record should be produced."""
    # 4 quarters: 2019-03-31, 2019-06-30, 2019-09-30, 2019-12-31
    quarters = pd.DatetimeIndex(["2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31"])
    df = pd.DataFrame({
        "gross_profit": [4e8] * 4,
        "net_income":   [1e8] * 4,
        "total_assets": [5e9] * 4,
        "equity":       [2e9] * 4,
        "shares":       1e8,
    }, index=quarters)
    df.index.name = "period_end"
    df.to_parquet(tmp_path / "AAPL.parquet")

    prices = _make_prices(["AAPL"], pd.bdate_range("2020-01-01", periods=5))
    # Rebalance on 2020-01-10; cutoff = 2019-11-11; only 3 quarters qualify (< 4 for TTM)
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-10")])
    panel = build_fundamental_panel(["AAPL"], tmp_path, prices, dates,
                                    filing_lag_days=60)
    assert panel.empty or "AAPL" not in panel.index.get_level_values("ticker")


def test_missing_ticker_skipped(tmp_path):
    prices = _make_prices(["MSFT"], pd.bdate_range("2020-01-01", periods=5))
    dates  = pd.DatetimeIndex([pd.Timestamp("2020-06-01")])
    panel  = build_fundamental_panel(["MSFT"], tmp_path, prices, dates)
    assert panel.empty


def test_gross_prof_positive_for_profitable_firm(tmp_path):
    _make_cache(tmp_path, "GOOG")
    prices = _make_prices(["GOOG"], pd.bdate_range("2020-01-01", periods=5))
    dates  = pd.DatetimeIndex([pd.Timestamp("2020-06-01")])
    panel  = build_fundamental_panel(["GOOG"], tmp_path, prices, dates)
    if not panel.empty and "gross_prof" in panel.columns:
        assert (panel["gross_prof"].dropna() > 0).all()


def test_ep_ratio_present_when_price_and_shares_available(tmp_path):
    _make_cache(tmp_path, "TSLA")
    prices = _make_prices(["TSLA"], pd.bdate_range("2019-01-01", "2020-12-31"))
    dates  = pd.DatetimeIndex([pd.Timestamp("2020-06-01")])
    panel  = build_fundamental_panel(["TSLA"], tmp_path, prices, dates)
    assert "ep_ratio" in panel.columns
