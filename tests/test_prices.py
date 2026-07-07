"""Tests for src/data/prices.py — all synthetic, no network calls."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.prices import (
    FULL_THRESHOLD,
    _chunk,
    _fetch_chunk,
    audit_coverage,
    load_panel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlcv(dates, close=100.0):
    """Minimal OHLCV DataFrame with a DatetimeIndex."""
    n = len(dates)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _membership(ticker, start, end=None):
    return pd.DataFrame([{
        "ticker": ticker,
        "start": pd.Timestamp(start),
        "end": pd.NaT if end is None else pd.Timestamp(end),
    }])


# ---------------------------------------------------------------------------
# _chunk
# ---------------------------------------------------------------------------

def test_chunk_splits_into_correct_sizes():
    chunks = list(_chunk(list(range(130)), 50))
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [50, 50, 30]


def test_chunk_empty_list_yields_nothing():
    assert list(_chunk([], 50)) == []


# ---------------------------------------------------------------------------
# _fetch_chunk (yfinance mocked — no network)
# ---------------------------------------------------------------------------

def test_fetch_chunk_single_ticker_flat_return(monkeypatch):
    """When yfinance returns a flat DataFrame (single ticker), result maps
    that ticker to the frame."""
    dates = pd.bdate_range("2020-01-02", periods=5)
    fake = _ohlcv(dates)

    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **kw: fake)

    result = _fetch_chunk(["AAPL"], "2020-01-01", "2020-01-10")
    assert list(result.keys()) == ["AAPL"]
    assert set(result["AAPL"].columns) == {"Open", "High", "Low", "Close", "Volume"}


def test_fetch_chunk_multi_ticker_multiindex(monkeypatch):
    """When yfinance returns a (field, ticker) MultiIndex DataFrame, each
    ticker is split out correctly."""
    dates = pd.bdate_range("2020-01-02", periods=3)
    mi = pd.MultiIndex.from_arrays([
        ["Close", "Close", "Volume", "Volume"],
        ["AAPL",  "MSFT",  "AAPL",   "MSFT"],
    ])
    fake = pd.DataFrame(
        [[100.0, 200.0, 1e6, 2e6]] * 3,
        index=dates,
        columns=mi,
    )

    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **kw: fake)

    result = _fetch_chunk(["AAPL", "MSFT"], "2020-01-01", "2020-01-10")
    assert set(result.keys()) == {"AAPL", "MSFT"}


def test_fetch_chunk_empty_download_returns_empty_dict(monkeypatch):
    """A ticker with no yfinance history (delisted, unknown symbol) must
    produce an empty dict, never write a parquet file."""
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **kw: pd.DataFrame())

    result = _fetch_chunk(["ENRN"], "2020-01-01", "2020-01-10")
    assert result == {}


# ---------------------------------------------------------------------------
# load_panel
# ---------------------------------------------------------------------------

def test_load_panel_correct_shape(tmp_path):
    dates = pd.bdate_range("2020-01-02", periods=10)
    for ticker in ["AAPL", "MSFT", "GOOG"]:
        _ohlcv(dates).to_parquet(tmp_path / f"{ticker}.parquet")

    panel = load_panel(["AAPL", "MSFT", "GOOG"], cache_dir=tmp_path)
    assert panel.shape == (10, 3)
    assert set(panel.columns) == {"AAPL", "MSFT", "GOOG"}


def test_load_panel_missing_ticker_becomes_nan_column(tmp_path):
    """A ticker with no cache file must appear as an all-NaN column rather
    than being silently dropped -- callers need to decide how to handle it."""
    dates = pd.bdate_range("2020-01-02", periods=5)
    _ohlcv(dates).to_parquet(tmp_path / "AAPL.parquet")

    panel = load_panel(["AAPL", "DELISTED"], cache_dir=tmp_path)
    assert "DELISTED" in panel.columns
    assert panel["DELISTED"].isna().all()


# ---------------------------------------------------------------------------
# audit_coverage
# ---------------------------------------------------------------------------

def test_audit_coverage_classifies_all_three_statuses(tmp_path):
    """full: price days cover >= FULL_THRESHOLD of expected trading days.
    partial: some data but below threshold.
    empty: no cache file (typical for delisted tickers yfinance can't reach)."""
    window_start, window_end = "2020-01-01", "2020-06-30"

    # FULL — data for the entire membership window
    full_dates = pd.bdate_range("2020-01-02", "2020-06-30")
    _ohlcv(full_dates).to_parquet(tmp_path / "FULL.parquet")

    # PARTIAL — only 10 days of data across a ~130-day membership window
    partial_dates = pd.bdate_range("2020-01-02", periods=10)
    _ohlcv(partial_dates).to_parquet(tmp_path / "PARTIAL.parquet")

    # EMPTY — no parquet file at all (delisted with no yfinance history)

    membership = pd.concat([
        _membership("FULL",    "2020-01-02", "2020-07-01"),
        _membership("PARTIAL", "2020-01-02", "2020-07-01"),
        _membership("EMPTY",   "2020-01-02", "2020-07-01"),
    ], ignore_index=True)

    cov = audit_coverage(membership, cache_dir=tmp_path,
                         start=window_start, end=window_end)

    by_ticker = cov.set_index("ticker")
    assert by_ticker.loc["FULL",    "status"] == "full"
    assert by_ticker.loc["PARTIAL", "status"] == "partial"
    assert by_ticker.loc["EMPTY",   "status"] == "empty"

    assert by_ticker.loc["FULL",    "coverage_pct"] >= FULL_THRESHOLD
    assert 0 < by_ticker.loc["PARTIAL", "coverage_pct"] < FULL_THRESHOLD
    assert by_ticker.loc["EMPTY",   "price_days"] == 0

    # Output is sorted worst-coverage first
    assert list(cov["status"]) == ["empty", "partial", "full"]
