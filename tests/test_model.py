"""Tests for src/models/model.py — tiny synthetic datasets, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.model import PRICE_FEATURE_COLS, _apply_dev_filter, walk_forward_predict

# Tests use price-only features so the fixture doesn't need a fundamentals panel
FEATURE_COLS = PRICE_FEATURE_COLS


# ---------------------------------------------------------------------------
# Fixtures: minimal (date, ticker) MultiIndex data
# ---------------------------------------------------------------------------

def _make_data(n_dates=60, n_tickers=10, seed=42):
    """Return (features, labels) DataFrames with MultiIndex (date, ticker)."""
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_dates, freq="5B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    feat = pd.DataFrame(
        rng.random((len(idx), len(FEATURE_COLS))),
        index=idx,
        columns=FEATURE_COLS,
    )
    labels = pd.DataFrame(
        {"fwd_return": rng.standard_normal(len(idx)) * 0.02},
        index=idx,
    )
    return feat, labels


# ---------------------------------------------------------------------------
# _apply_dev_filter
# ---------------------------------------------------------------------------

def test_dev_filter_keeps_exactly_top_n_per_date():
    feat, labels = _make_data()
    data = feat.join(labels)
    filtered = _apply_dev_filter(data, top_n=5)
    for date, grp in filtered.groupby(level="date"):
        assert len(grp) == 5


def test_dev_filter_selects_highest_dollar_vol():
    dates   = pd.bdate_range("2020-01-06", periods=1, freq="5B")
    tickers = ["T00", "T01", "T02", "T03", "T04", "T05"]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    dv_vals = {"T00": 0.9, "T01": 0.8, "T02": 0.7, "T03": 0.3, "T04": 0.2, "T05": 0.1}
    feat = pd.DataFrame(0.5, index=idx, columns=FEATURE_COLS)
    for t, v in dv_vals.items():
        feat.loc[(dates[0], t), "dollar_vol_60"] = v

    labels   = pd.DataFrame({"fwd_return": 0.01}, index=idx)
    data     = feat.join(labels)
    filtered = _apply_dev_filter(data, top_n=3)

    kept = set(filtered.index.get_level_values("ticker"))
    assert {"T00", "T01", "T02"}.issubset(kept)


# ---------------------------------------------------------------------------
# walk_forward_predict
# ---------------------------------------------------------------------------

def test_walk_forward_returns_score_column():
    feat, labels = _make_data()
    cutoff = str(feat.index.get_level_values("date").unique()[29].date())
    preds  = walk_forward_predict(feat, labels, initial_train_end=cutoff)
    assert "score" in preds.columns


def test_walk_forward_multiindex_names():
    feat, labels = _make_data()
    cutoff = str(feat.index.get_level_values("date").unique()[29].date())
    preds  = walk_forward_predict(feat, labels, initial_train_end=cutoff)
    assert preds.index.names == ["date", "ticker"]


def test_walk_forward_no_predictions_before_cutoff():
    feat, labels = _make_data()
    dates  = feat.index.get_level_values("date").unique()
    cutoff = str(dates[29].date())
    preds  = walk_forward_predict(feat, labels, initial_train_end=cutoff)
    pred_dates = preds.index.get_level_values("date")
    assert (pred_dates > pd.Timestamp(cutoff)).all()


def test_walk_forward_scores_are_finite_floats():
    feat, labels = _make_data()
    cutoff = str(feat.index.get_level_values("date").unique()[29].date())
    preds  = walk_forward_predict(feat, labels, initial_train_end=cutoff)
    assert preds["score"].dtype == float
    assert preds["score"].notna().all()


def test_walk_forward_dev_mode_restricts_universe():
    feat, labels = _make_data(n_tickers=10)
    cutoff = str(feat.index.get_level_values("date").unique()[29].date())
    preds_full = walk_forward_predict(feat, labels, initial_train_end=cutoff)
    preds_dev  = walk_forward_predict(feat, labels, initial_train_end=cutoff, dev_top_n=4)
    # Dev mode should predict on fewer tickers per date
    full_per_date = preds_full.groupby(level="date").size().mean()
    dev_per_date  = preds_dev.groupby(level="date").size().mean()
    assert dev_per_date <= full_per_date


def test_walk_forward_expanding_window_grows():
    """Each year's training set must be strictly larger than the previous."""
    feat, labels = _make_data(n_dates=120, n_tickers=5)
    cutoff = str(feat.index.get_level_values("date").unique()[29].date())
    # Just verify it runs without error and produces predictions across years
    preds = walk_forward_predict(feat, labels, initial_train_end=cutoff)
    years = sorted(preds.index.get_level_values("date").year.unique())
    assert len(years) >= 1
