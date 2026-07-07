"""Tests for src/backtest/backtest.py — all synthetic, no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtest import (
    _construct_weights,
    compute_metrics,
    feature_ic,
    information_coefficient,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictions(n_dates=20, n_tickers=10, seed=0):
    rng     = np.random.default_rng(seed)
    dates   = pd.bdate_range("2020-01-06", periods=n_dates, freq="5B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.DataFrame({"score": rng.random(len(idx))}, index=idx)


def _make_labels(predictions, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"fwd_return": rng.standard_normal(len(predictions)) * 0.02},
        index=predictions.index,
    )


# ---------------------------------------------------------------------------
# _construct_weights
# ---------------------------------------------------------------------------

def test_weights_long_short_approximately_zero_sum():
    scores  = pd.Series(np.arange(10, dtype=float), index=[f"T{i}" for i in range(10)])
    weights = _construct_weights(scores, top_q=0.2)
    assert abs(weights.sum()) < 1e-10


def test_weights_long_leg_positive_short_leg_negative():
    scores  = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=list("ABCDE"))
    weights = _construct_weights(scores, top_q=0.2)
    assert weights["E"] > 0     # highest score → long
    assert weights["A"] < 0     # lowest score → short


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------

def test_backtest_returns_expected_columns():
    preds  = _make_predictions()
    labels = _make_labels(preds)
    bt     = run_backtest(preds, labels, costs_bps=10)
    assert set(bt.columns) >= {"gross_return", "cost", "net_return",
                                "long_return", "short_return"}


def test_net_return_equals_gross_minus_cost():
    preds  = _make_predictions()
    labels = _make_labels(preds)
    bt     = run_backtest(preds, labels, costs_bps=10)
    residual = (bt["net_return"] - (bt["gross_return"] - bt["cost"])).abs()
    assert residual.max() < 1e-10


def test_costs_reduce_net_vs_gross():
    preds  = _make_predictions()
    labels = _make_labels(preds)
    bt     = run_backtest(preds, labels, costs_bps=10)
    assert (bt["net_return"] <= bt["gross_return"]).all()


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_all_keys_present():
    preds  = _make_predictions()
    labels = _make_labels(preds)
    bt      = run_backtest(preds, labels)
    metrics = compute_metrics(bt)
    for key in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "cagr",
                "hit_rate", "n_periods"):
        assert key in metrics


# ---------------------------------------------------------------------------
# information_coefficient
# ---------------------------------------------------------------------------

def test_ic_bounded_minus_one_to_one():
    preds  = _make_predictions()
    labels = _make_labels(preds)
    ic     = information_coefficient(preds, labels)
    assert (ic >= -1.0).all() and (ic <= 1.0).all()


def test_ic_perfect_prediction_gives_one():
    """If scores perfectly rank-predict returns, IC should be 1.0."""
    dates   = pd.bdate_range("2020-01-06", periods=5, freq="5B")
    tickers = ["A", "B", "C", "D", "E"]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    scores  = pd.Series(range(len(idx)), index=idx, dtype=float)
    returns = pd.Series(range(len(idx)), index=idx, dtype=float)
    preds   = pd.DataFrame({"score": scores})
    labels  = pd.DataFrame({"fwd_return": returns})
    ic      = information_coefficient(preds, labels)
    assert (ic == pytest.approx(1.0, abs=1e-6)).all()


# ---------------------------------------------------------------------------
# regime filter
# ---------------------------------------------------------------------------

def test_regime_false_produces_zero_gross_return():
    """When regime is always False, no positions are taken and gross return = 0."""
    preds  = _make_predictions(n_dates=10)
    labels = _make_labels(preds)
    # Regime always off — use dates strictly before first prediction date
    pred_dates = preds.index.get_level_values("date").unique().sort_values()
    regime_idx = pd.date_range(pred_dates[0] - pd.Timedelta(days=10),
                               periods=5, freq="B")
    regime = pd.Series(False, index=regime_idx)
    bt = run_backtest(preds, labels, regime=regime)
    assert (bt["gross_return"] == 0.0).all()


def test_regime_true_matches_no_regime():
    """When regime is always True the result equals running without a regime filter."""
    preds  = _make_predictions(n_dates=10)
    labels = _make_labels(preds)
    pred_dates = preds.index.get_level_values("date").unique().sort_values()
    regime_idx = pd.date_range(pred_dates[0] - pd.Timedelta(days=10),
                               periods=5, freq="B")
    regime = pd.Series(True, index=regime_idx)
    bt_regime = run_backtest(preds, labels, long_only=True, regime=regime)
    bt_plain  = run_backtest(preds, labels, long_only=True)
    pd.testing.assert_series_equal(
        bt_regime["net_return"], bt_plain["net_return"], check_names=False
    )


# ---------------------------------------------------------------------------
# feature_ic
# ---------------------------------------------------------------------------

def test_feature_ic_returns_one_column_per_feature():
    preds  = _make_predictions()
    labels = _make_labels(preds)
    feats  = pd.DataFrame(
        np.random.default_rng(99).random((len(preds), 3)),
        index=preds.index,
        columns=["f1", "f2", "f3"],
    )
    fic = feature_ic(feats, labels)
    assert set(fic.columns) == {"f1", "f2", "f3"}
