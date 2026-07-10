"""Tests for src/backtest/backtest.py — all synthetic, no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtest import (
    _construct_weights,
    compute_metrics,
    drawdown_trigger_exposure,
    feature_ic,
    information_coefficient,
    level_trigger_exposure,
    run_backtest,
    trend_exposure,
    vol_target_exposure,
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


def _returns_panel(values, start="2020-01-02"):
    """Single-ticker daily returns panel -- keeps equal_weight_index's
    cross-sectional mean exactly equal to `values`, so exposure functions
    built on top of it can be checked with exact arithmetic."""
    dates = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"T00": values}, index=dates)


def _monotonic_returns(n, daily_return):
    return _returns_panel([daily_return] * n)


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


def test_sector_neutral_long_weights_sum_to_one():
    """Sector-neutral long weights must still sum to 1."""
    tickers = list("ABCDEFGHIJ")
    scores  = pd.Series(np.arange(10, dtype=float), index=tickers)
    sectors = pd.Series(
        ["Tech"] * 5 + ["Finance"] * 5, index=tickers
    )
    weights = _construct_weights(scores, top_q=0.2, long_only=True, sectors=sectors)
    assert abs(weights[weights > 0].sum() - 1.0) < 1e-9


def test_sector_neutral_selects_within_each_sector():
    """Best scorer in each sector must be selected, worst must not."""
    tickers = list("ABCD")
    # A, B → Tech;  C, D → Finance
    scores  = pd.Series([1.0, 4.0, 2.0, 3.0], index=tickers)  # B best tech, D best finance
    sectors = pd.Series(["Tech", "Tech", "Finance", "Finance"], index=tickers)
    weights = _construct_weights(scores, top_q=0.5, long_only=True, sectors=sectors)
    assert weights["B"] > 0    # best in Tech
    assert weights["D"] > 0    # best in Finance
    assert weights["A"] == 0   # worst in Tech
    assert weights["C"] == 0   # worst in Finance


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


# ---------------------------------------------------------------------------
# vol_target_exposure
# ---------------------------------------------------------------------------

def test_vol_target_exposure_zero_vol_gives_max_exposure():
    """Zero realized vol -> target/0 -> inf, clipped to max_exposure. Also
    covers the rolling-window warmup rows (NaN std), which are filled to
    max_exposure rather than left NaN -- if they were NaN, run_backtest's
    `float(past_exp.iloc[-1])` lookup would raise."""
    returns = _returns_panel([0.0] * 40)
    exp = vol_target_exposure(returns, target_vol=0.15, window=20)
    assert (exp == 1.0).all()


def test_vol_target_exposure_never_exceeds_max_exposure():
    returns = _returns_panel([0.0] * 40)
    exp = vol_target_exposure(returns, target_vol=0.15, window=20, max_exposure=0.8)
    assert (exp <= 0.8).all()
    assert (exp == 0.8).all()  # zero vol -> clipped straight to the cap


def test_vol_target_exposure_scales_down_when_realized_vol_exceeds_target():
    """Large alternating daily returns (+/-5%) give a high realized vol --
    well above a 15% annualized target -- so exposure should shrink well
    below max_exposure once the rolling window has enough data."""
    values  = [0.05 if i % 2 == 0 else -0.05 for i in range(60)]
    returns = _returns_panel(values)
    exp = vol_target_exposure(returns, target_vol=0.15, window=20)
    assert (exp.iloc[25:] < 0.5).all()


# ---------------------------------------------------------------------------
# trend_exposure
# ---------------------------------------------------------------------------

def test_trend_exposure_warmup_period_defaults_to_floor():
    """Before `window` observations exist, the rolling mean is NaN and
    `idx > NaN` is False regardless of the real trend -- so exposure
    defaults to `floor` (a conservative "not yet confirmed" state), not 1.0."""
    returns = _monotonic_returns(40, 0.001)
    exp = trend_exposure(returns, window=20, floor=0.3)
    assert (exp.iloc[:19] == 0.3).all()


def test_trend_exposure_uptrend_gives_full_exposure_after_warmup():
    returns = _monotonic_returns(40, 0.001)
    exp = trend_exposure(returns, window=20, floor=0.3)
    assert (exp.iloc[19:] == 1.0).all()


def test_trend_exposure_downtrend_gives_floor_after_warmup():
    returns = _monotonic_returns(40, -0.001)
    exp = trend_exposure(returns, window=20, floor=0.4)
    assert (exp.iloc[19:] == 0.4).all()


def test_trend_exposure_default_floor_is_zero():
    returns = _monotonic_returns(40, -0.001)
    exp = trend_exposure(returns, window=20)
    assert (exp.iloc[19:] == 0.0).all()


# ---------------------------------------------------------------------------
# drawdown_trigger_exposure
# ---------------------------------------------------------------------------

def test_drawdown_trigger_no_drawdown_in_uptrend():
    """A monotonic uptrend is always at a new high, so drawdown-from-peak is
    always ~0 -- the trigger should never fire."""
    returns = _monotonic_returns(30, 0.001)
    exp = drawdown_trigger_exposure(returns, lookback=10, threshold=0.05, floor=0.0)
    assert (exp == 1.0).all()


def test_drawdown_trigger_small_dip_below_threshold_not_triggered():
    values  = [0.0] * 5 + [-0.03] + [0.0] * 4   # -3% dip, 8% threshold
    returns = _returns_panel(values)
    exp = drawdown_trigger_exposure(returns, lookback=20, threshold=0.08, floor=0.0)
    assert (exp == 1.0).all()


def test_drawdown_trigger_fires_on_crash_and_rearms_on_recovery():
    """Fast tail-trigger's whole point vs. a slow trend filter: it should
    re-arm as soon as the index makes a new high, not wait on a lagging
    average to catch up."""
    values  = [0.0] * 5 + [-0.10] + [0.0] * 4 + [0.15]
    returns = _returns_panel(values)
    exp = drawdown_trigger_exposure(returns, lookback=20, threshold=0.08, floor=0.0)
    assert (exp.iloc[:5] == 1.0).all()     # before the crash
    assert (exp.iloc[5:10] == 0.0).all()   # in an >8% drawdown from the prior peak
    assert exp.iloc[10] == 1.0             # new high -> re-armed


# ---------------------------------------------------------------------------
# level_trigger_exposure
# ---------------------------------------------------------------------------

def test_level_trigger_below_threshold_gives_full_exposure():
    dates     = pd.bdate_range("2020-01-02", periods=10)
    indicator = pd.Series([10.0] * 10, index=dates)
    exp = level_trigger_exposure(indicator, threshold=20.0, floor=0.0)
    assert (exp == 1.0).all()


def test_level_trigger_above_threshold_gives_floor_on_exact_days():
    dates     = pd.bdate_range("2020-01-02", periods=5)
    indicator = pd.Series([10.0, 25.0, 10.0, 30.0, 10.0], index=dates)
    exp = level_trigger_exposure(indicator, threshold=20.0, floor=0.5)
    expected = pd.Series([1.0, 0.5, 1.0, 0.5, 1.0], index=dates)
    pd.testing.assert_series_equal(exp, expected, check_names=False)


def test_level_trigger_exact_threshold_value_not_triggered():
    """Boundary is strict `>` -- a reading exactly at the threshold should
    not itself force a de-risking."""
    dates     = pd.bdate_range("2020-01-02", periods=3)
    indicator = pd.Series([19.9, 20.0, 20.1], index=dates)
    exp = level_trigger_exposure(indicator, threshold=20.0, floor=0.0)
    assert list(exp) == [1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# run_backtest's `exposure` kwarg (integration -- mirrors the `regime` tests)
# ---------------------------------------------------------------------------

def test_exposure_column_defaults_to_one_when_not_provided():
    preds  = _make_predictions(n_dates=5)
    labels = _make_labels(preds)
    bt = run_backtest(preds, labels, long_only=True)
    assert (bt["exposure"] == 1.0).all()


def test_exposure_zero_produces_zero_gross_return():
    preds  = _make_predictions(n_dates=10)
    labels = _make_labels(preds)
    pred_dates = preds.index.get_level_values("date").unique().sort_values()
    exposure_idx = pd.date_range(pred_dates[0] - pd.Timedelta(days=10),
                                 periods=5, freq="B")
    exposure = pd.Series(0.0, index=exposure_idx)
    bt = run_backtest(preds, labels, long_only=True, exposure=exposure)
    assert (bt["gross_return"] == 0.0).all()


def test_exposure_one_matches_no_exposure():
    preds  = _make_predictions(n_dates=10)
    labels = _make_labels(preds)
    pred_dates = preds.index.get_level_values("date").unique().sort_values()
    exposure_idx = pd.date_range(pred_dates[0] - pd.Timedelta(days=10),
                                 periods=5, freq="B")
    exposure = pd.Series(1.0, index=exposure_idx)
    bt_exp   = run_backtest(preds, labels, long_only=True, exposure=exposure)
    bt_plain = run_backtest(preds, labels, long_only=True)
    pd.testing.assert_series_equal(
        bt_exp["net_return"], bt_plain["net_return"], check_names=False
    )


def test_exposure_no_lookahead_uses_prior_bar_only():
    """Exposure is sampled strictly before each rebalance date, same
    guarantee as `regime`. A value set ON a rebalance date must not affect
    that same date's weights -- only the next one."""
    preds  = _make_predictions(n_dates=5)
    labels = _make_labels(preds)
    pred_dates = preds.index.get_level_values("date").unique().sort_values()
    exposure = pd.Series(
        [1.0, 0.0],
        index=[pred_dates[0] - pd.Timedelta(days=1), pred_dates[1]],
    )
    bt = run_backtest(preds, labels, long_only=True, exposure=exposure)
    assert bt.loc[pred_dates[1], "exposure"] == 1.0   # same-day value not yet "seen"
    assert bt.loc[pred_dates[2], "exposure"] == 0.0   # now strictly in the past
