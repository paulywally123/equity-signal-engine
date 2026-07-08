"""Walk-forward LightGBM model for cross-sectional return prediction.

Training uses an expanding window re-fit once per calendar year. Hyperparameters
are fixed (no tuning loop) to keep the walk-forward evaluation honest — the
biggest silent risk in rolling-window backtests is leaking test-set information
through hyperparameter search.

Dev mode: restricts each date's universe to the top-N tickers by dollar_vol_60,
enabling fast iteration without changing the model or backtest logic.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import pandas as pd

logger = logging.getLogger(__name__)

PRICE_FEATURE_COLS = ["mom_12_1", "dollar_vol_60"]
FUND_FEATURE_COLS  = ["gross_prof", "roe", "ep_ratio"]

# Resolved at training time: include only columns actually present in the panel
FEATURE_COLS = PRICE_FEATURE_COLS + FUND_FEATURE_COLS

PARAMS: dict = {
    "objective":        "regression",
    "n_estimators":     100,
    "learning_rate":    0.03,
    "max_depth":        3,
    "num_leaves":       7,
    "min_child_samples": 200,
    "subsample":        0.7,
    "colsample_bytree": 0.7,
    "reg_alpha":        0.5,
    "reg_lambda":       0.5,
    "random_state":     42,
    "verbose":          -1,
}


def _apply_dev_filter(data: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep only top-N tickers by dollar_vol_60 on each rebalance date."""
    out = []
    for _, grp in data.groupby(level="date"):
        ranked = grp["dollar_vol_60"].rank(ascending=False)
        out.append(grp[ranked <= top_n])
    return pd.concat(out)


def walk_forward_predict(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    initial_train_end: str,
    params: dict = PARAMS,
    dev_top_n: int | None = None,
) -> pd.DataFrame:
    """Walk-forward predictions with annual re-fitting on an expanding window.

    Parameters
    ----------
    features          : (date, ticker) MultiIndex, columns = feature names
    labels            : (date, ticker) MultiIndex, column 'fwd_return'
    initial_train_end : last date (inclusive) in the first training window
    params            : LightGBM params — fixed, not tuned on test data
    dev_top_n         : if set, restrict universe to top-N by dollar_vol_60

    Returns
    -------
    DataFrame with MultiIndex (date, ticker) and column 'score'.
    """
    data = features.join(labels, how="inner").dropna(subset=["fwd_return"])
    if dev_top_n is not None:
        data = _apply_dev_filter(data, top_n=dev_top_n)

    # Use only the feature columns that are actually present in the panel
    feat_cols = [c for c in FEATURE_COLS if c in data.columns]
    logger.info("Using %d features: %s", len(feat_cols), feat_cols)

    all_dates = data.index.get_level_values("date").unique().sort_values()
    cutoff = pd.Timestamp(initial_train_end)
    test_dates = all_dates[all_dates > cutoff]

    if test_dates.empty:
        raise ValueError(f"No test dates after initial_train_end={initial_train_end!r}")

    predictions: list[pd.DataFrame] = []

    for year in sorted(test_dates.year.unique()):
        year_start = pd.Timestamp(f"{year}-01-01")
        train_dates = data.index.get_level_values("date")
        train_mask = train_dates < year_start

        # Embargo: rebalance dates are spaced exactly horizon_days trading
        # days apart, so the single most recent pre-cutoff date has a label
        # (forward return) that resolves at approximately the same time as
        # the first test date of `year` -- training on it would mean the
        # model saw an outcome that wasn't actually knowable yet as of the
        # first test prediction. Purge that one date.
        if train_mask.any():
            last_train_date = train_dates[train_mask].max()
            train_mask &= train_dates < last_train_date

        train = data[train_mask]

        if len(train) < 50:
            logger.warning("Year %d: only %d training rows — skipping", year, len(train))
            continue

        model = lgb.LGBMRegressor(**params)
        model.fit(train[feat_cols], train["fwd_return"])
        logger.info("Year %d: trained on %d obs", year, len(train))

        year_test = data[data.index.get_level_values("date").year == year]
        if year_test.empty:
            continue

        scores = model.predict(year_test[feat_cols])
        predictions.append(pd.DataFrame({"score": scores}, index=year_test.index))

    if not predictions:
        raise RuntimeError("No predictions produced — check date range and initial_train_end")

    return pd.concat(predictions).sort_index()


def predict_latest(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    params: dict = PARAMS,
    dev_top_n: int | None = None,
) -> pd.DataFrame:
    """Score the most recent feature date(s) that don't yet have a resolved
    forward-return label.

    walk_forward_predict only ever scores dates that survive an inner join
    with labels -- by construction it can never produce a score newer than
    (latest price date - horizon_days), since a label needs that much future
    price history to exist at all. That's correct for backtesting but means
    it can never represent a genuinely current signal. This fits on every
    labeled observation available (same expanding-window discipline as the
    final year of walk_forward_predict) and scores whatever feature rows
    exist beyond the last labeled date -- the actual "right now."

    Returns
    -------
    DataFrame with MultiIndex (date, ticker) and column 'score', for date(s)
    strictly after the last labeled date.
    """
    labeled = features.join(labels, how="inner").dropna(subset=["fwd_return"])
    if dev_top_n is not None:
        labeled = _apply_dev_filter(labeled, top_n=dev_top_n)

    feat_cols = [c for c in FEATURE_COLS if c in labeled.columns]

    model = lgb.LGBMRegressor(**params)
    model.fit(labeled[feat_cols], labeled["fwd_return"])

    last_labeled_date = labeled.index.get_level_values("date").max()
    unlabeled = features[features.index.get_level_values("date") > last_labeled_date]
    if dev_top_n is not None:
        unlabeled = _apply_dev_filter(unlabeled, top_n=dev_top_n)

    if unlabeled.empty:
        raise RuntimeError(
            "No feature dates found after the last labeled date "
            f"({last_labeled_date.date()}) -- nothing new to score."
        )

    scores = model.predict(unlabeled[feat_cols])
    return pd.DataFrame({"score": scores}, index=unlabeled.index).sort_index()
