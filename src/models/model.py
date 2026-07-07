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

FEATURE_COLS = ["mom_12_1", "mom_6_1", "mom_3", "mom_1", "high_52w", "vol_21", "rsi_14", "dollar_vol_60"]

PARAMS: dict = {
    "objective":        "regression",
    "n_estimators":     200,
    "learning_rate":    0.05,
    "max_depth":        4,
    "num_leaves":       15,
    "min_child_samples": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
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

    all_dates = data.index.get_level_values("date").unique().sort_values()
    cutoff = pd.Timestamp(initial_train_end)
    test_dates = all_dates[all_dates > cutoff]

    if test_dates.empty:
        raise ValueError(f"No test dates after initial_train_end={initial_train_end!r}")

    predictions: list[pd.DataFrame] = []

    for year in sorted(test_dates.year.unique()):
        year_start = pd.Timestamp(f"{year}-01-01")
        train = data[data.index.get_level_values("date") < year_start]

        if len(train) < 50:
            logger.warning("Year %d: only %d training rows — skipping", year, len(train))
            continue

        model = lgb.LGBMRegressor(**params)
        model.fit(train[FEATURE_COLS], train["fwd_return"])
        logger.info("Year %d: trained on %d obs", year, len(train))

        year_test = data[data.index.get_level_values("date").year == year]
        if year_test.empty:
            continue

        scores = model.predict(year_test[FEATURE_COLS])
        predictions.append(pd.DataFrame({"score": scores}, index=year_test.index))

    if not predictions:
        raise RuntimeError("No predictions produced — check date range and initial_train_end")

    return pd.concat(predictions).sort_index()
