"""Long-short portfolio backtest with transaction costs and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight_index(returns: pd.DataFrame) -> pd.Series:
    """Compound equal-weight index level from daily log returns.

    Cross-sectional mean log return each day (NaN-skipping, so tickers
    outside their point-in-time membership window don't count), then
    compounded into a level series. Averaging raw price levels instead
    (as opposed to returns) would give every ticker influence proportional
    to its nominal share price rather than an equal weight -- a $500 stock
    would move the "index" 100x more than a $5 stock for the same percentage
    move, the classic price-weighted-index distortion.
    """
    daily = returns.mean(axis=1)
    return np.exp(daily.cumsum())


def _construct_weights(
    scores: pd.Series,
    top_q: float,
    long_only: bool = False,
    sectors: pd.Series | None = None,
) -> pd.Series:
    """Equal-weight long top quintile, optionally sector-neutralised.

    When `sectors` is provided (ticker → sector string) the selection is done
    within each sector: top `top_q` fraction of each sector goes long, so no
    single sector can dominate the portfolio.  Tickers with no sector label
    are treated as their own group.
    """
    weights = pd.Series(0.0, index=scores.index)

    if sectors is not None:
        sector_map = sectors.reindex(scores.index).fillna("Unknown")
        groups = sector_map.groupby(sector_map).groups
        for sec, members in groups.items():
            sec_scores = scores[members]
            n = len(sec_scores)
            if n == 0:
                continue
            n_long = max(1, int(round(n * top_q)))
            ranked  = sec_scores.rank(ascending=True)
            long_idx = ranked[ranked > (n - n_long)].index
            weights[long_idx] = 1.0 / len(long_idx)
            if not long_only:
                short_idx = ranked[ranked <= n_long].index
                weights[short_idx] = -1.0 / len(short_idx)
        # Re-normalise so total long weight = 1
        long_sum = weights[weights > 0].sum()
        if long_sum > 0:
            weights[weights > 0] /= long_sum
        if not long_only:
            short_sum = weights[weights < 0].abs().sum()
            if short_sum > 0:
                weights[weights < 0] /= short_sum
    else:
        n = len(scores)
        n_leg = max(1, int(round(n * top_q)))
        ranked = scores.rank(ascending=True)
        weights[ranked > (n - n_leg)] = 1.0 / n_leg
        if not long_only:
            weights[ranked <= n_leg] = -1.0 / n_leg

    return weights


def run_backtest(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    costs_bps: float = 10,
    top_q: float = 0.2,
    long_only: bool = False,
    regime: pd.Series | None = None,
    sectors: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute per-period gross and net returns.

    Parameters
    ----------
    predictions : (date, ticker) MultiIndex, column 'score'
    labels      : (date, ticker) MultiIndex, column 'fwd_return'
    costs_bps   : one-way transaction cost in basis points
    top_q       : fraction of universe in each leg
    regime      : optional boolean Series indexed by date; True = invest,
                  False = flat (cash).  Regime is sampled from the most
                  recent prior bar so there is no look-ahead.

    Returns
    -------
    DataFrame indexed by date with: gross_return, cost, net_return,
    long_return, short_return.
    """
    pred_dates  = predictions.index.get_level_values("date").unique().sort_values()
    label_dates = set(labels.index.get_level_values("date"))
    prev_weights: pd.Series | None = None
    results = []

    for date in pred_dates:
        # Market regime: look up most recent signal strictly before this date
        in_regime = True
        if regime is not None:
            past = regime[regime.index < date]
            if len(past):
                in_regime = bool(past.iloc[-1])

        if in_regime:
            scores  = predictions.xs(date, level="date")["score"]
            weights = _construct_weights(scores, top_q, long_only=long_only,
                                         sectors=sectors)
        else:
            # Flat / cash — zero weights, pay liquidation cost if we were invested
            scores  = predictions.xs(date, level="date")["score"]
            weights = pd.Series(0.0, index=scores.index)

        if date in label_dates:
            fwd = labels.xs(date, level="date")["fwd_return"]
            common = weights.index.intersection(fwd.index)
            w = weights[common]
            f = fwd[common]
            gross      = (w * f).sum()
            long_ret   = (w[w > 0] * f[w > 0]).sum()
            short_ret  = (w[w < 0] * f[w < 0]).sum()
        else:
            gross = long_ret = short_ret = np.nan

        if prev_weights is not None:
            prev     = prev_weights.reindex(weights.index, fill_value=0.0)
            turnover = (weights - prev).abs().sum()
        else:
            turnover = weights.abs().sum()
        cost = turnover * costs_bps / 10_000

        results.append({
            "date":         date,
            "gross_return": gross,
            "cost":         cost,
            "net_return":   gross - cost if not np.isnan(gross) else np.nan,
            "long_return":  long_ret,
            "short_return": short_ret,
            "in_regime":    in_regime,
        })
        prev_weights = weights

    return pd.DataFrame(results).set_index("date").dropna(subset=["net_return"])


def compute_metrics(port_returns: pd.DataFrame, periods_per_year: int = 52) -> dict:
    """Annualized performance metrics from a column of period net returns."""
    r = port_returns["net_return"]
    ann_ret = r.mean() * periods_per_year
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    cagr    = cum.iloc[-1] ** (periods_per_year / len(r)) - 1
    return {
        "ann_return":   round(ann_ret, 4),
        "ann_vol":      round(ann_vol, 4),
        "sharpe":       round(sharpe,  3),
        "max_drawdown": round(max_dd,  4),
        "cagr":         round(cagr,    4),
        "hit_rate":     round((port_returns["gross_return"] > 0).mean(), 4),
        "n_periods":    len(r),
    }


def information_coefficient(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.Series:
    """Spearman rank correlation between predicted score and realized return."""
    label_dates = set(labels.index.get_level_values("date"))
    ics: dict[pd.Timestamp, float] = {}

    for date in predictions.index.get_level_values("date").unique():
        if date not in label_dates:
            continue
        scores = predictions.xs(date, level="date")["score"]
        fwd    = labels.xs(date, level="date")["fwd_return"].dropna()
        common = scores.index.intersection(fwd.index)
        if len(common) < 10:
            continue
        ics[date] = scores[common].corr(fwd[common], method="spearman")

    return pd.Series(ics, name="ic").sort_index()


def feature_ic(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Spearman IC of each raw feature against realized returns, per date.

    Returns a DataFrame indexed by date, one column per feature.
    Useful for diagnosing which signals actually correlate with future returns.
    """
    label_dates = set(labels.index.get_level_values("date"))
    feat_dates  = features.index.get_level_values("date").unique()
    records: list[dict] = []

    for date in feat_dates:
        if date not in label_dates:
            continue
        feats = features.xs(date, level="date")
        fwd   = labels.xs(date, level="date")["fwd_return"].dropna()
        common = feats.index.intersection(fwd.index)
        if len(common) < 10:
            continue
        row = {"date": date}
        for col in feats.columns:
            row[col] = feats.loc[common, col].corr(fwd[common], method="spearman")
        records.append(row)

    return pd.DataFrame(records).set_index("date").sort_index()
