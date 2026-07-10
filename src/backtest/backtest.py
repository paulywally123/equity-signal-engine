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


def vol_target_exposure(
    returns: pd.DataFrame,
    target_vol: float,
    window: int = 20,
    max_exposure: float = 1.0,
) -> pd.Series:
    """Continuous exposure scale in [0, max_exposure] from trailing realized vol.

    Unlike a binary regime filter (fully in or fully flat), this shrinks
    position size smoothly as the equal-weight index's trailing realized
    volatility rises above `target_vol`, and holds `max_exposure` (capped at
    1.0 by default -- no leverage) when realized vol is at or below target.
    The portfolio is never fully out, so it can't miss a sharp post-selloff
    recovery entirely the way a binary in/out filter can -- it's just
    smaller sized during the stressed period.

    Sampled from the most recent prior bar at each use site (see
    `run_backtest`'s `exposure` param), so there is no look-ahead.
    """
    daily = returns.mean(axis=1)
    realized_vol = daily.rolling(window).std() * np.sqrt(252)
    exposure = (target_vol / realized_vol).clip(upper=max_exposure).fillna(max_exposure)
    exposure.index = pd.to_datetime(exposure.index)
    return exposure


def trend_exposure(
    returns: pd.DataFrame,
    window: int,
    floor: float = 0.0,
) -> pd.Series:
    """Two-level exposure from a trend filter: 1.0 above the SMA, `floor` below it.

    Generalises the binary 200d regime filter (floor=0.0, fully out) into a
    tunable partial-exposure version (floor=0.5, half-sized rather than
    flat) so the CAGR given up for a given amount of drawdown reduction is
    an explicit, chosen dial rather than an all-or-nothing switch.
    """
    idx = equal_weight_index(returns)
    above = idx > idx.rolling(window).mean()
    exposure = above.astype(float).where(above, floor)
    exposure.index = pd.to_datetime(exposure.index)
    return exposure


def drawdown_trigger_exposure(
    returns: pd.DataFrame,
    lookback: int,
    threshold: float,
    floor: float = 0.0,
) -> pd.Series:
    """Fast tail-only trigger: `floor` exposure only when the equal-weight
    index has fallen more than `threshold` from its trailing `lookback`-day
    peak; 1.0 otherwise.

    A genuinely different design from `trend_exposure`'s 200-250 day SMA:
    that filter is slow to re-enter after a rally (a long moving average
    takes months to turn back up), which is exactly what cost it the COVID
    holdout (see drawdown_cap_holdout.py). This uses the same short lookback
    window in both directions, so it re-arms as soon as the market has
    actually recovered rather than waiting on a lagging average -- intended
    to fire only on genuinely sharp drawdowns (tail risk) rather than
    ordinary trend dips, while not being structurally slow to un-fire.
    """
    idx = equal_weight_index(returns)
    trailing_peak = idx.rolling(lookback, min_periods=1).max()
    dd = idx / trailing_peak - 1.0
    exposure = pd.Series(1.0, index=idx.index)
    exposure[dd < -threshold] = floor
    exposure.index = pd.to_datetime(exposure.index)
    return exposure


def level_trigger_exposure(
    indicator: pd.Series,
    threshold: float,
    floor: float = 0.0,
) -> pd.Series:
    """`floor` exposure when an external indicator (e.g. VIX, credit spread)
    is above `threshold`; 1.0 otherwise.

    Unlike `trend_exposure` / `drawdown_trigger_exposure`, `indicator` is not
    derived from this strategy's own price/return history at all -- it's an
    independent, real-time-observable market signal, sampled the same
    no-look-ahead way (`run_backtest`'s `exposure` param looks up the most
    recent prior bar).
    """
    exposure = pd.Series(1.0, index=indicator.index)
    exposure[indicator > threshold] = floor
    exposure.index = pd.to_datetime(exposure.index)
    return exposure


def run_backtest(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    costs_bps: float = 10,
    top_q: float = 0.2,
    long_only: bool = False,
    regime: pd.Series | None = None,
    sectors: pd.Series | None = None,
    exposure: pd.Series | None = None,
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
    exposure    : optional float Series indexed by date, in [0, 1]; scales
                  position size continuously instead of the binary in/out
                  of `regime` (see `vol_target_exposure`). Sampled from the
                  most recent prior bar, no look-ahead. Composable with
                  `regime` -- both are applied if both are given.

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

        # Vol-target exposure: same no-look-ahead lookup, continuous scale
        exp_scale = 1.0
        if exposure is not None:
            past_exp = exposure[exposure.index < date]
            if len(past_exp):
                exp_scale = float(past_exp.iloc[-1])

        if in_regime:
            scores  = predictions.xs(date, level="date")["score"]
            weights = _construct_weights(scores, top_q, long_only=long_only,
                                         sectors=sectors) * exp_scale
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
            "exposure":     exp_scale,
            "turnover":     turnover,
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
    metrics = {
        "ann_return":   round(ann_ret, 4),
        "ann_vol":      round(ann_vol, 4),
        "sharpe":       round(sharpe,  3),
        "max_drawdown": round(max_dd,  4),
        "cagr":         round(cagr,    4),
        "hit_rate":     round((port_returns["gross_return"] > 0).mean(), 4),
        "n_periods":    len(r),
    }
    if "turnover" in port_returns.columns:
        avg_turnover = port_returns["turnover"].mean()
        metrics["avg_turnover"]    = round(avg_turnover, 4)
        metrics["annual_turnover"] = round(avg_turnover * periods_per_year, 4)
    return metrics


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
