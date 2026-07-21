# Weekly Equity Momentum Rotation

Paper-trading research. Descriptive, not predictive. Nothing here is a trade
list: a leader is a reason to look at a name, not a signal to buy.

A live paper-trading arm now trades this ranking mechanically on Alpaca
account 3 as an honest test of it -- see `rotation_trading_protocol.md`.

## Signal

Cross-sectional momentum ranked on the **12-1 month total return**:

    mom_12_1 = close[t-21] / close[t-252] - 1

The return from roughly 12 months ago to roughly 1 month ago, skipping the most
recent month because short-horizon returns mean-revert. This is the standard
academic momentum factor (Jegadeesh-Titman 1993, the UMD factor). The ranking is
recomputed weekly even though the lookback is long.

Computed for context only, never ranked on:

- 3-month return: `close[t] / close[t-63] - 1`
- 1-month return: `close[t] / close[t-21] - 1` (the excluded window, shown so a
  cracking leader is visible)
- price vs 200-day MA (trend confirmation; long-side momentum is historically
  more reliable above it)

## Two layers

1. **Layer 1**: rank the sector and theme ETFs in `universe.md` by 12-1
   momentum. SPY and RSP are benchmarks for the market read, not ranked.
2. **Layer 2**: take the top 3 ranked groups, rank each group's constituents by
   12-1 momentum, surface the top 5 per group.

## Week-over-week

Each run saves its full ranking to `history/YYYY-MM-DD_ranks.json`. The next run
diffs against the most recent prior file and reports movers: rank changes of 2+
places, and any group crossing its 200-day MA in either direction.

## Data

yfinance, auto-adjusted daily closes, 2 years of history, batch download.
Read-only. No POST/PUT/DELETE/PATCH to any broker API, ever.

## Running it

    cd weekly_rotation
    python scripts/weekly_rotation.py            # real data (needs internet)
    python scripts/weekly_rotation.py --synthetic  # offline pipeline test

Output: `reports/YYYY-MM-DD_rotation.md` and `history/YYYY-MM-DD_ranks.json`.

Dependencies: `pip install yfinance pandas numpy`

## Known limitations (deliberate, documented)

- **Static universe.** Constituent lists in `universe.md` are a snapshot, so
  Layer 2 has survivorship and point-in-time drift. Fine for a watchlist
  generator, not fine for a backtest. Do not backtest on this universe.
- **Adjusted closes restate history.** yfinance auto-adjustment re-scales past
  prices on every dividend, so a rerun of an old date can disagree slightly
  with the saved JSON from that date. The saved JSON is the record.
- **Weekly recompute of a 12-month signal is slow-moving.** Most week-over-week
  ETF rank changes are endpoint noise, not information. The 200-day MA
  crossings and the 1-month column are where a genuine break shows up first.

## Rules (non-negotiable)

- Not a trade list. No "will", no "guaranteed", no "sure thing". Rankings are
  probabilistic.
- No invented numbers. Every figure comes from the data. If data is missing or
  stale for a ticker, exclude it and say so in the report.
- Do not modify this file or `universe.md` without an explicit ask.
