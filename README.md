# Equity Signal Engine

Cross-sectional monthly long-only equity strategy on a point-in-time S&P 500
universe with an overfitting-resistant walk-forward backtest.

## Results (as of latest run)

| Metric | Value |
|---|---|
| Sharpe ratio | **0.705** |
| CAGR | **10.6%** |
| Ann. volatility | 16.2% |
| Max drawdown | -29.8% |
| Hit rate | 70.4% |
| IC (model vs realized) | +0.0047 (t = 0.46) |
| Excess return vs equal-weight BM | +1.92% / yr |
| Backtest period | 2013 – 2026 (169 periods) |
| Universe | Full point-in-time S&P 500 (607 current members, 814 ever-members) |

Strategy: long top-20% by model score, 20-day rebalance, no market-timing
overlay, 10 bps per-side transaction costs. **No short leg** — the long-short
design was the original intent (isolating the cross-sectional signal by
canceling market beta), but shorting costs and hard-to-borrow constraints on
smaller/delisted names aren't modeled in this backtest's data, so long-only is
the more realistic framing given that limitation, at the cost of not cleanly
separating stock-selection skill from market beta. See **Decomposition**
below for how much of the return is attributable to each.

## Decomposition: how much is the model actually contributing?

A long-only, trend-filtered strategy's Sharpe conflates three things: equity
beta over a strong bull period, any market-timing overlay, and the
cross-sectional model. This breaks it apart:

| Configuration | Sharpe | Ann. return | Vol | Max DD | CAGR |
|---|---|---|---|---|---|
| (a) Equal-weight buy-and-hold, no regime, no model | 0.570 | 9.83% | 17.25% | -37.8% | 8.60% |
| (b) Equal-weight + 200d regime filter, no model | 0.472 | 5.35% | 11.33% | -22.5% | 4.80% |
| **(c) Model portfolio, no regime filter (current default)** | **0.705** | **11.43%** | 16.22% | -29.8% | **10.57%** |
| (d) Model + 200d regime filter | 0.519 | 5.94% | 11.46% | -23.6% | 5.41% |

Two things this shows plainly:

1. **The regime filter hurts, in both cases it's applied** — (a)→(b) and
   (c)→(d) both lose Sharpe. It was originally intended as a risk-reducing
   market-timing overlay; once built on a correct equal-weight *return* index
   (see below), it doesn't earn its keep. It's not used by default.
2. **The model has a real, if modest, edge over passive** — (c) beats (a) by
   both Sharpe and absolute return. That's the actual claim this project can
   support: a weak but genuine cross-sectional signal, not "trend-filtered
   index fund with extra steps."

**A bug this decomposition surfaced**: the original regime filter averaged
raw *price levels* across the universe (`prices.mean(axis=1)`) rather than
compounding daily *returns* — a price-weighted index, not equal-weighted (a
$500 stock moved it 100x more than a $5 stock for the same % move). Fixed in
`equal_weight_index()` (`src/backtest/backtest.py`). Once fixed, the
200-day window — which happened to sit at the top of its own sensitivity
sweep under the old (buggy) index — no longer looks favorable at any window;
see **Sensitivity summary**.

## Phases completed

- [x] **2a** Point-in-time S&P 500 universe (Wikipedia change log, backward reconstruction)
- [x] **2b** Price ingestion + coverage audit (`src/data/prices.py`)
- [x] **2c** Cleaning & membership masking (`src/data/clean.py`)
- [x] **3a** Cross-sectional features — momentum, volatility, RSI, dollar-vol (`src/features/`)
- [x] **3b** Forward return labels with cross-sectional rank normalisation (`src/labels/`)
- [x] **3c** Walk-forward LightGBM model — annual re-fit, no hyperparameter tuning on test data (`src/models/`)
- [x] **3d** Long-only backtest + equity curve report (`src/backtest/`, `src/report/`)
- [x] **4a** SEC EDGAR fundamental data — point-in-time observations, 2007–2026 (`src/data/edgar.py`)
- [x] **4b** Feature pruning; sector-neutral construction (available, not default — costs ~0.54 Sharpe, see Known Limitations on selection methodology)
- [x] **5a** Annual attribution tearsheet + sector exposure + worst/best periods (`src/report/attribution.py`)
- [x] **5b** Sensitivity analysis — top_q, cost assumptions, regime window (`src/backtest/sensitivity.py`)
- [x] **5c** Live signal generation — current ranked portfolio (`src/signal/`)
- [x] **5d** Documentation (this file)
- [x] **6a** Fixed EDGAR filing-date bug — median point-in-time lag was 406 days (89% > 180 days) due to later filings' comparative-period restatements overwriting the original filing date; now 36 days median
- [x] **6b** Trimmed to 5 features with real IC, regularized the LightGBM model — reduced overfitting to the dev-mode ticker subset (train/test gap 2.7x → 1.2x on a matched liquidity slice)
- [x] **6c** Fixed regime index (price-weighted → equal-weight return) and dropped the regime filter given the decomposition above
- [x] **6d** Fixed live signal generation always lagging ~1 rebalance cycle behind reality regardless of data freshness (`predict_latest()` in `src/models/model.py`)
- [x] **6e** Alpaca paper-trading integration (`src/trading/`) — dry-run by default, equal-weight rebalance to the model's top-N

## Feature set (5 features, all rank-normalised cross-sectionally)

Started at 8; `mom_3`, `mom_1`, `rsi_14` were excluded from training from the
start (negative individual IC). `mom_6_1`, `high_52w`, `vol_21` were later
dropped after measuring ~zero individual IC — see **Known Limitations** for
why that specific decision has a methodology caveat of its own.

| Feature | Type | IC | t-stat |
|---|---|---|---|
| `mom_12_1` | Price — 12-1 month momentum | +0.012 | 0.85 |
| `dollar_vol_60` | Price — 60-day avg dollar volume | +0.006 | 0.85 |
| `gross_prof` | Fundamental — gross profit / assets (Novy-Marx) | +0.002 | 0.21 |
| `roe` | Fundamental — net income / equity | +0.011 | 1.34 |
| `ep_ratio` | Fundamental — earnings / market cap | -0.004 | -0.44 |

None of these individually clears a conventional significance bar, and with
5+ features tested, `roe`'s t=1.34 shouldn't be read as a standout finding —
it's what you'd expect to see somewhere in this set by chance alone. The
model's IC (t=0.46, combining all features nonlinearly via LightGBM) is the
number that actually matters, and it's modest.

## Sensitivity summary

| Dimension | Sharpe range | Baseline |
|---|---|---|
| top_q (0.10 – 0.30) | 0.645 – 0.738 | 0.705 (top_q=0.20) |
| costs_bps (5 – 20) | 0.670 – 0.723 | 0.705 (10 bps) |
| regime window (100 – 250d, vs. no filter) | 0.413 – 0.591 | **0.705 (no filter — best of all options)** |

Unlike the pre-fix version of this table, the baseline (no regime filter) is
no longer sitting at a suspicious optimum within its own sweep — adding *any*
regime window makes things worse, monotonically. top_q and cost sensitivity
are both smooth, unremarkable curves around the baseline.

## Quickstart

```bash
pip install -r requirements.txt
pytest                                        # 73 tests

# Run the full pipeline (requires network access):
py -m src.data.build_universe
py -m src.data.build_prices
py -m src.data.build_clean
py -m src.data.build_fundamentals            # ~10 min, SEC EDGAR
py -m src.features.build_features
py -m src.labels.build_labels
py -m src.models.build_model
py -m src.backtest.build_backtest            # prints attribution tearsheet
py -m src.backtest.sensitivity               # parameter sensitivity table
py -m src.signal.build_live_signal           # today's portfolio holdings

# Paper-trade the live signal via Alpaca (dry run by default):
py -m src.trading.rebalance                  # prints order plan only
py -m src.trading.rebalance --execute        # submits orders
```

## Key design decisions

**Point-in-time universe**: membership reconstructed backward from Wikipedia's
S&P 500 change log. Prices outside a ticker's membership window are masked to
NaN, preventing survivorship bias.

**Walk-forward evaluation**: model re-fits annually on an expanding training
window. No hyperparameters are tuned on test data — the biggest silent risk in
rolling-window backtests. Known gap: the last ~20 trading days of each
training window have labels that resolve at roughly the same time as the
first test predictions of the following year, i.e. close to zero embargo at
each of the 13 annual boundaries. Not yet purged; likely small effect (affects
the margin of ~13/169 periods) but not yet quantified.

**Ranked training labels**: forward returns are converted to cross-sectional
percentile ranks [0, 1] for model training, giving a stable target distribution
across volatile and quiet markets. Raw log returns are used for backtest P&L.

**No regime/market-timing filter**: tested a 200-day SMA cash overlay; once
built on a correct equal-weight return index it reduced Sharpe in every
configuration (see Decomposition). Not used by default.

**EDGAR fundamentals**: quarterly GAAP data fetched from the public SEC EDGAR
XBRL API (no key required). Filing date used as the availability date, taking
the *earliest* filing that reports each period (a later 10-K's five-year
comparative table can otherwise overwrite the true filing date with a much
later one — this was a real bug, see Phase 6a). The API only returns a filing
*date*, not time-of-day, so same-day after-hours filings aren't pushed to the
next trading day; given the 20-trading-day rebalance grid this has low
practical impact, but it's a known, undodged limitation of the data source.

**Dev mode** (`universe.mode: dev`): restricts each date's universe to the
top-100 tickers by trailing dollar volume, for fast iteration. Its metrics
run meaningfully hotter than full-universe (e.g. IC t-stat ~2 vs ~0.5) —
diagnosed this as partly a real liquidity effect and partly overfitting to a
small, persistent, recurring set of the same ~100-150 mega-cap names across
13 years. Full mode (the default) is the honest result to report; dev mode is
for fast local iteration only, not a claim about signal quality.

## Known limitations (not yet addressed)

- **Walk-forward embargo gap** at annual refit boundaries — see above.
- **No formal null baseline.** With IC this weak, a permutation/Monte Carlo
  null (shuffle labels, rerun, see where the actual IC falls) would give a
  real p-value instead of relying on a t-stat that assumes independent
  periods. Not yet built.
- **No momentum-only baseline** (e.g. a plain 12-1 sort) to compare the full
  model against a simple, well-known factor.
- **Feature pruning was in-sample.** Both the original 8→5 trim this session
  and the earlier Phase 4b pruning were decided using IC measured over the
  *same* 2013–2026 span the backtest reports performance over — textbook
  in-sample feature selection. A held-out window for feature decisions would
  fix this; not yet done.
- **Dev-mode overfitting hypothesis untested directly.** The liquidity-based
  diagnosis above is circumstantial. A direct test — rerun dev mode on a
  *different* random 100-ticker subset, not the top-100-by-liquidity one —
  would confirm whether the effect is genuine size/liquidity signal or
  fitting to that specific subset's quirks. Not yet run.

## Survivorship bias handling

1. Change-log completeness trusted only back to `universe.floor_date` (2010-01-01).
2. Ticker renames may appear as spurious remove/add pairs — inspect coverage audit.
3. Some delisted tickers have no retrievable price history (177 empty of 814 tickers).
