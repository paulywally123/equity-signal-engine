# Rotation Live Paper-Trading Protocol

`rotation_strategy.md` describes a descriptive research tool: rankings are
probabilistic, "a leader is a reason to look at a name, not a signal to buy."
This document is the deliberate exception -- it defines a live-executed test
of that ranking, the same way `equity-signal-engine`'s model and momentum
strategies are tested live in accounts 1 and 2. Running it does not change
what the ranking means; it's an honest experiment to see whether trading it
mechanically would have made money, nothing more.

## Account

Alpaca **paper** account 3 (`ALPACA_API_KEY_3` / `ALPACA_SECRET_KEY_3` in
`.env`, read via `src.trading.alpaca_client.get_trading_client(account=3)`).
Separate from accounts 1 (model) and 2 (momentum) so this strategy's P&L
never mixes with that comparison.

## Selection rule

Flat equal weight across the union of `layer2` tickers in the latest
`history/*_ranks.json` -- i.e. the top 5 names in each of the top 3 ranked
groups, up to 15 tickers, each ~1/N of account equity. Same equal-weight
rebalance mechanic as `src/trading/rebalance.py` (close positions that
dropped off the list, buy/sell toward target weight, skip moves under 5% of
target size).

## Cadence

Weekly, gated to ~5 days via `data/last_rebalance.txt` so re-running the
same week is a no-op. Scheduled via Windows Task Scheduler, task name
`EquityRotationWeekly`, **Monday 8:45am Central time** (the machine's local
zone) -- one hour behind ET, so this fires at 9:45am ET, shortly after the
market open. Re-adjust with `schtasks /change /tn EquityRotationWeekly /st HH:MM`
if the machine's timezone ever changes.

Task Scheduler default: "run only when user is logged on" (no password was
stored). The machine needs to be logged in at trigger time for this to fire.

Registered command:
```
python weekly_rotation/scripts/build_rotation_rebalance.py --execute
```
Dry run (omit `--execute`) any time to see the plan without trading:
```
python scripts/weekly_rotation.py
python scripts/rebalance.py
```

## Checking P&L

Pure observation, no side effects:
```python
from src.trading.alpaca_client import get_trading_client
client = get_trading_client(account=3)
client.get_account()          # equity, cash, etc.
client.get_all_positions()    # current holdings
```

## Known limitations

Inherits everything in `rotation_strategy.md`'s "Known limitations" section
(static universe, adjusted-close restatement, slow-moving weekly signal).
Additionally: 15 names on a weekly cadence is higher turnover than the
20-name/28-day cadence used by accounts 1 and 2 -- expect more trading
activity and more sensitivity to the 5% rebalance threshold.

## Credentials

Never paste Alpaca keys into chat or commit them. Add
`ALPACA_API_KEY_3`/`ALPACA_SECRET_KEY_3` to the local `.env` directly.
