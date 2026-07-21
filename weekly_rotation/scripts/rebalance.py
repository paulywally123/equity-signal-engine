"""Group-weighted rebalance of an Alpaca paper account to the rotation
report's current picks (top groups selected with hysteresis, from the
latest history/*_ranks.json). Each selected group gets an equal share of
the book; within a group that share splits across its surviving names (or
goes whole into one ETF for a terminal/hold-the-sleeve group). RUN ON YOUR
MACHINE.

This is a live-executed test of a tool (weekly_rotation.py) that is
documented as descriptive-only research. It never touches a broker itself;
this separate script is the only thing here that does.

    python scripts/rebalance.py                # dry run, account 3
    python scripts/rebalance.py --execute       # place orders
    python scripts/rebalance.py --account 4 --execute

Dry run by default -- always inspect the printed order plan before passing
--execute.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "history"
DATA_DIR = ROOT / "data"
GATE_LOG = DATA_DIR / "gate_log.txt"
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.trading.alpaca_client import get_trading_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Skip adjustments smaller than this fraction of the target position size --
# avoids churning tiny rebalancing trades on noise.
REBALANCE_THRESHOLD_FRAC = 0.05

DEFAULT_ACCOUNT = 3


def latest_snapshot() -> dict:
    files = sorted(HISTORY_DIR.glob("*_ranks.json"))
    if not files:
        sys.exit(
            "No history/*_ranks.json found -- run scripts/weekly_rotation.py "
            "first to generate a snapshot."
        )
    return json.loads(files[-1].read_text())


def _log_gate_decision(msg: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(GATE_LOG, "a") as f:
        f.write(f"\n{datetime.now()} rebalance.py {msg}\n")


def check_gate(snapshot: dict, execute: bool) -> dict[str, float]:
    """Return {ticker: weight_fraction}, honoring the pre-trade gate verdict
    and group-weighted allocation written by weekly_rotation.py. Fails
    closed: refuses --execute against a blocked, missing, or malformed gate
    verdict rather than falling back to an unweighted or terminal-blind
    target."""
    gate = snapshot.get("gate")
    weights = snapshot.get("target_weights")
    if gate is None or weights is None:
        _log_gate_decision(
            f"snapshot={snapshot['date']} execute={execute} "
            "decision=HARD_STOP_MISSING_GATE_OR_WEIGHTS"
        )
        sys.exit(
            f"Refusing to trade: snapshot {snapshot['date']} is missing "
            "'gate' and/or 'target_weights'. This should be impossible for "
            "a snapshot written by the current weekly_rotation.py -- "
            "something upstream changed. Not falling back to an "
            "unweighted or terminal-blind target. Investigate before "
            "re-running."
        )

    for w in gate.get("warnings", []):
        logger.warning("  gate warning: %s", w)
    for t, why in gate.get("dropped", {}).items():
        logger.info("  gate dropped %s: %s", t, why)

    if not gate.get("ok", False):
        logger.error("Pre-trade gate BLOCKED snapshot %s:", snapshot["date"])
        for b in gate.get("blocking", []):
            logger.error("  - %s", b)
        if execute:
            _log_gate_decision(
                f"snapshot={snapshot['date']} execute=True "
                f"decision=REFUSED blocking={gate.get('blocking', [])}"
            )
            sys.exit(
                "Refusing to execute: pre-trade gate blocked this snapshot. "
                "Re-run scripts/weekly_rotation.py once the underlying data "
                "issue is fixed, or investigate universe.md."
            )
        logger.warning(
            "Dry run only -- would refuse --execute against this snapshot "
            "for the reasons above."
        )
        _log_gate_decision(
            f"snapshot={snapshot['date']} execute=False "
            f"decision=BLOCKED_DRY_RUN blocking={gate.get('blocking', [])}"
        )
    else:
        _log_gate_decision(
            f"snapshot={snapshot['date']} execute={execute} decision=PASS"
        )

    return weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=int, default=DEFAULT_ACCOUNT,
                     help="Alpaca credential pair to use (default: 3)")
    ap.add_argument("--execute", action="store_true",
                     help="Actually submit orders (default: dry run / print plan only)")
    args = ap.parse_args()

    snapshot = latest_snapshot()
    weights = check_gate(snapshot, args.execute)
    if not weights:
        logger.info("No target names in the latest snapshot -- nothing to do.")
        return

    logger.info("Snapshot date: %s", snapshot["date"])
    plan_desc = ", ".join(
        f"{t}({w:.1%})" for t, w in sorted(weights.items(), key=lambda kv: -kv[1])
    )
    logger.info("Account: %d  |  Target portfolio (%d names, group-weighted): %s",
                args.account, len(weights), plan_desc)

    client = get_trading_client(account=args.account)
    account_info = client.get_account()
    equity = float(account_info.equity)
    target_values = {t: w * equity for t, w in weights.items()}
    logger.info("Account equity: $%.2f", equity)

    current = {p.symbol: float(p.market_value) for p in client.get_all_positions()}

    orders: list[tuple[str, str, float]] = []  # (symbol, side, notional)

    # Close positions that fell out of the target list entirely
    for symbol, value in current.items():
        if symbol not in weights:
            orders.append((symbol, "CLOSE", value))

    # Open new positions / top up or trim existing ones toward target weight
    for symbol, target_value in target_values.items():
        held = current.get(symbol, 0.0)
        diff = target_value - held
        if abs(diff) < target_value * REBALANCE_THRESHOLD_FRAC:
            continue
        side = "BUY" if diff > 0 else "SELL"
        orders.append((symbol, side, abs(diff)))

    if not orders:
        logger.info("No rebalancing needed -- portfolio already matches target.")
        return

    logger.info("Order plan (%d orders):", len(orders))
    for symbol, side, notional in orders:
        logger.info("  %-6s %-8s $%.2f", symbol, side, notional)

    if not args.execute:
        logger.info("Dry run -- no orders submitted. Re-run with --execute to place these.")
        return

    for symbol, side, notional in orders:
        try:
            if side == "CLOSE":
                client.close_position(symbol)
            else:
                order = MarketOrderRequest(
                    symbol=symbol,
                    notional=round(notional, 2),
                    side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                client.submit_order(order)
            logger.info("  Submitted: %-6s %-8s $%.2f", symbol, side, notional)
        except Exception as exc:
            logger.error("  FAILED: %-6s %-8s $%.2f -- %s", symbol, side, notional, exc)


if __name__ == "__main__":
    main()
