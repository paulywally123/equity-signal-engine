"""Equal-weight rebalance of an Alpaca paper account to the model's top-N
ranked long leg (reads data/processed/live_signal.csv). RUN ON YOUR MACHINE.

    python -m src.trading.rebalance --config config/config.yaml            # dry run
    python -m src.trading.rebalance --config config/config.yaml --execute  # place orders

Dry run by default -- always inspect the printed order plan before passing
--execute.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.trading.alpaca_client import get_trading_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Skip adjustments smaller than this fraction of the target position size --
# avoids churning tiny rebalancing trades on noise.
REBALANCE_THRESHOLD_FRAC = 0.05


def target_portfolio(processed_dir: Path, top_n: int) -> list[str]:
    signal = pd.read_csv(processed_dir / "live_signal.csv", index_col=0)
    signal = signal.sort_values("rank")
    return signal.index[:top_n].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--execute", action="store_true",
                     help="Actually submit orders (default: dry run / print plan only)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])

    targets = target_portfolio(processed_dir, args.top_n)
    logger.info("Target portfolio (%d names): %s", len(targets), ", ".join(targets))

    client = get_trading_client()
    account = client.get_account()
    equity = float(account.equity)
    target_value = equity / args.top_n
    logger.info("Account equity: $%.2f  |  target per position: $%.2f", equity, target_value)

    current = {p.symbol: float(p.market_value) for p in client.get_all_positions()}

    orders: list[tuple[str, str, float]] = []  # (symbol, side, notional)

    # Close positions that fell out of the target list entirely
    for symbol, value in current.items():
        if symbol not in targets:
            orders.append((symbol, "CLOSE", value))

    # Open new positions / top up or trim existing ones toward target weight
    for symbol in targets:
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
