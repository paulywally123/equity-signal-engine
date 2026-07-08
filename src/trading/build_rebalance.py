"""Automated rebalance cycle: refresh prices/features/model/live-signal, then
rebalance the Alpaca paper account to match. RUN ON YOUR MACHINE (or on a
schedule -- see cron setup).

Gates itself to actually rebalance only ~every 20 trading days via a state
file (data/processed/last_rebalance.txt), so it's safe to invoke more often
than that (e.g. from a weekly cron) without over-trading.

Does NOT refresh SEC EDGAR fundamentals (quarterly data doesn't change
week to week and re-fetching hits the SEC API for 600+ tickers) -- run
`python -m src.data.build_fundamentals --force-refresh` separately,
periodically, if you want fresher fundamentals.

    python -m src.trading.build_rebalance --config config/config.yaml            # dry run
    python -m src.trading.build_rebalance --config config/config.yaml --execute  # place orders
    python -m src.trading.build_rebalance --config config/config.yaml --force    # ignore cadence gate
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REBALANCE_EVERY_DAYS = 28  # ~20 trading days
STATE_FILE = "last_rebalance.txt"


def _run(module: str, config: str) -> None:
    logger.info("Running %s ...", module)
    subprocess.run([sys.executable, "-m", module, "--config", config], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--execute", action="store_true",
                    help="Actually submit orders (default: dry run / print plan only)")
    ap.add_argument("--force", action="store_true",
                    help="Rebalance even if the cadence gate says it isn't due yet")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    state_path = processed_dir / STATE_FILE

    if state_path.exists() and not args.force:
        last = datetime.fromisoformat(state_path.read_text().strip())
        elapsed = (datetime.now() - last).days
        if elapsed < REBALANCE_EVERY_DAYS:
            logger.info(
                "Last rebalance %d days ago (< %d) -- skipping. Use --force to override.",
                elapsed, REBALANCE_EVERY_DAYS,
            )
            return

    _run("src.data.build_prices", args.config)
    _run("src.data.build_clean", args.config)
    _run("src.features.build_features", args.config)
    _run("src.models.build_model", args.config)
    _run("src.signal.build_live_signal", args.config)

    rebalance_cmd = [sys.executable, "-m", "src.trading.rebalance", "--config", args.config]
    if args.execute:
        rebalance_cmd.append("--execute")
    subprocess.run(rebalance_cmd, check=True)

    if args.execute:
        state_path.write_text(datetime.now().isoformat())
    logger.info("Rebalance cycle complete.")


if __name__ == "__main__":
    main()
