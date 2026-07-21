"""Automated weekly cycle: refresh the rotation report/history snapshot, then
rebalance the Alpaca paper account (account 3 by default) to match its
current picks. RUN ON YOUR MACHINE (or on a schedule -- see Task Scheduler
setup in rotation_trading_protocol.md).

Gates itself to actually rebalance only ~every 5 days via a state file
(data/last_rebalance.txt), so it's safe to invoke more often than that
without over-trading.

    python scripts/build_rotation_rebalance.py                # dry run
    python scripts/build_rotation_rebalance.py --execute       # place orders
    python scripts/build_rotation_rebalance.py --force         # ignore cadence gate
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_PATH = DATA_DIR / "last_rebalance.txt"

REBALANCE_EVERY_DAYS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=int, default=3)
    ap.add_argument("--execute", action="store_true",
                     help="Actually submit orders (default: dry run / print plan only)")
    ap.add_argument("--force", action="store_true",
                     help="Rebalance even if the cadence gate says it isn't due yet")
    args = ap.parse_args()

    if STATE_PATH.exists() and not args.force:
        last = datetime.fromisoformat(STATE_PATH.read_text().strip())
        elapsed = (datetime.now() - last).days
        if elapsed < REBALANCE_EVERY_DAYS:
            logger.info(
                "Last rebalance %d days ago (< %d) -- skipping. Use --force to override.",
                elapsed, REBALANCE_EVERY_DAYS,
            )
            return

    logger.info("Running weekly_rotation.py ...")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "weekly_rotation.py")],
        check=True, cwd=ROOT,
    )

    logger.info("Running rebalance.py ...")
    rebalance_cmd = [
        sys.executable, str(ROOT / "scripts" / "rebalance.py"),
        "--account", str(args.account),
    ]
    if args.execute:
        rebalance_cmd.append("--execute")
    subprocess.run(rebalance_cmd, check=True, cwd=ROOT)

    if args.execute:
        DATA_DIR.mkdir(exist_ok=True)
        STATE_PATH.write_text(datetime.now().isoformat())
    logger.info("Rotation rebalance cycle complete.")


if __name__ == "__main__":
    main()
