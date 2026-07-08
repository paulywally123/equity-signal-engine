"""Alpaca paper-trading client.

Hardcoded to paper=True -- there is no parameter to point this at live
trading. Switching to real money would need a deliberate, separate change.

Requires environment variables (set these yourself, never paste keys into
chat or commit them):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

from __future__ import annotations

import os

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

load_dotenv()


def get_trading_client() -> TradingClient:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables "
            "(paper-trading keys from your Alpaca dashboard)."
        )
    return TradingClient(api_key, secret_key, paper=True)
