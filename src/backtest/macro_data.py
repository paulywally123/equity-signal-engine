"""Macro/market-stress indicators from FRED (free, no API key required via
the public fredgraph.csv download): https://fred.stlouisfed.org/

VIXCLS (CBOE Volatility Index) has full history back to 1990 via this
endpoint. The obvious credit-spread choice, ICE BofA US High Yield OAS
(BAMLH0A0HYM2), turned out to be capped by FRED/ICE's data-licensing terms
to a rolling ~3-year window regardless of requested start date (confirmed
empirically -- returns the same ~795 rows starting 2023-07 no matter what
`cosd` is passed), which would silently drop the entire pre-2020 selection
window and the COVID crash. Used Moody's Baa corporate bond yield minus the
10-year Treasury (BAA10Y) instead -- a standard, unrestricted credit-spread
stress proxy with full history back to 1986.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

_HEADERS = {"User-Agent": "equity-signal-engine research@example.com"}
_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id: str, cache_dir: Path, force_refresh: bool = False) -> pd.Series:
    """Return a FRED daily series indexed by date, missing ('.') rows dropped."""
    cache_path = cache_dir / f"{series_id}.csv"
    if cache_path.exists() and not force_refresh:
        text = cache_path.read_text()
    else:
        resp = requests.get(_URL, params={"id": series_id}, headers=_HEADERS, timeout=60)
        resp.raise_for_status()
        text = resp.text
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)

    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.dropna().set_index("date")[series_id].sort_index()
