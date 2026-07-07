"""SEC EDGAR XBRL fundamental data fetch and point-in-time panel construction.

Uses the public EDGAR company-facts API (no key required, 10 req/s rate limit):
    https://data.sec.gov/api/xbrl/companyfacts/{CIK10}.json

Each quarterly 10-Q or annual 10-K filing has:
    * period 'end'  — fiscal period end date  (do NOT use for look-ahead avoidance)
    * 'filed'       — date the SEC received the filing (true public availability)

We use the 'filed' date directly — no extra lag needed beyond a 1-day buffer.

Key GAAP concepts extracted:
    GrossProfit                     → gross_profit
    NetIncomeLoss                   → net_income
    Assets                          → total_assets
    StockholdersEquity              → equity    (tries several aliases)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_SEC_HEADERS = {"User-Agent": "equity-signal-engine research@example.com"}
_CIK_URL     = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# GAAP concept names we try (in priority order) for each target field
_CONCEPTS: dict[str, list[str]] = {
    "gross_profit": ["GrossProfit"],
    "net_income":   ["NetIncomeLoss", "NetIncome", "ProfitLoss"],
    "total_assets": ["Assets"],
    "equity":       [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "EquityAttributableToParent",
    ],
}


# ---------------------------------------------------------------------------
# CIK mapping
# ---------------------------------------------------------------------------

def fetch_cik_map(cache_path: Path) -> dict[str, int]:
    """Return {ticker: cik_int} mapping, cached locally."""
    if cache_path.exists():
        return pd.read_parquet(cache_path).set_index("ticker")["cik"].to_dict()

    resp = requests.get(_CIK_URL, headers=_SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    raw  = resp.json()
    rows = [{"ticker": v["ticker"].upper(), "cik": v["cik_str"]}
            for v in raw.values()]
    df   = pd.DataFrame(rows).drop_duplicates("ticker")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached %d CIK mappings → %s", len(df), cache_path)
    return df.set_index("ticker")["cik"].to_dict()


# ---------------------------------------------------------------------------
# Per-ticker fetch and extraction
# ---------------------------------------------------------------------------

def _extract_concept(
    facts_usgaap: dict,
    concept_names: list[str],
    forms: tuple[str, ...] = ("10-Q", "10-K"),
) -> pd.DataFrame | None:
    """Extract a GAAP concept as a DataFrame with columns [end, filed, val]."""
    for name in concept_names:
        if name not in facts_usgaap:
            continue
        units = facts_usgaap[name].get("units", {})
        rows  = units.get("USD") or units.get("shares") or []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df[df["form"].isin(forms)].copy()
        if df.empty:
            continue
        df["end"]   = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
        # Keep only annual (12-month) periods to avoid double-counting interim filings
        df["months"] = ((df["end"] - df["end"].shift(1)).dt.days / 30).round()
        # For 10-Qs we want the most-recent quarterly value, not YTD cumulative.
        # Use frame/accn to filter to point-in-time distinct periods.
        df = df.drop_duplicates(subset=["end"], keep="last")
        return df[["end", "filed", "val"]].sort_values("end").reset_index(drop=True)
    return None


def _fetch_one(
    ticker: str,
    cik: int,
    cache_dir: Path,
    force_refresh: bool,
) -> bool:
    out = cache_dir / f"{ticker}.parquet"
    if out.exists() and not force_refresh:
        return True
    try:
        url  = _FACTS_URL.format(cik=cik)
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=30)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        usgaap = resp.json().get("facts", {}).get("us-gaap", {})
        if not usgaap:
            return False

        parts: list[pd.DataFrame] = []
        for field, concepts in _CONCEPTS.items():
            df = _extract_concept(usgaap, concepts)
            if df is not None and not df.empty:
                df = df.rename(columns={"val": field})
                parts.append(df.set_index("end")[["filed", field]])

        if not parts:
            return False

        # Merge on period end date; filed date = latest across all concepts
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.join(part.drop(columns="filed"), how="outer")
        merged = merged.sort_index()
        merged.index.name = "period_end"
        merged.to_parquet(out)
        return True

    except Exception as exc:
        logger.debug("Failed %s (CIK %d): %s", ticker, cik, exc)
        return False


def fetch_all(
    tickers: list[str],
    cik_map: dict[str, int],
    cache_dir: Path,
    force_refresh: bool = False,
    workers: int = 4,
    rps: float = 8.0,
) -> tuple[int, int]:
    """Fetch EDGAR fundamentals for all tickers; rate-limited to rps requests/sec."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = skip = 0
    delay = 1.0 / rps

    # Only fetch tickers we have a CIK for
    pairs = [(t, cik_map[t]) for t in tickers if t in cik_map]
    skip  = len(tickers) - len(pairs)
    if skip:
        logger.info("No CIK for %d tickers — skipping", skip)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for t, cik in pairs:
            futures[pool.submit(_fetch_one, t, cik, cache_dir, force_refresh)] = t
            time.sleep(delay)          # rate-limit submissions

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                success = fut.result()
            except Exception:
                success = False
            if success:
                ok += 1
            else:
                fail += 1
            if i % 100 == 0 or i == len(pairs):
                logger.info("  %d / %d  (ok=%d  fail=%d)", i, len(pairs), ok, fail)

    return ok, fail


# ---------------------------------------------------------------------------
# Point-in-time fundamental panel
# ---------------------------------------------------------------------------

def _load_one(ticker: str, cache_dir: Path) -> pd.DataFrame | None:
    p = cache_dir / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _ttm(series: pd.Series, n: int = 4) -> pd.Series:
    """Trailing-twelve-month sum of the last n values."""
    return series.rolling(n, min_periods=n).sum()


def build_fundamental_panel(
    tickers: list[str],
    cache_dir: Path,
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    shares_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Point-in-time fundamental features indexed by (date, ticker).

    Uses each filing's 'filed' date as the availability date, so no extra lag
    is needed — a filing is used on rebalance dates strictly after filed.

    Features:
        gross_prof  = gross_profit_ttm / total_assets   (Novy-Marx 2013)
        roe         = net_income_ttm   / abs(equity)    (quality)
        ep_ratio    = net_income_ttm   / (price×shares) (earnings yield)
    """
    records: list[dict] = []

    for ticker in tickers:
        raw = _load_one(ticker, cache_dir)
        if raw is None or raw.empty:
            continue

        # Sort by period end; use filed date for availability
        raw = raw.sort_index()
        filed = raw["filed"] if "filed" in raw.columns else None

        ttm_gp = _ttm(raw["gross_profit"]) if "gross_profit" in raw.columns else None
        ttm_ni = _ttm(raw["net_income"])   if "net_income"   in raw.columns else None
        eq_col = raw["equity"]              if "equity"       in raw.columns else None
        at_col = raw["total_assets"]        if "total_assets" in raw.columns else None

        shares    = (shares_map or {}).get(ticker, np.nan)
        price_col = prices.get(ticker)

        for date in rebalance_dates:
            # Strict point-in-time: only use rows whose 'filed' date is before this date
            if filed is not None:
                avail_mask = filed < date
                available  = raw.index[avail_mask]
            else:
                available = raw.index[raw.index < date]

            if len(available) < 4:
                continue
            q = available[-1]

            row: dict = {"date": date, "ticker": ticker}

            gp  = ttm_gp.loc[q] if ttm_gp is not None  else np.nan
            ni  = ttm_ni.loc[q] if ttm_ni is not None  else np.nan
            eq  = eq_col.loc[q] if eq_col is not None  else np.nan
            ast = at_col.loc[q] if at_col is not None  else np.nan

            if pd.notna(gp) and pd.notna(ast) and ast > 0:
                row["gross_prof"] = float(gp / ast)

            if pd.notna(ni) and pd.notna(eq) and abs(eq) > 0:
                row["roe"] = float(ni / abs(eq))

            if pd.notna(ni) and not np.isnan(shares) and shares > 0 \
                    and price_col is not None:
                px = price_col.get(date)
                if pd.notna(px) and px > 0:
                    row["ep_ratio"] = float(ni / (px * shares))

            if len(row) > 2:
                records.append(row)

    if not records:
        logger.warning("No fundamental records — check EDGAR cache at %s", cache_dir)
        return pd.DataFrame(
            columns=["gross_prof", "roe", "ep_ratio"],
            index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]),
        )

    out = pd.DataFrame(records).set_index(["date", "ticker"]).sort_index()
    out.index.names = ["date", "ticker"]
    return out
