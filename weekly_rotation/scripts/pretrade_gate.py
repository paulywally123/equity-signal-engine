#!/usr/bin/env python3
"""Pre-trade validation gate for the weekly rotation rebalancer.

The rebalancer runs unattended on a schedule. Its entire risk lives in the
quality of the data it rebalances against. This module is the guard: it
inspects the downloaded prices and the proposed target, and returns a decision
the rebalancer must respect before it touches the broker.

Design principle: FAIL CLOSED. If the data looks stale, a group thinned out, or
a target name looks deal-pinned, the gate blocks. A blocked run writes its
report, logs why, and places no trades. A skipped rebalance is cheap. A
rebalance against corrupted input is not.

No network, no broker calls. Pure pandas/numpy.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# All thresholds in one place so they are auditable and testable.
DEFAULTS = {
    "max_stale_bdays": 4,          # latest close must be this fresh, else block
    "min_history_days": 257,       # 12-1 needs ~252 days; a little slack
    "min_constituents_per_group": 6,   # a ranked group thinner than this = block
    "vol_window_days": 21,
    "daily_vol_floor": 0.0015,     # 0.15%/day. Annualized ~2.4%. Nothing live sits this quiet.
    # A single non-zero return (e.g. the halt/deal-announcement jump) inside
    # the vol window inflates std enough to hide an otherwise-frozen tape --
    # confirmed empirically: 20 flat days + 1 jump day still clears
    # daily_vol_floor. This second signal catches that: a name where most of
    # the window's daily closes didn't move at all is pinned regardless of
    # what std says.
    "flat_return_eps": 1e-6,       # |return| below this counts as "unchanged"
    "flat_day_frac_floor": 0.8,    # >= this fraction of flat days -> pinned
    "expected_benchmarks": ("SPY", "RSP"),
}


@dataclass
class GateResult:
    ok: bool = True
    blocking: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)        # ticker -> why removed
    clean_target: list = field(default_factory=list)   # target minus dropped
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        head = "PASS" if self.ok else "BLOCK"
        lines = [f"[pretrade gate] {head}"]
        lines += [f"  BLOCK: {b}" for b in self.blocking]
        lines += [f"  warn:  {w}" for w in self.warnings]
        lines += [f"  drop:  {t} ({why})" for t, why in self.dropped.items()]
        return "\n".join(lines)


def _latest_date(close):
    idx = close.dropna(how="all").index
    return idx[-1] if len(idx) else None


def _check_staleness(close, cfg, res):
    latest = _latest_date(close)
    if latest is None:
        res.ok = False
        res.blocking.append("price feed is empty, no dated rows at all.")
        return
    today = pd.Timestamp.today().normalize()
    stale_by = int(np.busday_count(latest.date(), today.date()))
    res.diagnostics["latest_close"] = str(latest.date())
    res.diagnostics["stale_bdays"] = stale_by
    if stale_by > cfg["max_stale_bdays"]:
        res.ok = False
        res.blocking.append(
            f"feed is stale: latest close {latest.date()} is {stale_by} "
            f"business days old (limit {cfg['max_stale_bdays']})."
        )


def _usable(series, cfg):
    return series.dropna().shape[0] >= cfg["min_history_days"]


def _check_group_coverage(close, groups, constituents, cfg, res, terminal=frozenset()):
    # Check ALL ranked groups, not just today's top 3: tomorrow's top 3 are
    # chosen from all of them, and a silently thinned group can rank in on a
    # distorted average. Terminal groups (breadth sleeves held whole, no
    # single-name drill-down) have no constituents to check -- instead
    # confirm the sleeve's own ETF series is present and fresh.
    for etf in groups:
        if etf in terminal:
            if etf not in close.columns or not _usable(close[etf], cfg):
                res.ok = False
                res.blocking.append(f"terminal sleeve {etf} missing or thin.")
            continue
        members = constituents.get(etf, [])
        usable = [t for t in members if t in close.columns and _usable(close[t], cfg)]
        missing = [t for t in members if t not in usable]
        res.diagnostics.setdefault("group_usable", {})[etf] = {
            "usable": len(usable), "of": len(members), "missing": missing,
        }
        if len(usable) < cfg["min_constituents_per_group"]:
            res.ok = False
            res.blocking.append(
                f"group {etf} returned only {len(usable)} usable constituents "
                f"(min {cfg['min_constituents_per_group']}). Missing: "
                f"{', '.join(missing) if missing else 'none named'}."
            )


def _daily_realized_vol(series, window):
    rets = series.dropna().pct_change().dropna()
    if len(rets) < window:
        return float("nan")
    return float(rets.iloc[-window:].std())


def _flat_day_fraction(series, window, eps):
    rets = series.dropna().pct_change().dropna()
    if len(rets) < window:
        return float("nan")
    recent = rets.iloc[-window:]
    return float((recent.abs() < eps).mean())


def _check_deal_pinned(close, target_tickers, cfg, res):
    # A takeover target halted at the cash deal price shows near-zero daily
    # variance in its last prints. A delisted name is simply absent. Neither
    # belongs in an equal-weight target. Drop (not block) per name, then block
    # if the target got gutted.
    for t in target_tickers:
        if t not in close.columns or not _usable(close[t], cfg):
            res.dropped[t] = "missing or insufficient history"
            continue
        vol = _daily_realized_vol(close[t], cfg["vol_window_days"])
        flat_frac = _flat_day_fraction(
            close[t], cfg["vol_window_days"], cfg["flat_return_eps"]
        )
        res.diagnostics.setdefault("target_vol", {})[t] = (
            None if np.isnan(vol) else round(vol, 5)
        )
        res.diagnostics.setdefault("target_flat_frac", {})[t] = (
            None if np.isnan(flat_frac) else round(flat_frac, 3)
        )
        if np.isnan(vol) or np.isnan(flat_frac):
            res.dropped[t] = "cannot compute recent vol"
            continue
        reasons = []
        if vol < cfg["daily_vol_floor"]:
            reasons.append(
                f"{cfg['vol_window_days']}d daily vol {vol:.4f} < floor "
                f"{cfg['daily_vol_floor']}"
            )
        if flat_frac >= cfg["flat_day_frac_floor"]:
            reasons.append(
                f"{flat_frac:.0%} of last {cfg['vol_window_days']}d closes "
                f"unchanged (>= {cfg['flat_day_frac_floor']:.0%})"
            )
        if reasons:
            res.dropped[t] = "deal-pinned/dead: " + "; ".join(reasons)

    res.clean_target = [t for t in target_tickers if t not in res.dropped]

    if len(res.clean_target) == 0:
        res.ok = False
        res.blocking.append("target list is empty after filtering. Not trading.")
    elif len(res.clean_target) < 0.5 * max(len(target_tickers), 1):
        res.ok = False
        res.blocking.append(
            f"more than half the target was filtered "
            f"({len(res.dropped)}/{len(target_tickers)}). Feed looks "
            f"unreliable, not trading."
        )


def _check_benchmarks(close, cfg, res):
    for b in cfg["expected_benchmarks"]:
        if b not in close.columns or not _usable(close[b], cfg):
            res.warnings.append(f"benchmark {b} missing or thin; market read degraded.")


def run_pretrade_gate(close, groups, constituents, target_tickers, config=None,
                       terminal=frozenset()):
    """Validate feed and target before the rebalancer may trade.
    Respect .ok before trading. Use .clean_target as the target."""
    cfg = {**DEFAULTS, **(config or {})}
    res = GateResult()
    _check_staleness(close, cfg, res)
    _check_group_coverage(close, groups, constituents, cfg, res, terminal=terminal)
    _check_deal_pinned(close, target_tickers, cfg, res)
    _check_benchmarks(close, cfg, res)
    return res
