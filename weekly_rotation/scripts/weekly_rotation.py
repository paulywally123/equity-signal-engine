#!/usr/bin/env python3
"""Weekly equity momentum rotation report generator.

Descriptive research tool. Ranks sector/theme ETFs and their constituents on
12-1 month momentum, writes a markdown report and a JSON history snapshot, and
diffs against the prior run. Read-only market data. Never touches a broker.

Usage:
    python scripts/weekly_rotation.py               # real data via yfinance
    python scripts/weekly_rotation.py --synthetic   # offline pipeline test
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pretrade_gate import run_pretrade_gate

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_FILE = ROOT / "universe.md"
REPORTS_DIR = ROOT / "reports"
HISTORY_DIR = ROOT / "history"
DATA_DIR = ROOT / "data"
GATE_LOG = DATA_DIR / "gate_log.txt"

LOOKBACK_12M = 252   # trading days, ~12 months
SKIP_1M = 21         # trading days, ~1 month
LOOKBACK_3M = 63
MA_WINDOW = 200
MIN_HISTORY = LOOKBACK_12M + 5  # a little slack for holidays

TOP_GROUPS = 3
TOP_NAMES = 5
RANK_MOVE_THRESHOLD = 2

# Group-level hysteresis on the TOP_GROUPS cutoff. Without this, a group
# sliding from rank 3 to rank 4 dumps its entire Layer-2 basket in one week
# -- a far bigger trade than any single-name drift, and vol-scaling makes it
# worse since the 63d vol denominator moves faster than the 11-month
# momentum numerator, so a score near the cutoff can flip on vol-estimate
# noise rather than a real trend change. A group held last cycle keeps its
# slot as long as it hasn't fallen past this rank; deliberately NOT paired
# with a separate score-ratio "entry margin" for a new group, since the exit
# band alone already prevents the marginal-flip churn without inventing an
# unvalidated second threshold.
GROUP_EXIT_RANK = 5

# "raw" | "vol_scaled". Ranking equities on raw return and bonds/gold on
# vol-scaled return in the same list uses two different rulers and the
# ranking stops meaning anything -- vol-scale the WHOLE universe at once.
# Cross-asset (Phase 2) is live as of 2026-07-21, so this is vol_scaled now.
RANK_METRIC = "vol_scaled"   # "raw" | "vol_scaled". Cross-asset live => vol_scaled.
VOL_WINDOW = 63       # ~3 months of daily returns for the vol denominator


# ---------------------------------------------------------------- universe ---

def _is_comment_line(line: str) -> bool:
    """A real markdown heading (##, ###) also starts with '#' -- only a
    line that is a SINGLE # (not ## or ###) is an actual comment."""
    s = line.lstrip()
    return s.startswith("#") and not s.startswith("##")


def parse_universe(path: Path):
    """Parse universe.md into benchmarks, ordered groups, constituents, and
    the set of terminal (hold-the-sleeve, no single-name drill-down) ETFs."""
    text = path.read_text()

    bench_match = re.search(
        r"## Benchmarks.*?\n+([A-Z0-9,\-\s]+?)\n", text, re.DOTALL
    )
    benchmarks = [t.strip() for t in bench_match.group(1).split(",")] if bench_match else []

    groups = {}          # etf -> group name
    constituents = {}    # etf -> [tickers], empty for terminal groups
    terminal = set()     # etf's that rank in Layer 1 but skip Layer 2
    # Strip commented-out (single #) lines so the staged Phase 2 block stays
    # inert until uncommented -- real ##/### headings are left alone.
    live = "\n".join(l for l in text.splitlines() if not _is_comment_line(l))

    for m in re.finditer(
        r"### ([A-Z0-9\-]+) \| (.+?)\n(terminal:\s*true\n)?([A-Z0-9,\-\s]*?)(?=\n###|\n##|\Z)",
        live,
    ):
        etf, name, is_terminal, members = m.group(1), m.group(2).strip(), m.group(3), m.group(4)
        groups[etf] = name
        if is_terminal:
            terminal.add(etf)
            constituents[etf] = []
        else:
            constituents[etf] = [t.strip() for t in members.split(",") if t.strip()]
    if not groups:
        sys.exit("universe.md parsed to zero groups. Check the format.")
    return benchmarks, groups, constituents, terminal


# -------------------------------------------------------------------- data ---

def download_prices(tickers, synthetic=False):
    """Return a DataFrame of daily adjusted closes, columns = tickers."""
    if synthetic:
        return synthetic_prices(tickers)
    import yfinance as yf
    data = yf.download(
        tickers, period="2y", auto_adjust=True, progress=False, threads=True
    )
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(how="all")


def synthetic_prices(tickers, n_days=520, seed=42):
    """Geometric random walks with per-ticker drift, for offline testing."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    frames = {}
    for i, t in enumerate(sorted(tickers)):
        drift = rng.normal(0.0004, 0.0006)
        vol = rng.uniform(0.008, 0.025)
        rets = rng.normal(drift, vol, n_days)
        frames[t] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(frames, index=idx)


# ----------------------------------------------------------------- metrics ---

def compute_metrics(close: pd.DataFrame):
    """Per-ticker metrics from a close-price DataFrame. NaN if insufficient."""
    rows = []
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < MIN_HISTORY:
            rows.append({"ticker": t, "ok": False, "n_days": len(s)})
            continue
        last = s.iloc[-1]
        mom_12_1 = s.iloc[-SKIP_1M] / s.iloc[-LOOKBACK_12M] - 1
        r3m = last / s.iloc[-LOOKBACK_3M] - 1
        r1m = last / s.iloc[-SKIP_1M] - 1
        ma200 = s.rolling(MA_WINDOW).mean().iloc[-1]
        rets = s.pct_change().dropna()
        vol_63 = float(rets.iloc[-VOL_WINDOW:].std()) if len(rets) >= VOL_WINDOW else float("nan")
        vol_valid = vol_63 and vol_63 > 0 and not np.isnan(vol_63)
        if RANK_METRIC == "vol_scaled":
            if not vol_valid:
                # Can't produce a comparable score in this mode -- excluding
                # beats silently falling back to raw mom_12_1, which is a
                # different scale entirely (typically ~10-70x smaller than a
                # vol-scaled score) and would make this name look like a
                # relative dead weight rather than genuinely incomparable.
                rows.append({"ticker": t, "ok": False, "n_days": len(s)})
                continue
            rank_score = mom_12_1 / vol_63
        else:
            rank_score = mom_12_1   # raw mode: vol_63 not used for ranking
        rows.append({
            "ticker": t, "ok": True, "n_days": len(s),
            "last_date": str(s.index[-1].date()),
            "mom_12_1": float(mom_12_1),
            "vol_63": vol_63,
            "rank_score": float(rank_score),
            "r3m": float(r3m),
            "r1m": float(r1m),
            "above_200dma": bool(last > ma200),
            "pct_vs_200dma": float(last / ma200 - 1),
        })
    df = pd.DataFrame(rows).set_index("ticker")
    return df


def rank_frame(metrics: pd.DataFrame, tickers):
    """Slice to tickers, keep usable rows, rank descending by rank_score
    (raw 12-1 momentum, or vol-scaled per RANK_METRIC)."""
    sub = metrics.loc[[t for t in tickers if t in metrics.index]]
    good = sub[sub["ok"] == True].copy()  # noqa: E712
    good = good.sort_values("rank_score", ascending=False)
    good["rank"] = range(1, len(good) + 1)
    excluded = [t for t in tickers if t not in good.index]
    return good, excluded


def select_top_groups(l1: pd.DataFrame, prior_top3: set) -> list:
    """Group-level hysteresis: a group held last cycle (prior_top3) keeps
    its slot as long as its current rank hasn't fallen past GROUP_EXIT_RANK.
    Remaining slots fill with the next best-ranked groups not already kept.
    With no prior (first run, or nothing in prior_top3 still ranks), this
    reduces to the plain top-TOP_GROUPS by rank."""
    ranked_order = list(l1.index)   # already sorted best-first
    rank_of = {t: int(r) for t, r in l1["rank"].items()}

    kept = [g for g in prior_top3 if g in rank_of and rank_of[g] <= GROUP_EXIT_RANK]
    kept.sort(key=lambda g: rank_of[g])
    kept = kept[:TOP_GROUPS]

    admitted = []
    for g in ranked_order:
        if len(kept) + len(admitted) >= TOP_GROUPS:
            break
        if g not in kept:
            admitted.append(g)

    return kept + admitted


# ----------------------------------------------------------------- history ---

def load_prior_history(run_date: str):
    files = sorted(HISTORY_DIR.glob("*_ranks.json"))
    files = [f for f in files if not f.name.startswith(run_date)]
    if not files:
        return None
    return json.loads(files[-1].read_text())


def diff_ranks(current: dict, prior: dict):
    """Rank movers of 2+ places and 200DMA crossings, Layer 1 only."""
    movers, crossings = [], []
    prev = {r["ticker"]: r for r in prior.get("layer1", [])}
    for row in current["layer1"]:
        t = row["ticker"]
        if t not in prev:
            continue
        delta = prev[t]["rank"] - row["rank"]  # positive = moved up
        if abs(delta) >= RANK_MOVE_THRESHOLD:
            movers.append((t, prev[t]["rank"], row["rank"], delta))
        if prev[t]["above_200dma"] != row["above_200dma"]:
            crossings.append((t, "above" if row["above_200dma"] else "below"))
    return movers, crossings


# ------------------------------------------------------------------ report ---

def pct(x):
    return f"{x * 100:+.1f}%"


def layer1_table(ranked, groups):
    lines = [
        "| Rank | ETF | Group | 12-1 mom | 3m | 1m | vs 200DMA |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for t, r in ranked.iterrows():
        flag = "above" if r["above_200dma"] else "below"
        lines.append(
            f"| {int(r['rank'])} | {t} | {groups.get(t, '')} | "
            f"{pct(r['mom_12_1'])} | {pct(r['r3m'])} | {pct(r['r1m'])} | "
            f"{pct(r['pct_vs_200dma'])} ({flag}) |"
        )
    return "\n".join(lines)


def layer2_table(ranked):
    lines = [
        "| Rank | Ticker | 12-1 mom | 3m | 1m | vs 200DMA |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for t, r in ranked.head(TOP_NAMES).iterrows():
        flag = "above" if r["above_200dma"] else "below"
        lines.append(
            f"| {int(r['rank'])} | {t} | {pct(r['mom_12_1'])} | "
            f"{pct(r['r3m'])} | {pct(r['r1m'])} | "
            f"{pct(r['pct_vs_200dma'])} ({flag}) |"
        )
    return "\n".join(lines)


def market_read(metrics, ranked_etfs):
    lines = []
    for b in ("SPY", "RSP", "ACWI"):
        if b in metrics.index and metrics.loc[b, "ok"]:
            r = metrics.loc[b]
            flag = "above" if r["above_200dma"] else "below"
            lines.append(
                f"- **{b}**: 12-1 {pct(r['mom_12_1'])}, 3m {pct(r['r3m'])}, "
                f"1m {pct(r['r1m'])}, {flag} its 200DMA "
                f"({pct(r['pct_vs_200dma'])})."
            )
    if {"SPY", "RSP"}.issubset(metrics.index):
        spread = metrics.loc["SPY", "r3m"] - metrics.loc["RSP", "r3m"]
        lines.append(
            f"- **Cap vs equal weight**: SPY minus RSP, 3m spread {pct(spread)} "
            f"({'mega-cap led' if spread > 0 else 'broad market led'})."
        )
    if {"SPY", "ACWI"}.issubset(metrics.index):
        spread = metrics.loc["SPY", "r3m"] - metrics.loc["ACWI", "r3m"]
        lines.append(
            f"- **US vs world**: SPY minus ACWI, 3m spread {pct(spread)} "
            f"({'US led' if spread > 0 else 'rest-of-world led'})."
        )
    n_above = int(ranked_etfs["above_200dma"].sum())
    lines.append(
        f"- **Breadth proxy**: {n_above}/{len(ranked_etfs)} ranked groups above "
        f"their 200DMA."
    )
    return "\n".join(lines)


def gate_section(gate, proposed_target):
    lines = ["## Pre-trade data safety gate\n"]
    lines.append(
        f"Proposed target ({len(proposed_target)}): {', '.join(proposed_target)}\n"
    )
    if gate.ok:
        lines.append("**PASS** -- rebalance.py may trade `clean_target` below.\n")
    else:
        lines.append(
            "**BLOCK** -- rebalance.py will refuse `--execute` against this "
            "snapshot until the issue is resolved.\n"
        )
    if gate.blocking:
        lines.append("Blocking reasons:\n")
        for b in gate.blocking:
            lines.append(f"- {b}")
        lines.append("")
    if gate.warnings:
        lines.append("Warnings (non-blocking):\n")
        for w in gate.warnings:
            lines.append(f"- {w}")
        lines.append("")
    if gate.dropped:
        lines.append("Dropped from target:\n")
        for t, why in gate.dropped.items():
            lines.append(f"- {t}: {why}")
        lines.append("")
    lines.append(f"Clean target ({len(gate.clean_target)}): "
                 f"{', '.join(gate.clean_target) or '(none)'}\n")
    return "\n".join(lines)


def weights_section(target_weights):
    lines = ["## Portfolio weights (group-weighted)\n"]
    lines.append(
        "Each selected group gets an equal share of the book; within a "
        "non-terminal group that share splits evenly across its surviving "
        "picks. A terminal sleeve's whole share goes into the one ETF.\n"
    )
    if not target_weights:
        lines.append("(none -- see gate section above)\n")
        return "\n".join(lines)
    lines.append("| Ticker | Weight |")
    lines.append("|---|---:|")
    for t, w in sorted(target_weights.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {t} | {w:.1%} |")
    lines.append(f"\nTotal: {sum(target_weights.values()):.1%}\n")
    return "\n".join(lines)


def build_report(run_date, metrics, l1, l1_excluded, l2, l2_excluded,
                 l2_terminal, top_etfs, groups, movers, crossings, prior_date,
                 synthetic, gate, proposed_target, target_weights):
    parts = [f"# Weekly Rotation, {run_date}\n"]
    if synthetic:
        parts.append(
            "> **SYNTHETIC DATA.** Pipeline test with random walks. "
            "Every number below is fake.\n"
        )
    parts.append(
        "> Descriptive research, not a trade list and not financial advice. "
        "A leader here is a reason to look at a name, nothing more. "
        "Rankings are probabilistic.\n"
    )
    metric_note = f"Ranked on: {RANK_METRIC} 12-1 momentum"
    if RANK_METRIC == "vol_scaled":
        metric_note += f", vol denominator {VOL_WINDOW}d"
    metric_note += (
        ". The 12-1 mom column below is always raw return for readability "
        "-- row order comes from the ranking metric above, so it can look "
        "out of order by that column alone when vol-scaled is active."
    )
    parts.append(f"> {metric_note}\n")

    parts.append("## Market read\n")
    parts.append(market_read(metrics, l1) + "\n")

    parts.append("## Layer 1: group ranking (12-1 momentum)\n")
    parts.append(layer1_table(l1, groups) + "\n")
    if l1_excluded:
        parts.append(
            f"Excluded for missing/insufficient data: {', '.join(l1_excluded)}\n"
        )

    parts.append("## Layer 2: top names inside the leaders\n")
    for etf in top_etfs:
        if etf in l2_terminal:
            parts.append(
                f"### {etf}, {groups[etf]}\n\n"
                f"Terminal sleeve. Hold {etf}, no single-name selection.\n"
            )
            continue
        if etf not in l2:
            continue
        parts.append(f"### {etf}, {groups[etf]}\n")
        parts.append(layer2_table(l2[etf]) + "\n")
        if l2_excluded.get(etf):
            parts.append(
                f"Excluded: {', '.join(l2_excluded[etf])}\n"
            )

    parts.append(gate_section(gate, proposed_target))
    parts.append(weights_section(target_weights))

    parts.append("## Week over week\n")
    if prior_date is None:
        parts.append("First run on record. No prior ranking to diff against.\n")
    else:
        parts.append(f"Compared against {prior_date}.\n")
        if movers:
            for t, old, new, delta in movers:
                arrow = "up" if delta > 0 else "down"
                parts.append(f"- {t}: rank {old} to {new} ({arrow} {abs(delta)}).")
        else:
            parts.append("- No group moved 2+ ranks.")
        if crossings:
            for t, direction in crossings:
                parts.append(f"- {t} crossed {direction} its 200-day MA.")
        else:
            parts.append("- No 200-day MA crossings.")
        parts.append("")

    return "\n".join(parts)


# -------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run with random-walk data, no network")
    args = ap.parse_args()

    run_date = str(date.today())
    REPORTS_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(exist_ok=True)

    benchmarks, groups, constituents, terminal = parse_universe(UNIVERSE_FILE)
    all_tickers = sorted(
        set(benchmarks) | set(groups) | {t for c in constituents.values() for t in c}
    )
    print(f"Universe: {len(groups)} groups ({len(terminal)} terminal), "
          f"{len(all_tickers)} tickers total.")

    close = download_prices(all_tickers, synthetic=args.synthetic)
    print(f"Prices: {close.shape[0]} days x {close.shape[1]} tickers.")

    metrics = compute_metrics(close)

    # Layer 1
    l1, l1_excluded = rank_frame(metrics, list(groups))

    # Loaded early (not just for the week-over-week diff below) because
    # group-level hysteresis needs to know which groups were held last
    # cycle before deciding this cycle's top-3.
    prior = load_prior_history(run_date)
    prior_top3 = (
        set(prior.get("layer2", {}).keys()) | set(prior.get("terminal_groups", []))
        if prior else set()
    )

    # Layer 2 -- terminal groups (breadth sleeves: international, small/mid
    # cap) rank in Layer 1 but skip single-name drill-down entirely; hold the
    # ETF instead. See universe.md for why.
    top_etfs = select_top_groups(l1, prior_top3)
    l2, l2_excluded, l2_terminal = {}, {}, []
    for etf in top_etfs:
        if etf in terminal:
            l2_terminal.append(etf)
            continue
        ranked, excluded = rank_frame(metrics, constituents[etf])
        l2[etf] = ranked
        l2_excluded[etf] = excluded

    # Proposed target: for each top-3 group, either the ETF itself (terminal
    # -- hold the sleeve whole, e.g. VEA, GLD) or the union of its top
    # TOP_NAMES constituents. This is what rebalance.py actually trades (via
    # the gate's clean_target below), so a terminal group ranking into the
    # top 3 must contribute something here or it silently falls through and
    # the live portfolio stops matching what the report says to hold.
    proposed_target = []
    for etf in top_etfs:
        if etf in terminal:
            if etf not in proposed_target:
                proposed_target.append(etf)
            continue
        for t in l2[etf].head(TOP_NAMES).index:
            if t not in proposed_target:
                proposed_target.append(t)

    gate = run_pretrade_gate(
        close, groups, constituents, proposed_target, terminal=terminal
    )
    print(gate.summary())

    DATA_DIR.mkdir(exist_ok=True)
    with open(GATE_LOG, "a") as f:
        f.write(f"\n{pd.Timestamp.now()} weekly_rotation.py run_date={run_date}\n")
        f.write(gate.summary() + "\n")

    # Group-weighted allocation: each of the (up to TOP_GROUPS) selected
    # groups gets an equal share of the book; within a non-terminal group,
    # its share splits evenly across whichever of its names survived the
    # gate. A flat per-name split let a single-ETF terminal sleeve like IWM
    # end up a ~9% token position instead of the real weight its own group
    # rank implies -- this is what "hold the sleeve" means at the portfolio
    # level, not just the report level. Ticker overlap across groups (e.g.
    # AMD in both XLK and SMH) accumulates weight from each group, which is
    # intended: independently ranking top-5 in two groups is more
    # conviction, not double-counting.
    clean_set = set(gate.clean_target)
    target_weights: dict = {}
    zero_survivor_groups = []
    n_groups = len(top_etfs) or 1
    per_group_weight = 1.0 / n_groups
    for etf in top_etfs:
        if etf in terminal:
            if etf in clean_set:
                target_weights[etf] = target_weights.get(etf, 0.0) + per_group_weight
            else:
                zero_survivor_groups.append(etf)
            continue
        names = [t for t in l2[etf].head(TOP_NAMES).index if t in clean_set]
        if not names:
            zero_survivor_groups.append(etf)
            continue
        per_name = per_group_weight / len(names)
        for t in names:
            target_weights[t] = target_weights.get(t, 0.0) + per_name
    if zero_survivor_groups:
        print(f"WARNING: groups with zero surviving picks (their share goes "
              f"unallocated this cycle): {', '.join(zero_survivor_groups)}")

    # History snapshot and diff
    snapshot = {
        "date": run_date,
        "synthetic": args.synthetic,
        "layer1": [
            {"ticker": t, "rank": int(r["rank"]),
             "mom_12_1": r["mom_12_1"], "above_200dma": r["above_200dma"]}
            for t, r in l1.iterrows()
        ],
        "layer2": {
            etf: [
                {"ticker": t, "rank": int(r["rank"]), "mom_12_1": r["mom_12_1"]}
                for t, r in l2[etf].iterrows()
            ]
            for etf in l2
        },
        "terminal_groups": l2_terminal,
        "top_groups": top_etfs,
        "gate": {
            "ok": gate.ok,
            "blocking": gate.blocking,
            "warnings": gate.warnings,
            "dropped": gate.dropped,
            "proposed_target": proposed_target,
            "clean_target": gate.clean_target,
            "diagnostics": gate.diagnostics,
        },
        "target_weights": target_weights,
    }
    movers, crossings = diff_ranks(snapshot, prior) if prior else ([], [])
    prior_date = prior["date"] if prior else None

    report = build_report(
        run_date, metrics, l1, l1_excluded, l2, l2_excluded, l2_terminal,
        top_etfs, groups, movers, crossings, prior_date, args.synthetic,
        gate, proposed_target, target_weights,
    )

    report_path = REPORTS_DIR / f"{run_date}_rotation.md"
    report_path.write_text(report)
    (HISTORY_DIR / f"{run_date}_ranks.json").write_text(
        json.dumps(snapshot, indent=2)
    )
    print(f"Wrote {report_path}")
    print(f"Wrote history/{run_date}_ranks.json")


if __name__ == "__main__":
    main()
