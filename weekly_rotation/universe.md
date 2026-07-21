# Universe

Static snapshot. Layer 2 (single-name selection) runs only on US large-cap
sector and theme ETFs, where constituent data is deep and clean. All other
sleeves are TERMINAL: they rank in Layer 1, and you hold the ETF. No single-name
drill-down, because foreign ADRs and small caps do not give reliable
point-in-time data.

Format: `### TICKER | Group Name`, optional `terminal: true` on the next line,
then a comma-separated constituent line for non-terminal groups only.

## Benchmarks (not ranked)

SPY, RSP, ACWI
# SPY = US cap-weight, RSP = US equal-weight, ACWI = global cap-weight.
# The three together give the market read: US breadth (SPY vs RSP) and
# US-vs-world (SPY vs ACWI).

## Ranked, Layer 2 (US single-name selection)

### XLK | Technology
MSFT, AAPL, NVDA, AVGO, ORCL, CRM, AMD, ADBE, CSCO, ACN

### XLF | Financials
BRK-B, JPM, V, MA, BAC, WFC, GS, MS, SPGI, AXP

### XLV | Health Care
LLY, UNH, JNJ, ABBV, MRK, TMO, ABT, ISRG, AMGN, DHR

### XLY | Consumer Discretionary
AMZN, TSLA, HD, MCD, BKNG, TJX, LOW, SBUX, NKE, CMG

### XLC | Communication Services
META, GOOGL, NFLX, TMUS, DIS, CMCSA, VZ, T, EA, TTWO

### XLI | Industrials
GE, CAT, UBER, RTX, HON, UNP, ETN, BA, DE, LMT

### XLE | Energy
XOM, CVX, COP, WMB, EOG, SLB, PSX, MPC, KMI, OKE

### XLP | Consumer Staples
PG, COST, WMT, KO, PEP, PM, MDLZ, MO, CL, TGT

### XLU | Utilities
NEE, SO, DUK, CEG, SRE, AEP, VST, D, PCG, EXC

### XLB | Materials
LIN, SHW, APD, ECL, FCX, NEM, CTVA, DD, MLM, VMC

### XLRE | Real Estate
PLD, AMT, EQIX, WELL, SPG, DLR, PSA, O, CCI, CBRE

### SMH | Semiconductors
NVDA, TSM, AVGO, AMD, ASML, QCOM, TXN, MU, LRCX, AMAT

### XBI | Biotech
VRTX, REGN, GILD, ALNY, MRNA, BMRN, INCY, SRPT, NBIX, UTHR

### ITA | Aerospace & Defense
RTX, BA, LMT, GD, NOC, HWM, TDG, LHX, AXON, HEI

### XHB | Homebuilders & Housing
DHI, LEN, NVR, PHM, TOL, BLDR, MAS, SHW, HD, LOW

## Ranked, terminal equity breadth (Phase 1, LIVE, hold the sleeve)

### VEA | Developed International
terminal: true

### VWO | Emerging Markets
terminal: true

### EWJ | Japan
terminal: true

### IWM | US Small Cap
terminal: true

### IJH | US Mid Cap
terminal: true

## Ranked, terminal cross-asset (Phase 2, LIVE)

### TLT | Long Treasuries
terminal: true

### IEF | Intermediate Treasuries
terminal: true

### LQD | Investment-Grade Credit
terminal: true

### HYG | High-Yield Credit
terminal: true

### GLD | Gold
terminal: true

### PDBC | Broad Commodities
terminal: true

### TIP | Inflation-Linked Treasuries
terminal: true

## Known limitations (deliberate)
# - Static snapshot: US constituent lists drift (survivorship). Refresh
#   periodically. Never backtest on this file.
# - EXAS removed 2026-03: acquired by Abbott, delisted. Replaced in XBI by UTHR.
# - Terminal sleeves are held whole by design, not drilled into.
# - Phase 2 cross-asset live as of 2026-07-21: Layer 1 now ranks on vol-scaled
#   12-1 (RANK_METRIC = "vol_scaled" in weekly_rotation.py), not raw return --
#   otherwise bonds/gold never out-rank equities except mid-drawdown.
