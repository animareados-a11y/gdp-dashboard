"""
MarketSentinel
TEST MIRATO V5 RESET - ISRG 2026
================================

OBIETTIVO
---------
Caricare dal FULL200 esclusivamente ISRG,
applicare la nuova PHASE Dynamics V5 RESET
e mostrare il periodo giugno-agosto 2026.

NON modifica file di produzione.
NON salva output.
NON modifica engine.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v5 as v5


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/v40_35_full200_weekly.csv"
)

TICKER = "ISRG"

DATE_FROM = "2026-05-15"
DATE_TO = "2026-08-31"


# ============================================================
# LOAD SOLO ISRG
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File non trovato: {INPUT_FILE}"
    )

print()
print("=" * 100)
print("LOAD ISRG")
print("=" * 100)

chunks = []

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=10000,
    low_memory=False,
):

    ticker_col = None

    for candidate in [
        "Ticker",
        "TICKER",
        "ticker",
    ]:
        if candidate in chunk.columns:
            ticker_col = candidate
            break

    if ticker_col is None:
        raise KeyError(
            "Colonna ticker non trovata."
        )

    part = chunk[
        chunk[ticker_col]
        .astype(str)
        .eq(TICKER)
    ].copy()

    if len(part):
        chunks.append(part)


if not chunks:
    raise ValueError(
        f"{TICKER} non trovato in {INPUT_FILE}"
    )

raw = pd.concat(
    chunks,
    ignore_index=True,
)

print(
    "Righe ISRG trovate:",
    len(raw),
)


# ============================================================
# DATE
# ============================================================

date_col = None

for candidate in [
    "Date",
    "DATE",
    "date",
]:
    if candidate in raw.columns:
        date_col = candidate
        break

if date_col is None:
    raise KeyError(
        "Colonna data non trovata."
    )

raw[date_col] = pd.to_datetime(
    raw[date_col],
    errors="coerce",
)

raw = (
    raw
    .sort_values(date_col)
    .reset_index(drop=True)
)


# ============================================================
# PROCESS V5
# ============================================================

print()
print("=" * 100)
print("PROCESS V5 RESET")
print("=" * 100)

out = v5.process_ticker(
    raw.copy()
)

print(
    "Righe processate:",
    len(out),
)


# ============================================================
# FILTER 2026
# ============================================================

out[date_col] = pd.to_datetime(
    out[date_col],
    errors="coerce",
)

x = out[
    out[date_col].between(
        DATE_FROM,
        DATE_TO,
    )
].copy()


# ============================================================
# OUTPUT
# ============================================================

cols = [
    date_col,
    "SAR_SIDE",
    "SAR_AGE",
    "V4010_PREV_SAR_AGE",
    "V4010_RUN_QUALIFIED",
    "HA_DIRECTION",
    "HA_INDECISION",
    "PHASE_DYNAMICS_V4",
    "PHASE_DYNAMICS_V5",
    "V5_BUY_LONG_BEAR_PROTECTED",
    "V5_BUY_CONFIRM_COUNT",
    "V5_BUY_CONFIRM_RESET",
    "V5_BUY_RESET_COUNT",
    "V5_BUY_CONFIRMATION_AGE",
    "V5_BUY_STAGE",
    "V5_BUY_REASON",
]

cols = [
    c for c in cols
    if c in x.columns
]

print()
print("=" * 100)
print("ISRG | 2026-05-15 -> 2026-08-31")
print("=" * 100)

print(
    x[cols]
    .to_string(index=False)
)


# ============================================================
# CHECK MIRATO LUGLIO
# ============================================================

target = x[
    x[date_col].between(
        "2026-07-01",
        "2026-07-31",
    )
].copy()

print()
print("=" * 100)
print("CHECK LUGLIO 2026")
print("=" * 100)

if len(target):

    print(
        target[cols]
        .to_string(index=False)
    )

else:

    print(
        "Nessuna riga trovata."
    )


print()
print("=" * 100)
print("TEST COMPLETATO")
print("=" * 100)