"""
MarketSentinel
VALIDAZIONE PHASE DYNAMICS V5 RESET - FULL200
=============================================

OBIETTIVO
---------
Validare lo SHADOW V5 RESET sull'intero universo FULL200.

La V5 applica esclusivamente la protezione BUY dopo un precedente
regime SAR bearish sufficientemente lungo:

- precedente SAR bearish >= 6 settimane
- servono 3 HA bullish decisive
- HA indecisione non incrementa e non azzera
- HA bearish decisiva azzera il conteggio
- la settimana della terza conferma diventa BUY1
- poi BUY2 / BUY3
- SELL non viene modificato

IMPORTANTE
----------
- NON modifica production
- NON modifica V40.10
- NON modifica V4
- NON modifica engine.py
- NON modifica il dataset di input
- lavora esclusivamente in SHADOW

CHECKPOINT
----------
Ogni ticker completato viene salvato singolarmente in:

    data/v5_full200_checkpoint/

Se l'esecuzione viene interrotta, rilanciando questo stesso script
i ticker già completati vengono recuperati dal checkpoint.

OUTPUT
------
data/validazione_phase_dynamics_v5_full200_rows.csv
data/validazione_phase_dynamics_v5_full200_changed.csv
data/validazione_phase_dynamics_v5_full200_protected.csv
data/validazione_phase_dynamics_v5_full200_summary.csv
data/validazione_phase_dynamics_v5_full200_reclassification.csv
data/validazione_phase_dynamics_v5_full200_stage_counts.csv
data/validazione_phase_dynamics_v5_full200_latest.csv
data/validazione_phase_dynamics_v5_full200_errors.csv
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v5 as v5

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

OUT_DIR = Path("data")

CHECKPOINT_DIR = OUT_DIR / "v5_full200_checkpoint"

OUT_ROWS = OUT_DIR / "validazione_phase_dynamics_v5_full200_rows.csv"
OUT_CHANGED = OUT_DIR / "validazione_phase_dynamics_v5_full200_changed.csv"
OUT_PROTECTED = OUT_DIR / "validazione_phase_dynamics_v5_full200_protected.csv"
OUT_SUMMARY = OUT_DIR / "validazione_phase_dynamics_v5_full200_summary.csv"
OUT_RECLASS = OUT_DIR / "validazione_phase_dynamics_v5_full200_reclassification.csv"
OUT_STAGE = OUT_DIR / "validazione_phase_dynamics_v5_full200_stage_counts.csv"
OUT_LATEST = OUT_DIR / "validazione_phase_dynamics_v5_full200_latest.csv"
OUT_ERRORS = OUT_DIR / "validazione_phase_dynamics_v5_full200_errors.csv"

CHECKPOINT_META = CHECKPOINT_DIR / "checkpoint_metadata.json"


# ============================================================
# UTILS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 130)
    print(title)
    print("=" * 130)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"Nessuna delle colonne trovata: {candidates}"
    )


def safe_filename(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value


def checkpoint_path(ticker: str) -> Path:
    return CHECKPOINT_DIR / f"{safe_filename(ticker)}.pkl"


def normalize_phase(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# PREPARE
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File input non trovato: {INPUT_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

section("LOAD DATA")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

ticker_col = find_column(
    df,
    [
        "Ticker",
        "TICKER",
        "ticker",
    ],
)

date_col = find_column(
    df,
    [
        "Date",
        "DATE",
        "date",
    ],
)

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce",
)

tickers = sorted(
    df[ticker_col]
    .dropna()
    .astype(str)
    .unique()
)

print(f"Input rows: {len(df):,}")
print(f"Ticker input: {len(tickers)}")


# ============================================================
# CHECKPOINT METADATA
# ============================================================

metadata = {
    "input_file": str(INPUT_FILE),
    "input_rows": int(len(df)),
    "ticker_count": int(len(tickers)),
    "v5_n_prev_bear_min": int(
        getattr(
            v5,
            "N_PREV_BEAR_MIN",
            -1,
        )
    ),
    "v5_k_bull_confirmations": int(
        getattr(
            v5,
            "K_BULL_HA_CONFIRMATIONS",
            -1,
        )
    ),
}

CHECKPOINT_META.write_text(
    json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# PROCESS FULL200
# ============================================================

section("PROCESS FULL200")

errors = []

for i, ticker in enumerate(
    tickers,
    start=1,
):

    cp = checkpoint_path(ticker)

    if cp.exists():
        print(
            f"[{i:03d}/{len(tickers):03d}] "
            f"{ticker:12s} CHECKPOINT"
        )
        continue

    g = (
        df[
            df[ticker_col]
            .astype(str)
            .eq(ticker)
        ]
        .copy()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    try:

        out = v5.process_ticker(
            g.copy()
        )

        if ticker_col not in out.columns:
            out[ticker_col] = ticker

        out.to_pickle(cp)

        print(
            f"[{i:03d}/{len(tickers):03d}] "
            f"{ticker:12s} "
            f"rows={len(out):4d} OK"
        )

    except Exception as exc:

        errors.append(
            {
                "Ticker": ticker,
                "Error": repr(exc),
            }
        )

        print(
            f"[{i:03d}/{len(tickers):03d}] "
            f"{ticker:12s} ERROR: {repr(exc)}"
        )


# ============================================================
# LOAD CHECKPOINTS
# ============================================================

section("LOAD CHECKPOINTS")

parts = []
missing_checkpoints = []

for ticker in tickers:

    cp = checkpoint_path(ticker)

    if not cp.exists():
        missing_checkpoints.append(
            ticker
        )
        continue

    try:

        part = pd.read_pickle(cp)

        parts.append(part)

    except Exception as exc:

        errors.append(
            {
                "Ticker": ticker,
                "Error": (
                    "CHECKPOINT_READ_ERROR: "
                    + repr(exc)
                ),
            }
        )

        missing_checkpoints.append(
            ticker
        )


if not parts:
    raise RuntimeError(
        "Nessun checkpoint valido disponibile."
    )

out = pd.concat(
    parts,
    ignore_index=True,
)

out[date_col] = pd.to_datetime(
    out[date_col],
    errors="coerce",
)

out = (
    out
    .sort_values(
        [
            ticker_col,
            date_col,
        ]
    )
    .reset_index(drop=True)
)

print(
    f"Checkpoint caricati: "
    f"{len(parts)}/{len(tickers)}"
)

print(
    f"Righe aggregate: "
    f"{len(out):,}"
)

if missing_checkpoints:
    print(
        "Checkpoint mancanti:",
        missing_checkpoints,
    )


# ============================================================
# REQUIRED V5 COLUMNS
# ============================================================

required_columns = [
    "PHASE_DYNAMICS_V4",
    "PHASE_DYNAMICS_V5",
    "V5_BUY_LONG_BEAR_PROTECTED",
    "V5_BUY_CONFIRM_COUNT",
    "V5_BUY_CONFIRMATION_REACHED",
    "V5_BUY_CONFIRMATION_AGE",
    "V5_BUY_STAGE",
    "V5_BUY_CONFIRM_RESET",
    "V5_BUY_RESET_COUNT",
]

missing_cols = [
    c for c in required_columns
    if c not in out.columns
]

if missing_cols:
    raise KeyError(
        "Colonne V5 mancanti: "
        + repr(missing_cols)
    )


# ============================================================
# NORMALIZED PHASE
# ============================================================

phase_v4 = normalize_phase(
    out["PHASE_DYNAMICS_V4"]
)

phase_v5 = normalize_phase(
    out["PHASE_DYNAMICS_V5"]
)

changed_mask = (
    phase_v4 != phase_v5
)

protected_mask = (
    pd.to_numeric(
        out["V5_BUY_LONG_BEAR_PROTECTED"],
        errors="coerce",
    )
    .fillna(0)
    .eq(1)
)


# ============================================================
# SAFETY
# ============================================================

# Qualunque ingresso o uscita dalla categoria SELL
# sarebbe una modifica della parte SELL.
sell_v4 = phase_v4.eq("SELL")
sell_v5 = phase_v5.eq("SELL")

sell_changed_mask = (
    sell_v4 != sell_v5
)

outside_protected_mask = (
    changed_mask
    & ~protected_mask
)


# ============================================================
# AGGREGATE METRICS
# ============================================================

rows_n = len(out)

ticker_n = (
    out[ticker_col]
    .astype(str)
    .nunique()
)

process_errors = (
    len(errors)
    + len(
        set(missing_checkpoints)
    )
)

protected_rows = int(
    protected_mask.sum()
)

changed_rows = int(
    changed_mask.sum()
)

changed_pct = (
    100.0
    * changed_rows
    / rows_n
    if rows_n
    else np.nan
)

sell_changed = int(
    sell_changed_mask.sum()
)

changes_outside_protected = int(
    outside_protected_mask.sum()
)

confirmation_buy1_mask = (
    out["V5_BUY_STAGE"]
    .fillna("")
    .astype(str)
    .eq("BUY1")
)

confirmation_buy1_rows = int(
    confirmation_buy1_mask.sum()
)

reset_rows = int(
    pd.to_numeric(
        out["V5_BUY_CONFIRM_RESET"],
        errors="coerce",
    )
    .fillna(0)
    .eq(1)
    .sum()
)

protected_tickers = int(
    out.loc[
        protected_mask,
        ticker_col,
    ]
    .astype(str)
    .nunique()
)


# ============================================================
# DISTRIBUTION
# ============================================================

phase_order = [
    "BUY",
    "DEBOLEZZA_RIALZISTA",
    "DEBOLEZZA_RIBASSISTA",
    "INDECISIONE",
    "SELL",
    "TREND_RIALZISTA",
    "TREND_RIBASSISTA",
]

all_phases = list(
    dict.fromkeys(
        phase_order
        + sorted(
            set(phase_v4.unique())
            | set(phase_v5.unique())
        )
    )
)

distribution_rows = []

for phase in all_phases:

    if phase == "":
        continue

    distribution_rows.append(
        {
            "PHASE": phase,
            "V4": int(
                phase_v4.eq(phase).sum()
            ),
            "V5": int(
                phase_v5.eq(phase).sum()
            ),
        }
    )

distribution = pd.DataFrame(
    distribution_rows
)


# ============================================================
# RECLASSIFICATION
# ============================================================

changed = out.loc[
    changed_mask
].copy()

if len(changed):

    reclass = (
        changed
        .groupby(
            [
                "PHASE_DYNAMICS_V4",
                "PHASE_DYNAMICS_V5",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="COUNT"
        )
        .sort_values(
            "COUNT",
            ascending=False,
        )
        .reset_index(drop=True)
    )

else:

    reclass = pd.DataFrame(
        columns=[
            "PHASE_DYNAMICS_V4",
            "PHASE_DYNAMICS_V5",
            "COUNT",
        ]
    )


# ============================================================
# STAGE COUNTS
# ============================================================

protected = out.loc[
    protected_mask
].copy()

if len(protected):

    stage_counts = (
        protected[
            "V5_BUY_STAGE"
        ]
        .fillna("")
        .astype(str)
        .value_counts()
        .rename_axis(
            "V5_BUY_STAGE"
        )
        .reset_index(
            name="COUNT"
        )
    )

else:

    stage_counts = pd.DataFrame(
        columns=[
            "V5_BUY_STAGE",
            "COUNT",
        ]
    )


# ============================================================
# LATEST ROW PER TICKER
# ============================================================

latest = (
    out
    .sort_values(
        [
            ticker_col,
            date_col,
        ]
    )
    .groupby(
        ticker_col,
        as_index=False,
    )
    .tail(1)
    .copy()
)

latest_cols = [
    ticker_col,
    date_col,
    "SAR_SIDE",
    "SAR_AGE",
    "V4010_RUN_QUALIFIED",
    "PHASE_DYNAMICS_V4",
    "PHASE_DYNAMICS_V5",
    "V5_BUY_LONG_BEAR_PROTECTED",
    "V5_BUY_CONFIRM_COUNT",
    "V5_BUY_CONFIRMATION_AGE",
    "V5_BUY_STAGE",
]

latest_cols = [
    c for c in latest_cols
    if c in latest.columns
]

latest = latest[
    latest_cols
].copy()


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "ROWS": rows_n,
            "TICKERS": ticker_n,
            "INPUT_TICKERS": len(tickers),
            "CHECKPOINTS_LOADED": len(parts),
            "PROCESS_ERRORS": process_errors,
            "PROTECTED_ROWS": protected_rows,
            "PROTECTED_TICKERS": protected_tickers,
            "BUY1_CONFIRMED_ROWS": confirmation_buy1_rows,
            "RESET_ROWS": reset_rows,
            "CHANGED_V4_TO_V5": changed_rows,
            "CHANGED_PCT": changed_pct,
            "SELL_CHANGED_FROM_V4": sell_changed,
            "CHANGES_OUTSIDE_PROTECTED": changes_outside_protected,
        }
    ]
)


# ============================================================
# PRINT RESULTS
# ============================================================

section("RISULTATO AGGREGATO")

print(
    f"ROWS: {rows_n}"
)

print(
    f"TICKERS: {ticker_n}"
)

print(
    f"INPUT_TICKERS: {len(tickers)}"
)

print(
    f"CHECKPOINTS_LOADED: {len(parts)}"
)

print(
    f"PROCESS_ERRORS: {process_errors}"
)

print()

print(
    f"PROTECTED_ROWS: {protected_rows}"
)

print(
    f"PROTECTED_TICKERS: {protected_tickers}"
)

print(
    f"BUY1_CONFIRMED_ROWS: "
    f"{confirmation_buy1_rows}"
)

print(
    f"RESET_ROWS: {reset_rows}"
)

print()

print(
    f"CHANGED_V4_TO_V5: "
    f"{changed_rows} "
    f"({changed_pct:.6f}%)"
)

print()

print(
    f"SELL_CHANGED_FROM_V4: "
    f"{sell_changed}"
)

print(
    f"CHANGES_OUTSIDE_PROTECTED: "
    f"{changes_outside_protected}"
)


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

section("DISTRIBUZIONE PHASE V4 vs V5")

print(
    distribution
    .to_string(index=False)
)


# ============================================================
# PRINT RECLASSIFICATION
# ============================================================

section("CAMBIAMENTI V4 -> V5")

if len(reclass):

    print(
        reclass
        .to_string(index=False)
    )

else:

    print(
        "Nessuna modifica V4 -> V5."
    )


# ============================================================
# PRINT STAGES
# ============================================================

section("BUY LONG-BEAR PROTECTED")

print(
    stage_counts
    .to_string(index=False)
)

print()

print(
    "Ticker con almeno una riga protetta:",
    protected_tickers,
)


# ============================================================
# SAFETY CHECKS
# ============================================================

section("SAFETY CHECKS")

print(
    f"SELL modificati da V5: "
    f"{sell_changed}"
)

print(
    "Modifiche V4->V5 fuori "
    f"dai run protetti: "
    f"{changes_outside_protected}"
)

if sell_changed == 0:
    print("SELL SAFETY: OK")
else:
    print("SELL SAFETY: FAIL")

if changes_outside_protected == 0:
    print("COLLATERAL SAFETY: OK")
else:
    print("COLLATERAL SAFETY: FAIL")


# ============================================================
# SAVE OUTPUT
# ============================================================

section("SAVE OUTPUT")

out.to_csv(
    OUT_ROWS,
    index=False,
)

changed.to_csv(
    OUT_CHANGED,
    index=False,
)

protected.to_csv(
    OUT_PROTECTED,
    index=False,
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

reclass.to_csv(
    OUT_RECLASS,
    index=False,
)

stage_counts.to_csv(
    OUT_STAGE,
    index=False,
)

latest.to_csv(
    OUT_LATEST,
    index=False,
)

pd.DataFrame(
    errors
).to_csv(
    OUT_ERRORS,
    index=False,
)

print(OUT_ROWS)
print(OUT_CHANGED)
print(OUT_PROTECTED)
print(OUT_SUMMARY)
print(OUT_RECLASS)
print(OUT_STAGE)
print(OUT_LATEST)
print(OUT_ERRORS)


# ============================================================
# FINAL STATUS
# ============================================================

section("VALIDAZIONE V5 FULL200 COMPLETATA")

if process_errors == 0:
    print("PROCESSING: OK")
else:
    print(
        "PROCESSING: ATTENZIONE - "
        f"{process_errors} errori/mancanti"
    )

if (
    sell_changed == 0
    and changes_outside_protected == 0
):
    print("STRUCTURAL SAFETY: OK")
else:
    print("STRUCTURAL SAFETY: FAIL")