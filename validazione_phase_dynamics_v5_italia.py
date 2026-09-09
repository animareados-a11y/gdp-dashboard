"""
MarketSentinel
VALIDAZIONE PHASE DYNAMICS V5 - ITALIA40
========================================

OBIETTIVO
---------
Validare lo SHADOW V5 su Italia40 confrontandolo con V4.

Controlli principali:

1. Quante righe cambiano da V4 a V5
2. Quanti BUY vengono modificati
3. Verificare che SELL non venga modificato dalla nuova logica BUY-only
4. Distribuzione PHASE V4 vs V5
5. Controllo sicurezza:
   - nessun SELL modificato
   - nessuna modifica fuori dai run BUY protetti
6. Ultima settimana per ticker
7. Casi BUY protetti N=6 / K=3

IMPORTANTE
----------
- NON modifica production
- NON modifica V40.10
- NON modifica V4
- NON modifica engine.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v5 as v5

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

OUT_DIR = Path("data")

OUT_ROWS = OUT_DIR / "validazione_phase_dynamics_v5_italia_rows.csv"
OUT_CHANGED = OUT_DIR / "validazione_phase_dynamics_v5_italia_changed.csv"
OUT_PROTECTED = OUT_DIR / "validazione_phase_dynamics_v5_italia_protected.csv"
OUT_SUMMARY = OUT_DIR / "validazione_phase_dynamics_v5_italia_summary.csv"


# Italia40
ITALIA40 = [
    'A2A.MI',
    'AMP.MI',
    'AVIO.MI',
    'AZM.MI',
    'BAMI.MI',
    'BC.MI',
    'BMED.MI',
    'BMPS.MI',
    'BPE.MI',
    'BZU.MI',
    'CPR.MI',
    'DIA.MI',
    'ENEL.MI',
    'ENI.MI',
    'FBK.MI',
    'FCT.MI',
    'G.MI',
    'HER.MI',
    'IG.MI',
    'INW.MI',
    'ISP.MI',
    'IVG.MI',
    'LDO.MI',
    'LTMC.MI',
    'MB.MI',
    'MONC.MI',
    'NEXI.MI',
    'PRY.MI',
    'PST.MI',
    'RACE.MI',
    'REC.MI',
    'SPM.MI',
    'SRG.MI',
    'STLAM.MI',
    'STMMI.MI',
    'TEN.MI',
    'TIT.MI',
    'TRN.MI',
    'UCG.MI',
    'UNI.MI',
]


# ============================================================
# UTILS
# ============================================================

def section(title: str):
    print()
    print("=" * 130)
    print(title)
    print("=" * 130)


def detect_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ============================================================
# LOAD
# ============================================================

section("LOAD DATA")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input non trovato: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

ticker_col = detect_col(
    df,
    ["Ticker", "TICKER", "ticker"],
)

date_col = detect_col(
    df,
    ["Date", "DATE", "date"],
)

if ticker_col is None:
    raise KeyError("Colonna ticker non trovata.")

if date_col is None:
    raise KeyError("Colonna date non trovata.")

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce",
)

available = sorted(
    set(df[ticker_col].astype(str).unique())
)

italia = [
    t for t in ITALIA40
    if t in available
]

missing = [
    t for t in ITALIA40
    if t not in available
]

print(f"Input rows: {len(df):,}")
print(f"Ticker input: {df[ticker_col].nunique()}")
print(f"Italia40 trovati: {len(italia)}")

if missing:
    print("Ticker Italia40 non trovati:")
    print(missing)


# ============================================================
# PROCESS
# ============================================================

section("PROCESS ITALIA40")

parts = []
errors = []

for i, ticker in enumerate(italia, 1):

    g = (
        df[df[ticker_col].astype(str) == ticker]
        .copy()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    try:
        out = v5.process_ticker(g)

        if ticker_col not in out.columns:
            out[ticker_col] = ticker

        parts.append(out)

        print(
            f"[{i:02d}/{len(italia):02d}] "
            f"{ticker:10s} rows={len(out):4d}"
        )

    except Exception as exc:

        errors.append(
            {
                "Ticker": ticker,
                "Error": repr(exc),
            }
        )

        print(
            f"[ERROR] {ticker}: {repr(exc)}"
        )


if not parts:
    raise RuntimeError(
        "Nessun ticker processato correttamente."
    )

out = pd.concat(
    parts,
    ignore_index=True,
)


# ============================================================
# BASIC FLAGS
# ============================================================

if "PHASE_DYNAMICS_V4" not in out.columns:
    raise KeyError(
        "PHASE_DYNAMICS_V4 non presente nell'output V5."
    )

if "PHASE_DYNAMICS_V5" not in out.columns:
    raise KeyError(
        "PHASE_DYNAMICS_V5 non presente nell'output V5."
    )

out["V5_CHANGED_FROM_V4"] = (
    out["PHASE_DYNAMICS_V4"].astype(str)
    !=
    out["PHASE_DYNAMICS_V5"].astype(str)
).astype(int)

out["V5_SELL_CHANGED"] = (
    (out["PHASE_DYNAMICS_V4"].astype(str) == "SELL")
    &
    (
        out["PHASE_DYNAMICS_V5"].astype(str)
        != "SELL"
    )
).astype(int)

out["V5_CHANGED_OUTSIDE_PROTECTED"] = (
    (out["V5_CHANGED_FROM_V4"] == 1)
    &
    (
        pd.to_numeric(
            out["V5_BUY_LONG_BEAR_PROTECTED"],
            errors="coerce",
        ).fillna(0) != 1
    )
).astype(int)


# ============================================================
# SUMMARY
# ============================================================

section("RISULTATO AGGREGATO")

rows = len(out)

changed = int(
    out["V5_CHANGED_FROM_V4"].sum()
)

changed_pct = (
    100 * changed / rows
    if rows
    else np.nan
)

protected = int(
    pd.to_numeric(
        out["V5_BUY_LONG_BEAR_PROTECTED"],
        errors="coerce",
    ).fillna(0).sum()
)

confirm_reached_rows = int(
    pd.to_numeric(
        out["V5_BUY_CONFIRMATION_REACHED"],
        errors="coerce",
    ).fillna(0).sum()
)

sell_changed = int(
    out["V5_SELL_CHANGED"].sum()
)

outside = int(
    out["V5_CHANGED_OUTSIDE_PROTECTED"].sum()
)

print(f"ROWS: {rows}")
print(f"TICKERS: {out[ticker_col].nunique()}")
print(f"PROCESS_ERRORS: {len(errors)}")
print()
print(f"PROTECTED_ROWS: {protected}")
print(f"CONFIRMATION_REACHED_ROWS: {confirm_reached_rows}")
print()
print(
    f"CHANGED_V4_TO_V5: {changed} "
    f"({changed_pct:.6f}%)"
)
print()
print(
    f"SELL_CHANGED_FROM_V4: {sell_changed}"
)
print(
    f"CHANGES_OUTSIDE_PROTECTED: {outside}"
)


# ============================================================
# DISTRIBUTION
# ============================================================

section("DISTRIBUZIONE PHASE V4 vs V5")

phase_values = sorted(
    set(
        out["PHASE_DYNAMICS_V4"]
        .dropna()
        .astype(str)
    )
    |
    set(
        out["PHASE_DYNAMICS_V5"]
        .dropna()
        .astype(str)
    )
)

dist_rows = []

for phase in phase_values:

    dist_rows.append(
        {
            "PHASE": phase,
            "V4": int(
                (
                    out["PHASE_DYNAMICS_V4"]
                    .astype(str)
                    == phase
                ).sum()
            ),
            "V5": int(
                (
                    out["PHASE_DYNAMICS_V5"]
                    .astype(str)
                    == phase
                ).sum()
            ),
        }
    )

dist = pd.DataFrame(dist_rows)

print(
    dist.to_string(index=False)
)


# ============================================================
# RECLASSIFICATION MATRIX
# ============================================================

section("CAMBIAMENTI V4 -> V5")

changed_df = out[
    out["V5_CHANGED_FROM_V4"] == 1
].copy()

if len(changed_df):

    matrix = (
        changed_df
        .groupby(
            [
                "PHASE_DYNAMICS_V4",
                "PHASE_DYNAMICS_V5",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="COUNT")
        .sort_values(
            "COUNT",
            ascending=False,
        )
    )

    print(
        matrix.to_string(index=False)
    )

else:
    print("Nessun cambiamento.")


# ============================================================
# PROTECTED BUY RUNS
# ============================================================

section("BUY LONG-BEAR PROTECTED")

protected_df = out[
    pd.to_numeric(
        out["V5_BUY_LONG_BEAR_PROTECTED"],
        errors="coerce",
    ).fillna(0) == 1
].copy()

if len(protected_df):

    stage_counts = (
        protected_df[
            "V5_BUY_STAGE"
        ]
        .astype(str)
        .value_counts(
            dropna=False
        )
        .rename_axis("V5_BUY_STAGE")
        .reset_index(name="COUNT")
    )

    print(stage_counts.to_string(index=False))

    print()
    print(
        "Ticker con almeno una riga protetta:",
        protected_df[ticker_col].nunique(),
    )

else:
    print("Nessuna riga protetta.")


# ============================================================
# SAFETY CHECKS
# ============================================================

section("SAFETY CHECKS")

print(
    "SELL modificati da V5:",
    sell_changed,
)

print(
    "Modifiche V4->V5 fuori dai run protetti:",
    outside,
)

if sell_changed == 0:
    print("SELL SAFETY: OK")
else:
    print("SELL SAFETY: FAIL")

if outside == 0:
    print("COLLATERAL SAFETY: OK")
else:
    print("COLLATERAL SAFETY: FAIL")


# ============================================================
# LATEST WEEK
# ============================================================

section("ULTIMA SETTIMANA PER TICKER")

latest_idx = (
    out
    .sort_values(date_col)
    .groupby(ticker_col)[date_col]
    .idxmax()
)

latest = (
    out.loc[latest_idx]
    .copy()
    .sort_values(ticker_col)
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

print(
    latest[latest_cols]
    .to_string(index=False)
)


# ============================================================
# IMPORTANT TARGETED CASES
# ============================================================

section("TARGETED CASES")

targets = [
    ("ISRG", "2026-06-01", "2026-08-31"),
    ("CRWD", "2026-06-01", "2026-08-31"),
    ("UCG.MI", "2023-11-01", "2024-01-31"),
    ("FBK.MI", "2026-05-01", "2026-08-31"),
]

for ticker, start, end in targets:

    q = out[
        (
            out[ticker_col].astype(str)
            == ticker
        )
        &
        (
            out[date_col]
            >= pd.Timestamp(start)
        )
        &
        (
            out[date_col]
            <= pd.Timestamp(end)
        )
    ].copy()

    if not len(q):
        continue

    print()
    print("-" * 130)
    print(
        f"{ticker} | {start} -> {end}"
    )
    print("-" * 130)

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
        "V5_BUY_CONFIRM_COUNT",
        "V5_BUY_CONFIRMATION_AGE",
        "V5_BUY_STAGE",
        "V5_BUY_REASON",
    ]

    cols = [
        c for c in cols
        if c in q.columns
    ]

    print(
        q[cols].to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

section("SAVE OUTPUT")

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT_ROWS,
    index=False,
)

changed_df.to_csv(
    OUT_CHANGED,
    index=False,
)

protected_df.to_csv(
    OUT_PROTECTED,
    index=False,
)

summary = pd.DataFrame(
    [
        {
            "ROWS": rows,
            "TICKERS": int(
                out[ticker_col].nunique()
            ),
            "PROCESS_ERRORS": len(errors),
            "PROTECTED_ROWS": protected,
            "CONFIRMATION_REACHED_ROWS": confirm_reached_rows,
            "CHANGED_V4_TO_V5": changed,
            "CHANGED_PCT": changed_pct,
            "SELL_CHANGED_FROM_V4": sell_changed,
            "CHANGES_OUTSIDE_PROTECTED": outside,
        }
    ]
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

print(OUT_ROWS)
print(OUT_CHANGED)
print(OUT_PROTECTED)
print(OUT_SUMMARY)

if errors:
    print()
    print("ERRORI:")
    print(
        pd.DataFrame(errors)
        .to_string(index=False)
    )


section("VALIDAZIONE V5 ITALIA COMPLETATA")