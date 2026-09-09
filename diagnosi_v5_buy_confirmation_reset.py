"""
MarketSentinel
DIAGNOSI V5 - BUY CONFIRMATION RESET
====================================

OBIETTIVO
---------
Testare una variante della logica BUY long-bear V5:

- dopo almeno N=6 settimane di precedente regime SAR bearish
- il nuovo run SAR bullish NON genera subito BUY
- servono K=3 HA bullish decisive

CONTEGGIO:
- HA bullish decisiva:
    +1
- HA indecisione:
    mantiene il conteggio
- HA bearish decisiva:
    NON resetta automaticamente
- HA bearish decisiva + debolezza rialzista riconosciuta:
    RESET del conteggio a 0

La debolezza rialzista viene considerata riconosciuta quando:

    PHASE_DYNAMICS_V4 == "DEBOLEZZA_RIALZISTA"

oppure, come supporto diagnostico:

    WEAK_UP_TRIGGER == 1

Il test confronta:

1. CUMULATIVE originale V5
2. RESET candidato

NON modifica V5.
NON modifica V4.
NON modifica production.
NON modifica engine.py.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/validazione_phase_dynamics_v5_italia_rows.csv"
)

N_PREV_BEAR_MIN = 6
K_CONFIRM = 3


# ============================================================
# UTILS
# ============================================================

def detect_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def num(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# LOAD
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File non trovato: {INPUT_FILE}"
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

df = (
    df
    .sort_values(
        [ticker_col, date_col]
    )
    .reset_index(drop=True)
)


required = [
    "SAR_SIDE",
    "SAR_AGE",
    "HA_DIRECTION",
    "HA_INDECISION",
    "PHASE_DYNAMICS_V4",
    "V5_BUY_LONG_BEAR_PROTECTED",
    "V5_BUY_CONFIRM_COUNT",
    "V5_BUY_STAGE",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise KeyError(
        f"Colonne mancanti: {missing}"
    )


# ============================================================
# RESET LOGIC
# ============================================================

df["RESET_CONFIRM_COUNT"] = 0
df["RESET_OCCURRED"] = 0
df["RESET_CONFIRM_REACHED"] = 0
df["RESET_CONFIRM_WEEK"] = np.nan
df["RESET_STAGE"] = ""
df["RESET_REASON"] = ""


for ticker, idx in df.groupby(
    ticker_col,
    sort=False,
).groups.items():

    idx = list(idx)

    protected = (
        num(
            df.loc[
                idx,
                "V5_BUY_LONG_BEAR_PROTECTED",
            ]
        )
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    sar_side = (
        num(
            df.loc[idx, "SAR_SIDE"]
        )
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    ha_dir = (
        num(
            df.loc[idx, "HA_DIRECTION"]
        )
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    ha_ind = (
        num(
            df.loc[idx, "HA_INDECISION"]
        )
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    phase_v4 = (
        df.loc[
            idx,
            "PHASE_DYNAMICS_V4",
        ]
        .astype(str)
        .to_numpy()
    )

    if "WEAK_UP_TRIGGER" in df.columns:
        weak_up_trigger = (
            num(
                df.loc[
                    idx,
                    "WEAK_UP_TRIGGER",
                ]
            )
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
    else:
        weak_up_trigger = np.zeros(
            len(idx),
            dtype=int,
        )

    count = 0
    reached = False
    confirmation_week = None

    for j, row_idx in enumerate(idx):

        # fuori dal run protetto:
        # resetta stato interno
        if protected[j] != 1:
            count = 0
            reached = False
            confirmation_week = None
            continue

        # se il SAR non è bullish,
        # il run protetto è terminato
        if sar_side[j] != 1:
            count = 0
            reached = False
            confirmation_week = None
            continue

        decisive_bull = (
            ha_dir[j] == 1
            and ha_ind[j] == 0
        )

        decisive_bear = (
            ha_dir[j] == -1
            and ha_ind[j] == 0
        )

        weakness_recognized = (
            phase_v4[j]
            == "DEBOLEZZA_RIALZISTA"
        )

        weakness_trigger = (
            weak_up_trigger[j] == 1
        )

        # ====================================================
        # RESET
        # ====================================================
        #
        # Una HA bearish decisiva da sola
        # NON cancella il test.
        #
        # Il reset avviene quando abbiamo:
        #
        # - HA bearish decisiva
        # - e debolezza rialzista già riconosciuta
        #
        # PHASE V4 è la condizione principale.
        # WEAK_UP_TRIGGER viene salvato nel motivo
        # per analisi diagnostica.
        # ====================================================

        if (
            not reached
            and decisive_bear
            and weakness_recognized
        ):
            if count > 0:
                df.at[
                    row_idx,
                    "RESET_OCCURRED",
                ] = 1

            count = 0

            if weakness_trigger:
                reason = (
                    "RESET_DECISIVE_BEAR_"
                    "PLUS_WEAK_UP_PHASE_"
                    "PLUS_TRIGGER"
                )
            else:
                reason = (
                    "RESET_DECISIVE_BEAR_"
                    "PLUS_WEAK_UP_PHASE"
                )

            df.at[
                row_idx,
                "RESET_REASON",
            ] = reason

        # ====================================================
        # BULLISH CONFIRMATION
        # ====================================================

        elif (
            not reached
            and decisive_bull
        ):
            count += 1

            if count >= K_CONFIRM:
                reached = True
                confirmation_week = row_idx

                df.at[
                    row_idx,
                    "RESET_CONFIRM_REACHED",
                ] = 1

                df.at[
                    row_idx,
                    "RESET_CONFIRM_WEEK",
                ] = 1

                df.at[
                    row_idx,
                    "RESET_REASON",
                ] = (
                    "BUY_CONFIRMATION_REACHED"
                )

        # indecisione:
        # nessun incremento
        # nessun reset

        df.at[
            row_idx,
            "RESET_CONFIRM_COUNT",
        ] = count

        # ====================================================
        # STAGE
        # ====================================================

        if not reached:

            df.at[
                row_idx,
                "RESET_STAGE",
            ] = f"TEST_{count}"

        else:

            # calcoliamo l'età dal punto
            # in cui è stata raggiunta K=3
            if confirmation_week is None:
                continue

            positions = idx.index(
                confirmation_week
            )

            confirm_age = (
                j - positions + 1
            )

            df.at[
                row_idx,
                "RESET_CONFIRM_WEEK",
            ] = confirm_age

            if confirm_age == 1:
                stage = "BUY1"

            elif confirm_age == 2:
                stage = "BUY2"

            elif confirm_age == 3:
                stage = "BUY3"

            else:
                stage = "POST_BUY3"

            df.at[
                row_idx,
                "RESET_STAGE",
            ] = stage


# ============================================================
# SUMMARY
# ============================================================

protected_df = df[
    num(
        df[
            "V5_BUY_LONG_BEAR_PROTECTED"
        ]
    )
    .fillna(0)
    .eq(1)
].copy()

print()
print("=" * 120)
print("SUMMARY")
print("=" * 120)

print(
    "RIGHE PROTETTE:",
    len(protected_df),
)

print(
    "RESET TOTALI:",
    int(
        protected_df[
            "RESET_OCCURRED"
        ].sum()
    ),
)

print(
    "CONFERME V5 CUMULATIVE:",
    int(
        (
            protected_df[
                "V5_BUY_STAGE"
            ]
            .astype(str)
            == "BUY1"
        ).sum()
    ),
)

print(
    "CONFERME RESET:",
    int(
        (
            protected_df[
                "RESET_STAGE"
            ]
            .astype(str)
            == "BUY1"
        ).sum()
    ),
)


# ============================================================
# BUY1 DISTRIBUTION
# ============================================================

reset_buy1 = protected_df[
    protected_df[
        "RESET_STAGE"
    ].astype(str).eq("BUY1")
].copy()

if len(reset_buy1):

    ages = num(
        reset_buy1["SAR_AGE"]
    )

    print()
    print(
        "RESET BUY1 TOTALI:",
        len(reset_buy1),
    )

    print(
        "SAR AGE MEDIANA:",
        ages.median(),
    )

    print(
        "SAR AGE MEDIA:",
        round(
            ages.mean(),
            2,
        ),
    )

    print(
        "SAR AGE MIN/MAX:",
        int(ages.min()),
        int(ages.max()),
    )

    print(
        "BUY1 AGE 3:",
        int(
            (ages == 3).sum()
        ),
    )

    print(
        "BUY1 AGE 4-5:",
        int(
            ages.between(
                4,
                5,
            ).sum()
        ),
    )

    print(
        "BUY1 AGE 6-8:",
        int(
            ages.between(
                6,
                8,
            ).sum()
        ),
    )

    print(
        "BUY1 AGE 9-10:",
        int(
            ages.between(
                9,
                10,
            ).sum()
        ),
    )

    print(
        "BUY1 AGE 11+:",
        int(
            (ages >= 11).sum()
        ),
    )


# ============================================================
# TARGETED A2A
# ============================================================

print()
print("=" * 120)
print("A2A.MI 2025")
print("=" * 120)

a2a = df[
    (
        df[ticker_col].astype(str)
        == "A2A.MI"
    )
    &
    (
        df[date_col]
        >= pd.Timestamp(
            "2025-06-01"
        )
    )
    &
    (
        df[date_col]
        <= pd.Timestamp(
            "2025-09-05"
        )
    )
].copy()

cols = [
    date_col,
    "SAR_SIDE",
    "SAR_AGE",
    "HA_DIRECTION",
    "HA_INDECISION",
    "WEAK_UP_TRIGGER",
    "PHASE_DYNAMICS_V4",
    "V5_BUY_CONFIRM_COUNT",
    "V5_BUY_STAGE",
    "RESET_OCCURRED",
    "RESET_CONFIRM_COUNT",
    "RESET_STAGE",
    "RESET_REASON",
]

cols = [
    c for c in cols
    if c in a2a.columns
]

print(
    a2a[cols]
    .to_string(index=False)
)


# ============================================================
# TARGETED ISRG
# ============================================================

print()
print("=" * 120)
print("ISRG 2026")
print("=" * 120)

isrg = df[
    (
        df[ticker_col].astype(str)
        == "ISRG"
    )
    &
    (
        df[date_col]
        >= pd.Timestamp(
            "2026-06-01"
        )
    )
    &
    (
        df[date_col]
        <= pd.Timestamp(
            "2026-08-31"
        )
    )
].copy()

print(
    isrg[cols]
    .to_string(index=False)
)


# ============================================================
# DIFFERENCE BETWEEN CUMULATIVE AND RESET
# ============================================================

print()
print("=" * 120)
print("DIFFERENZE BUY1 CUMULATIVE vs RESET")
print("=" * 120)

cum_buy1 = protected_df[
    protected_df[
        "V5_BUY_STAGE"
    ].astype(str).eq("BUY1")
].copy()

cum_keys = set(
    zip(
        cum_buy1[ticker_col].astype(str),
        cum_buy1[date_col].astype(str),
    )
)

reset_keys = set(
    zip(
        reset_buy1[ticker_col].astype(str),
        reset_buy1[date_col].astype(str),
    )
)

lost = cum_keys - reset_keys
gained = reset_keys - cum_keys

print(
    "BUY1 CUMULATIVE NON PIU' BUY1 CON RESET:",
    len(lost),
)

print(
    "BUY1 RESET SU DATA DIVERSA:",
    len(gained),
)


# ============================================================
# LATE CASES RESET
# ============================================================

print()
print("=" * 120)
print("BUY1 RESET CON SAR_AGE > 10")
print("=" * 120)

late = reset_buy1[
    num(
        reset_buy1["SAR_AGE"]
    ) > 10
].copy()

if len(late):

    show = [
        ticker_col,
        date_col,
        "SAR_AGE",
        "HA_DIRECTION",
        "HA_INDECISION",
        "PHASE_DYNAMICS_V4",
        "RESET_CONFIRM_COUNT",
        "RESET_STAGE",
    ]

    print(
        late[show]
        .to_string(index=False)
    )

else:
    print("NESSUN CASO")


print()
print("=" * 120)
print("DIAGNOSI COMPLETATA")
print("=" * 120)