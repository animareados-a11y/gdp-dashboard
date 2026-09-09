"""
MarketSentinel
DIAGNOSI V5 FULL200 - BUY1 TARDIVI
==================================

OBIETTIVO
---------
Analizzare semanticamente i BUY1 confermati dalla V5 RESET,
con particolare attenzione a quelli che arrivano tardi rispetto
all'inizio del nuovo run SAR bullish.

NON modifica alcun file di produzione.
NON ricalcola V5.
Legge esclusivamente l'output già prodotto dalla validazione FULL200.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/validazione_phase_dynamics_v5_full200_rows.csv"
)

LATE_THRESHOLD = 8
VERY_LATE_THRESHOLD = 10
TOP_N = 30


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

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)

buy1 = df[
    df["V5_BUY_STAGE"]
    .fillna("")
    .astype(str)
    .eq("BUY1")
].copy()

buy1["SAR_AGE"] = pd.to_numeric(
    buy1["SAR_AGE"],
    errors="coerce",
)

buy1["V4010_PREV_SAR_AGE"] = pd.to_numeric(
    buy1["V4010_PREV_SAR_AGE"],
    errors="coerce",
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 120)
print("BUY1 V5 - DISTRIBUZIONE SAR AGE")
print("=" * 120)

print(f"BUY1 TOTALI: {len(buy1)}")

if len(buy1):

    print(
        f"SAR AGE MEDIANA: "
        f"{buy1['SAR_AGE'].median():.2f}"
    )

    print(
        f"SAR AGE MEDIA: "
        f"{buy1['SAR_AGE'].mean():.2f}"
    )

    print(
        f"SAR AGE MIN: "
        f"{buy1['SAR_AGE'].min():.0f}"
    )

    print(
        f"SAR AGE MAX: "
        f"{buy1['SAR_AGE'].max():.0f}"
    )


# ============================================================
# BUCKETS
# ============================================================

def age_bucket(x):

    if pd.isna(x):
        return "NA"

    x = int(x)

    if x == 3:
        return "AGE_3"

    if 4 <= x <= 5:
        return "AGE_4_5"

    if 6 <= x <= 8:
        return "AGE_6_8"

    if 9 <= x <= 10:
        return "AGE_9_10"

    if x >= 11:
        return "AGE_11_PLUS"

    return "AGE_LT3"


buy1["AGE_BUCKET"] = (
    buy1["SAR_AGE"]
    .apply(age_bucket)
)

bucket_order = [
    "AGE_LT3",
    "AGE_3",
    "AGE_4_5",
    "AGE_6_8",
    "AGE_9_10",
    "AGE_11_PLUS",
    "NA",
]

bucket_counts = (
    buy1["AGE_BUCKET"]
    .value_counts()
    .reindex(
        bucket_order,
        fill_value=0,
    )
)

print()
print("=" * 120)
print("DISTRIBUZIONE BUY1 PER SAR AGE")
print("=" * 120)

for bucket, count in bucket_counts.items():

    pct = (
        100.0 * count / len(buy1)
        if len(buy1)
        else 0.0
    )

    print(
        f"{bucket:15s} "
        f"{count:5d} "
        f"{pct:8.3f}%"
    )


# ============================================================
# LATE / VERY LATE
# ============================================================

late = buy1[
    buy1["SAR_AGE"] > LATE_THRESHOLD
].copy()

very_late = buy1[
    buy1["SAR_AGE"] > VERY_LATE_THRESHOLD
].copy()

print()
print("=" * 120)
print("BUY1 TARDIVI")
print("=" * 120)

print(
    f"SAR AGE > {LATE_THRESHOLD}: "
    f"{len(late)}"
)

print(
    f"SAR AGE > {VERY_LATE_THRESHOLD}: "
    f"{len(very_late)}"
)


# ============================================================
# RESET HISTORY
# ============================================================

reset_col = "V5_BUY_RESET_COUNT"

if reset_col in buy1.columns:

    buy1[reset_col] = pd.to_numeric(
        buy1[reset_col],
        errors="coerce",
    ).fillna(0)

    very_late = buy1[
        buy1["SAR_AGE"] > VERY_LATE_THRESHOLD
    ].copy()

    with_reset = int(
        very_late[reset_col]
        .gt(0)
        .sum()
    )

    without_reset = int(
        very_late[reset_col]
        .eq(0)
        .sum()
    )

    print()
    print("=" * 120)
    print("BUY1 AGE > 10 - RESET PRECEDENTI")
    print("=" * 120)

    print(
        f"CON ALMENO UN RESET: {with_reset}"
    )

    print(
        f"SENZA RESET: {without_reset}"
    )


# ============================================================
# TOP VERY LATE
# ============================================================

cols = [
    "Ticker",
    "Date",
    "SAR_SIDE",
    "SAR_AGE",
    "V4010_PREV_SAR_AGE",
    "V4010_RUN_QUALIFIED",
    "HA_DIRECTION",
    "HA_INDECISION",
    "PHASE_DYNAMICS_V4",
    "PHASE_DYNAMICS_V5",
    "V5_BUY_CONFIRM_COUNT",
    "V5_BUY_CONFIRM_RESET",
    "V5_BUY_RESET_COUNT",
    "V5_BUY_CONFIRMATION_AGE",
    "V5_BUY_STAGE",
    "V5_BUY_REASON",
]

cols = [
    c for c in cols
    if c in buy1.columns
]

top = (
    very_late
    .sort_values(
        [
            "SAR_AGE",
            "Ticker",
            "Date",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )
    .head(TOP_N)
)

print()
print("=" * 120)
print(
    f"TOP {TOP_N} BUY1 CON SAR AGE > "
    f"{VERY_LATE_THRESHOLD}"
)
print("=" * 120)

if len(top):

    print(
        top[cols]
        .to_string(index=False)
    )

else:

    print(
        "Nessun BUY1 con SAR AGE > "
        f"{VERY_LATE_THRESHOLD}."
    )


# ============================================================
# TICKER CON PIU BUY1 TARDIVI
# ============================================================

print()
print("=" * 120)
print("TICKER CON PIU BUY1 AGE > 10")
print("=" * 120)

if len(very_late):

    counts = (
        very_late["Ticker"]
        .astype(str)
        .value_counts()
        .head(20)
    )

    print(
        counts.to_string()
    )

else:

    print("Nessun caso.")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 120)
print("DIAGNOSI COMPLETATA")
print("=" * 120)