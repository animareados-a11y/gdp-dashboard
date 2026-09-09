"""
MarketSentinel
DIAGNOSI V5 - BUY CONFIRMATION RESET V2
======================================

REGOLA TESTATA
--------------
Nei BUY protetti dopo un precedente lungo regime bearish:

- HA bullish decisiva  -> +1
- HA indecisione       -> mantiene il conteggio
- HA bearish decisiva  -> RESET a 0

Quindi K=3 significa:
3 HA bullish decisive senza che tra loro compaia
una HA bearish decisiva.

Le HA di indecisione sono neutrali.

NON modifica V5.
NON modifica V4.
NON modifica production.
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/validazione_phase_dynamics_v5_italia_rows.csv"
)

K_CONFIRM = 3


def detect_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def num(s):
    return pd.to_numeric(
        s,
        errors="coerce",
    )


# ============================================================
# LOAD
# ============================================================

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
    raise KeyError("Colonna data non trovata.")

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


# ============================================================
# OUTPUT DIAGNOSTICI
# ============================================================

df["RESET_V2_COUNT"] = 0
df["RESET_V2_OCCURRED"] = 0
df["RESET_V2_STAGE"] = ""
df["RESET_V2_CONFIRM_AGE"] = 0


# ============================================================
# PROCESS
# ============================================================

for ticker, positions in df.groupby(
    ticker_col,
    sort=False,
).indices.items():

    positions = list(positions)

    count = 0
    confirmed = False
    confirm_pos = None

    for local_pos, row_idx in enumerate(positions):

        protected = int(
            num(
                pd.Series(
                    [
                        df.at[
                            row_idx,
                            "V5_BUY_LONG_BEAR_PROTECTED",
                        ]
                    ]
                )
            )
            .fillna(0)
            .iloc[0]
        )

        sar_side = int(
            num(
                pd.Series(
                    [df.at[row_idx, "SAR_SIDE"]]
                )
            )
            .fillna(0)
            .iloc[0]
        )

        # fuori dal run BUY protetto
        if protected != 1 or sar_side != 1:
            count = 0
            confirmed = False
            confirm_pos = None
            continue

        ha_dir = int(
            num(
                pd.Series(
                    [
                        df.at[
                            row_idx,
                            "HA_DIRECTION",
                        ]
                    ]
                )
            )
            .fillna(0)
            .iloc[0]
        )

        ha_ind = int(
            num(
                pd.Series(
                    [
                        df.at[
                            row_idx,
                            "HA_INDECISION",
                        ]
                    ]
                )
            )
            .fillna(0)
            .iloc[0]
        )

        decisive_bull = (
            ha_dir == 1
            and ha_ind == 0
        )

        decisive_bear = (
            ha_dir == -1
            and ha_ind == 0
        )

        # ====================================================
        # PRIMA DELLA CONFERMA
        # ====================================================

        if not confirmed:

            # una vera HA bearish invalida
            # le conferme bullish precedenti
            if decisive_bear:

                if count > 0:
                    df.at[
                        row_idx,
                        "RESET_V2_OCCURRED",
                    ] = 1

                count = 0

            elif decisive_bull:

                count += 1

                if count >= K_CONFIRM:
                    confirmed = True
                    confirm_pos = local_pos

        df.at[
            row_idx,
            "RESET_V2_COUNT",
        ] = count

        # ====================================================
        # STAGE
        # ====================================================

        if not confirmed:

            df.at[
                row_idx,
                "RESET_V2_STAGE",
            ] = f"TEST_{count}"

        else:

            confirm_age = (
                local_pos
                - confirm_pos
                + 1
            )

            df.at[
                row_idx,
                "RESET_V2_CONFIRM_AGE",
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
                "RESET_V2_STAGE",
            ] = stage


# ============================================================
# SUMMARY
# ============================================================

protected = df[
    num(
        df["V5_BUY_LONG_BEAR_PROTECTED"]
    )
    .fillna(0)
    .eq(1)
].copy()

cum_buy1 = protected[
    protected[
        "V5_BUY_STAGE"
    ].astype(str).eq("BUY1")
].copy()

reset_buy1 = protected[
    protected[
        "RESET_V2_STAGE"
    ].astype(str).eq("BUY1")
].copy()


print()
print("=" * 100)
print("SUMMARY RESET V2")
print("=" * 100)

print(
    "RIGHE PROTETTE:",
    len(protected),
)

print(
    "RESET TOTALI:",
    int(
        protected[
            "RESET_V2_OCCURRED"
        ].sum()
    ),
)

print(
    "BUY1 CUMULATIVE V5:",
    len(cum_buy1),
)

print(
    "BUY1 RESET V2:",
    len(reset_buy1),
)


if len(reset_buy1):

    ages = num(
        reset_buy1["SAR_AGE"]
    )

    print()
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
        "AGE 3:",
        int((ages == 3).sum()),
    )

    print(
        "AGE 4-5:",
        int(
            ages.between(4, 5).sum()
        ),
    )

    print(
        "AGE 6-8:",
        int(
            ages.between(6, 8).sum()
        ),
    )

    print(
        "AGE 9-10:",
        int(
            ages.between(9, 10).sum()
        ),
    )

    print(
        "AGE 11+:",
        int(
            (ages >= 11).sum()
        ),
    )

    print(
        "MAX AGE:",
        int(ages.max()),
    )


# ============================================================
# DIFFERENZE
# ============================================================

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

print()
print(
    "BUY1 V5 eliminati/spostati:",
    len(cum_keys - reset_keys),
)

print(
    "BUY1 V2 su nuova data:",
    len(reset_keys - cum_keys),
)


# ============================================================
# A2A 2025
# ============================================================

print()
print("=" * 100)
print("A2A.MI 2025")
print("=" * 100)

x = df[
    (
        df[ticker_col].astype(str)
        == "A2A.MI"
    )
    &
    (
        df[date_col]
        >= pd.Timestamp("2025-06-01")
    )
    &
    (
        df[date_col]
        <= pd.Timestamp("2025-09-05")
    )
].copy()

cols = [
    date_col,
    "SAR_AGE",
    "HA_DIRECTION",
    "HA_INDECISION",
    "PHASE_DYNAMICS_V4",
    "V5_BUY_CONFIRM_COUNT",
    "V5_BUY_STAGE",
    "RESET_V2_OCCURRED",
    "RESET_V2_COUNT",
    "RESET_V2_STAGE",
]

print(
    x[cols]
    .to_string(index=False)
)


# ============================================================
# LATE BUY1
# ============================================================

print()
print("=" * 100)
print("RESET V2 BUY1 SAR_AGE > 10")
print("=" * 100)

late = reset_buy1[
    num(
        reset_buy1["SAR_AGE"]
    ) > 10
].copy()

if len(late):

    cols = [
        ticker_col,
        date_col,
        "SAR_AGE",
        "HA_DIRECTION",
        "HA_INDECISION",
        "PHASE_DYNAMICS_V4",
        "RESET_V2_COUNT",
        "RESET_V2_STAGE",
    ]

    print(
        late[cols]
        .to_string(index=False)
    )

else:
    print("NESSUN CASO")


print()
print("=" * 100)
print("DIAGNOSI COMPLETATA")
print("=" * 100)