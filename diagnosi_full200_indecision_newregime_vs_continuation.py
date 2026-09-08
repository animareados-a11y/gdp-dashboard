"""
MarketSentinel
FULL200 - INDECISION
NEW REGIME vs CONTINUATION DIAGNOSTIC

INPUT
-----
data/diagnosi_full200_indecision_exits.csv

OBIETTIVO
---------
Capire cosa succede realmente alla prima settimana dopo INDECISIONE:

1. nuovo regime SAR?
2. continuazione dello stesso regime?
3. BUY/SELL con quale SAR_AGE?
4. TREND_RIALZISTA / TREND_RIBASSISTA con quale SAR_AGE?
5. quanto spesso cambia effettivamente il lato SAR?

Nessun motore viene ricalcolato o modificato.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/diagnosi_full200_indecision_exits.csv"
)

OUT_DETAIL = Path(
    "data/diagnosi_full200_indecision_newregime_vs_continuation.csv"
)

OUT_SUMMARY = Path(
    "data/diagnosi_full200_indecision_newregime_vs_continuation_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 105)
    print(text)
    print("=" * 105)


def classify_sar_transition(row):

    last_side = row["last_sar_side"]
    exit_side = row["exit_sar_side"]
    exit_age = row["exit_sar_age"]

    if pd.isna(exit_side):
        return "NO_EXIT"

    if pd.isna(last_side):
        return "UNKNOWN"

    # Nuovo lato SAR
    if exit_side != last_side:
        return "SAR_SIDE_CHANGED"

    # Stesso lato ma SAR_AGE riparte da 1
    if pd.notna(exit_age) and int(exit_age) == 1:
        return "SAR_RESTART_AGE1"

    # Stesso regime che prosegue
    return "SAME_SAR_REGIME"


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "MARKETSENTINEL - INDECISION: NEW REGIME vs CONTINUATION"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Non trovo {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    print(
        f"\nEpisodi caricati: {len(df):,}"
    )

    # ========================================================
    # CLASSIFICAZIONE TRANSIZIONE SAR
    # ========================================================

    df["sar_transition"] = df.apply(
        classify_sar_transition,
        axis=1,
    )

    df["is_new_sar_regime"] = (
        df["sar_transition"].isin(
            [
                "SAR_SIDE_CHANGED",
                "SAR_RESTART_AGE1",
            ]
        )
    ).astype(int)

    df["is_same_sar_regime"] = (
        df["sar_transition"]
        ==
        "SAME_SAR_REGIME"
    ).astype(int)

    # ========================================================
    # SALVA DETTAGLIO
    # ========================================================

    OUT_DETAIL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT_DETAIL,
        index=False,
    )

    # ========================================================
    # 1. TRANSIZIONE SAR GENERALE
    # ========================================================

    banner(
        "1 - TRANSIZIONE SAR ALL'USCITA DA INDECISIONE"
    )

    sar_counts = (
        df["sar_transition"]
        .value_counts(dropna=False)
        .rename_axis("sar_transition")
        .reset_index(name="episodes")
    )

    sar_counts["pct"] = (
        sar_counts["episodes"]
        /
        len(df)
        *
        100
    )

    print(
        sar_counts.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}",
        )
    )

    # ========================================================
    # 2. EXIT PHASE vs TRANSIZIONE SAR
    # ========================================================

    banner(
        "2 - EXIT PHASE vs TRANSIZIONE SAR"
    )

    phase_sar = (
        df
        .groupby(
            [
                "exit_phase",
                "sar_transition",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="episodes")
        .sort_values(
            [
                "exit_phase",
                "episodes",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print(
        phase_sar.to_string(
            index=False,
        )
    )

    # ========================================================
    # 3. DISTRIBUZIONE SAR AGE PER BUY
    # ========================================================

    banner(
        "3 - BUY: DISTRIBUZIONE EXIT SAR_AGE"
    )

    buy = df[
        df["exit_phase"] == "BUY"
    ].copy()

    buy_age = (
        buy["exit_sar_age"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("exit_sar_age")
        .reset_index(name="episodes")
    )

    buy_age["pct_buy"] = (
        buy_age["episodes"]
        /
        len(buy)
        *
        100
    )

    print(
        buy_age.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}",
        )
    )

    # ========================================================
    # 4. DISTRIBUZIONE SAR AGE PER SELL
    # ========================================================

    banner(
        "4 - SELL: DISTRIBUZIONE EXIT SAR_AGE"
    )

    sell = df[
        df["exit_phase"] == "SELL"
    ].copy()

    sell_age = (
        sell["exit_sar_age"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("exit_sar_age")
        .reset_index(name="episodes")
    )

    sell_age["pct_sell"] = (
        sell_age["episodes"]
        /
        len(sell)
        *
        100
    )

    print(
        sell_age.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}",
        )
    )

    # ========================================================
    # 5. TREND RIALZISTA: SAR AGE
    # ========================================================

    banner(
        "5 - TREND_RIALZISTA: EXIT SAR_AGE"
    )

    up = df[
        df["exit_phase"]
        ==
        "TREND_RIALZISTA"
    ].copy()

    if up.empty:
        print("Nessun caso.")
    else:
        up_age = (
            up["exit_sar_age"]
            .value_counts(dropna=False)
            .sort_index()
            .rename_axis("exit_sar_age")
            .reset_index(name="episodes")
        )

        print(
            up_age.to_string(
                index=False,
            )
        )

    # ========================================================
    # 6. TREND RIBASSISTA: SAR AGE
    # ========================================================

    banner(
        "6 - TREND_RIBASSISTA: EXIT SAR_AGE"
    )

    down = df[
        df["exit_phase"]
        ==
        "TREND_RIBASSISTA"
    ].copy()

    if down.empty:
        print("Nessun caso.")
    else:
        down_age = (
            down["exit_sar_age"]
            .value_counts(dropna=False)
            .sort_index()
            .rename_axis("exit_sar_age")
            .reset_index(name="episodes")
        )

        print(
            down_age.to_string(
                index=False,
            )
        )

    # ========================================================
    # 7. DIREZIONE PRECEDENTE + EXIT PHASE + SAR TRANSITION
    # ========================================================

    banner(
        "7 - DIREZIONE PRECEDENTE / EXIT PHASE / SAR"
    )

    combo = (
        df
        .groupby(
            [
                "prev_direction",
                "exit_phase",
                "sar_transition",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="episodes")
        .sort_values(
            "episodes",
            ascending=False,
        )
    )

    print(
        combo.to_string(
            index=False,
        )
    )

    # ========================================================
    # 8. BUY/SELL SENZA CAMBIO SAR
    # ========================================================

    banner(
        "8 - BUY/SELL SENZA CAMBIO DI REGIME SAR"
    )

    suspicious = df[
        df["exit_phase"].isin(
            ["BUY", "SELL"]
        )
        &
        (
            df["sar_transition"]
            ==
            "SAME_SAR_REGIME"
        )
    ].copy()

    print(
        f"\nCasi: {len(suspicious):,}"
    )

    if not suspicious.empty:

        print(
            suspicious[
                [
                    "Ticker",
                    "indecision_start",
                    "indecision_end",
                    "prev_phase",
                    "last_sar_side",
                    "last_sar_age",
                    "exit_date",
                    "exit_phase",
                    "exit_sar_side",
                    "exit_sar_age",
                ]
            ]
            .head(40)
            .to_string(
                index=False,
            )
        )

    # ========================================================
    # 9. SUMMARY COMPATTA
    # ========================================================

    summary = (
        df
        .groupby(
            [
                "exit_phase",
                "sar_transition",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="episodes")
    )

    phase_totals = (
        summary
        .groupby(
            "exit_phase",
            dropna=False,
        )["episodes"]
        .transform("sum")
    )

    summary["pct_within_exit_phase"] = (
        summary["episodes"]
        /
        phase_totals
        *
        100
    )

    summary = summary.sort_values(
        [
            "exit_phase",
            "episodes",
        ],
        ascending=[
            True,
            False,
        ],
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # 10. NUMERI FINALI
    # ========================================================

    banner(
        "10 - NUMERI FINALI"
    )

    valid = df[
        df["sar_transition"]
        !=
        "NO_EXIT"
    ]

    new_regime = int(
        valid["is_new_sar_regime"].sum()
    )

    same_regime = int(
        valid["is_same_sar_regime"].sum()
    )

    print(
        f"\nEpisodi con uscita valida: "
        f"{len(valid):,}"
    )

    print(
        f"Nuovo regime SAR: "
        f"{new_regime:,} "
        f"({new_regime / len(valid) * 100:.2f}%)"
    )

    print(
        f"Stesso regime SAR: "
        f"{same_regime:,} "
        f"({same_regime / len(valid) * 100:.2f}%)"
    )

    banner(
        "FILE SALVATI"
    )

    print(
        f"\nDettaglio: {OUT_DETAIL}"
    )

    print(
        f"Summary : {OUT_SUMMARY}"
    )

    print(
        "\nNESSUN MOTORE MARKETSentinel È STATO MODIFICATO."
    )


if __name__ == "__main__":
    main()