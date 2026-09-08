"""
MarketSentinel
DIAGNOSTIC - CAUSAL SHADOW
3 anomalie + storico UCG

INPUT
-----
data/validazione_full200_v405_causal_shadow_changed.csv

OBIETTIVO
---------
1. Identificare le 3 righe anomale:
   TREND_RIALZISTA  -> INDECISIONE
   TREND_RIBASSISTA -> INDECISIONE

2. Capire perché la causal shadow le ha prodotte.

3. Analizzare tutte le modifiche storiche UCG.MI:
   - periodo
   - transizioni
   - reason
   - sequenze temporali

Nessun motore viene eseguito.
Nessun file di produzione viene modificato.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/validazione_full200_v405_causal_shadow_changed.csv"
)

OUT_ANOMALIES = Path(
    "data/diagnosi_causal_shadow_3_anomalie.csv"
)

OUT_UCG = Path(
    "data/diagnosi_causal_shadow_ucg_history.csv"
)


# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 130)
    print(text)
    print("=" * 130)


def show_columns(df, columns):
    existing = [
        c for c in columns
        if c in df.columns
    ]

    if df.empty:
        print("Nessuna riga.")
        return

    print(
        df[existing].to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "MARKETSENTINEL - DIAGNOSI ANOMALIE + UCG"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Non trovo {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    print(
        f"\nRighe modificate caricate: {len(df):,}"
    )

    # ========================================================
    # 1. ANOMALIE TREND -> INDECISIONE
    # ========================================================

    banner(
        "1 - ANOMALIE TREND -> INDECISIONE"
    )

    anomalies = df[
        (
            df["PHASE_V405_ORIGINAL"].isin(
                [
                    "TREND_RIALZISTA",
                    "TREND_RIBASSISTA",
                ]
            )
        )
        &
        (
            df["PHASE_CAUSAL_SHADOW"]
            ==
            "INDECISIONE"
        )
    ].copy()

    print(
        f"\nCasi trovati: {len(anomalies):,}\n"
    )

    anomaly_columns = [
        "Ticker",
        "Date",
        "Close",

        "SAR_SIDE",
        "SAR_AGE",

        "WEAK_UP_TRIGGER",
        "WEAK_DOWN_TRIGGER",

        "LATERAL_SIGNAL",

        "BULL_RECOVERY_RAW",
        "BULL_RECOVERY_CONFIRMED",

        "BEAR_RECOVERY_RAW",
        "BEAR_RECOVERY_CONFIRMED",

        "PHASE_V405_ORIGINAL",
        "PHASE_CAUSAL_SHADOW",

        "CAUSAL_SHADOW_REASON",
        "CAUSAL_SHADOW_ACTIVE",
        "WEAK_ORIGIN_INDECISION",
    ]

    show_columns(
        anomalies,
        anomaly_columns,
    )

    anomalies.to_csv(
        OUT_ANOMALIES,
        index=False,
    )

    # ========================================================
    # 2. CONTESTO DELLE ANOMALIE
    # ========================================================

    banner(
        "2 - CONTESTO MODIFICHE PER I TICKER ANOMALI"
    )

    anomaly_tickers = (
        anomalies["Ticker"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(anomaly_tickers) == 0:

        print(
            "Nessun ticker anomalo."
        )

    else:

        for ticker in anomaly_tickers:

            z = df[
                df["Ticker"] == ticker
            ].copy()

            anomaly_dates = anomalies[
                anomalies["Ticker"] == ticker
            ]["Date"]

            min_date = (
                anomaly_dates.min()
                -
                pd.Timedelta(weeks=8)
            )

            max_date = (
                anomaly_dates.max()
                +
                pd.Timedelta(weeks=8)
            )

            z = z[
                (z["Date"] >= min_date)
                &
                (z["Date"] <= max_date)
            ].copy()

            banner(
                f"CONTESTO {ticker}"
            )

            show_columns(
                z,
                anomaly_columns,
            )

    # ========================================================
    # 3. UCG - TUTTE LE MODIFICHE
    # ========================================================

    banner(
        "3 - UCG.MI: TUTTE LE MODIFICHE STORICHE"
    )

    ucg = df[
        df["Ticker"] == "UCG.MI"
    ].copy()

    print(
        f"\nRighe UCG modificate: {len(ucg):,}"
    )

    if not ucg.empty:

        print(
            f"Prima modifica: {ucg['Date'].min()}"
        )

        print(
            f"Ultima modifica: {ucg['Date'].max()}"
        )

    ucg_columns = [
        "Date",
        "Close",

        "SAR_SIDE",
        "SAR_AGE",

        "WEAK_UP_TRIGGER",
        "WEAK_DOWN_TRIGGER",

        "LATERAL_SIGNAL",

        "BULL_RECOVERY_RAW",
        "BULL_RECOVERY_CONFIRMED",

        "BEAR_RECOVERY_RAW",
        "BEAR_RECOVERY_CONFIRMED",

        "PHASE_V405_ORIGINAL",
        "PHASE_CAUSAL_SHADOW",

        "CAUSAL_SHADOW_REASON",
        "WEAK_ORIGIN_INDECISION",
    ]

    show_columns(
        ucg,
        ucg_columns,
    )

    ucg.to_csv(
        OUT_UCG,
        index=False,
    )

    # ========================================================
    # 4. UCG - TRANSIZIONI
    # ========================================================

    banner(
        "4 - UCG.MI: TRANSIZIONI ORIGINAL -> SHADOW"
    )

    if not ucg.empty:

        transitions = (
            ucg
            .groupby(
                [
                    "PHASE_V405_ORIGINAL",
                    "PHASE_CAUSAL_SHADOW",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="rows"
            )
            .sort_values(
                "rows",
                ascending=False,
            )
        )

        print(
            transitions.to_string(
                index=False
            )
        )

    # ========================================================
    # 5. UCG - REASONS
    # ========================================================

    banner(
        "5 - UCG.MI: REASONS"
    )

    if not ucg.empty:

        reasons = (
            ucg[
                "CAUSAL_SHADOW_REASON"
            ]
            .value_counts(
                dropna=False
            )
            .rename_axis(
                "reason"
            )
            .reset_index(
                name="rows"
            )
        )

        print(
            reasons.to_string(
                index=False
            )
        )

    # ========================================================
    # 6. UCG - MODIFICHE PER ANNO
    # ========================================================

    banner(
        "6 - UCG.MI: MODIFICHE PER ANNO"
    )

    if not ucg.empty:

        ucg["Year"] = (
            ucg["Date"].dt.year
        )

        by_year = (
            ucg
            .groupby("Year")
            .size()
            .reset_index(
                name="changed_rows"
            )
        )

        print(
            by_year.to_string(
                index=False
            )
        )

    # ========================================================
    # 7. CHECK FINALE ANOMALIE
    # ========================================================

    banner(
        "7 - CHECK FINALE"
    )

    print(
        f"\nTREND -> INDECISIONE: "
        f"{len(anomalies):,}"
    )

    print(
        f"Ticker coinvolti: "
        f"{len(anomaly_tickers):,}"
    )

    print(
        f"\nUCG righe modificate: "
        f"{len(ucg):,}"
    )

    # ========================================================
    # FILES
    # ========================================================

    banner(
        "FILE SALVATI"
    )

    print(
        f"\nANOMALIE : {OUT_ANOMALIES}"
    )

    print(
        f"UCG      : {OUT_UCG}"
    )

    print(
        "\nNESSUN MOTORE RICALCOLATO."
    )

    print(
        "NESSUN FILE DI PRODUZIONE MODIFICATO."
    )


if __name__ == "__main__":
    main()