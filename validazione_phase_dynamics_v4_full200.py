"""
MarketSentinel
VALIDAZIONE PHASE DYNAMICS V4 - FULL200

OBIETTIVO
---------
Validare la PHASE Dynamics V4 sull'intero universo FULL200
senza modificare produzione.

Controlli principali:
- numero ticker e righe processate;
- BUY/SELL non qualificati rielaborati;
- BUY finali non validi;
- SELL finali non validi;
- distribuzione delle riclassificazioni;
- modifiche V3.1 -> V4;
- eventuali modifiche PRE-V40.10 fuori dai casi rielaborati;
- fotografia dell'ultima settimana disponibile.

OUTPUT
------
data/validazione_phase_dynamics_v4_full200_summary.csv
data/validazione_phase_dynamics_v4_full200_reprocessed.csv
data/validazione_phase_dynamics_v4_full200_transitions.csv
data/validazione_phase_dynamics_v4_full200_pre_v4010_anomalies.csv
data/validazione_phase_dynamics_v4_full200_latest.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v4 as v4


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/v40_35_full200_weekly.csv"
)

OUTPUT_DIR = Path("data")

OUT_SUMMARY = OUTPUT_DIR / (
    "validazione_phase_dynamics_v4_full200_summary.csv"
)

OUT_REPROCESSED = OUTPUT_DIR / (
    "validazione_phase_dynamics_v4_full200_reprocessed.csv"
)

OUT_TRANSITIONS = OUTPUT_DIR / (
    "validazione_phase_dynamics_v4_full200_transitions.csv"
)

OUT_PRE_ANOMALIES = OUTPUT_DIR / (
    "validazione_phase_dynamics_v4_full200_pre_v4010_anomalies.csv"
)

OUT_LATEST = OUTPUT_DIR / (
    "validazione_phase_dynamics_v4_full200_latest.csv"
)


# ============================================================
# HELPERS
# ============================================================

def banner(text: str) -> None:
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    banner(
        "MARKETSENTINEL - VALIDAZIONE PHASE DYNAMICS V4 FULL200"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File non trovato: {INPUT_FILE}"
        )

    print(
        f"Input: {INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    if "Ticker" not in df.columns:
        raise RuntimeError(
            "Colonna Ticker non presente."
        )

    if "Date" not in df.columns:
        raise RuntimeError(
            "Colonna Date non presente."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        utc=True,
    )

    tickers = (
        df["Ticker"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Ticker trovati: {len(tickers)}"
    )

    print(
        f"Righe input: {len(df):,}"
    )

    # ========================================================
    # PROCESSAMENTO TICKER PER TICKER
    # ========================================================

    all_rows = []

    errors = []

    for n, ticker in enumerate(
        tickers,
        start=1,
    ):

        g = (
            df[
                df["Ticker"].astype(str).eq(
                    ticker
                )
            ]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if g.empty:
            continue

        try:

            z = (
                v4.process_ticker(
                    g
                )
                .reset_index(drop=True)
            )

            z["Ticker"] = ticker

            all_rows.append(z)

        except Exception as exc:

            errors.append(
                {
                    "Ticker": ticker,
                    "Error": str(exc),
                }
            )

        if (
            n == 1
            or n % 20 == 0
            or n == len(tickers)
        ):
            print(
                f"Processati {n}/{len(tickers)} ticker"
            )

    if not all_rows:
        raise RuntimeError(
            "Nessun ticker processato."
        )

    out = pd.concat(
        all_rows,
        ignore_index=True,
    )

    # ========================================================
    # COLONNE OBBLIGATORIE
    # ========================================================

    required = [
        "Ticker",
        "Date",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_DYNAMICS_V31",
        "PHASE_DYNAMICS_V4_PRE_V4010",
        "PHASE_DYNAMICS_V4",
        "V4_INVALID_BUY_SELL",
        "V4_INVALID_FINAL_BUY",
        "V4_INVALID_FINAL_SELL",
    ]

    missing = [
        c
        for c in required
        if c not in out.columns
    ]

    if missing:
        raise RuntimeError(
            "Colonne mancanti: "
            + ", ".join(missing)
        )

    # ========================================================
    # MASCHERE
    # ========================================================

    reprocessed_mask = (
        numeric(
            out[
                "V4_INVALID_BUY_SELL"
            ]
        )
        .eq(1)
    )

    changed_mask = (
        out[
            "PHASE_DYNAMICS_V31"
        ].astype(str)
        !=
        out[
            "PHASE_DYNAMICS_V4"
        ].astype(str)
    )

    pre_changed_mask = (
        out[
            "PHASE_DYNAMICS_V31"
        ].astype(str)
        !=
        out[
            "PHASE_DYNAMICS_V4_PRE_V4010"
        ].astype(str)
    )

    pre_anomaly_mask = (
        pre_changed_mask
        &
        ~reprocessed_mask
    )

    invalid_final_buy = int(
        numeric(
            out[
                "V4_INVALID_FINAL_BUY"
            ]
        ).sum()
    )

    invalid_final_sell = int(
        numeric(
            out[
                "V4_INVALID_FINAL_SELL"
            ]
        ).sum()
    )

    # ========================================================
    # CONTEGGI
    # ========================================================

    total_rows = len(out)

    processed_tickers = int(
        out["Ticker"].nunique()
    )

    reprocessed_rows = int(
        reprocessed_mask.sum()
    )

    changed_rows = int(
        changed_mask.sum()
    )

    pre_anomaly_rows = int(
        pre_anomaly_mask.sum()
    )

    # ========================================================
    # RICLASSIFICAZIONI DEI CASI INVALIDI
    # ========================================================

    reprocessed = (
        out.loc[
            reprocessed_mask
        ]
        .copy()
    )

    transitions = (
        reprocessed
        .groupby(
            [
                "PHASE_DYNAMICS_V31",
                "PHASE_DYNAMICS_V4_PRE_V4010",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="N"
        )
        .sort_values(
            "N",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # ANOMALIE PRE-V40.10 FUORI DAI CASI RIELABORATI
    # ========================================================

    pre_anomalies = (
        out.loc[
            pre_anomaly_mask
        ]
        .copy()
    )

    # ========================================================
    # ULTIMA SETTIMANA PER TICKER
    # ========================================================

    latest = (
        out
        .sort_values(
            [
                "Ticker",
                "Date",
            ]
        )
        .groupby(
            "Ticker",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    latest_cols = [
        "Ticker",
        "Date",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_DYNAMICS_V31",
        "PHASE_DYNAMICS_V4",
    ]

    latest = latest[
        latest_cols
    ].reset_index(drop=True)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "ROWS": total_rows,
                "TICKERS": processed_tickers,
                "INPUT_TICKERS": len(tickers),
                "PROCESS_ERRORS": len(errors),
                "REPROCESSED_INVALID_BUY_SELL":
                    reprocessed_rows,
                "CHANGED_V31_TO_V4":
                    changed_rows,
                "CHANGED_PCT":
                    (
                        changed_rows
                        /
                        total_rows
                        *
                        100
                    )
                    if total_rows
                    else 0,
                "PRE_V4010_CHANGES_OUTSIDE_REPROCESSED":
                    pre_anomaly_rows,
                "INVALID_FINAL_BUY":
                    invalid_final_buy,
                "INVALID_FINAL_SELL":
                    invalid_final_sell,
            }
        ]
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    reprocessed.to_csv(
        OUT_REPROCESSED,
        index=False,
    )

    transitions.to_csv(
        OUT_TRANSITIONS,
        index=False,
    )

    pre_anomalies.to_csv(
        OUT_PRE_ANOMALIES,
        index=False,
    )

    latest.to_csv(
        OUT_LATEST,
        index=False,
    )

    # ========================================================
    # RISULTATO TERMINALE
    # ========================================================

    banner(
        "RISULTATO FULL200"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    banner(
        "RICLASSIFICAZIONE BUY / SELL NON QUALIFICATI"
    )

    if transitions.empty:
        print(
            "Nessun caso rielaborato."
        )
    else:
        print(
            transitions.to_string(
                index=False
            )
        )

    banner(
        "SAFETY"
    )

    print(
        f"BUY finali non validi: "
        f"{invalid_final_buy}"
    )

    print(
        f"SELL finali non validi: "
        f"{invalid_final_sell}"
    )

    print(
        "Cambi PRE-V40.10 fuori dai casi rielaborati: "
        f"{pre_anomaly_rows}"
    )

    if errors:

        banner(
            "ERRORI"
        )

        for item in errors[:20]:

            print(
                f"{item['Ticker']}: "
                f"{item['Error']}"
            )

        if len(errors) > 20:

            print(
                f"... altri {len(errors) - 20} errori"
            )

    banner(
        "FILE SALVATI"
    )

    print(
        OUT_SUMMARY
    )

    print(
        OUT_TRANSITIONS
    )

    print(
        OUT_REPROCESSED
    )

    print(
        OUT_PRE_ANOMALIES
    )

    print(
        OUT_LATEST
    )


if __name__ == "__main__":
    main()