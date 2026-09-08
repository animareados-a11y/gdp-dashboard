from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_4_core_phase_engine as v404
import weekly_v40_7_weakness_severity_engine as v407
import weekly_v40_9_phase_confirmation_fix as v409
import weekly_v40_10_sar_phase_fix as v4010


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

TICKER = "FBK.MI"

START_DATE = "2026-05-01"
END_DATE = "2026-08-31"

OUTPUT_FILE = Path("data/diagnosi_phase_fineco_completa.csv")


# ============================================================
# HELPERS
# ============================================================

def banner(text: str) -> None:

    print("\n" + "=" * 180)
    print(text)
    print("=" * 180)


def safe_series(
    df: pd.DataFrame,
    column: str,
    default=None,
):

    if column in df.columns:
        return df[column].values

    return [default] * len(df)


def normalize_date(df: pd.DataFrame) -> pd.DataFrame:

    x = df.copy()

    x["Date"] = pd.to_datetime(
        x["Date"],
        utc=True,
        errors="coerce",
    )

    return x


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "MARKETSENTINEL - DIAGNOSI COMPLETA PHASE FINECO"
    )

    # --------------------------------------------------------
    # CARICAMENTO FULL200
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Non trovo il file: {INPUT_FILE}"
        )

    raw = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    raw = normalize_date(raw)

    g = (
        raw[
            raw["Ticker"] == TICKER
        ]
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    if g.empty:

        raise RuntimeError(
            f"Nessun dato trovato per {TICKER}"
        )

    print(
        f"\nTicker: {TICKER}"
    )

    print(
        f"Righe disponibili: {len(g):,}"
    )

    print(
        f"Da: {g['Date'].min()}"
    )

    print(
        f"A : {g['Date'].max()}"
    )

    # ========================================================
    # V40.4
    # ========================================================

    banner(
        "CALCOLO V40.4"
    )

    out404 = v404.process_ticker(
        g.copy()
    )

    # ========================================================
    # V40.7
    # ========================================================

    banner(
        "CALCOLO V40.7"
    )

    out407 = v407.process_ticker(
        g.copy()
    )

    # ========================================================
    # V40.9
    # ========================================================

    banner(
        "CALCOLO V40.9"
    )

    out409 = v409.process_ticker(
        g.copy()
    )

    # ========================================================
    # V40.10
    # ========================================================

    banner(
        "CALCOLO V40.10"
    )

    out4010 = v4010.process_ticker(
        g.copy()
    )

    # ========================================================
    # INTERVALLO DI ANALISI
    # ========================================================

    start = pd.Timestamp(
        START_DATE,
        tz="UTC",
    )

    end = pd.Timestamp(
        END_DATE,
        tz="UTC",
    )

    mask404 = (
        (out404["Date"] >= start)
        &
        (out404["Date"] <= end)
    )

    mask407 = (
        (out407["Date"] >= start)
        &
        (out407["Date"] <= end)
    )

    mask409 = (
        (out409["Date"] >= start)
        &
        (out409["Date"] <= end)
    )

    mask4010 = (
        (out4010["Date"] >= start)
        &
        (out4010["Date"] <= end)
    )

    a404 = (
        out404.loc[mask404]
        .reset_index(drop=True)
        .copy()
    )

    a407 = (
        out407.loc[mask407]
        .reset_index(drop=True)
        .copy()
    )

    a409 = (
        out409.loc[mask409]
        .reset_index(drop=True)
        .copy()
    )

    a4010 = (
        out4010.loc[mask4010]
        .reset_index(drop=True)
        .copy()
    )

    if not (
        len(a404)
        ==
        len(a407)
        ==
        len(a409)
        ==
        len(a4010)
    ):

        raise RuntimeError(
            "Le versioni hanno un numero diverso di righe "
            "nell'intervallo analizzato."
        )

    # ========================================================
    # TABELLA COMPLETA
    # ========================================================

    diag = pd.DataFrame({

        "Date":
            a404["Date"]
            .dt.strftime("%Y-%m-%d"),

        "Close":
            safe_series(
                a404,
                "Close",
            ),

        # ----------------------------------------------------
        # SAR
        # ----------------------------------------------------

        "SAR":
            safe_series(
                a404,
                "SAR",
            ),

        "SAR_SIDE":
            safe_series(
                a404,
                "SAR_SIDE",
            ),

        "SAR_AGE":
            safe_series(
                a404,
                "SAR_AGE",
            ),

        # ----------------------------------------------------
        # HEIKIN ASHI
        # ----------------------------------------------------

        "HA_DIRECTION":
            safe_series(
                a404,
                "HA_DIRECTION",
            ),

        "HA_BODY_RATIO":
            safe_series(
                a404,
                "HA_BODY_RATIO",
            ),

        "HA_INDECISION":
            safe_series(
                a404,
                "HA_INDECISION",
            ),

        "HA_BODY_SHRINK_2":
            safe_series(
                a404,
                "HA_BODY_SHRINK_2",
            ),

        "HA_GREEN_COUNT4":
            safe_series(
                a404,
                "HA_GREEN_COUNT4",
            ),

        "HA_RED_COUNT4":
            safe_series(
                a404,
                "HA_RED_COUNT4",
            ),

        # ----------------------------------------------------
        # BOLLINGER
        # ----------------------------------------------------

        "BB_POSITION":
            safe_series(
                a404,
                "BB_POSITION",
            ),

        "BB_POSITION_MAX4":
            safe_series(
                a404,
                "BB_POSITION_MAX4",
            ),

        "BB_DETACH_UPPER":
            safe_series(
                a404,
                "BB_DETACH_UPPER",
            ),

        "BB_ABOVE_MIDDLE":
            safe_series(
                a404,
                "BB_ABOVE_MIDDLE",
            ),

        "BB_PROGRESS_UP":
            safe_series(
                a404,
                "BB_PROGRESS_UP",
            ),

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        "RSI_14":
            safe_series(
                a404,
                "RSI_14",
            ),

        "RSI_D1":
            safe_series(
                a404,
                "RSI_D1",
            ),

        "RSI_SLOPE3":
            safe_series(
                a404,
                "RSI_SLOPE3",
            ),

        "RSI_WEAK_UP_FILM":
            safe_series(
                a404,
                "RSI_WEAK_UP_FILM",
            ),

        # ----------------------------------------------------
        # MACD
        # ----------------------------------------------------

        "MACD":
            safe_series(
                a404,
                "MACD",
            ),

        "MACD_SIGNAL":
            safe_series(
                a404,
                "MACD_SIGNAL",
            ),

        "MACD_HIST":
            safe_series(
                a404,
                "MACD_HIST",
            ),

        "MACD_HIST_D1":
            safe_series(
                a404,
                "MACD_HIST_D1",
            ),

        "MACD_SLOPE3":
            safe_series(
                a404,
                "MACD_SLOPE3",
            ),

        "MACD_WEAK_UP_FILM":
            safe_series(
                a404,
                "MACD_WEAK_UP_FILM",
            ),

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        "VOLUME_OSC":
            safe_series(
                a404,
                "VOLUME_OSC",
            ),

        "VOLUME_OSC_SLOPE3":
            safe_series(
                a404,
                "VOLUME_OSC_SLOPE3",
            ),

        "VOL_WEAK_UP_FILM":
            safe_series(
                a404,
                "VOL_WEAK_UP_FILM",
            ),

        # ----------------------------------------------------
        # DEBOLEZZA
        # ----------------------------------------------------

        "MOM_WEAK_UP_COUNT":
            safe_series(
                a404,
                "MOM_WEAK_UP_COUNT",
            ),

        "MOM_WEAK_UP_2OF3":
            safe_series(
                a404,
                "MOM_WEAK_UP_2OF3",
            ),

        "WEAK_UP_BB_PATH":
            safe_series(
                a404,
                "WEAK_UP_BB_PATH",
            ),

        "WEAK_UP_HA_PATH":
            safe_series(
                a404,
                "WEAK_UP_HA_PATH",
            ),

        "WEAK_UP_TRIGGER":
            safe_series(
                a404,
                "WEAK_UP_TRIGGER",
            ),

        # ----------------------------------------------------
        # LATERALITA'
        # ----------------------------------------------------

        "LATERAL_SIGNAL":
            safe_series(
                a404,
                "LATERAL_SIGNAL",
            ),

        # ----------------------------------------------------
        # RECOVERY
        # ----------------------------------------------------

        "BULL_RECOVERY_SCORE":
            safe_series(
                a404,
                "BULL_RECOVERY_SCORE",
            ),

        "BULL_RECOVERY_RAW":
            safe_series(
                a404,
                "BULL_RECOVERY_RAW",
            ),

        "BULL_RECOVERY_CONFIRMED":
            safe_series(
                a404,
                "BULL_RECOVERY_CONFIRMED",
            ),

        # ----------------------------------------------------
        # PHASE PER VERSIONE
        # ----------------------------------------------------

        "PHASE_V404":
            safe_series(
                a404,
                "PHASE",
            ),

        "PHASE_V407":
            safe_series(
                a407,
                "PHASE",
            ),

        "PHASE_V409":
            safe_series(
                a409,
                "PHASE",
            ),

        "PHASE_V4010":
            safe_series(
                a4010,
                "PHASE",
            ),

        # ----------------------------------------------------
        # V40.9 - RESCUE
        # ----------------------------------------------------

        "V409_CORRECTED":
            safe_series(
                a409,
                "V409_CORRECTED",
            ),

        "V409_BULL_HA":
            safe_series(
                a409,
                "V409_BULL_HA",
            ),

        "V409_BULL_BB":
            safe_series(
                a409,
                "V409_BULL_BB",
            ),

        "V409_BULL_RSI":
            safe_series(
                a409,
                "V409_BULL_RSI",
            ),

        "V409_BULL_MACD":
            safe_series(
                a409,
                "V409_BULL_MACD",
            ),

        "V409_BULL_CONFIRMATIONS":
            safe_series(
                a409,
                "V409_BULL_CONFIRMATIONS",
            ),

        "V409_BULL_DAMAGE_HA":
            safe_series(
                a409,
                "V409_BULL_DAMAGE_HA",
            ),

        "V409_BULL_DAMAGE_RSI":
            safe_series(
                a409,
                "V409_BULL_DAMAGE_RSI",
            ),

        "V409_BULL_DAMAGE_MACD":
            safe_series(
                a409,
                "V409_BULL_DAMAGE_MACD",
            ),

        "V409_BULL_DAMAGE_BB":
            safe_series(
                a409,
                "V409_BULL_DAMAGE_BB",
            ),

        "V409_BULL_DAMAGE_COUNT":
            safe_series(
                a409,
                "V409_BULL_DAMAGE_COUNT",
            ),

        "V409_REASON":
            safe_series(
                a409,
                "V409_REASON",
                "",
            ),

    })

    # ========================================================
    # SALVATAGGIO
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diag.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # TABELLA 1 - FILM PHASE
    # ========================================================

    banner(
        "1 - FILM PHASE V40.4 -> V40.7 -> V40.9 -> V40.10"
    )

    cols_phase = [
        "Date",
        "Close",
        "SAR_SIDE",
        "SAR_AGE",
        "WEAK_UP_TRIGGER",
        "LATERAL_SIGNAL",
        "BULL_RECOVERY_RAW",
        "BULL_RECOVERY_CONFIRMED",
        "PHASE_V404",
        "PHASE_V407",
        "PHASE_V409",
        "PHASE_V4010",
        "V409_CORRECTED",
    ]

    print(
        diag[
            cols_phase
        ].to_string(
            index=False,
        )
    )

    # ========================================================
    # TABELLA 2 - INDICATORI TECNICI
    # ========================================================

    banner(
        "2 - INDICATORI TECNICI DI DEBOLEZZA"
    )

    cols_technical = [
        "Date",
        "HA_DIRECTION",
        "HA_BODY_RATIO",
        "HA_INDECISION",
        "HA_BODY_SHRINK_2",
        "BB_POSITION",
        "BB_DETACH_UPPER",
        "RSI_14",
        "RSI_D1",
        "MACD_HIST",
        "MACD_HIST_D1",
        "VOLUME_OSC",
        "MOM_WEAK_UP_COUNT",
        "WEAK_UP_BB_PATH",
        "WEAK_UP_HA_PATH",
        "WEAK_UP_TRIGGER",
    ]

    print(
        diag[
            cols_technical
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.3f}",
        )
    )

    # ========================================================
    # TABELLA 3 - COSA FA V40.9
    # ========================================================

    banner(
        "3 - V40.9: CONFERME E DANNI"
    )

    cols_v409 = [
        "Date",
        "WEAK_UP_TRIGGER",
        "PHASE_V407",
        "V409_BULL_HA",
        "V409_BULL_BB",
        "V409_BULL_RSI",
        "V409_BULL_MACD",
        "V409_BULL_CONFIRMATIONS",
        "V409_BULL_DAMAGE_HA",
        "V409_BULL_DAMAGE_RSI",
        "V409_BULL_DAMAGE_MACD",
        "V409_BULL_DAMAGE_BB",
        "V409_BULL_DAMAGE_COUNT",
        "V409_CORRECTED",
        "PHASE_V409",
        "V409_REASON",
    ]

    print(
        diag[
            cols_v409
        ].to_string(
            index=False,
        )
    )

    # ========================================================
    # FOCUS AGOSTO
    # ========================================================

    banner(
        "4 - FOCUS FINECO: AGOSTO 2026"
    )

    august = diag[
        diag["Date"].isin(
            [
                "2026-08-07",
                "2026-08-14",
                "2026-08-21",
                "2026-08-28",
            ]
        )
    ].copy()

    print(
        august.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.3f}",
        )
    )

    # ========================================================
    # FINE
    # ========================================================

    banner(
        "DIAGNOSI COMPLETATA"
    )

    print(
        f"\nFile salvato: {OUTPUT_FILE}"
    )

    print(
        "\nNON è stato modificato alcun motore PHASE."
    )


if __name__ == "__main__":

    main()