from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_4_core_phase_engine as v404
import weekly_v40_5_core_phase_engine as v405
import weekly_v40_9_phase_confirmation_fix as v409
import weekly_v40_10_sar_phase_fix as v4010


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

TICKERS = [
    "FBK.MI",
    "UCG.MI",
]

START_DATE = "2026-07-03"
END_DATE = "2026-08-28"

OUTPUT_FILE = Path(
    "data/diagnosi_fineco_ucg_v405.csv"
)


# ============================================================
# HELPERS
# ============================================================

def banner(text: str) -> None:
    print("\n" + "=" * 175)
    print(text)
    print("=" * 175)


def normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["Date"] = pd.to_datetime(
        x["Date"],
        utc=True,
        errors="coerce",
    )

    return x


def val(
    df: pd.DataFrame,
    date: pd.Timestamp,
    column: str,
):
    if column not in df.columns:
        return None

    z = df.loc[df["Date"] == date]

    if z.empty:
        return None

    return z.iloc[0][column]


# ============================================================
# ANALISI TICKER
# ============================================================

def analyse_ticker(
    raw: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:

    g = (
        raw[
            raw["Ticker"] == ticker
        ]
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    if g.empty:
        raise RuntimeError(
            f"Nessun dato per {ticker}"
        )

    print(
        f"\n{ticker}: {len(g):,} righe"
    )

    o4 = v404.process_ticker(
        g.copy()
    )

    o5 = v405.process_ticker(
        g.copy()
    )

    o9 = v409.process_ticker(
        g.copy()
    )

    o10 = v4010.process_ticker(
        g.copy()
    )

    start = pd.Timestamp(
        START_DATE,
        tz="UTC",
    )

    end = pd.Timestamp(
        END_DATE,
        tz="UTC",
    )

    dates = (
        o4[
            (o4["Date"] >= start)
            &
            (o4["Date"] <= end)
        ]["Date"]
        .tolist()
    )

    rows = []

    for d in dates:

        rows.append(
            {
                "Ticker": ticker,

                "Date":
                    d.strftime("%Y-%m-%d"),

                "Close":
                    val(o5, d, "Close"),

                # --------------------------------------------
                # SAR
                # --------------------------------------------

                "SAR_SIDE":
                    val(o5, d, "SAR_SIDE"),

                "SAR_AGE":
                    val(o5, d, "SAR_AGE"),

                # --------------------------------------------
                # HA
                # --------------------------------------------

                "HA_DIRECTION":
                    val(o5, d, "HA_DIRECTION"),

                "HA_BODY_RATIO":
                    val(o5, d, "HA_BODY_RATIO"),

                "HA_INDECISION":
                    val(o5, d, "HA_INDECISION"),

                # --------------------------------------------
                # BOLLINGER
                # --------------------------------------------

                "BB_POSITION":
                    val(o5, d, "BB_POSITION"),

                "V405_BB_DETACH_UPPER":
                    val(
                        o5,
                        d,
                        "V405_BB_DETACH_UPPER",
                    ),

                # --------------------------------------------
                # RSI
                # --------------------------------------------

                "RSI_14":
                    val(o5, d, "RSI_14"),

                "RSI_D1":
                    val(o5, d, "RSI_D1"),

                "RSI_SLOPE3":
                    val(o5, d, "RSI_SLOPE3"),

                "V405_RSI_HARD_WEAK_UP":
                    val(
                        o5,
                        d,
                        "V405_RSI_HARD_WEAK_UP",
                    ),

                # --------------------------------------------
                # MACD
                # --------------------------------------------

                "MACD_HIST_D1":
                    val(
                        o5,
                        d,
                        "MACD_HIST_D1",
                    ),

                "V405_MACD_HARD_WEAK_UP":
                    val(
                        o5,
                        d,
                        "V405_MACD_HARD_WEAK_UP",
                    ),

                # --------------------------------------------
                # VOLUME
                # --------------------------------------------

                "VOLUME_OSC_SLOPE3":
                    val(
                        o5,
                        d,
                        "VOLUME_OSC_SLOPE3",
                    ),

                "V405_VOL_PARTICIPATION_FALLING":
                    val(
                        o5,
                        d,
                        "V405_VOL_PARTICIPATION_FALLING",
                    ),

                # --------------------------------------------
                # WEAKNESS
                # --------------------------------------------

                "V405_WEAK_UP_MOM_COUNT":
                    val(
                        o5,
                        d,
                        "V405_WEAK_UP_MOM_COUNT",
                    ),

                "V405_WEAK_UP_BB_PATH":
                    val(
                        o5,
                        d,
                        "V405_WEAK_UP_BB_PATH",
                    ),

                "V405_WEAK_UP_HA_PATH":
                    val(
                        o5,
                        d,
                        "V405_WEAK_UP_HA_PATH",
                    ),

                "WEAK_UP_TRIGGER":
                    val(
                        o5,
                        d,
                        "WEAK_UP_TRIGGER",
                    ),

                # --------------------------------------------
                # LATERALITA'
                # --------------------------------------------

                "LATERAL_SIGNAL":
                    val(
                        o5,
                        d,
                        "LATERAL_SIGNAL",
                    ),

                # --------------------------------------------
                # RECOVERY COMPONENTS
                # --------------------------------------------

                "V405_BULL_REC_HA":
                    val(
                        o5,
                        d,
                        "V405_BULL_REC_HA",
                    ),

                "V405_BULL_REC_BB":
                    val(
                        o5,
                        d,
                        "V405_BULL_REC_BB",
                    ),

                "V405_BULL_REC_RSI":
                    val(
                        o5,
                        d,
                        "V405_BULL_REC_RSI",
                    ),

                "V405_BULL_REC_MACD":
                    val(
                        o5,
                        d,
                        "V405_BULL_REC_MACD",
                    ),

                "V405_BULL_REC_VOLUME":
                    val(
                        o5,
                        d,
                        "V405_BULL_REC_VOLUME",
                    ),

                "BULL_RECOVERY_RAW":
                    val(
                        o5,
                        d,
                        "BULL_RECOVERY_RAW",
                    ),

                "BULL_RECOVERY_CONFIRMED":
                    val(
                        o5,
                        d,
                        "BULL_RECOVERY_CONFIRMED",
                    ),

                # --------------------------------------------
                # PHASE
                # --------------------------------------------

                "PHASE_V404":
                    val(
                        o4,
                        d,
                        "PHASE",
                    ),

                "PHASE_V405":
                    val(
                        o5,
                        d,
                        "PHASE",
                    ),

                "PHASE_V409":
                    val(
                        o9,
                        d,
                        "PHASE",
                    ),

                "PHASE_V4010":
                    val(
                        o10,
                        d,
                        "PHASE",
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "MARKETSENTINEL - FINECO vs UCG - DIAGNOSI V40.5"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Non trovo {INPUT_FILE}"
        )

    raw = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    raw = normalize_date(raw)

    parts = []

    for ticker in TICKERS:

        banner(
            f"CALCOLO {ticker}"
        )

        part = analyse_ticker(
            raw,
            ticker,
        )

        parts.append(part)

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # 1 - FILM COMPLETO
    # ========================================================

    for ticker in TICKERS:

        z = result[
            result["Ticker"] == ticker
        ].copy()

        banner(
            f"1 - FILM PHASE {ticker}"
        )

        cols = [
            "Date",
            "Close",
            "SAR_SIDE",
            "SAR_AGE",
            "WEAK_UP_TRIGGER",
            "LATERAL_SIGNAL",
            "BULL_RECOVERY_RAW",
            "BULL_RECOVERY_CONFIRMED",
            "PHASE_V404",
            "PHASE_V405",
            "PHASE_V409",
            "PHASE_V4010",
        ]

        print(
            z[cols].to_string(
                index=False,
            )
        )

    # ========================================================
    # 2 - DEBOLEZZA
    # ========================================================

    for ticker in TICKERS:

        z = result[
            result["Ticker"] == ticker
        ].copy()

        banner(
            f"2 - DEBOLEZZA {ticker}"
        )

        cols = [
            "Date",
            "HA_DIRECTION",
            "HA_INDECISION",
            "BB_POSITION",
            "V405_BB_DETACH_UPPER",
            "RSI_D1",
            "RSI_SLOPE3",
            "V405_RSI_HARD_WEAK_UP",
            "MACD_HIST_D1",
            "V405_MACD_HARD_WEAK_UP",
            "VOLUME_OSC_SLOPE3",
            "V405_VOL_PARTICIPATION_FALLING",
            "V405_WEAK_UP_MOM_COUNT",
            "V405_WEAK_UP_BB_PATH",
            "V405_WEAK_UP_HA_PATH",
            "WEAK_UP_TRIGGER",
        ]

        print(
            z[cols].to_string(
                index=False,
                float_format=lambda x:
                f"{x:.3f}",
            )
        )

    # ========================================================
    # 3 - RECOVERY
    # ========================================================

    for ticker in TICKERS:

        z = result[
            result["Ticker"] == ticker
        ].copy()

        banner(
            f"3 - RECOVERY {ticker}"
        )

        cols = [
            "Date",
            "V405_BULL_REC_HA",
            "V405_BULL_REC_BB",
            "V405_BULL_REC_RSI",
            "V405_BULL_REC_MACD",
            "V405_BULL_REC_VOLUME",
            "BULL_RECOVERY_RAW",
            "BULL_RECOVERY_CONFIRMED",
            "PHASE_V405",
        ]

        print(
            z[cols].to_string(
                index=False,
            )
        )

    # ========================================================
    # 4 - DIFFERENZE V40.4 vs V40.5
    # ========================================================

    banner(
        "4 - DOVE V40.5 CAMBIA V40.4"
    )

    diff = result[
        result["PHASE_V404"]
        !=
        result["PHASE_V405"]
    ].copy()

    print(
        diff[
            [
                "Ticker",
                "Date",
                "WEAK_UP_TRIGGER",
                "LATERAL_SIGNAL",
                "BULL_RECOVERY_RAW",
                "BULL_RECOVERY_CONFIRMED",
                "PHASE_V404",
                "PHASE_V405",
                "PHASE_V409",
                "PHASE_V4010",
            ]
        ].to_string(
            index=False,
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
        "\nNessun file del motore è stato modificato."
    )


if __name__ == "__main__":
    main()