from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_4_core_phase_engine as v404
import weekly_v40_5_core_phase_engine as v405


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

TICKER = "FBK.MI"

START_DATE = "2026-03-01"
END_DATE = "2026-08-28"

OUTPUT_FILE = Path(
    "data/diagnosi_fineco_origine_indecisione.csv"
)


# ============================================================
# HELPERS
# ============================================================

def banner(text: str) -> None:
    print("\n" + "=" * 180)
    print(text)
    print("=" * 180)


def normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["Date"] = pd.to_datetime(
        x["Date"],
        utc=True,
        errors="coerce",
    )

    return x


def safe_value(
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
# MAIN
# ============================================================

def main():

    banner(
        "MARKETSENTINEL - ORIGINE INDECISIONE FINECO"
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

    print(f"\nTicker: {TICKER}")
    print(f"Righe: {len(g):,}")
    print(f"Da: {g['Date'].min()}")
    print(f"A : {g['Date'].max()}")

    # ========================================================
    # CALCOLO V40.4 E V40.5
    # ========================================================

    banner("CALCOLO V40.4")
    o4 = v404.process_ticker(
        g.copy()
    )

    banner("CALCOLO V40.5")
    o5 = v405.process_ticker(
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
        o5[
            (o5["Date"] >= start)
            &
            (o5["Date"] <= end)
        ]["Date"]
        .tolist()
    )

    rows = []

    for d in dates:

        rows.append(
            {
                "Date":
                    d.strftime("%Y-%m-%d"),

                "Close":
                    safe_value(o5, d, "Close"),

                # --------------------------------------------
                # PHASE
                # --------------------------------------------

                "PHASE_V404":
                    safe_value(o4, d, "PHASE"),

                "PHASE_V405":
                    safe_value(o5, d, "PHASE"),

                # --------------------------------------------
                # SAR
                # --------------------------------------------

                "SAR":
                    safe_value(o5, d, "SAR"),

                "SAR_SIDE":
                    safe_value(o5, d, "SAR_SIDE"),

                "SAR_AGE":
                    safe_value(o5, d, "SAR_AGE"),

                "SAR_FLIP_UP":
                    safe_value(o5, d, "SAR_FLIP_UP"),

                "SAR_FLIP_DOWN":
                    safe_value(o5, d, "SAR_FLIP_DOWN"),

                # --------------------------------------------
                # HA
                # --------------------------------------------

                "HA_DIRECTION":
                    safe_value(o5, d, "HA_DIRECTION"),

                "HA_BODY_RATIO":
                    safe_value(o5, d, "HA_BODY_RATIO"),

                "HA_INDECISION":
                    safe_value(o5, d, "HA_INDECISION"),

                "HA_INDECISION_COUNT3":
                    safe_value(
                        o5,
                        d,
                        "HA_INDECISION_COUNT3",
                    ),

                "HA_BODY_SHRINK_2":
                    safe_value(
                        o5,
                        d,
                        "HA_BODY_SHRINK_2",
                    ),

                # --------------------------------------------
                # BOLLINGER
                # --------------------------------------------

                "BB_POSITION":
                    safe_value(o5, d, "BB_POSITION"),

                "BB_POSITION_D1":
                    safe_value(
                        o5,
                        d,
                        "BB_POSITION_D1",
                    ),

                "BB_POSITION_MAX4":
                    safe_value(
                        o5,
                        d,
                        "BB_POSITION_MAX4",
                    ),

                "V405_BB_DETACH_UPPER":
                    safe_value(
                        o5,
                        d,
                        "V405_BB_DETACH_UPPER",
                    ),

                # --------------------------------------------
                # RSI
                # --------------------------------------------

                "RSI_14":
                    safe_value(o5, d, "RSI_14"),

                "RSI_D1":
                    safe_value(o5, d, "RSI_D1"),

                "RSI_SLOPE3":
                    safe_value(o5, d, "RSI_SLOPE3"),

                "V405_RSI_HARD_WEAK_UP":
                    safe_value(
                        o5,
                        d,
                        "V405_RSI_HARD_WEAK_UP",
                    ),

                # --------------------------------------------
                # MACD
                # --------------------------------------------

                "MACD_D1":
                    safe_value(o5, d, "MACD_D1"),

                "MACD_HIST_D1":
                    safe_value(
                        o5,
                        d,
                        "MACD_HIST_D1",
                    ),

                "V405_MACD_HARD_WEAK_UP":
                    safe_value(
                        o5,
                        d,
                        "V405_MACD_HARD_WEAK_UP",
                    ),

                # --------------------------------------------
                # VOLUME
                # --------------------------------------------

                "VOLUME_OSC_D1":
                    safe_value(
                        o5,
                        d,
                        "VOLUME_OSC_D1",
                    ),

                "VOLUME_OSC_SLOPE3":
                    safe_value(
                        o5,
                        d,
                        "VOLUME_OSC_SLOPE3",
                    ),

                "V405_VOL_PARTICIPATION_FALLING":
                    safe_value(
                        o5,
                        d,
                        "V405_VOL_PARTICIPATION_FALLING",
                    ),

                # --------------------------------------------
                # WEAKNESS
                # --------------------------------------------

                "V405_WEAK_UP_MOM_COUNT":
                    safe_value(
                        o5,
                        d,
                        "V405_WEAK_UP_MOM_COUNT",
                    ),

                "V405_HA_WEAK_UP_STRUCTURE":
                    safe_value(
                        o5,
                        d,
                        "V405_HA_WEAK_UP_STRUCTURE",
                    ),

                "V405_WEAK_UP_BB_PATH":
                    safe_value(
                        o5,
                        d,
                        "V405_WEAK_UP_BB_PATH",
                    ),

                "V405_WEAK_UP_HA_PATH":
                    safe_value(
                        o5,
                        d,
                        "V405_WEAK_UP_HA_PATH",
                    ),

                "WEAK_UP_TRIGGER":
                    safe_value(
                        o5,
                        d,
                        "WEAK_UP_TRIGGER",
                    ),

                # --------------------------------------------
                # LATERALITA'
                # --------------------------------------------

                "LATERAL_SCORE":
                    safe_value(
                        o5,
                        d,
                        "LATERAL_SCORE",
                    ),

                "LATERAL_RAW":
                    safe_value(
                        o5,
                        d,
                        "LATERAL_RAW",
                    ),

                "LATERAL_SIGNAL":
                    safe_value(
                        o5,
                        d,
                        "LATERAL_SIGNAL",
                    ),

                # --------------------------------------------
                # RECOVERY
                # --------------------------------------------

                "V405_BULL_REC_HA":
                    safe_value(
                        o5,
                        d,
                        "V405_BULL_REC_HA",
                    ),

                "V405_BULL_REC_BB":
                    safe_value(
                        o5,
                        d,
                        "V405_BULL_REC_BB",
                    ),

                "V405_BULL_REC_RSI":
                    safe_value(
                        o5,
                        d,
                        "V405_BULL_REC_RSI",
                    ),

                "V405_BULL_REC_MACD":
                    safe_value(
                        o5,
                        d,
                        "V405_BULL_REC_MACD",
                    ),

                "V405_BULL_REC_VOLUME":
                    safe_value(
                        o5,
                        d,
                        "V405_BULL_REC_VOLUME",
                    ),

                "BULL_RECOVERY_RAW":
                    safe_value(
                        o5,
                        d,
                        "BULL_RECOVERY_RAW",
                    ),

                "BULL_RECOVERY_CONFIRMED":
                    safe_value(
                        o5,
                        d,
                        "BULL_RECOVERY_CONFIRMED",
                    ),
            }
        )

    result = pd.DataFrame(rows)

    # ========================================================
    # PREVIOUS PHASE
    # ========================================================

    result["PREV_PHASE_V405"] = (
        result["PHASE_V405"]
        .shift(1)
    )

    result["PHASE_CHANGED_V405"] = (
        result["PHASE_V405"]
        !=
        result["PREV_PHASE_V405"]
    )

    # ========================================================
    # SALVA
    # ========================================================

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

    banner(
        "1 - FILM PHASE MARZO -> AGOSTO 2026"
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
    ]

    print(
        result[cols].to_string(
            index=False,
        )
    )

    # ========================================================
    # 2 - CAMBI DI STATO V40.5
    # ========================================================

    banner(
        "2 - TUTTI I CAMBI DI STATO V40.5"
    )

    changes = result[
        result["PHASE_CHANGED_V405"]
        == True
    ].copy()

    cols_changes = [
        "Date",
        "Close",
        "PREV_PHASE_V405",
        "PHASE_V405",
        "SAR_SIDE",
        "SAR_AGE",
        "WEAK_UP_TRIGGER",
        "LATERAL_SCORE",
        "LATERAL_RAW",
        "LATERAL_SIGNAL",
        "BULL_RECOVERY_RAW",
        "BULL_RECOVERY_CONFIRMED",
    ]

    print(
        changes[cols_changes].to_string(
            index=False,
        )
    )

    # ========================================================
    # 3 - PRIMO INGRESSO IN INDECISIONE
    # ========================================================

    banner(
        "3 - PRIMO INGRESSO IN INDECISIONE"
    )

    indecision_entries = result[
        (result["PHASE_V405"] == "INDECISIONE")
        &
        (
            result["PREV_PHASE_V405"]
            !=
            "INDECISIONE"
        )
    ].copy()

    if indecision_entries.empty:

        print(
            "Nessun ingresso in INDECISIONE "
            "nell'intervallo analizzato."
        )

    else:

        first = indecision_entries.iloc[0]

        first_date = first["Date"]

        print(
            "\nPRIMA DATA DI INGRESSO:"
        )

        print(first_date)

        # --------------------------------------------
        # mostra 3 settimane prima + 5 dopo
        # --------------------------------------------

        idx = result.index[
            result["Date"] == first_date
        ][0]

        lo = max(
            0,
            idx - 3,
        )

        hi = min(
            len(result),
            idx + 6,
        )

        focus = result.iloc[
            lo:hi
        ].copy()

        focus_cols = [
            "Date",
            "Close",
            "PREV_PHASE_V405",
            "PHASE_V404",
            "PHASE_V405",

            "SAR_SIDE",
            "SAR_AGE",

            "HA_DIRECTION",
            "HA_BODY_RATIO",
            "HA_INDECISION",
            "HA_INDECISION_COUNT3",
            "HA_BODY_SHRINK_2",

            "BB_POSITION",
            "BB_POSITION_D1",
            "V405_BB_DETACH_UPPER",

            "RSI_D1",
            "RSI_SLOPE3",
            "V405_RSI_HARD_WEAK_UP",

            "MACD_D1",
            "MACD_HIST_D1",
            "V405_MACD_HARD_WEAK_UP",

            "VOLUME_OSC_D1",
            "VOLUME_OSC_SLOPE3",
            "V405_VOL_PARTICIPATION_FALLING",

            "V405_WEAK_UP_MOM_COUNT",
            "V405_HA_WEAK_UP_STRUCTURE",
            "V405_WEAK_UP_BB_PATH",
            "V405_WEAK_UP_HA_PATH",
            "WEAK_UP_TRIGGER",

            "LATERAL_SCORE",
            "LATERAL_RAW",
            "LATERAL_SIGNAL",

            "V405_BULL_REC_HA",
            "V405_BULL_REC_BB",
            "V405_BULL_REC_RSI",
            "V405_BULL_REC_MACD",
            "V405_BULL_REC_VOLUME",
            "BULL_RECOVERY_RAW",
            "BULL_RECOVERY_CONFIRMED",
        ]

        print(
            focus[focus_cols].to_string(
                index=False,
                float_format=lambda x:
                f"{x:.3f}",
            )
        )

    # ========================================================
    # 4 - SETTIMANE INDECISIONE CON TRIGGER DEBOLEZZA
    # ========================================================

    banner(
        "4 - INDECISIONE + WEAK_UP_TRIGGER = 1"
    )

    trapped_weak = result[
        (result["PHASE_V405"] == "INDECISIONE")
        &
        (result["WEAK_UP_TRIGGER"] == 1)
    ].copy()

    if trapped_weak.empty:

        print(
            "Nessun caso."
        )

    else:

        print(
            trapped_weak[
                [
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
                ]
            ].to_string(
                index=False,
            )
        )

    # ========================================================
    # 5 - INDECISIONE CON RECOVERY RAW MA NON CONFERMATO
    # ========================================================

    banner(
        "5 - INDECISIONE + RECOVERY RAW = 1 MA NON CONFERMATO"
    )

    raw_not_confirmed = result[
        (result["PHASE_V405"] == "INDECISIONE")
        &
        (result["BULL_RECOVERY_RAW"] == 1)
        &
        (result["BULL_RECOVERY_CONFIRMED"] == 0)
    ].copy()

    if raw_not_confirmed.empty:

        print(
            "Nessun caso."
        )

    else:

        print(
            raw_not_confirmed[
                [
                    "Date",
                    "Close",
                    "V405_BULL_REC_HA",
                    "V405_BULL_REC_BB",
                    "V405_BULL_REC_RSI",
                    "V405_BULL_REC_MACD",
                    "V405_BULL_REC_VOLUME",
                    "BULL_RECOVERY_RAW",
                    "BULL_RECOVERY_CONFIRMED",
                    "PHASE_V405",
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
        "\nNessun motore MarketSentinel è stato modificato."
    )


if __name__ == "__main__":
    main()
    