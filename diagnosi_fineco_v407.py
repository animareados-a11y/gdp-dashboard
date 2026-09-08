from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_4_core_phase_engine as v404
import weekly_v40_5_core_phase_engine as v405
import weekly_v40_6_transition_engine as v406
import weekly_v40_7_weakness_severity_engine as v407


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

TICKER = "FBK.MI"

START_DATE = "2026-07-03"
END_DATE = "2026-08-28"

FOCUS_DATES = [
    "2026-08-07",
    "2026-08-14",
    "2026-08-21",
    "2026-08-28",
]

OUTPUT_FILE = Path(
    "data/diagnosi_fineco_v407.csv"
)


# ============================================================
# HELPERS
# ============================================================

def banner(text: str) -> None:
    print("\n" + "=" * 170)
    print(text)
    print("=" * 170)


def normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["Date"] = pd.to_datetime(
        x["Date"],
        utc=True,
        errors="coerce",
    )

    return x


def get_value(
    df: pd.DataFrame,
    date: pd.Timestamp,
    column: str,
):
    if column not in df.columns:
        return None

    row = df.loc[df["Date"] == date]

    if row.empty:
        return None

    return row.iloc[0][column]


def phase_table(
    o404: pd.DataFrame,
    o405: pd.DataFrame,
    o406: pd.DataFrame,
    o407: pd.DataFrame,
) -> pd.DataFrame:

    dates = (
        o404[
            (o404["Date"] >= pd.Timestamp(START_DATE, tz="UTC"))
            &
            (o404["Date"] <= pd.Timestamp(END_DATE, tz="UTC"))
        ]["Date"]
        .tolist()
    )

    rows = []

    for d in dates:

        rows.append(
            {
                "Date": d.strftime("%Y-%m-%d"),

                "Close":
                    get_value(o404, d, "Close"),

                "SAR_SIDE":
                    get_value(o404, d, "SAR_SIDE"),

                "SAR_AGE":
                    get_value(o404, d, "SAR_AGE"),

                "WEAK_UP_TRIGGER":
                    get_value(o404, d, "WEAK_UP_TRIGGER"),

                "LATERAL_SIGNAL":
                    get_value(o404, d, "LATERAL_SIGNAL"),

                "BULL_RECOVERY_RAW":
                    get_value(
                        o404,
                        d,
                        "BULL_RECOVERY_RAW",
                    ),

                "BULL_RECOVERY_CONFIRMED":
                    get_value(
                        o404,
                        d,
                        "BULL_RECOVERY_CONFIRMED",
                    ),

                "PHASE_V404":
                    get_value(o404, d, "PHASE"),

                "PHASE_V405":
                    get_value(o405, d, "PHASE"),

                "PHASE_V406":
                    get_value(o406, d, "PHASE"),

                "PHASE_V407":
                    get_value(o407, d, "PHASE"),
            }
        )

    return pd.DataFrame(rows)


def interesting_columns(df: pd.DataFrame) -> list[str]:

    keywords = [
        "PHASE",
        "WEAK",
        "LATERAL",
        "RECOVERY",
        "TRANS",
        "INDEC",
        "DAMAGE",
        "CONFIRM",
        "SEVER",
        "LEVEL",
        "SAR",
    ]

    cols = []

    for col in df.columns:

        upper = col.upper()

        if any(k in upper for k in keywords):
            cols.append(col)

    return cols


def print_focus(
    name: str,
    df: pd.DataFrame,
) -> None:

    banner(
        f"DETTAGLIO COLONNE INTERESSANTI - {name}"
    )

    focus_dates = [
        pd.Timestamp(x, tz="UTC")
        for x in FOCUS_DATES
    ]

    x = df[
        df["Date"].isin(focus_dates)
    ].copy()

    cols = interesting_columns(df)

    base = [
        c
        for c in [
            "Date",
            "Close",
        ]
        if c in x.columns
    ]

    cols = base + [
        c
        for c in cols
        if c not in base
    ]

    print(
        f"\nNumero colonne interessanti: {len(cols)}"
    )

    print("\nNomi colonne:")
    for c in cols:
        print(" -", c)

    print("\nVALORI:")

    if not x.empty:
        print(
            x[cols].to_string(
                index=False,
            )
        )
    else:
        print("Nessuna riga trovata.")


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "MARKETSENTINEL - DIAGNOSI FINECO V40.4 -> V40.7"
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
    # ESECUZIONE SEPARATA DEI 4 MOTORI
    # ========================================================

    banner("CALCOLO V40.4")
    o404 = v404.process_ticker(
        g.copy()
    )

    banner("CALCOLO V40.5")
    o405 = v405.process_ticker(
        g.copy()
    )

    banner("CALCOLO V40.6")
    o406 = v406.process_ticker(
        g.copy()
    )

    banner("CALCOLO V40.7")
    o407 = v407.process_ticker(
        g.copy()
    )

    # ========================================================
    # 1 - CATENA DELLA PHASE
    # ========================================================

    result = phase_table(
        o404,
        o405,
        o406,
        o407,
    )

    banner(
        "1 - CATENA PHASE V40.4 -> V40.5 -> V40.6 -> V40.7"
    )

    print(
        result.to_string(
            index=False,
        )
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # 2 - SOLO DOVE CAMBIA
    # ========================================================

    banner(
        "2 - SETTIMANE IN CUI UNA VERSIONE CAMBIA LA PHASE"
    )

    changed = result[
        (result["PHASE_V404"] != result["PHASE_V405"])
        |
        (result["PHASE_V405"] != result["PHASE_V406"])
        |
        (result["PHASE_V406"] != result["PHASE_V407"])
    ].copy()

    if changed.empty:
        print(
            "Nessuna differenza trovata."
        )
    else:
        print(
            changed.to_string(
                index=False,
            )
        )

    # ========================================================
    # 3 - FOCUS 21 E 28 AGOSTO
    # ========================================================

    banner(
        "3 - FOCUS 7 / 14 / 21 / 28 AGOSTO"
    )

    focus = result[
        result["Date"].isin(
            FOCUS_DATES
        )
    ]

    print(
        focus.to_string(
            index=False,
        )
    )

    # ========================================================
    # 4 - COLONNE INTERNE DEI MOTORI
    # ========================================================

    print_focus(
        "V40.5",
        o405,
    )

    print_focus(
        "V40.6",
        o406,
    )

    print_focus(
        "V40.7",
        o407,
    )

    # ========================================================
    # 5 - IDENTIFICAZIONE AUTOMATICA DEL PRIMO PASSAGGIO
    # ========================================================

    banner(
        "5 - CHI TRASFORMA DEBOLEZZA_RIALZISTA IN INDECISIONE?"
    )

    for ds in [
        "2026-08-21",
        "2026-08-28",
    ]:

        d = pd.Timestamp(
            ds,
            tz="UTC",
        )

        p4 = get_value(
            o404,
            d,
            "PHASE",
        )

        p5 = get_value(
            o405,
            d,
            "PHASE",
        )

        p6 = get_value(
            o406,
            d,
            "PHASE",
        )

        p7 = get_value(
            o407,
            d,
            "PHASE",
        )

        print(f"\n{ds}")
        print(f"V40.4 = {p4}")
        print(f"V40.5 = {p5}")
        print(f"V40.6 = {p6}")
        print(f"V40.7 = {p7}")

        if p4 != p5:
            print(
                ">>> PRIMO CAMBIO: V40.4 -> V40.5"
            )

        elif p5 != p6:
            print(
                ">>> PRIMO CAMBIO: V40.5 -> V40.6"
            )

        elif p6 != p7:
            print(
                ">>> PRIMO CAMBIO: V40.6 -> V40.7"
            )

        else:
            print(
                ">>> Nessun cambio tra queste versioni."
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