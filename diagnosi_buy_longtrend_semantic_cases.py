"""
MarketSentinel
DIAGNOSI SEMANTICA BUY DOPO LUNGO TREND RIBASSISTA
==================================================

OBIETTIVO
---------
Verifica finale del candidato:

    SOLO BUY
    N = 8 settimane minimo di precedente regime SAR bearish
    K = 3 HA rialziste decisive
    MODE = CUMULATIVE

Non modifica alcun motore di produzione.

Vogliamo osservare soprattutto i veri nuovi run bullish abbastanza lunghi
e capire quando la nuova regola farebbe partire BUY1.

La logica diagnostica è:

- precedente SAR bearish >= 8 settimane
- nuovo SAR bullish qualificato da V40.10
- HA bullish decisiva -> +1 conferma
- HA indecisione -> non conta
- HA bearish decisiva -> non conta
- terza HA bullish decisiva -> BUY1 candidato

OUTPUT
------
data/diagnosi_buy_longtrend_semantic_cases.csv
data/diagnosi_buy_longtrend_semantic_summary.csv
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v4 as v4

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

OUT_CASES = Path(
    "data/diagnosi_buy_longtrend_semantic_cases.csv"
)

OUT_SUMMARY = Path(
    "data/diagnosi_buy_longtrend_semantic_summary.csv"
)

N_PREV_TREND_MIN = 8
K_HA_CONFIRMATIONS = 3

# Per la verifica semantica consideriamo "persistente"
# un nuovo SAR bullish che dura almeno 8 settimane.
MIN_NEW_BULL_RUN = 8


# ============================================================
# UTILS
# ============================================================

def first_existing(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None


def ticker_col(df):
    c = first_existing(
        df,
        ["Ticker", "TICKER", "ticker", "Symbol", "SYMBOL"],
    )
    if c is None:
        raise KeyError("Colonna ticker non trovata.")
    return c


def date_col(df):
    c = first_existing(
        df,
        ["Date", "DATE", "date", "Datetime", "DATETIME"],
    )
    if c is None:
        raise KeyError("Colonna data non trovata.")
    return c


def num(x, default=np.nan):
    try:
        v = pd.to_numeric(
            pd.Series([x]),
            errors="coerce",
        ).iloc[0]
    except Exception:
        return default

    if pd.isna(v):
        return default

    return v


def phase_value(row):
    for c in [
        "PHASE_DYNAMICS_V4",
        "PHASE",
        "PHASE_V409",
    ]:
        if c in row.index:
            return row.get(c)

    return np.nan


# ============================================================
# HA
# ============================================================

def bullish_decisive_ha(row):
    ha_dir = num(
        row.get("HA_DIRECTION", np.nan)
    )

    ha_ind = num(
        row.get("HA_INDECISION", np.nan)
    )

    if pd.isna(ha_dir) or pd.isna(ha_ind):
        return False

    return (
        int(ha_dir) == 1
        and int(ha_ind) == 0
    )


def ha_label(row):
    ha_dir = num(
        row.get("HA_DIRECTION", np.nan)
    )

    ha_ind = num(
        row.get("HA_INDECISION", np.nan)
    )

    if pd.isna(ha_dir) or pd.isna(ha_ind):
        return "UNKNOWN"

    if int(ha_ind) == 1:
        return "INDECISION"

    if int(ha_dir) == 1:
        return "BULL"

    if int(ha_dir) == -1:
        return "BEAR"

    return "UNKNOWN"


# ============================================================
# EXTRACT CURRENT SAR RUN
# ============================================================

def same_bull_sar_run(g, start_pos):

    positions = []

    for pos in range(start_pos, len(g)):

        side = num(
            g.iloc[pos].get("SAR_SIDE", np.nan)
        )

        if pd.isna(side):
            break

        if int(side) != 1:
            break

        positions.append(pos)

    return positions


# ============================================================
# PROCESS TICKER
# ============================================================

def process_ticker(raw, ticker):

    g = v4.process_ticker(
        raw.copy()
    ).reset_index(drop=True)

    dcol = date_col(g)

    prev_age_col = first_existing(
        g,
        [
            "V4010_PREV_SAR_AGE",
            "PREV_SAR_AGE",
            "V4010_PREVIOUS_SAR_AGE",
        ],
    )

    qualified_col = first_existing(
        g,
        [
            "V4010_RUN_QUALIFIED",
            "RUN_QUALIFIED",
        ],
    )

    if prev_age_col is None:
        raise KeyError(
            f"{ticker}: PREV_SAR_AGE non trovato."
        )

    if qualified_col is None:
        raise KeyError(
            f"{ticker}: RUN_QUALIFIED non trovato."
        )

    sar_side = pd.to_numeric(
        g["SAR_SIDE"],
        errors="coerce",
    )

    sar_age = pd.to_numeric(
        g["SAR_AGE"],
        errors="coerce",
    )

    qualified = pd.to_numeric(
        g[qualified_col],
        errors="coerce",
    ).fillna(0)

    flip_positions = np.where(
        (sar_side == 1)
        & (sar_age == 1)
        & (qualified == 1)
    )[0]

    records = []

    for start_pos in flip_positions:

        start_row = g.iloc[start_pos]

        prev_age = int(
            num(
                start_row.get(prev_age_col, 0),
                default=0,
            )
        )

        if prev_age < N_PREV_TREND_MIN:
            continue

        run_positions = same_bull_sar_run(
            g,
            start_pos,
        )

        run_length = len(run_positions)

        # Qui ci interessano i movimenti bullish persistenti.
        if run_length < MIN_NEW_BULL_RUN:
            continue

        confirmation_count = 0
        confirmation_week = None
        confirmation_pos = None

        sequence = []

        for week_no, pos in enumerate(
            run_positions,
            start=1,
        ):

            row = g.iloc[pos]

            label = ha_label(row)
            sequence.append(label)

            if bullish_decisive_ha(row):
                confirmation_count += 1

            if (
                confirmation_week is None
                and confirmation_count
                >= K_HA_CONFIRMATIONS
            ):
                confirmation_week = week_no
                confirmation_pos = pos

        if confirmation_week is None:
            continue

        conf_row = g.iloc[confirmation_pos]

        # Raccogliamo le prime settimane per poterle leggere
        # semanticamente senza stampare l'intero dataset.
        first_weeks = []

        for week_no, pos in enumerate(
            run_positions[:8],
            start=1,
        ):
            row = g.iloc[pos]

            first_weeks.append(
                {
                    "WEEK": week_no,
                    "DATE": row.get(dcol),
                    "PHASE": phase_value(row),
                    "HA": ha_label(row),
                    "RSI": num(
                        row.get("RSI", np.nan)
                    ),
                    "MACD_HIST": num(
                        row.get(
                            "MACD_HIST",
                            row.get(
                                "MACD_HISTOGRAM",
                                np.nan,
                            ),
                        )
                    ),
                }
            )

        records.append(
            {
                "TICKER": ticker,

                "FLIP_DATE":
                    start_row.get(dcol),

                "PREV_BEAR_SAR_AGE":
                    prev_age,

                "NEW_BULL_SAR_RUN_LENGTH":
                    run_length,

                "CONFIRMATION_WEEK":
                    confirmation_week,

                "BUY1_CANDIDATE_DATE":
                    conf_row.get(dcol),

                "PHASE_AT_FLIP":
                    phase_value(start_row),

                "PHASE_AT_CONFIRMATION":
                    phase_value(conf_row),

                "HA_SEQUENCE_FIRST_8W":
                    "|".join(
                        sequence[:8]
                    ),

                "W1_DATE":
                    first_weeks[0]["DATE"]
                    if len(first_weeks) >= 1
                    else np.nan,

                "W1_PHASE":
                    first_weeks[0]["PHASE"]
                    if len(first_weeks) >= 1
                    else np.nan,

                "W1_HA":
                    first_weeks[0]["HA"]
                    if len(first_weeks) >= 1
                    else np.nan,

                "W2_DATE":
                    first_weeks[1]["DATE"]
                    if len(first_weeks) >= 2
                    else np.nan,

                "W2_PHASE":
                    first_weeks[1]["PHASE"]
                    if len(first_weeks) >= 2
                    else np.nan,

                "W2_HA":
                    first_weeks[1]["HA"]
                    if len(first_weeks) >= 2
                    else np.nan,

                "W3_DATE":
                    first_weeks[2]["DATE"]
                    if len(first_weeks) >= 3
                    else np.nan,

                "W3_PHASE":
                    first_weeks[2]["PHASE"]
                    if len(first_weeks) >= 3
                    else np.nan,

                "W3_HA":
                    first_weeks[2]["HA"]
                    if len(first_weeks) >= 3
                    else np.nan,

                "W4_DATE":
                    first_weeks[3]["DATE"]
                    if len(first_weeks) >= 4
                    else np.nan,

                "W4_PHASE":
                    first_weeks[3]["PHASE"]
                    if len(first_weeks) >= 4
                    else np.nan,

                "W4_HA":
                    first_weeks[3]["HA"]
                    if len(first_weeks) >= 4
                    else np.nan,

                "W5_DATE":
                    first_weeks[4]["DATE"]
                    if len(first_weeks) >= 5
                    else np.nan,

                "W5_PHASE":
                    first_weeks[4]["PHASE"]
                    if len(first_weeks) >= 5
                    else np.nan,

                "W5_HA":
                    first_weeks[4]["HA"]
                    if len(first_weeks) >= 5
                    else np.nan,

                "RSI_AT_CONFIRMATION":
                    num(
                        conf_row.get(
                            "RSI",
                            np.nan,
                        )
                    ),

                "MACD_HIST_AT_CONFIRMATION":
                    num(
                        conf_row.get(
                            "MACD_HIST",
                            conf_row.get(
                                "MACD_HISTOGRAM",
                                np.nan,
                            ),
                        )
                    ),
            }
        )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("MARKETSENTINEL")
    print("SEMANTIC CHECK - BUY AFTER LONG BEAR TREND")
    print("=" * 78)

    print()
    print(
        "Regola candidata: "
        "BUY only / N=8 / K=3 / CUMULATIVE"
    )
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    print("Lettura FULL200...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    tcol = ticker_col(df)
    dcol = date_col(df)

    df[dcol] = pd.to_datetime(
        df[dcol],
        errors="coerce",
    )

    tickers = sorted(
        df[tcol]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    all_records = []
    errors = []

    for i, ticker in enumerate(
        tickers,
        start=1,
    ):

        print(
            f"[{i:03d}/{len(tickers):03d}] {ticker}",
            flush=True,
        )

        raw = (
            df.loc[
                df[tcol].astype(str)
                == ticker
            ]
            .copy()
            .sort_values(dcol)
            .reset_index(drop=True)
        )

        try:
            all_records.extend(
                process_ticker(
                    raw,
                    ticker,
                )
            )

        except Exception as exc:
            errors.append(
                {
                    "TICKER": ticker,
                    "ERROR": repr(exc),
                }
            )

            print(
                f"  ERRORE: {repr(exc)}",
                flush=True,
            )

    cases = pd.DataFrame(
        all_records
    )

    if len(cases):

        cases = cases.sort_values(
            [
                "CONFIRMATION_WEEK",
                "NEW_BULL_SAR_RUN_LENGTH",
                "TICKER",
                "FLIP_DATE",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        ).reset_index(drop=True)

    cases.to_csv(
        OUT_CASES,
        index=False,
    )

    if len(cases):

        summary = pd.DataFrame(
            [
                {
                    "CASES":
                        len(cases),

                    "MEDIAN_CONFIRMATION_WEEK":
                        cases[
                            "CONFIRMATION_WEEK"
                        ].median(),

                    "MEAN_CONFIRMATION_WEEK":
                        cases[
                            "CONFIRMATION_WEEK"
                        ].mean(),

                    "CONFIRMED_WEEK_3":
                        int(
                            (
                                cases[
                                    "CONFIRMATION_WEEK"
                                ] == 3
                            ).sum()
                        ),

                    "CONFIRMED_BY_WEEK_4":
                        int(
                            (
                                cases[
                                    "CONFIRMATION_WEEK"
                                ] <= 4
                            ).sum()
                        ),

                    "CONFIRMED_BY_WEEK_4_PCT":
                        100.0
                        * (
                            cases[
                                "CONFIRMATION_WEEK"
                            ] <= 4
                        ).mean(),

                    "CONFIRMED_AFTER_WEEK_5":
                        int(
                            (
                                cases[
                                    "CONFIRMATION_WEEK"
                                ] > 5
                            ).sum()
                        ),

                    "CONFIRMED_AFTER_WEEK_5_PCT":
                        100.0
                        * (
                            cases[
                                "CONFIRMATION_WEEK"
                            ] > 5
                        ).mean(),

                    "MEDIAN_NEW_BULL_RUN":
                        cases[
                            "NEW_BULL_SAR_RUN_LENGTH"
                        ].median(),
                }
            ]
        )

    else:

        summary = pd.DataFrame(
            [
                {
                    "CASES": 0,
                }
            ]
        )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    print()
    print("=" * 78)
    print("RISULTATI")
    print("=" * 78)
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Errori: {len(errors)}"
    )

    print()
    print("Output:")
    print(f"  {OUT_CASES}")
    print(f"  {OUT_SUMMARY}")

    print()
    print(
        "SEMANTIC BUY CHECK COMPLETATO."
    )


if __name__ == "__main__":
    main()