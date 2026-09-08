"""
MarketSentinel
DIAGNOSI V40.10 - RUN SAR QUALIFICATI / NON QUALIFICATI

OBIETTIVO
---------
Capire PERCHÉ alcuni nuovi run SAR vengono classificati da V40.10 come:

    V4010_RUN_QUALIFIED = 0

pur arrivando dalla PHASE precedente come BUY / SELL.

IMPORTANTE
----------
- NON modifica production
- NON modifica Dynamics V3.1
- NON modifica V40.10
- NON introduce nuove regole
- legge semplicemente le variabili che V40.10 produce

INPUT
-----
data/v40_35_full200_weekly.csv

FOCUS
-----
40 ticker italiani (.MI)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import weekly_v40_5_phase_dynamics_shadow as dynamics
import weekly_v40_10_sar_phase_fix as v4010


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")


# ============================================================
# HELPERS
# ============================================================

def safe_value_counts(series):
    return (
        series
        .fillna("<NA>")
        .astype(str)
        .value_counts(dropna=False)
    )


def process_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stessa catena usata nella validazione precedente:
    Dynamics V3.1 -> componenti V40.10.
    """

    g = dynamics.process_ticker(df.copy())

    g["PHASE_BEFORE_V4010"] = g["PHASE_DYNAMICS_SHADOW"].copy()
    g["PHASE"] = g["PHASE_BEFORE_V4010"].copy()

    g = v4010.add_previous_sar_run_information(g)
    g = v4010.qualify_sar_flips(g)
    g = v4010.propagate_qualified_signal(g)

    # Salviamo lo stato PRIMA dell'ultimo overwrite V40.10
    g["PHASE_PRE_FIX_V4010"] = g["PHASE"].copy()

    g = v4010.apply_v4010_fix(g)

    g["PHASE_FINAL_V4010"] = g["PHASE"].copy()

    return g


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 140)
    print("DIAGNOSI V40.10 - RUN SAR QUALIFICATI / NON QUALIFICATI")
    print("=" * 140)

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    if "Ticker" not in df.columns:
        raise RuntimeError("Colonna Ticker non trovata.")

    if "Date" not in df.columns:
        raise RuntimeError("Colonna Date non trovata.")

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    italy_tickers = sorted(
        [
            ticker
            for ticker in df["Ticker"].dropna().unique()
            if str(ticker).endswith(".MI")
        ]
    )

    print()
    print(f"Ticker italiani trovati: {len(italy_tickers)}")
    print()

    all_results = []

    for i, ticker in enumerate(
        italy_tickers,
        start=1,
    ):

        print(
            f"[{i:02d}/{len(italy_tickers):02d}] {ticker}"
        )

        g = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if g.empty:
            continue

        try:
            z = process_chain(g)

        except Exception as exc:
            print(f"  ERRORE: {exc}")
            continue

        z["Ticker"] = ticker

        all_results.append(z)

    if not all_results:
        raise RuntimeError("Nessun ticker processato.")

    out = pd.concat(
        all_results,
        ignore_index=True,
    )

    # ========================================================
    # COLONNE V40.10 DISPONIBILI
    # ========================================================

    v4010_cols = sorted(
        [
            c
            for c in out.columns
            if str(c).startswith("V4010_")
        ]
    )

    print()
    print("=" * 140)
    print("COLONNE V40.10 TROVATE")
    print("=" * 140)

    for col in v4010_cols:
        print(col)

    # ========================================================
    # RUN START
    # ========================================================

    required = [
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_DYNAMICS_BASE",
        "PHASE_DYNAMICS_SHADOW",
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

    run_start = (
        out["SAR_AGE"] == 1
    )

    qualified_start = (
        run_start
        &
        (out["V4010_RUN_QUALIFIED"] == 1)
    )

    rejected_start = (
        run_start
        &
        (out["V4010_RUN_QUALIFIED"] == 0)
    )

    print()
    print("=" * 140)
    print("TOTALI RUN SAR")
    print("=" * 140)

    print()
    print(
        f"Run SAR AGE 1 totali: "
        f"{int(run_start.sum())}"
    )

    print(
        f"Run qualificati: "
        f"{int(qualified_start.sum())}"
    )

    print(
        f"Run NON qualificati: "
        f"{int(rejected_start.sum())}"
    )

    # ========================================================
    # DIREZIONE
    # ========================================================

    print()
    print("=" * 140)
    print("RUN NON QUALIFICATI PER DIREZIONE")
    print("=" * 140)

    rejected_direction = (
        out.loc[
            rejected_start,
            "SAR_SIDE",
        ]
        .value_counts(dropna=False)
        .reset_index()
    )

    rejected_direction.columns = [
        "SAR_SIDE",
        "COUNT",
    ]

    print()
    print(
        rejected_direction.to_string(
            index=False
        )
    )

    # ========================================================
    # QUANTI ERANO GIÀ BUY / SELL A MONTE
    # ========================================================

    rejected_buy = (
        rejected_start
        &
        (out["SAR_SIDE"] == 1)
        &
        (
            out["PHASE_DYNAMICS_SHADOW"]
            == "BUY"
        )
    )

    rejected_sell = (
        rejected_start
        &
        (out["SAR_SIDE"] == -1)
        &
        (
            out["PHASE_DYNAMICS_SHADOW"]
            == "SELL"
        )
    )

    print()
    print("=" * 140)
    print("RUN NON QUALIFICATI MA GIÀ BUY / SELL A MONTE")
    print("=" * 140)

    print()
    print(
        f"Run bullish non qualificati ma già BUY: "
        f"{int(rejected_buy.sum())}"
    )

    print(
        f"Run bearish non qualificati ma già SELL: "
        f"{int(rejected_sell.sum())}"
    )

    # ========================================================
    # DISTRIBUZIONE PHASE A MONTE
    # ========================================================

    print()
    print("=" * 140)
    print("PHASE A MONTE DEI RUN NON QUALIFICATI")
    print("=" * 140)

    phase_origin = (
        out.loc[
            rejected_start,
            [
                "SAR_SIDE",
                "PHASE_DYNAMICS_BASE",
                "PHASE_DYNAMICS_SHADOW",
            ],
        ]
        .value_counts()
        .reset_index(name="COUNT")
        .sort_values(
            "COUNT",
            ascending=False,
        )
    )

    print()
    print(
        phase_origin.to_string(
            index=False
        )
    )

    # ========================================================
    # ANALISI DI TUTTE LE VARIABILI V40.10
    # ========================================================

    print()
    print("=" * 140)
    print("CONFRONTO VARIABILI V40.10 - QUALIFICATI VS NON QUALIFICATI")
    print("=" * 140)

    analysis_rows = []

    for col in v4010_cols:

        if col == "V4010_RUN_QUALIFIED":
            continue

        q = out.loc[
            qualified_start,
            col,
        ]

        r = out.loc[
            rejected_start,
            col,
        ]

        # Proviamo lettura numerica
        q_num = pd.to_numeric(
            q,
            errors="coerce",
        )

        r_num = pd.to_numeric(
            r,
            errors="coerce",
        )

        q_numeric_count = int(
            q_num.notna().sum()
        )

        r_numeric_count = int(
            r_num.notna().sum()
        )

        if (
            q_numeric_count > 0
            or
            r_numeric_count > 0
        ):

            analysis_rows.append(
                {
                    "COLUMN": col,
                    "QUAL_MEAN": (
                        float(q_num.mean())
                        if q_numeric_count
                        else np.nan
                    ),
                    "REJECT_MEAN": (
                        float(r_num.mean())
                        if r_numeric_count
                        else np.nan
                    ),
                    "QUAL_MIN": (
                        float(q_num.min())
                        if q_numeric_count
                        else np.nan
                    ),
                    "REJECT_MIN": (
                        float(r_num.min())
                        if r_numeric_count
                        else np.nan
                    ),
                    "QUAL_MAX": (
                        float(q_num.max())
                        if q_numeric_count
                        else np.nan
                    ),
                    "REJECT_MAX": (
                        float(r_num.max())
                        if r_numeric_count
                        else np.nan
                    ),
                }
            )

    comparison = pd.DataFrame(
        analysis_rows
    )

    if not comparison.empty:

        pd.set_option(
            "display.max_rows",
            500,
        )

        pd.set_option(
            "display.max_columns",
            100,
        )

        pd.set_option(
            "display.width",
            250,
        )

        print()
        print(
            comparison.to_string(
                index=False
            )
        )

    # ========================================================
    # VALUE COUNTS DELLE COLONNE V40.10
    # ========================================================

    print()
    print("=" * 140)
    print("VALORI DELLE VARIABILI V40.10 NEI RUN NON QUALIFICATI")
    print("=" * 140)

    for col in v4010_cols:

        if col == "V4010_RUN_QUALIFIED":
            continue

        print()
        print("-" * 120)
        print(col)
        print("-" * 120)

        vc = safe_value_counts(
            out.loc[
                rejected_start,
                col,
            ]
        )

        print(
            vc.head(20).to_string()
        )

    # ========================================================
    # ESEMPI BULLISH NON QUALIFICATI
    # ========================================================

    diagnostic_cols = [
        "Ticker",
        "Date",
        "SAR_SIDE",
        "SAR_AGE",
        "PHASE_DYNAMICS_BASE",
        "PHASE_DYNAMICS_SHADOW",
    ]

    for col in v4010_cols:
        if col not in diagnostic_cols:
            diagnostic_cols.append(col)

    for optional in [
        "PHASE_PRE_FIX_V4010",
        "PHASE_FINAL_V4010",
    ]:
        if optional in out.columns:
            diagnostic_cols.append(optional)

    diagnostic_cols = [
        c
        for c in diagnostic_cols
        if c in out.columns
    ]

    print()
    print("=" * 140)
    print("PRIMI 40 RUN BULLISH NON QUALIFICATI")
    print("=" * 140)

    bull_rejected = (
        rejected_start
        &
        (out["SAR_SIDE"] == 1)
    )

    print()
    print(
        out.loc[
            bull_rejected,
            diagnostic_cols,
        ]
        .head(40)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 140)
    print("PRIMI 40 RUN BEARISH NON QUALIFICATI")
    print("=" * 140)

    bear_rejected = (
        rejected_start
        &
        (out["SAR_SIDE"] == -1)
    )

    print()
    print(
        out.loc[
            bear_rejected,
            diagnostic_cols,
        ]
        .head(40)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # CONFRONTO QUALIFICATI / NON QUALIFICATI
    # PER LE SINGOLE COLONNE BINARIE
    # ========================================================

    print()
    print("=" * 140)
    print("COLONNE V40.10 POTENZIALMENTE DISCRIMINANTI")
    print("=" * 140)

    discriminating_rows = []

    for col in v4010_cols:

        if col == "V4010_RUN_QUALIFIED":
            continue

        q_num = pd.to_numeric(
            out.loc[
                qualified_start,
                col,
            ],
            errors="coerce",
        )

        r_num = pd.to_numeric(
            out.loc[
                rejected_start,
                col,
            ],
            errors="coerce",
        )

        if (
            q_num.notna().sum() == 0
            or
            r_num.notna().sum() == 0
        ):
            continue

        q_mean = q_num.mean()
        r_mean = r_num.mean()

        if (
            pd.isna(q_mean)
            or
            pd.isna(r_mean)
        ):
            continue

        diff = abs(
            float(q_mean)
            -
            float(r_mean)
        )

        discriminating_rows.append(
            {
                "COLUMN": col,
                "QUALIFIED_MEAN": float(q_mean),
                "REJECTED_MEAN": float(r_mean),
                "ABS_DIFF": diff,
            }
        )

    discr = pd.DataFrame(
        discriminating_rows
    )

    if not discr.empty:

        discr = discr.sort_values(
            "ABS_DIFF",
            ascending=False,
        )

        print()
        print(
            discr.head(30).to_string(
                index=False
            )
        )

    print()
    print("=" * 140)
    print("FINE DIAGNOSI")
    print("=" * 140)


if __name__ == "__main__":
    main()