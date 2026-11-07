"""
MARKETSENTINEL - CLASSIFICA OPERATIVA 200 TITOLI

Legge gli archivi V160 esistenti.
Non scarica ancora nuove quotazioni.
Non modifica PHASE canonica o modelli scientifici.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from operational.phase_nuova import calculate_phase_nuova
from operational.build_usa80_v160_ratings import (
    NAMES,
    DIRECTIONS,
    load_thresholds,
    values_for_row,
    score_from_thresholds,
)

BASE = ROOT / "data" / "operational_market"

MARKETS = {
    "ITALIA": "italy40",
    "EUROPA": "europe40",
    "USA80": "usa80",
    "USA40": "usa40",
}

BUY_PHASES = ("BUY1", "BUY2", "BUY3")


def calculate_rating(g, i, phase, thresholds):
    try:
        values = values_for_row(g, i, phase)

        scores = [
            score_from_thresholds(v, t, d)
            for v, t, d in zip(
                values,
                thresholds[phase],
                DIRECTIONS,
            )
        ]

        if len(scores) != len(NAMES):
            raise ValueError("Numero indicatori V160 errato")

        return round(float(np.mean(scores)), 3)

    except (ValueError, KeyError, IndexError, TypeError):
        return np.nan


def main():
    print("=== MARKETSENTINEL 200 ===", flush=True)

    frames = []

    for market, prefix in MARKETS.items():
        path = BASE / f"{prefix}_v160_features.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        print("CARICAMENTO:", market, flush=True)

        required_columns = {
            "Ticker", "Date", "PHASE", "Close",
            "SAR", "SAR_SIDE", "SAR_FLIP_UP", "V4015_HA_DIRECTION", "V4015_HA_INDECISION",
            "V4015_ATR_PCT",
            "V4012_SAR_DISTANCE_PCT",
            "V4015_BB_WIDTH_PCT",
            "V4012_DISTANCE_TO_BB_LOWER_PCT",
            "V4015_DIST_26W_LOW_PCT",
        }

        available = set(pd.read_csv(path, nrows=0).columns)
        missing = required_columns - available
        if missing:
            raise ValueError(
                f"{market}: colonne mancanti: {sorted(missing)}"
            )

        df = pd.read_csv(
            path,
            usecols=lambda col: col in required_columns,
            low_memory=False,
        )
        df["MERCATO"] = market
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    data["Date"] = pd.to_datetime(
        data["Date"], format="mixed", utc=True
    )

    data = data.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    if data["Ticker"].nunique() != 199:
        raise RuntimeError("Universo diverso da 199 titoli")

    if data.duplicated(["Ticker", "Date"]).any():
        raise RuntimeError("Date duplicate per ticker")

    thresholds = {
        phase: load_thresholds(phase)
        for phase in BUY_PHASES
    }

    print("CALCOLO V160 FAST...", flush=True)

    from operational.v160_fast import calculate_all_ratings

    data = calculate_all_ratings(data)

    print("APPLICAZIONE PHASE NUOVA...", flush=True)

    history = calculate_phase_nuova(
        data,
        phase_col="PHASE",
        sar_col="SAR_FLIP_UP",
        ha_col="V4015_HA_DIRECTION",
        rating_col="RATING_V160_BUY1",
    )

    # PR WEEKLY: MA52, applicata dopo PHASE NUOVA scientifica.
    from ms_pr_filter import apply_pr_filter
    history = apply_pr_filter(history, timeframe="WEEKLY")
    phase = history["PHASE_PRE_PR"]

    history["RATING_V160"] = np.select(
        [
            phase.eq("BUY1"),
            phase.eq("BUY2"),
            phase.eq("BUY3"),
        ],
        [
            history["RATING_V160_BUY1"],
            history["RATING_V160_BUY2"],
            history["RATING_V160_BUY3"],
        ],
        default=np.nan,
    )



    latest = (
        history.sort_values("Date")
        .groupby("Ticker", as_index=False)
        .tail(1)
        .copy()
    )

    reference_date = history["Date"].max()
    latest["GIORNI_RITARDO"] = (
        reference_date - latest["Date"]
    ).dt.days

    # Stato aggiornamento rispetto alla settimana
    # provvisoria più recente nell'intero universo.
    # I titoli ancora fermi alla settimana definitiva
    # conservano PHASE, V160, prezzi e data originali.
    latest["STATO_DATI"] = np.where(
        latest["Date"] < reference_date,
        "IN_ATTESA_AGGIORNAMENTO",
        "AGGIORNATO",
    )

    priority = {
        "BUY1": 0,
        "BUY1(PR)": 0,
        "BUY2": 1,
        "BUY2(PR)": 1,
        "BUY3": 2,
        "BUY3(PR)": 2,
    }

    latest["PRIORITA"] = (
        latest["PHASE_NUOVA"]
        .map(priority)
        .fillna(3)
        .astype(int)
    )

    latest = latest.sort_values(
        ["PRIORITA", "RATING_V160", "Ticker"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    latest.insert(
        0, "RANK", range(1, len(latest) + 1)
    )

    columns = [
        "RANK",
        "Ticker",
        "MERCATO",
        "Date",
        "PHASE",
        "PHASE_NUOVA",
        "EARLY_BUY",
        "EARLY_BUY_RATING",
        "PHASE_NUOVA_REASON",
        "RATING_V160",
        "RATING_V160_BUY1",
        "STATO_DATI",
        "GIORNI_RITARDO",
        "Close",
        "SAR",
        "V4015_ATR_PCT",
        "V4015_BB_WIDTH_PCT",
        "V4012_SAR_DISTANCE_PCT",
    ]

    columns = [
        c for c in columns
        if c in latest.columns
    ]

    output = BASE / "marketsentinel_200_phase_nuova.csv"
    historical_output = (
        BASE / "marketsentinel_200_phase_nuova_history.csv"
    )

    latest[columns].to_csv(output, index=False)
    history_columns = [
        "Ticker",
        "MERCATO",
        "Date",
        "Close",
        "PHASE",
        "PHASE_NUOVA",
        "EARLY_BUY",
        "EARLY_BUY_RATING",
        "PHASE_NUOVA_REASON",
        "RATING_V160",
        "RATING_V160_BUY1",
        "RATING_V160_BUY2",
        "RATING_V160_BUY3",
    ]

    history_columns = [
        c for c in history_columns
        if c in history.columns
    ]

    history[history_columns].to_csv(
        historical_output,
        index=False,
    )

    print("\n=== RISULTATO ===")
    print("TITOLI:", len(latest))
    print("DATA RIFERIMENTO:", reference_date.date())
    print("EARLY BUY ATTIVI:", int(latest["EARLY_BUY"].sum()))
    print(
        "DATI NON AGGIORNATI:",
        int((latest["STATO_DATI"] == "IN_ATTESA_AGGIORNAMENTO").sum())
    )

    print("\n=== PRIME 30 POSIZIONI ===")
    print(
        latest[[
            "Ticker",
            "MERCATO",
            "PHASE",
            "PHASE_NUOVA",
            "RATING_V160",
            "STATO_DATI",
        ]].head(30).to_string(index=False)
    )

    print("\nCLASSIFICA:", output)
    print("STORICO:", historical_output)
    print("CALCOLO COMPLETATO")


if __name__ == "__main__":
    main()
