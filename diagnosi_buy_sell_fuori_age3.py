"""
MarketSentinel
DIAGNOSI BUY / SELL FUORI DALLE PRIME 3 SETTIMANE

OBIETTIVO
---------
Capire da dove arrivano i BUY e SELL finali che risultano
fuori da un regime SAR qualificato AGE 1-3.

Il file NON modifica nessun file production.
Fa solo diagnosi.

INPUT
-----
data/v40_35_full200_weekly.csv

CATENA
------
Dynamics V3.1
-> V40.10 componenti
-> analisi BUY/SELL finali fuori AGE 1-3
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow as dynamics
import weekly_v40_10_sar_phase_fix as v4010


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")


def process_complete_chain(df: pd.DataFrame) -> pd.DataFrame:
    g = dynamics.process_ticker(df.copy())

    g["PHASE_BEFORE_V4010"] = g["PHASE_DYNAMICS_SHADOW"].copy()
    g["PHASE"] = g["PHASE_BEFORE_V4010"].copy()

    g = v4010.add_previous_sar_run_information(g)
    g = v4010.qualify_sar_flips(g)
    g = v4010.propagate_qualified_signal(g)
    g = v4010.apply_v4010_fix(g)

    g["PHASE_FINAL_V4010"] = g["PHASE"].copy()

    return g


def main():
    print("=" * 120)
    print("DIAGNOSI BUY / SELL FUORI DALLE PRIME 3 SETTIMANE")
    print("=" * 120)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    if "Ticker" not in df.columns:
        raise RuntimeError("Colonna Ticker non trovata.")

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    italy_tickers = sorted(
        [
            t
            for t in df["Ticker"].dropna().unique()
            if str(t).endswith(".MI")
        ]
    )

    print()
    print(f"Ticker italiani trovati: {len(italy_tickers)}")
    print()

    all_rows = []

    for i, ticker in enumerate(italy_tickers, start=1):
        print(f"[{i:02d}/{len(italy_tickers):02d}] {ticker}")

        g = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if g.empty:
            continue

        try:
            z = process_complete_chain(g)
        except Exception as exc:
            print(f"  ERRORE: {exc}")
            continue

        z["Ticker"] = ticker
        all_rows.append(z)

    if not all_rows:
        raise RuntimeError("Nessun ticker processato.")

    out = pd.concat(all_rows, ignore_index=True)

    required = [
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_DYNAMICS_BASE",
        "PHASE_DYNAMICS_SHADOW",
        "PHASE_FINAL_V4010",
    ]

    missing = [c for c in required if c not in out.columns]

    if missing:
        raise RuntimeError(
            "Colonne mancanti: " + ", ".join(missing)
        )

    valid_buy = (
        (out["V4010_RUN_QUALIFIED"] == 1)
        &
        (out["SAR_SIDE"] == 1)
        &
        (out["SAR_AGE"].isin([1, 2, 3]))
    )

    valid_sell = (
        (out["V4010_RUN_QUALIFIED"] == 1)
        &
        (out["SAR_SIDE"] == -1)
        &
        (out["SAR_AGE"].isin([1, 2, 3]))
    )

    buy_outside = (
        (out["PHASE_FINAL_V4010"] == "BUY")
        &
        (~valid_buy)
    )

    sell_outside = (
        (out["PHASE_FINAL_V4010"] == "SELL")
        &
        (~valid_sell)
    )

    print()
    print("=" * 120)
    print("TOTALI")
    print("=" * 120)

    print(f"BUY finali fuori AGE 1-3: {int(buy_outside.sum())}")
    print(f"SELL finali fuori AGE 1-3: {int(sell_outside.sum())}")

    print()
    print("=" * 120)
    print("ORIGINE DEI BUY FUORI AGE 1-3")
    print("=" * 120)

    buy_origin = (
        out.loc[
            buy_outside,
            [
                "PHASE_DYNAMICS_BASE",
                "PHASE_DYNAMICS_SHADOW",
                "PHASE_FINAL_V4010",
            ],
        ]
        .value_counts()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    print()
    print(buy_origin.to_string(index=False))

    print()
    print("=" * 120)
    print("ORIGINE DEI SELL FUORI AGE 1-3")
    print("=" * 120)

    sell_origin = (
        out.loc[
            sell_outside,
            [
                "PHASE_DYNAMICS_BASE",
                "PHASE_DYNAMICS_SHADOW",
                "PHASE_FINAL_V4010",
            ],
        ]
        .value_counts()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    print()
    print(sell_origin.to_string(index=False))

    print()
    print("=" * 120)
    print("BUY FUORI AGE 1-3 - SAR SIDE / AGE / QUALIFIED")
    print("=" * 120)

    buy_sar = (
        out.loc[
            buy_outside,
            [
                "SAR_SIDE",
                "SAR_AGE",
                "V4010_RUN_QUALIFIED",
            ],
        ]
        .value_counts()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    print()
    print(buy_sar.to_string(index=False))

    print()
    print("=" * 120)
    print("SELL FUORI AGE 1-3 - SAR SIDE / AGE / QUALIFIED")
    print("=" * 120)

    sell_sar = (
        out.loc[
            sell_outside,
            [
                "SAR_SIDE",
                "SAR_AGE",
                "V4010_RUN_QUALIFIED",
            ],
        ]
        .value_counts()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    print()
    print(sell_sar.to_string(index=False))

    print()
    print("=" * 120)
    print("TOP TICKER CON BUY FUORI AGE 1-3")
    print("=" * 120)

    buy_ticker = (
        out.loc[buy_outside, "Ticker"]
        .value_counts()
        .reset_index()
    )
    buy_ticker.columns = ["Ticker", "Count"]

    print()
    print(buy_ticker.head(20).to_string(index=False))

    print()
    print("=" * 120)
    print("TOP TICKER CON SELL FUORI AGE 1-3")
    print("=" * 120)

    sell_ticker = (
        out.loc[sell_outside, "Ticker"]
        .value_counts()
        .reset_index()
    )
    sell_ticker.columns = ["Ticker", "Count"]

    print()
    print(sell_ticker.head(20).to_string(index=False))

    print()
    print("=" * 120)
    print("ESEMPI BUY FUORI AGE 1-3")
    print("=" * 120)

    cols = [
        "Ticker",
        "Date",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_DYNAMICS_BASE",
        "PHASE_DYNAMICS_SHADOW",
        "PHASE_FINAL_V4010",
    ]

    if "PHASE_DYNAMICS_REASON" in out.columns:
        cols.append("PHASE_DYNAMICS_REASON")

    print()
    print(
        out.loc[buy_outside, cols]
        .head(40)
        .to_string(index=False)
    )

    print()
    print("=" * 120)
    print("ESEMPI SELL FUORI AGE 1-3")
    print("=" * 120)

    print()
    print(
        out.loc[sell_outside, cols]
        .head(40)
        .to_string(index=False)
    )

    print()
    print("=" * 120)
    print("CONTROLLO CHIAVE")
    print("=" * 120)

    inherited_buy = (
        buy_outside
        &
        (out["PHASE_DYNAMICS_SHADOW"] == "BUY")
    )

    inherited_sell = (
        sell_outside
        &
        (out["PHASE_DYNAMICS_SHADOW"] == "SELL")
    )

    print()
    print(
        "BUY fuori AGE 1-3 già presenti prima di V40.10: "
        f"{int(inherited_buy.sum())}"
    )

    print(
        "SELL fuori AGE 1-3 già presenti prima di V40.10: "
        f"{int(inherited_sell.sum())}"
    )

    created_buy = (
        buy_outside
        &
        (out["PHASE_DYNAMICS_SHADOW"] != "BUY")
    )

    created_sell = (
        sell_outside
        &
        (out["PHASE_DYNAMICS_SHADOW"] != "SELL")
    )

    print()
    print(
        "BUY fuori AGE 1-3 creati da V40.10: "
        f"{int(created_buy.sum())}"
    )

    print(
        "SELL fuori AGE 1-3 creati da V40.10: "
        f"{int(created_sell.sum())}"
    )


if __name__ == "__main__":
    main()