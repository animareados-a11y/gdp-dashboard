"""
MarketSentinel
VALIDAZIONE PHASE DYNAMICS V3.1 + V40.10 - ITALIA

OBIETTIVO
---------
Applicare la nuova PHASE Dynamics Shadow V3.1
e successivamente la logica V40.10 per ottenere
la PHASE finale, inclusi BUY/SELL nelle prime
tre settimane del nuovo regime qualificato.

Nessun file production viene modificato.

INPUT
-----
data/v40_35_full200_weekly.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow as dynamics
import weekly_v40_10_sar_phase_fix as v4010


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")


# ============================================================
# HELPERS
# ============================================================

def pct(n, d):
    if d == 0:
        return 0.0
    return 100.0 * n / d


# ============================================================
# PROCESS COMPLETE CHAIN
# ============================================================

def process_complete_chain(df):

    # 1. Nuova Dynamics V3.1
    g = dynamics.process_ticker(df.copy())

    # La PHASE risultante della Dynamics diventa
    # la PHASE di ingresso per V40.10
    g["PHASE_BEFORE_V4010"] = (
        g["PHASE_DYNAMICS_SHADOW"].copy()
    )

    g["PHASE"] = (
        g["PHASE_BEFORE_V4010"].copy()
    )

    # 2. Logica V40.10
    g = v4010.add_previous_sar_run_information(g)

    g = v4010.qualify_sar_flips(g)

    g = v4010.propagate_qualified_signal(g)

    g = v4010.apply_v4010_fix(g)

    g["PHASE_FINAL_V4010"] = g["PHASE"].copy()

    return g


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 130)
    print("VALIDAZIONE PHASE DYNAMICS V3.1 + V40.10 - ITALIA")
    print("=" * 130)

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    if "Ticker" not in df.columns:
        raise RuntimeError("Colonna Ticker non presente.")

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    italy_tickers = sorted(
        [
            x
            for x in df["Ticker"].dropna().unique()
            if str(x).endswith(".MI")
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
            z = process_complete_chain(g)

        except Exception as exc:
            print(f"  ERRORE: {exc}")
            continue

        z["Ticker"] = ticker

        all_results.append(z)

    if not all_results:
        raise RuntimeError(
            "Nessun ticker processato."
        )

    out = pd.concat(
        all_results,
        ignore_index=True,
    )

    # ========================================================
    # AGGREGATE
    # ========================================================

    print()
    print("=" * 130)
    print("RISULTATO AGGREGATO")
    print("=" * 130)

    total_rows = len(out)

    dyn_changed = (
        out["PHASE_DYNAMICS_BASE"].astype(str)
        !=
        out["PHASE_DYNAMICS_SHADOW"].astype(str)
    )

    final_changed = (
        out["PHASE_DYNAMICS_BASE"].astype(str)
        !=
        out["PHASE_FINAL_V4010"].astype(str)
    )

    print()
    print(f"Righe totali: {total_rows:,}")

    print(
        f"Modificate dalla Dynamics: "
        f"{int(dyn_changed.sum()):,} "
        f"({pct(int(dyn_changed.sum()), total_rows):.2f}%)"
    )

    print(
        f"Modificate nella PHASE finale: "
        f"{int(final_changed.sum()):,} "
        f"({pct(int(final_changed.sum()), total_rows):.2f}%)"
    )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    print()
    print("=" * 130)
    print("DISTRIBUZIONE PHASE")
    print("=" * 130)

    base_counts = (
        out["PHASE_DYNAMICS_BASE"]
        .value_counts()
    )

    dynamics_counts = (
        out["PHASE_DYNAMICS_SHADOW"]
        .value_counts()
    )

    final_counts = (
        out["PHASE_FINAL_V4010"]
        .value_counts()
    )

    labels = sorted(
        set(base_counts.index)
        |
        set(dynamics_counts.index)
        |
        set(final_counts.index)
    )

    rows = []

    for phase in labels:

        rows.append(
            {
                "PHASE": phase,
                "BASE": int(
                    base_counts.get(phase, 0)
                ),
                "DYNAMICS": int(
                    dynamics_counts.get(phase, 0)
                ),
                "FINAL_V4010": int(
                    final_counts.get(phase, 0)
                ),
            }
        )

    dist = pd.DataFrame(rows)

    print()
    print(
        dist.to_string(
            index=False
        )
    )

    # ========================================================
    # CHANGES DYNAMICS -> FINAL
    # ========================================================

    print()
    print("=" * 130)
    print("CAMBIAMENTI DYNAMICS -> V40.10")
    print("=" * 130)

    mask = (
        out["PHASE_DYNAMICS_SHADOW"].astype(str)
        !=
        out["PHASE_FINAL_V4010"].astype(str)
    )

    changes = (
        out.loc[
            mask,
            [
                "PHASE_DYNAMICS_SHADOW",
                "PHASE_FINAL_V4010",
            ],
        ]
        .value_counts()
        .reset_index(name="Count")
        .sort_values(
            "Count",
            ascending=False,
        )
    )

    print()
    print(
        changes.to_string(
            index=False
        )
    )

    # ========================================================
    # QUALIFIED SAR SAFETY
    # ========================================================

    print()
    print("=" * 130)
    print("CONTROLLO BUY / SELL PRIME 3 SETTIMANE")
    print("=" * 130)

    required_cols = [
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
    ]

    for col in required_cols:
        if col not in out.columns:
            print(
                f"ATTENZIONE: colonna {col} assente."
            )
            continue

    if all(
        col in out.columns
        for col in required_cols
    ):

        bull_qualified = (
            (out["V4010_RUN_QUALIFIED"] == 1)
            &
            (out["SAR_SIDE"] == 1)
            &
            (out["SAR_AGE"].isin([1, 2, 3]))
        )

        bear_qualified = (
            (out["V4010_RUN_QUALIFIED"] == 1)
            &
            (out["SAR_SIDE"] == -1)
            &
            (out["SAR_AGE"].isin([1, 2, 3]))
        )

        bull_not_buy = (
            bull_qualified
            &
            (
                out["PHASE_FINAL_V4010"]
                != "BUY"
            )
        )

        bear_not_sell = (
            bear_qualified
            &
            (
                out["PHASE_FINAL_V4010"]
                != "SELL"
            )
        )

        print()
        print(
            f"Settimane bullish qualificate AGE 1-3: "
            f"{int(bull_qualified.sum())}"
        )
        print(
            f"di cui NON BUY: "
            f"{int(bull_not_buy.sum())}"
        )

        print()
        print(
            f"Settimane bearish qualificate AGE 1-3: "
            f"{int(bear_qualified.sum())}"
        )
        print(
            f"di cui NON SELL: "
            f"{int(bear_not_sell.sum())}"
        )

    # ========================================================
    # BUY / SELL OUTSIDE AGE 1-3
    # ========================================================

    print()
    print("=" * 130)
    print("BUY / SELL FUORI DALLE PRIME 3 SETTIMANE")
    print("=" * 130)

    if all(
        col in out.columns
        for col in required_cols
    ):

        final_buy = (
            out["PHASE_FINAL_V4010"] == "BUY"
        )

        final_sell = (
            out["PHASE_FINAL_V4010"] == "SELL"
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
            final_buy
            &
            (~valid_buy)
        )

        sell_outside = (
            final_sell
            &
            (~valid_sell)
        )

        print()
        print(
            f"BUY finali: {int(final_buy.sum())}"
        )
        print(
            f"BUY fuori da qualified AGE 1-3: "
            f"{int(buy_outside.sum())}"
        )

        print()
        print(
            f"SELL finali: {int(final_sell.sum())}"
        )
        print(
            f"SELL fuori da qualified AGE 1-3: "
            f"{int(sell_outside.sum())}"
        )

    # ========================================================
    # LATEST WEEK
    # ========================================================

    print()
    print("=" * 130)
    print("ULTIMA SETTIMANA PER TICKER")
    print("=" * 130)

    latest_cols = [
        "Ticker",
        "Date",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_DYNAMICS_BASE",
        "PHASE_DYNAMICS_SHADOW",
        "PHASE_FINAL_V4010",
    ]

    latest_cols = [
        c
        for c in latest_cols
        if c in out.columns
    ]

    latest = (
        out.sort_values("Date")
        .groupby("Ticker")
        .tail(1)
        [latest_cols]
        .sort_values("Ticker")
    )

    print()
    print(
        latest.to_string(
            index=False
        )
    )

    # ========================================================
    # ENEL DETAIL
    # ========================================================

    print()
    print("=" * 130)
    print("ENEL - DETTAGLIO MAGGIO / AGOSTO 2026")
    print("=" * 130)

    enel = out[
        (
            out["Ticker"] == "ENEL.MI"
        )
        &
        (
            out["Date"]
            >= pd.Timestamp(
                "2026-04-01",
                tz="UTC",
            )
        )
    ].copy()

    enel_cols = [
        "Date",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_DYNAMICS_BASE",
        "PHASE_DYNAMICS_SHADOW",
        "PHASE_FINAL_V4010",
        "PHASE_DYNAMICS_REASON",
    ]

    enel_cols = [
        c
        for c in enel_cols
        if c in enel.columns
    ]

    print()
    print(
        enel[enel_cols]
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()