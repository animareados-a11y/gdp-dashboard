"""
MarketSentinel
TEST V40.5 CAUSAL INDECISION EXIT SHADOW
Fineco + UniCredit

OBIETTIVO
---------
Confrontare settimana per settimana:

- PHASE originale V40.5
- PHASE nuova causal shadow
- SAR
- weakness
- lateralità
- recovery

Focus:
FBK.MI
UCG.MI

Nessun file di produzione viene modificato.
"""

from pathlib import Path

import pandas as pd

import weekly_v40_5_indecision_exit_causal_shadow as shadow


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/v40_35_full200_weekly.csv"
)

OUTPUT_FILE = Path(
    "data/test_v405_causal_shadow_fineco_ucg.csv"
)

TICKERS = [
    "FBK.MI",
    "UCG.MI",
]

START_DATE = "2026-03-06"
END_DATE = "2026-08-28"


# ============================================================
# HELPERS
# ============================================================

def banner(text):

    print("\n" + "=" * 175)
    print(text)
    print("=" * 175)


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "MARKETSENTINEL - TEST V40.5 CAUSAL SHADOW"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Non trovo {INPUT_FILE}"
        )

    raw = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    raw["Date"] = pd.to_datetime(
        raw["Date"],
        utc=True,
        errors="coerce",
    )

    start = pd.Timestamp(
        START_DATE,
        tz="UTC",
    )

    end = pd.Timestamp(
        END_DATE,
        tz="UTC",
    )

    all_results = []

    # ========================================================
    # TICKER
    # ========================================================

    for ticker in TICKERS:

        banner(ticker)

        g = (
            raw[
                raw["Ticker"] == ticker
            ]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if g.empty:

            print(
                f"Nessun dato per {ticker}"
            )

            continue

        out = shadow.process_ticker(g)

        z = out[
            (out["Date"] >= start)
            &
            (out["Date"] <= end)
        ].copy()

        # ----------------------------------------------------
        # Colonne principali
        # ----------------------------------------------------

        columns = [
            "Date",
            "Close",

            "SAR_SIDE",
            "SAR_AGE",

            "WEAK_UP_TRIGGER",
            "WEAK_DOWN_TRIGGER",

            "LATERAL_SIGNAL",

            "BULL_RECOVERY_RAW",
            "BULL_RECOVERY_CONFIRMED",

            "BEAR_RECOVERY_RAW",
            "BEAR_RECOVERY_CONFIRMED",

            "PHASE_V405_ORIGINAL",
            "PHASE_CAUSAL_SHADOW",

            "CAUSAL_SHADOW_CHANGED",
            "CAUSAL_SHADOW_ACTIVE",

            "WEAK_ORIGIN_INDECISION",

            "CAUSAL_SHADOW_REASON",
        ]

        existing = [
            c
            for c in columns
            if c in z.columns
        ]

        print(
            z[existing].to_string(
                index=False,
            )
        )

        save = z[existing].copy()

        save.insert(
            0,
            "Ticker",
            ticker,
        )

        all_results.append(save)

    # ========================================================
    # COMBINA
    # ========================================================

    if not all_results:

        raise RuntimeError(
            "Nessun risultato prodotto."
        )

    result = pd.concat(
        all_results,
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
    # SOLE DIFFERENZE
    # ========================================================

    banner(
        "SOLE RIGHE MODIFICATE DALLA CAUSAL SHADOW"
    )

    changed = result[
        result["CAUSAL_SHADOW_CHANGED"] == 1
    ].copy()

    if changed.empty:

        print(
            "Nessuna differenza."
        )

    else:

        changed_columns = [
            "Ticker",
            "Date",
            "Close",

            "SAR_SIDE",
            "SAR_AGE",

            "WEAK_UP_TRIGGER",
            "WEAK_DOWN_TRIGGER",

            "LATERAL_SIGNAL",

            "BULL_RECOVERY_RAW",
            "BULL_RECOVERY_CONFIRMED",

            "PHASE_V405_ORIGINAL",
            "PHASE_CAUSAL_SHADOW",

            "WEAK_ORIGIN_INDECISION",

            "CAUSAL_SHADOW_REASON",
        ]

        existing_changed = [
            c
            for c in changed_columns
            if c in changed.columns
        ]

        print(
            changed[
                existing_changed
            ].to_string(
                index=False,
            )
        )

    # ========================================================
    # FINECO FOCUS
    # ========================================================

    banner(
        "FINECO - FOCUS 2026-05-15 -> 2026-08-28"
    )

    fineco = result[
        (
            result["Ticker"]
            ==
            "FBK.MI"
        )
        &
        (
            result["Date"]
            >=
            pd.Timestamp(
                "2026-05-15",
                tz="UTC",
            )
        )
    ].copy()

    focus_columns = [
        "Date",
        "Close",
        "SAR_SIDE",
        "SAR_AGE",
        "WEAK_UP_TRIGGER",
        "LATERAL_SIGNAL",
        "BULL_RECOVERY_RAW",
        "BULL_RECOVERY_CONFIRMED",
        "PHASE_V405_ORIGINAL",
        "PHASE_CAUSAL_SHADOW",
        "CAUSAL_SHADOW_REASON",
    ]

    existing_focus = [
        c
        for c in focus_columns
        if c in fineco.columns
    ]

    print(
        fineco[
            existing_focus
        ].to_string(
            index=False,
        )
    )

    # ========================================================
    # UCG FOCUS
    # ========================================================

    banner(
        "UCG - FOCUS 2026-07-03 -> 2026-08-28"
    )

    ucg = result[
        (
            result["Ticker"]
            ==
            "UCG.MI"
        )
        &
        (
            result["Date"]
            >=
            pd.Timestamp(
                "2026-07-03",
                tz="UTC",
            )
        )
    ].copy()

    print(
        ucg[
            existing_focus
        ].to_string(
            index=False,
        )
    )

    # ========================================================
    # CONTEGGIO
    # ========================================================

    banner(
        "RIEPILOGO"
    )

    summary = (
        result
        .groupby("Ticker")
        .agg(
            rows=(
                "Date",
                "size",
            ),
            changed_rows=(
                "CAUSAL_SHADOW_CHANGED",
                "sum",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
        )
    )

    banner(
        "TEST COMPLETATO"
    )

    print(
        f"\nSalvato: {OUTPUT_FILE}"
    )

    print(
        "\nNESSUN FILE DI PRODUZIONE MODIFICATO."
    )


if __name__ == "__main__":
    main()