"""
MarketSentinel
TEST V40.5 INDECISION EXIT SHADOW
Fineco + UniCredit

Nessun file di produzione viene modificato.
"""

from pathlib import Path

import pandas as pd

import weekly_v40_5_indecision_exit_shadow as shadow


INPUT_FILE = Path(
    "data/v40_35_full200_weekly.csv"
)

TICKERS = [
    "FBK.MI",
    "UCG.MI",
]

START_DATE = "2026-05-01"
END_DATE = "2026-08-28"

OUTPUT_FILE = Path(
    "data/test_v405_indecision_exit_fineco_ucg.csv"
)


def banner(text):
    print("\n" + "=" * 155)
    print(text)
    print("=" * 155)


def main():

    banner(
        "MARKETSENTINEL - TEST INDECISION EXIT SHADOW"
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

    all_rows = []

    for ticker in TICKERS:

        banner(ticker)

        g = (
            raw[
                raw["Ticker"] == ticker
            ]
            .sort_values("Date")
            .reset_index(drop=True)
            .copy()
        )

        if g.empty:
            print(
                f"Nessun dato trovato per {ticker}"
            )
            continue

        out = shadow.process_ticker(g)

        z = out[
            (out["Date"] >= start)
            &
            (out["Date"] <= end)
        ].copy()

        cols = [
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
            "PHASE_SHADOW",
            "SHADOW_CHANGED",
            "SHADOW_REASON",
        ]

        existing = [
            c
            for c in cols
            if c in z.columns
        ]

        print(
            z[existing].to_string(
                index=False,
            )
        )

        keep = z[existing].copy()

        keep.insert(
            0,
            "Ticker",
            ticker,
        )

        all_rows.append(
            keep
        )

    if not all_rows:
        raise RuntimeError(
            "Nessun risultato prodotto."
        )

    result = pd.concat(
        all_rows,
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

    banner(
        "SOLE RIGHE MODIFICATE DALLA SHADOW"
    )

    changed = result[
        result["SHADOW_CHANGED"] == 1
    ].copy()

    if changed.empty:

        print(
            "Nessuna modifica."
        )

    else:

        cols_changed = [
            "Ticker",
            "Date",
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
            "PHASE_SHADOW",
            "SHADOW_REASON",
        ]

        existing_changed = [
            c
            for c in cols_changed
            if c in changed.columns
        ]

        print(
            changed[
                existing_changed
            ].to_string(
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