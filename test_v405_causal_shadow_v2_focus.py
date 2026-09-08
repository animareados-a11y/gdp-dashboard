"""
MarketSentinel
TEST MIRATO - V40.5 CAUSAL SHADOW V2

Ticker:
- FBK.MI
- UCG.MI
- BBVA.MC
- TIT.MI

OBIETTIVO
---------
Verificare che la V2:

1. mantenga Fineco coerente;
2. non crei la trappola DEBOLEZZA vista su UCG;
3. elimini le anomalie BBVA/TIT;
4. non inventi BUY/SELL;
5. non modifichi file di produzione.
"""

from pathlib import Path

import pandas as pd

import weekly_v40_5_indecision_exit_causal_shadow_v2 as shadow


INPUT_FILE = Path(
    "data/v40_35_full200_weekly.csv"
)

TICKERS = [
    "FBK.MI",
    "UCG.MI",
    "BBVA.MC",
    "TIT.MI",
]


def banner(text):
    print("\n" + "=" * 140)
    print(text)
    print("=" * 140)


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Non trovo {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    df = df[
        df["Ticker"].isin(TICKERS)
    ].copy()

    banner(
        "MARKETSENTINEL - TEST V40.5 CAUSAL SHADOW V2"
    )

    print(
        f"\nRighe input: {len(df):,}"
    )

    print(
        f"Ticker trovati: {df['Ticker'].nunique()}"
    )

    results = []

    for ticker in TICKERS:

        g = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if g.empty:
            print(
                f"\nATTENZIONE: {ticker} non trovato."
            )
            continue

        out = shadow.process_ticker(g)

        results.append(out)

        banner(
            f"{ticker} - RIGHE MODIFICATE"
        )

        changed = out[
            out["CAUSAL_SHADOW_CHANGED"] == 1
        ].copy()

        print(
            f"\nRighe modificate: {len(changed):,}"
        )

        cols = [
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
            "BEAR_RECOVERY_RAW",
            "BEAR_RECOVERY_CONFIRMED",
            "PHASE_V405_ORIGINAL",
            "PHASE_CAUSAL_SHADOW",
            "CAUSAL_SHADOW_REASON",
            "CAUSAL_SHADOW_ACTIVE",
            "WEAK_ORIGIN_INDECISION",
        ]

        existing = [
            c for c in cols
            if c in changed.columns
        ]

        if changed.empty:
            print(
                "Nessuna modifica."
            )
        else:
            print(
                changed[existing].to_string(
                    index=False
                )
            )

    if not results:
        raise RuntimeError(
            "Nessun ticker processato."
        )

    full = pd.concat(
        results,
        ignore_index=True,
    )

    # ========================================================
    # FINECO CHECK
    # ========================================================

    banner(
        "FINECO CHECK - 2026"
    )

    fineco = full[
        (full["Ticker"] == "FBK.MI")
        &
        (
            full["Date"]
            >=
            pd.Timestamp(
                "2026-05-01",
                tz="UTC",
            )
        )
    ].copy()

    cols_f = [
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

    print(
        fineco[
            [c for c in cols_f if c in fineco.columns]
        ].tail(20).to_string(
            index=False
        )
    )

    # ========================================================
    # UCG CHECK 2023-2024
    # ========================================================

    banner(
        "UCG CHECK - 2023/2024"
    )

    ucg = full[
        (full["Ticker"] == "UCG.MI")
        &
        (
            full["Date"]
            >=
            pd.Timestamp(
                "2023-11-01",
                tz="UTC",
            )
        )
        &
        (
            full["Date"]
            <=
            pd.Timestamp(
                "2024-06-30",
                tz="UTC",
            )
        )
    ].copy()

    cols_u = [
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

    print(
        ucg[
            [c for c in cols_u if c in ucg.columns]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BBVA / TIT ANOMALY CHECK
    # ========================================================

    banner(
        "CHECK ANOMALIE BBVA / TIT"
    )

    check = full[
        (
            (
                full["Ticker"] == "BBVA.MC"
            )
            &
            (
                full["Date"].between(
                    pd.Timestamp(
                        "2015-12-01",
                        tz="UTC",
                    ),
                    pd.Timestamp(
                        "2016-01-31",
                        tz="UTC",
                    ),
                )
            )
        )
        |
        (
            (
                full["Ticker"] == "TIT.MI"
            )
            &
            (
                full["Date"].between(
                    pd.Timestamp(
                        "2025-08-01",
                        tz="UTC",
                    ),
                    pd.Timestamp(
                        "2025-09-15",
                        tz="UTC",
                    ),
                )
            )
        )
    ].copy()

    cols_c = [
        "Ticker",
        "Date",
        "Close",
        "SAR_SIDE",
        "SAR_AGE",
        "LATERAL_SIGNAL",
        "PHASE_V405_ORIGINAL",
        "PHASE_CAUSAL_SHADOW",
        "CAUSAL_SHADOW_REASON",
        "CAUSAL_SHADOW_ACTIVE",
    ]

    print(
        check[
            [c for c in cols_c if c in check.columns]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BUY / SELL SAFETY
    # ========================================================

    banner(
        "BUY / SELL SAFETY CHECK"
    )

    dangerous = full[
        (
            full["PHASE_CAUSAL_SHADOW"].isin(
                ["BUY", "SELL"]
            )
        )
        &
        (
            full["PHASE_CAUSAL_SHADOW"]
            !=
            full["PHASE_V405_ORIGINAL"]
        )
    ].copy()

    print(
        f"\nBUY/SELL inventati dalla shadow: "
        f"{len(dangerous):,}"
    )

    if not dangerous.empty:

        print(
            dangerous[
                [
                    "Ticker",
                    "Date",
                    "PHASE_V405_ORIGINAL",
                    "PHASE_CAUSAL_SHADOW",
                    "CAUSAL_SHADOW_REASON",
                ]
            ].to_string(
                index=False
            )
        )

    banner(
        "FINE TEST"
    )

    print(
        "\nNESSUN FILE DI PRODUZIONE MODIFICATO."
    )


if __name__ == "__main__":
    main()