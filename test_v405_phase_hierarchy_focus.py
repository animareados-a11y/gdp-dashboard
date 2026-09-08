"""
MarketSentinel
TEST FOCUS - V40.5 PHASE HIERARCHY SHADOW

Testa la nuova gerarchia PHASE su:

    FBK.MI   Fineco
    UCG.MI   UniCredit
    BBVA.MC  BBVA
    TIT.MI   Telecom Italia

OBIETTIVO
---------
Verificare che:

1. BUY / SELL originali NON vengano inventati o modificati.
2. La vera lateralita' resti INDECISIONE.
3. INDECISIONE possa uscire verso:
       TREND_RIALZISTA
       DEBOLEZZA_RIALZISTA
       TREND_RIBASSISTA
       DEBOLEZZA_RIBASSISTA
4. Fineco 2026 venga classificata in modo coerente.
5. Non ricompaiano le anomalie BBVA/TIT viste nelle shadow precedenti.

NESSUN FILE DI PRODUZIONE VIENE MODIFICATO.
"""

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_hierarchy_shadow as shadow


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

TICKERS = [
    "FBK.MI",
    "UCG.MI",
    "BBVA.MC",
    "TIT.MI",
]


def banner(text):
    print("\n" + "=" * 170)
    print(text)
    print("=" * 170)


def flag_series(df, col):
    if col not in df.columns:
        return pd.Series(0, index=df.index)

    return (
        pd.to_numeric(
            df[col],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )


def main():

    banner("CARICAMENTO")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    print(f"Input: {INPUT_FILE}")
    print(f"Righe input: {len(df):,}")

    results = []

    # ============================================================
    # PROCESSO I 4 TICKER
    # ============================================================

    for ticker in TICKERS:

        raw = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if raw.empty:
            print(f"{ticker}: NON TROVATO")
            continue

        g = shadow.process_ticker(raw)

        results.append(g)

        changed = int(
            g["PHASE_HIERARCHY_CHANGED"].sum()
        )

        print(
            f"{ticker}: "
            f"{len(g):,} righe - "
            f"cambiate {changed:,}"
        )

    all_df = pd.concat(
        results,
        ignore_index=True,
    )

    # ============================================================
    # TRANSIZIONI MODIFICATE
    # ============================================================

    banner("TRANSIZIONI MODIFICATE")

    changed = all_df[
        all_df["PHASE_HIERARCHY_CHANGED"] == 1
    ].copy()

    if changed.empty:

        print("Nessuna modifica.")

    else:

        transitions = (
            changed.groupby(
                [
                    "PHASE_V405_ORIGINAL",
                    "PHASE_HIERARCHY_SHADOW",
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        print(transitions.to_string())

    # ============================================================
    # MOTIVI DELLE MODIFICHE
    # ============================================================

    banner("MOTIVI DELLE MODIFICHE")

    if changed.empty:

        print("Nessuna modifica.")

    else:

        print(
            changed[
                "PHASE_HIERARCHY_REASON"
            ]
            .value_counts()
            .to_string()
        )

    # ============================================================
    # SAFETY BUY / SELL
    # ============================================================

    banner("SAFETY CHECK BUY / SELL")

    invented_buy_sell = all_df[
        (
            all_df[
                "PHASE_HIERARCHY_SHADOW"
            ].isin(["BUY", "SELL"])
        )
        &
        (
            all_df[
                "PHASE_HIERARCHY_SHADOW"
            ]
            !=
            all_df[
                "PHASE_V405_ORIGINAL"
            ]
        )
    ]

    altered_original_buy_sell = all_df[
        (
            all_df[
                "PHASE_V405_ORIGINAL"
            ].isin(["BUY", "SELL"])
        )
        &
        (
            all_df[
                "PHASE_HIERARCHY_SHADOW"
            ]
            !=
            all_df[
                "PHASE_V405_ORIGINAL"
            ]
        )
    ]

    print(
        "BUY/SELL inventati dalla shadow:",
        len(invented_buy_sell),
    )

    print(
        "BUY/SELL originali modificati:",
        len(altered_original_buy_sell),
    )

    # ============================================================
    # CHECK:
    # TREND ORIGINALE NON DEVE ESSERE TRASFORMATO IN INDECISIONE
    # ============================================================

    banner(
        "SAFETY CHECK TREND ORIGINALE -> INDECISIONE"
    )

    trend_to_ind = all_df[
        (
            all_df[
                "PHASE_V405_ORIGINAL"
            ].isin(
                [
                    "TREND_RIALZISTA",
                    "TREND_RIBASSISTA",
                ]
            )
        )
        &
        (
            all_df[
                "PHASE_HIERARCHY_SHADOW"
            ]
            == "INDECISIONE"
        )
    ]

    print(
        "TREND originali trasformati in INDECISIONE:",
        len(trend_to_ind),
    )

    if not trend_to_ind.empty:

        print(
            trend_to_ind[
                [
                    "Ticker",
                    "Date",
                    "Close",
                    "PHASE_V405_ORIGINAL",
                    "PHASE_HIERARCHY_SHADOW",
                    "PHASE_HIERARCHY_REASON",
                ]
            ]
            .to_string(index=False)
        )

    # ============================================================
    # COLONNE DI DETTAGLIO
    # ============================================================

    detail_cols = [
        "Ticker",
        "Date",
        "Close",
        "SAR_SIDE",
        "SAR_AGE",
        "PHASE_V405_ORIGINAL",
        "PHASE_HIERARCHY_SHADOW",
        "LATERAL_SIGNAL",
        "WEAK_UP_TRIGGER",
        "WEAK_DOWN_TRIGGER",
        "BULL_RECOVERY_RAW",
        "BEAR_RECOVERY_RAW",
        "BULL_RECOVERY_CONFIRMED",
        "BEAR_RECOVERY_CONFIRMED",
        "PHASE_HIERARCHY_REASON",
    ]

    detail_cols = [
        c
        for c in detail_cols
        if c in all_df.columns
    ]

    # ============================================================
    # FINECO 2026
    # ============================================================

    banner("FINECO 2026")

    fineco = all_df[
        (all_df["Ticker"] == "FBK.MI")
        &
        (
            all_df["Date"]
            >= pd.Timestamp(
                "2026-05-01",
                tz="UTC",
            )
        )
    ]

    print(
        fineco[
            detail_cols
        ]
        .to_string(index=False)
    )

    # ============================================================
    # UCG 2023-2024
    # ============================================================

    banner("UNICREDIT 2023-11 -> 2024-06")

    ucg = all_df[
        (all_df["Ticker"] == "UCG.MI")
        &
        (
            all_df["Date"]
            >= pd.Timestamp(
                "2023-11-01",
                tz="UTC",
            )
        )
        &
        (
            all_df["Date"]
            <= pd.Timestamp(
                "2024-06-30",
                tz="UTC",
            )
        )
    ]

    print(
        ucg[
            detail_cols
        ]
        .to_string(index=False)
    )

    # ============================================================
    # BBVA - ANOMALIA PRECEDENTE
    # ============================================================

    banner("BBVA 2015-12 -> 2016-02")

    bbva = all_df[
        (all_df["Ticker"] == "BBVA.MC")
        &
        (
            all_df["Date"]
            >= pd.Timestamp(
                "2015-12-01",
                tz="UTC",
            )
        )
        &
        (
            all_df["Date"]
            <= pd.Timestamp(
                "2016-02-29",
                tz="UTC",
            )
        )
    ]

    print(
        bbva[
            detail_cols
        ]
        .to_string(index=False)
    )

    # ============================================================
    # TIT - ANOMALIA PRECEDENTE
    # ============================================================

    banner("TELECOM ITALIA 2025-08 -> 2025-09")

    tit = all_df[
        (all_df["Ticker"] == "TIT.MI")
        &
        (
            all_df["Date"]
            >= pd.Timestamp(
                "2025-08-01",
                tz="UTC",
            )
        )
        &
        (
            all_df["Date"]
            <= pd.Timestamp(
                "2025-09-30",
                tz="UTC",
            )
        )
    ]

    print(
        tit[
            detail_cols
        ]
        .to_string(index=False)
    )

    # ============================================================
    # LATERALITA' CHE RESTA LATERALITA'
    # ============================================================

    banner(
        "LATERAL_SIGNAL ATTIVO - DEVE RESTARE INDECISIONE"
    )

    lateral = flag_series(
        all_df,
        "LATERAL_SIGNAL",
    )

    lateral_cases = all_df[
        (
            all_df[
                "PHASE_V405_ORIGINAL"
            ]
            == "INDECISIONE"
        )
        &
        (lateral == 1)
    ]

    wrong_lateral = lateral_cases[
        lateral_cases[
            "PHASE_HIERARCHY_SHADOW"
        ]
        != "INDECISIONE"
    ]

    print(
        "Righe originali INDECISIONE con "
        "LATERAL_SIGNAL=1:",
        len(lateral_cases),
    )

    print(
        "Di queste trasformate erroneamente "
        "in fase direzionale:",
        len(wrong_lateral),
    )

    # ============================================================
    # DISTRIBUZIONE FINALE
    # ============================================================

    banner("DISTRIBUZIONE PHASE")

    comparison = pd.DataFrame(
        {
            "V405_ORIGINAL":
                all_df[
                    "PHASE_V405_ORIGINAL"
                ]
                .value_counts(),

            "HIERARCHY_SHADOW":
                all_df[
                    "PHASE_HIERARCHY_SHADOW"
                ]
                .value_counts(),
        }
    ).fillna(0).astype(int)

    print(comparison.to_string())

    banner("FINE TEST")

    print(
        "Nessun file di produzione modificato."
    )


if __name__ == "__main__":
    main()