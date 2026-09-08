"""
MarketSentinel
VALIDAZIONE FULL200 - PHASE HIERARCHY SHADOW

OBIETTIVO
---------
Validare sui 200 titoli la nuova gerarchia PHASE:

    BUY
    TREND_RIALZISTA
    DEBOLEZZA_RIALZISTA
    SELL
    TREND_RIBASSISTA
    DEBOLEZZA_RIBASSISTA
    INDECISIONE / LATERALIZZAZIONE residuale

La shadow NON modifica produzione.
NON modifica V40.5 originale.
NON modifica V40.10.
NON inventa BUY / SELL.
NON usa rendimenti futuri.
NON introduce nuove soglie numeriche.

OUTPUT
------
data/validazione_full200_phase_hierarchy_shadow_changed.csv
data/validazione_full200_phase_hierarchy_shadow_latest.csv
data/validazione_full200_phase_hierarchy_shadow_summary.csv
"""

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_hierarchy_shadow as shadow


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

OUT_CHANGED = Path(
    "data/validazione_full200_phase_hierarchy_shadow_changed.csv"
)

OUT_LATEST = Path(
    "data/validazione_full200_phase_hierarchy_shadow_latest.csv"
)

OUT_SUMMARY = Path(
    "data/validazione_full200_phase_hierarchy_shadow_summary.csv"
)


def banner(text):
    print("\n" + "=" * 170)
    print(text)
    print("=" * 170)


def main():

    banner("CARICAMENTO FULL200")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    tickers = sorted(
        df["Ticker"]
        .dropna()
        .astype(str)
        .unique()
    )

    print(f"Input: {INPUT_FILE}")
    print(f"Righe input: {len(df):,}")
    print(f"Ticker: {len(tickers)}")

    results = []

    # ============================================================
    # PROCESSO FULL200
    # ============================================================

    for i, ticker in enumerate(
        tickers,
        start=1,
    ):

        raw = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        try:

            g = shadow.process_ticker(raw)

            results.append(g)

            if (
                i == 1
                or i % 10 == 0
                or i == len(tickers)
            ):
                print(
                    f"[{i:03d}/{len(tickers):03d}] "
                    f"{ticker} OK"
                )

        except Exception as exc:

            print(
                f"[{i:03d}/{len(tickers):03d}] "
                f"{ticker} ERRORE: {exc}"
            )

            raise

    all_df = pd.concat(
        results,
        ignore_index=True,
    )

    banner("RISULTATO GENERALE")

    total_rows = len(all_df)

    changed = all_df[
        all_df[
            "PHASE_HIERARCHY_CHANGED"
        ] == 1
    ].copy()

    changed_rows = len(changed)

    changed_pct = (
        changed_rows / total_rows * 100
        if total_rows
        else 0
    )

    changed_tickers = (
        changed["Ticker"]
        .nunique()
        if not changed.empty
        else 0
    )

    print(
        f"Righe totali: {total_rows:,}"
    )

    print(
        f"Righe cambiate: "
        f"{changed_rows:,} "
        f"({changed_pct:.2f}%)"
    )

    print(
        f"Ticker con almeno una modifica: "
        f"{changed_tickers}/{len(tickers)}"
    )

    # ============================================================
    # TRANSIZIONI
    # ============================================================

    banner("TRANSIZIONI MODIFICATE")

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

        print(
            transitions.to_string()
        )

    # ============================================================
    # MOTIVI
    # ============================================================

    banner("MOTIVI DELLE MODIFICHE")

    if changed.empty:

        print("Nessuna modifica.")

    else:

        reasons = (
            changed[
                "PHASE_HIERARCHY_REASON"
            ]
            .value_counts()
        )

        print(
            reasons.to_string()
        )

    # ============================================================
    # SAFETY BUY / SELL
    # ============================================================

    banner("SAFETY BUY / SELL")

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
        "BUY/SELL inventati:",
        len(invented_buy_sell),
    )

    print(
        "BUY/SELL originali modificati:",
        len(altered_original_buy_sell),
    )

    # ============================================================
    # SAFETY TREND -> INDECISIONE
    # ============================================================

    banner(
        "SAFETY TREND ORIGINALE -> INDECISIONE"
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
        "TREND originali trasformati "
        "in INDECISIONE:",
        len(trend_to_ind),
    )

    # ============================================================
    # SAFETY LATERAL_SIGNAL
    # ============================================================

    banner(
        "SAFETY VERA LATERALITA"
    )

    lateral_signal = pd.to_numeric(
        all_df.get(
            "LATERAL_SIGNAL",
            0,
        ),
        errors="coerce",
    ).fillna(0)

    original_ind_lateral = all_df[
        (
            all_df[
                "PHASE_V405_ORIGINAL"
            ]
            == "INDECISIONE"
        )
        &
        (
            lateral_signal == 1
        )
    ]

    lateral_wrong = original_ind_lateral[
        original_ind_lateral[
            "PHASE_HIERARCHY_SHADOW"
        ]
        != "INDECISIONE"
    ]

    print(
        "INDECISIONE originali "
        "con LATERAL_SIGNAL=1:",
        len(original_ind_lateral),
    )

    print(
        "Trasformate erroneamente "
        "in fase direzionale:",
        len(lateral_wrong),
    )

    # ============================================================
    # DISTRIBUZIONE PHASE
    # ============================================================

    banner("DISTRIBUZIONE PHASE FULL200")

    distribution = pd.DataFrame(
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

    print(
        distribution.to_string()
    )

    # ============================================================
    # MODIFICHE PER TICKER
    # ============================================================

    banner("TOP 30 TICKER PER NUMERO MODIFICHE")

    if changed.empty:

        print("Nessuna modifica.")

    else:

        top_tickers = (
            changed.groupby("Ticker")
            .size()
            .sort_values(
                ascending=False
            )
            .head(30)
        )

        print(
            top_tickers.to_string()
        )

    # ============================================================
    # ULTIMA DATA DI OGNI TICKER
    # ============================================================

    banner("ULTIMA SETTIMANA - CONFRONTO")

    latest = (
        all_df
        .sort_values("Date")
        .groupby(
            "Ticker",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    latest_changed = latest[
        latest[
            "PHASE_V405_ORIGINAL"
        ]
        !=
        latest[
            "PHASE_HIERARCHY_SHADOW"
        ]
    ].copy()

    print(
        f"Ticker differenti "
        f"all'ultima settimana: "
        f"{len(latest_changed)}/{len(latest)}"
    )

    latest_cols = [
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
        "PHASE_HIERARCHY_REASON",
    ]

    latest_cols = [
        c
        for c in latest_cols
        if c in latest.columns
    ]

    if not latest_changed.empty:

        print(
            latest_changed[
                latest_cols
            ]
            .sort_values("Ticker")
            .to_string(index=False)
        )

    # ============================================================
    # FINECO ULTIME SETTIMANE
    # ============================================================

    banner("FINECO - 2026")

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

    if not fineco.empty:

        print(
            fineco[
                latest_cols
            ]
            .to_string(index=False)
        )

    # ============================================================
    # SALVATAGGIO OUTPUT
    # ============================================================

    banner("SALVATAGGIO")

    changed.to_csv(
        OUT_CHANGED,
        index=False,
    )

    latest.to_csv(
        OUT_LATEST,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "total_rows":
                    total_rows,

                "changed_rows":
                    changed_rows,

                "changed_pct":
                    changed_pct,

                "changed_tickers":
                    changed_tickers,

                "total_tickers":
                    len(tickers),

                "invented_buy_sell":
                    len(
                        invented_buy_sell
                    ),

                "altered_original_buy_sell":
                    len(
                        altered_original_buy_sell
                    ),

                "trend_to_indecision":
                    len(
                        trend_to_ind
                    ),

                "lateral_signal_rows":
                    len(
                        original_ind_lateral
                    ),

                "lateral_wrong":
                    len(
                        lateral_wrong
                    ),

                "latest_changed":
                    len(
                        latest_changed
                    ),
            }
        ]
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    print(f"Salvato: {OUT_CHANGED}")
    print(f"Salvato: {OUT_LATEST}")
    print(f"Salvato: {OUT_SUMMARY}")

    banner("FINE VALIDAZIONE")

    print(
        "Nessun file di produzione modificato."
    )


if __name__ == "__main__":
    main()