"""
MarketSentinel
VALIDAZIONE FULL200
PHASE HIERARCHY SHADOW -> V40.6 -> V40.7 -> V40.9 -> V40.10

OBIETTIVO
---------
Validare la nuova gerarchia PHASE attraverso tutta la catena finale:

    V40.5 originale
        ↓
    PHASE HIERARCHY SHADOW
        ↓
    V40.6 TRANSITION
        ↓
    V40.7 WEAKNESS SEVERITY
        ↓
    V40.9 PHASE CONFIRMATION
        ↓
    V40.10 SAR PHASE FIX

IMPORTANTE
----------
- NON modifica file di produzione.
- NON modifica engine.py.
- NON modifica V40.5 / V40.6 / V40.7 / V40.9 / V40.10.
- NON ricalcola Reversal.
- NON ricalcola Strength.
- NON introduce nuove soglie.

La differenza fondamentale rispetto a chiamare direttamente
weekly_v40_10_sar_phase_fix.process_ticker() è che qui
NON viene richiamata nuovamente la V40.5 originale dopo
l'applicazione della PHASE HIERARCHY SHADOW.

La catena viene costruita manualmente usando le funzioni
pubbliche già esistenti nei moduli di produzione.
"""

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_hierarchy_shadow as hierarchy
import weekly_v40_6_transition_engine as v406
import weekly_v40_7_weakness_severity_engine as v407
import weekly_v40_9_phase_confirmation_fix as v409
import weekly_v40_10_sar_phase_fix as v4010


# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = Path(
    "data/v40_35_full200_weekly.csv"
)

OUTPUT_DIR = Path("data")

OUT_CHANGED = OUTPUT_DIR / (
    "validazione_full200_phase_hierarchy_chain_v4010_changed.csv"
)

OUT_LATEST = OUTPUT_DIR / (
    "validazione_full200_phase_hierarchy_chain_v4010_latest.csv"
)

OUT_SUMMARY = OUTPUT_DIR / (
    "validazione_full200_phase_hierarchy_chain_v4010_summary.csv"
)


PHASES = [
    "BUY",
    "TREND_RIALZISTA",
    "DEBOLEZZA_RIALZISTA",
    "SELL",
    "TREND_RIBASSISTA",
    "DEBOLEZZA_RIBASSISTA",
    "INDECISIONE",
]


def banner(text):
    print("\n" + "=" * 150)
    print(text)
    print("=" * 150)


# =============================================================================
# CATENA SHADOW COMPLETA
# =============================================================================

def process_ticker_shadow_chain(raw):

    # =========================================================================
    # 1. V40.5 + PHASE HIERARCHY SHADOW
    # =========================================================================

    g = hierarchy.process_ticker(
        raw.copy()
    )

    # Salviamo la PHASE prodotta dalla hierarchy.
    g["PHASE_AFTER_HIERARCHY"] = (
        g["PHASE_HIERARCHY_SHADOW"]
        .copy()
    )

    # La rendiamo PHASE corrente per gli stadi successivi.
    g["PHASE"] = (
        g["PHASE_AFTER_HIERARCHY"]
        .copy()
    )

    # =========================================================================
    # 2. V40.6
    #
    # NON chiamiamo v406.process_ticker(), perché richiamerebbe V40.5 originale.
    # =========================================================================

    g = v406.add_middle_band_structure(g)
    g = v406.add_ha_transition(g)
    g = v406.add_rsi_transition(g)
    g = v406.add_macd_transition(g)
    g = v406.add_volume_transition(g)
    g = v406.add_transition_pressure(g)
    g = v406.add_transition_confirmation(g)

    g["PHASE_AFTER_V406"] = (
        g["PHASE"]
        .copy()
    )

    # =========================================================================
    # 3. V40.7
    #
    # NON chiamiamo v407.process_ticker(), perché richiamerebbe V40.6 e V40.5.
    # =========================================================================

    g = v407.add_weakness_components(g)
    g = v407.add_weakness_scores(g)
    g = v407.add_weakness_level(g)
    g = v407.add_weakness_labels(g)
    g = v407.add_pre_signal_history(g)

    g["PHASE_AFTER_V407"] = (
        g["PHASE"]
        .copy()
    )

    # =========================================================================
    # 4. V40.9
    #
    # Aggiunge le feature di conferma e applica il rescue PHASE.
    # =========================================================================

    g = v409.add_phase_confirmation_features(g)

    g = v409.apply_v409_phase_fix(g)

    g["PHASE_AFTER_V409"] = (
        g["PHASE"]
        .copy()
    )

    # =========================================================================
    # 5. V40.10
    #
    # Ordine identico al process_ticker originale V40.10.
    # =========================================================================

    g = v4010.add_previous_sar_run_information(g)

    g = v4010.qualify_sar_flips(g)

    g = v4010.propagate_qualified_signal(g)

    g = v4010.apply_v4010_fix(g)

    g["PHASE_FINAL_SHADOW"] = (
        g["PHASE"]
        .copy()
    )

    return g


# =============================================================================
# MAIN
# =============================================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Non trovo {INPUT_FILE}"
        )

    banner(
        "CARICAMENTO FULL200"
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

    df = (
        df
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    tickers = (
        df["Ticker"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Input: {INPUT_FILE}"
    )

    print(
        f"Righe input: {len(df):,}"
    )

    print(
        f"Ticker: {len(tickers)}"
    )

    # =========================================================================
    # PROCESSAMENTO
    # =========================================================================

    results = []

    for idx, ticker in enumerate(
        tickers,
        start=1,
    ):

        raw = (
            df[
                df["Ticker"] == ticker
            ]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        out = process_ticker_shadow_chain(
            raw
        )

        results.append(
            out
        )

        if (
            idx == 1
            or idx % 10 == 0
            or idx == len(tickers)
        ):
            print(
                f"[{idx:03d}/{len(tickers):03d}] "
                f"{ticker} OK"
            )

    full = pd.concat(
        results,
        ignore_index=True,
    )

    # =========================================================================
    # DIFFERENZA HIERARCHY -> FINAL
    # =========================================================================

    full[
        "FINAL_CHANGED_FROM_HIERARCHY"
    ] = (
        full["PHASE_FINAL_SHADOW"]
        !=
        full["PHASE_AFTER_HIERARCHY"]
    ).astype(int)

    # =========================================================================
    # DIFFERENZA V40.5 ORIGINALE -> FINAL
    # =========================================================================

    full[
        "FINAL_CHANGED_FROM_V405"
    ] = (
        full["PHASE_FINAL_SHADOW"]
        !=
        full["PHASE_V405_ORIGINAL"]
    ).astype(int)

    # =========================================================================
    # RISULTATO GENERALE
    # =========================================================================

    banner(
        "RISULTATO GENERALE"
    )

    total_rows = len(full)

    changed_from_v405 = int(
        full[
            "FINAL_CHANGED_FROM_V405"
        ].sum()
    )

    changed_from_hierarchy = int(
        full[
            "FINAL_CHANGED_FROM_HIERARCHY"
        ].sum()
    )

    ticker_changed = (
        full.loc[
            full[
                "FINAL_CHANGED_FROM_V405"
            ] == 1,
            "Ticker",
        ]
        .nunique()
    )

    print(
        f"Righe totali: {total_rows:,}"
    )

    print(
        "Finali diverse da V40.5 originale: "
        f"{changed_from_v405:,} "
        f"({changed_from_v405 / total_rows * 100:.2f}%)"
    )

    print(
        "Finali diverse dalla Hierarchy pre-V40.10: "
        f"{changed_from_hierarchy:,} "
        f"({changed_from_hierarchy / total_rows * 100:.2f}%)"
    )

    print(
        "Ticker con almeno una modifica vs V40.5: "
        f"{ticker_changed}/{len(tickers)}"
    )

    # =========================================================================
    # TRANSIZIONI HIERARCHY -> V409
    # =========================================================================

    banner(
        "TRANSIZIONI HIERARCHY -> V40.9"
    )

    v409_changes = full[
        full["PHASE_AFTER_V409"]
        !=
        full["PHASE_AFTER_HIERARCHY"]
    ].copy()

    if v409_changes.empty:
        print(
            "Nessuna modifica V40.9 rispetto alla Hierarchy."
        )
    else:
        print(
            v409_changes
            .groupby(
                [
                    "PHASE_AFTER_HIERARCHY",
                    "PHASE_AFTER_V409",
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
            .to_string()
        )

    # =========================================================================
    # TRANSIZIONI V409 -> V4010
    # =========================================================================

    banner(
        "TRANSIZIONI V40.9 -> V40.10"
    )

    v4010_changes = full[
        full["PHASE_FINAL_SHADOW"]
        !=
        full["PHASE_AFTER_V409"]
    ].copy()

    if v4010_changes.empty:
        print(
            "Nessuna modifica V40.10."
        )
    else:
        print(
            v4010_changes
            .groupby(
                [
                    "PHASE_AFTER_V409",
                    "PHASE_FINAL_SHADOW",
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
            .to_string()
        )

    # =========================================================================
    # SAFETY V40.10
    # =========================================================================

    banner(
        "SAFETY V40.10 - PRIME 3 SETTIMANE"
    )

    qualified_first3 = full[
        (full["V4010_RUN_QUALIFIED"] == 1)
        &
        (full["SAR_AGE"].between(1, 3))
        &
        (full["SAR_SIDE"].isin([1, -1]))
    ].copy()

    wrong_buy = qualified_first3[
        (qualified_first3["SAR_SIDE"] == 1)
        &
        (
            qualified_first3[
                "PHASE_FINAL_SHADOW"
            ]
            != "BUY"
        )
    ]

    wrong_sell = qualified_first3[
        (qualified_first3["SAR_SIDE"] == -1)
        &
        (
            qualified_first3[
                "PHASE_FINAL_SHADOW"
            ]
            != "SELL"
        )
    ]

    print(
        "Run qualificati, settimane 1-3: "
        f"{len(qualified_first3):,}"
    )

    print(
        "Bull qualificati 1-3 NON BUY: "
        f"{len(wrong_buy):,}"
    )

    print(
        "Bear qualificati 1-3 NON SELL: "
        f"{len(wrong_sell):,}"
    )

    # =========================================================================
    # BUY / SELL FUORI PRIME 3 SETTIMANE QUALIFICATE
    # =========================================================================

    banner(
        "CONTROLLO BUY / SELL FINALI"
    )

    final_buy = full[
        full["PHASE_FINAL_SHADOW"]
        == "BUY"
    ].copy()

    final_sell = full[
        full["PHASE_FINAL_SHADOW"]
        == "SELL"
    ].copy()

    buy_outside = final_buy[
        ~(
            (final_buy["V4010_RUN_QUALIFIED"] == 1)
            &
            (final_buy["SAR_SIDE"] == 1)
            &
            (final_buy["SAR_AGE"].between(1, 3))
        )
    ]

    sell_outside = final_sell[
        ~(
            (final_sell["V4010_RUN_QUALIFIED"] == 1)
            &
            (final_sell["SAR_SIDE"] == -1)
            &
            (final_sell["SAR_AGE"].between(1, 3))
        )
    ]

    print(
        f"BUY finali: {len(final_buy):,}"
    )

    print(
        "BUY fuori da run qualificato AGE 1-3: "
        f"{len(buy_outside):,}"
    )

    print(
        f"SELL finali: {len(final_sell):,}"
    )

    print(
        "SELL fuori da run qualificato AGE 1-3: "
        f"{len(sell_outside):,}"
    )

    # =========================================================================
    # SAFETY LATERALITA
    # =========================================================================

    banner(
        "SAFETY LATERALITA ESPLICITA"
    )

    lateral = full[
        (
            full[
                "PHASE_V405_ORIGINAL"
            ]
            == "INDECISIONE"
        )
        &
        (
            full[
                "LATERAL_SIGNAL"
            ]
            == 1
        )
    ].copy()

    lateral_final_directional = lateral[
        lateral[
            "PHASE_FINAL_SHADOW"
        ]
        != "INDECISIONE"
    ].copy()

    print(
        "V40.5 INDECISIONE con LATERAL_SIGNAL=1: "
        f"{len(lateral):,}"
    )

    print(
        "Diventate direzionali nella PHASE finale: "
        f"{len(lateral_final_directional):,}"
    )

    # =========================================================================
    # DISTRIBUZIONE
    # =========================================================================

    banner(
        "DISTRIBUZIONE PHASE FULL200"
    )

    distribution = pd.DataFrame(
        {
            "V405_ORIGINAL": (
                full[
                    "PHASE_V405_ORIGINAL"
                ]
                .value_counts()
            ),
            "HIERARCHY": (
                full[
                    "PHASE_AFTER_HIERARCHY"
                ]
                .value_counts()
            ),
            "AFTER_V409": (
                full[
                    "PHASE_AFTER_V409"
                ]
                .value_counts()
            ),
            "FINAL_V4010": (
                full[
                    "PHASE_FINAL_SHADOW"
                ]
                .value_counts()
            ),
        }
    ).fillna(0).astype(int)

    distribution = distribution.reindex(
        PHASES
    ).fillna(0).astype(int)

    print(
        distribution.to_string()
    )

    # =========================================================================
    # ULTIMA SETTIMANA
    # =========================================================================

    banner(
        "ULTIMA SETTIMANA - PHASE FINALE"
    )

    latest = (
        full
        .sort_values(
            ["Ticker", "Date"]
        )
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
            "PHASE_FINAL_SHADOW"
        ]
    ].copy()

    print(
        "Ticker differenti all'ultima settimana: "
        f"{len(latest_changed)}/{len(latest)}"
    )

    latest_cols = [
        "Ticker",
        "Date",
        "Close",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_V405_ORIGINAL",
        "PHASE_AFTER_HIERARCHY",
        "PHASE_AFTER_V409",
        "PHASE_FINAL_SHADOW",
        "LATERAL_SIGNAL",
        "WEAK_UP_TRIGGER",
        "WEAK_DOWN_TRIGGER",
        "BULL_RECOVERY_RAW",
        "BEAR_RECOVERY_RAW",
        "V409_PHASE_CORRECTED",
        "V4010_PHASE_CORRECTED",
    ]

    latest_existing = [
        c
        for c in latest_cols
        if c in latest_changed.columns
    ]

    if latest_changed.empty:
        print(
            "Nessuna differenza."
        )
    else:
        print(
            latest_changed[
                latest_existing
            ]
            .sort_values("Ticker")
            .to_string(
                index=False
            )
        )

    # =========================================================================
    # FINECO 2026
    # =========================================================================

    banner(
        "FINECO - 2026 - CATENA COMPLETA"
    )

    fbk = full[
        (full["Ticker"] == "FBK.MI")
        &
        (
            full["Date"]
            >= pd.Timestamp(
                "2026-05-01",
                tz="UTC",
            )
        )
    ].copy()

    fbk_cols = [
        "Ticker",
        "Date",
        "Close",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_V405_ORIGINAL",
        "PHASE_AFTER_HIERARCHY",
        "PHASE_AFTER_V409",
        "PHASE_FINAL_SHADOW",
        "LATERAL_SIGNAL",
        "WEAK_UP_TRIGGER",
        "WEAK_DOWN_TRIGGER",
        "BULL_RECOVERY_RAW",
        "BEAR_RECOVERY_RAW",
        "V409_PHASE_CORRECTED",
        "V4010_PHASE_CORRECTED",
        "V4010_CORRECTION_REASON",
    ]

    fbk_existing = [
        c
        for c in fbk_cols
        if c in fbk.columns
    ]

    print(
        fbk[
            fbk_existing
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # CASI SAR AGE 1-3 ALL'ULTIMA SETTIMANA
    # =========================================================================

    banner(
        "ULTIMA SETTIMANA - SAR AGE 1-3"
    )

    latest_age3 = latest[
        latest["SAR_AGE"].between(
            1,
            3,
        )
    ].copy()

    age_cols = [
        "Ticker",
        "Date",
        "Close",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_V405_ORIGINAL",
        "PHASE_AFTER_HIERARCHY",
        "PHASE_AFTER_V409",
        "PHASE_FINAL_SHADOW",
        "V4010_PHASE_CORRECTED",
        "V4010_CORRECTION_REASON",
    ]

    age_existing = [
        c
        for c in age_cols
        if c in latest_age3.columns
    ]

    print(
        latest_age3[
            age_existing
        ]
        .sort_values("Ticker")
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # SALVATAGGIO
    # =========================================================================

    banner(
        "SALVATAGGIO"
    )

    changed = full[
        full[
            "FINAL_CHANGED_FROM_V405"
        ]
        == 1
    ].copy()

    changed.to_csv(
        OUT_CHANGED,
        index=False,
    )

    latest.to_csv(
        OUT_LATEST,
        index=False,
    )

    summary_rows = [
        {
            "METRIC": "ROWS_TOTAL",
            "VALUE": total_rows,
        },
        {
            "METRIC": "ROWS_CHANGED_FROM_V405",
            "VALUE": changed_from_v405,
        },
        {
            "METRIC": "ROWS_CHANGED_FROM_HIERARCHY",
            "VALUE": changed_from_hierarchy,
        },
        {
            "METRIC": "TICKERS_CHANGED_FROM_V405",
            "VALUE": ticker_changed,
        },
        {
            "METRIC": "QUALIFIED_FIRST3",
            "VALUE": len(
                qualified_first3
            ),
        },
        {
            "METRIC": "QUALIFIED_BULL_NOT_BUY",
            "VALUE": len(
                wrong_buy
            ),
        },
        {
            "METRIC": "QUALIFIED_BEAR_NOT_SELL",
            "VALUE": len(
                wrong_sell
            ),
        },
        {
            "METRIC": "LATERAL_FINAL_DIRECTIONAL",
            "VALUE": len(
                lateral_final_directional
            ),
        },
    ]

    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUT_SUMMARY,
        index=False,
    )

    print(
        f"Salvato: {OUT_CHANGED}"
    )

    print(
        f"Salvato: {OUT_LATEST}"
    )

    print(
        f"Salvato: {OUT_SUMMARY}"
    )

    banner(
        "FINE VALIDAZIONE"
    )

    print(
        "Nessun file di produzione modificato."
    )


if __name__ == "__main__":
    main()