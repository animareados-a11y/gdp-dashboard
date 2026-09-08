"""
MarketSentinel
FULL200 VALIDATION - V40.5 CAUSAL INDECISION EXIT SHADOW

OBIETTIVO
---------
Applicare la nuova causal shadow ai 200 titoli e misurare:

- quante righe cambiano PHASE;
- quanti ticker vengono modificati;
- quali transizioni ORIGINAL -> SHADOW compaiono;
- quante nuove DEBOLEZZA_RIALZISTA / DEBOLEZZA_RIBASSISTA;
- quante nuove TREND_RIALZISTA / TREND_RIBASSISTA;
- distribuzione delle ragioni della shadow;
- situazione dell'ultima settimana disponibile.

IMPORTANTE
----------
- NON modifica V40.4
- NON modifica V40.5
- NON modifica V40.9
- NON modifica V40.10
- NON modifica engine.py
- NON modifica i file di produzione

INPUT
-----
data/v40_35_full200_weekly.csv

OUTPUT
------
data/validazione_full200_v405_causal_shadow.csv
data/validazione_full200_v405_causal_shadow_changed.csv
data/validazione_full200_v405_causal_shadow_summary.csv
data/validazione_full200_v405_causal_shadow_latest.csv
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

OUT_FULL = Path(
    "data/validazione_full200_v405_causal_shadow.csv"
)

OUT_CHANGED = Path(
    "data/validazione_full200_v405_causal_shadow_changed.csv"
)

OUT_SUMMARY = Path(
    "data/validazione_full200_v405_causal_shadow_summary.csv"
)

OUT_LATEST = Path(
    "data/validazione_full200_v405_causal_shadow_latest.csv"
)


# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120)


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "MARKETSENTINEL - FULL200 VALIDATION V40.5 CAUSAL SHADOW"
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

    tickers = sorted(
        raw["Ticker"]
        .dropna()
        .astype(str)
        .unique()
    )

    print(
        f"\nTicker trovati: {len(tickers):,}"
    )

    results = []

    # ========================================================
    # PROCESS FULL200
    # ========================================================

    for n, ticker in enumerate(
        tickers,
        start=1,
    ):

        g = (
            raw[
                raw["Ticker"] == ticker
            ]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if g.empty:
            continue

        try:

            out = shadow.process_ticker(g)

        except Exception as exc:

            print(
                f"\nERRORE {ticker}: {exc}"
            )
            raise

        out["Ticker"] = ticker

        results.append(out)

        if (
            n % 20 == 0
            or n == len(tickers)
        ):

            print(
                f"Processati {n}/{len(tickers)} ticker"
            )

    if not results:
        raise RuntimeError(
            "Nessun risultato prodotto."
        )

    full = pd.concat(
        results,
        ignore_index=True,
    )

    OUT_FULL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    full.to_csv(
        OUT_FULL,
        index=False,
    )

    # ========================================================
    # SOLE RIGHE MODIFICATE
    # ========================================================

    changed = full[
        full["CAUSAL_SHADOW_CHANGED"] == 1
    ].copy()

    changed.to_csv(
        OUT_CHANGED,
        index=False,
    )

    # ========================================================
    # 1. NUMERI GENERALI
    # ========================================================

    banner(
        "1 - NUMERI GENERALI"
    )

    total_rows = len(full)

    changed_rows = len(changed)

    changed_tickers = (
        changed["Ticker"]
        .nunique()
        if not changed.empty
        else 0
    )

    print(
        f"\nRighe totali: {total_rows:,}"
    )

    print(
        f"Righe modificate: {changed_rows:,}"
    )

    print(
        f"Percentuale righe modificate: "
        f"{changed_rows / total_rows * 100:.2f}%"
    )

    print(
        f"Ticker modificati: "
        f"{changed_tickers:,} / {len(tickers):,}"
    )

    # ========================================================
    # 2. TRANSIZIONI ORIGINAL -> SHADOW
    # ========================================================

    banner(
        "2 - TRANSIZIONI PHASE ORIGINAL -> SHADOW"
    )

    transitions = (
        changed
        .groupby(
            [
                "PHASE_V405_ORIGINAL",
                "PHASE_CAUSAL_SHADOW",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .sort_values(
            "rows",
            ascending=False,
        )
    )

    if transitions.empty:

        print(
            "Nessuna transizione."
        )

    else:

        transitions["pct_changed"] = (
            transitions["rows"]
            /
            changed_rows
            *
            100
        )

        print(
            transitions.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.2f}",
            )
        )

    # ========================================================
    # 3. REASON
    # ========================================================

    banner(
        "3 - CAUSAL SHADOW REASONS"
    )

    reasons = (
        changed[
            "CAUSAL_SHADOW_REASON"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "reason"
        )
        .reset_index(
            name="rows"
        )
    )

    if not reasons.empty:

        reasons["pct_changed"] = (
            reasons["rows"]
            /
            changed_rows
            *
            100
        )

        print(
            reasons.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.2f}",
            )
        )

    # ========================================================
    # 4. NUOVE DEBOLEZZE
    # ========================================================

    banner(
        "4 - NUOVE DEBOLEZZE"
    )

    new_weak_up = changed[
        changed["PHASE_CAUSAL_SHADOW"]
        ==
        "DEBOLEZZA_RIALZISTA"
    ]

    new_weak_down = changed[
        changed["PHASE_CAUSAL_SHADOW"]
        ==
        "DEBOLEZZA_RIBASSISTA"
    ]

    print(
        f"\nDEBOLEZZA_RIALZISTA nuove righe: "
        f"{len(new_weak_up):,}"
    )

    print(
        f"Ticker coinvolti: "
        f"{new_weak_up['Ticker'].nunique():,}"
    )

    print(
        f"\nDEBOLEZZA_RIBASSISTA nuove righe: "
        f"{len(new_weak_down):,}"
    )

    print(
        f"Ticker coinvolti: "
        f"{new_weak_down['Ticker'].nunique():,}"
    )

    # ========================================================
    # 5. NUOVI TREND
    # ========================================================

    banner(
        "5 - NUOVI TREND"
    )

    new_trend_up = changed[
        changed["PHASE_CAUSAL_SHADOW"]
        ==
        "TREND_RIALZISTA"
    ]

    new_trend_down = changed[
        changed["PHASE_CAUSAL_SHADOW"]
        ==
        "TREND_RIBASSISTA"
    ]

    print(
        f"\nTREND_RIALZISTA nuove righe: "
        f"{len(new_trend_up):,}"
    )

    print(
        f"Ticker coinvolti: "
        f"{new_trend_up['Ticker'].nunique():,}"
    )

    print(
        f"\nTREND_RIBASSISTA nuove righe: "
        f"{len(new_trend_down):,}"
    )

    print(
        f"Ticker coinvolti: "
        f"{new_trend_down['Ticker'].nunique():,}"
    )

    # ========================================================
    # 6. DURATA MODIFICHE PER TICKER
    # ========================================================

    banner(
        "6 - TICKER PIU' MODIFICATI"
    )

    ticker_changes = (
        changed
        .groupby(
            "Ticker"
        )
        .size()
        .reset_index(
            name="changed_rows"
        )
        .sort_values(
            "changed_rows",
            ascending=False,
        )
    )

    print(
        ticker_changes
        .head(30)
        .to_string(
            index=False,
        )
    )

    # ========================================================
    # 7. ULTIMA SETTIMANA DISPONIBILE
    # ========================================================

    banner(
        "7 - ULTIMA SETTIMANA DISPONIBILE"
    )

    latest_date = full["Date"].max()

    latest = full[
        full["Date"] == latest_date
    ].copy()

    latest.to_csv(
        OUT_LATEST,
        index=False,
    )

    latest_changed = latest[
        latest["CAUSAL_SHADOW_CHANGED"] == 1
    ].copy()

    print(
        f"\nUltima data: {latest_date}"
    )

    print(
        f"Ticker ultima settimana: "
        f"{len(latest):,}"
    )

    print(
        f"Ticker con PHASE diversa: "
        f"{len(latest_changed):,}"
    )

    if not latest_changed.empty:

        latest_cols = [
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
        ]

        existing = [
            c
            for c in latest_cols
            if c in latest_changed.columns
        ]

        print(
            latest_changed[
                existing
            ]
            .sort_values(
                "Ticker"
            )
            .to_string(
                index=False,
            )
        )

    # ========================================================
    # 8. FINECO CHECK
    # ========================================================

    banner(
        "8 - FINECO CHECK"
    )

    fineco = full[
        (
            full["Ticker"]
            ==
            "FBK.MI"
        )
        &
        (
            full["Date"]
            >=
            pd.Timestamp(
                "2026-08-07",
                tz="UTC",
            )
        )
    ].copy()

    fineco_cols = [
        "Ticker",
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

    existing_fineco = [
        c
        for c in fineco_cols
        if c in fineco.columns
    ]

    print(
        fineco[
            existing_fineco
        ].to_string(
            index=False,
        )
    )

    # ========================================================
    # 9. SUMMARY CSV
    # ========================================================

    summary_rows = []

    summary_rows.append(
        {
            "metric": "total_rows",
            "value": total_rows,
        }
    )

    summary_rows.append(
        {
            "metric": "changed_rows",
            "value": changed_rows,
        }
    )

    summary_rows.append(
        {
            "metric": "changed_rows_pct",
            "value": (
                changed_rows
                /
                total_rows
                *
                100
            ),
        }
    )

    summary_rows.append(
        {
            "metric": "changed_tickers",
            "value": changed_tickers,
        }
    )

    summary_rows.append(
        {
            "metric": "new_weak_up_rows",
            "value": len(new_weak_up),
        }
    )

    summary_rows.append(
        {
            "metric": "new_weak_down_rows",
            "value": len(new_weak_down),
        }
    )

    summary_rows.append(
        {
            "metric": "new_trend_up_rows",
            "value": len(new_trend_up),
        }
    )

    summary_rows.append(
        {
            "metric": "new_trend_down_rows",
            "value": len(new_trend_down),
        }
    )

    summary_rows.append(
        {
            "metric": "latest_changed_tickers",
            "value": len(latest_changed),
        }
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # FILES
    # ========================================================

    banner(
        "FILE SALVATI"
    )

    print(
        f"\nFULL    : {OUT_FULL}"
    )

    print(
        f"CHANGED : {OUT_CHANGED}"
    )

    print(
        f"SUMMARY : {OUT_SUMMARY}"
    )

    print(
        f"LATEST  : {OUT_LATEST}"
    )

    print(
        "\nNESSUN FILE DI PRODUZIONE MODIFICATO."
    )


if __name__ == "__main__":
    main()