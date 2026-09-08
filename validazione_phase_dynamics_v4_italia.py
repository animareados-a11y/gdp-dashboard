"""
MarketSentinel
VALIDAZIONE PHASE DYNAMICS V4 - ITALIA

OBIETTIVO
---------
Validare la V4 sui ticker italiani prima di qualunque FULL200.

Controlli:
1. quanti BUY / SELL non qualificati vengono rielaborati;
2. quanti BUY / SELL finali restano fuori da run qualificati AGE 1-3;
3. distribuzione PHASE V3.1 vs V4;
4. transizioni V3.1 -> V4;
5. esito dei casi rielaborati;
6. ultima settimana per ticker;
7. focus Fineco / UCG / ENEL.

IMPORTANTE
----------
- NON modifica production
- NON modifica V40.10
- NON modifica Dynamics V3.1
- NON modifica engine.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v4 as v4


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

OUT_CHANGED = Path(
    "data/validazione_phase_dynamics_v4_italia_changed.csv"
)

OUT_REPROCESSED = Path(
    "data/validazione_phase_dynamics_v4_italia_reprocessed.csv"
)

OUT_LATEST = Path(
    "data/validazione_phase_dynamics_v4_italia_latest.csv"
)

OUT_SUMMARY = Path(
    "data/validazione_phase_dynamics_v4_italia_summary.csv"
)


FOCUS_TICKERS = [
    "FBK.MI",
    "UCG.MI",
    "ENEL.MI",
]


def banner(title: str):
    print()
    print("=" * 140)
    print(title)
    print("=" * 140)


def main():

    banner(
        "VALIDAZIONE PHASE DYNAMICS V4 - ITALIA"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input non trovato: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    if "Ticker" not in df.columns:
        raise RuntimeError(
            "Colonna Ticker non trovata."
        )

    if "Date" not in df.columns:
        raise RuntimeError(
            "Colonna Date non trovata."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    italy_tickers = sorted(
        [
            ticker
            for ticker in df["Ticker"]
            .dropna()
            .unique()
            if str(ticker).endswith(".MI")
        ]
    )

    print()
    print(
        f"Ticker italiani trovati: "
        f"{len(italy_tickers)}"
    )

    all_rows = []

    for i, ticker in enumerate(
        italy_tickers,
        start=1,
    ):

        print(
            f"[{i:02d}/{len(italy_tickers):02d}] "
            f"{ticker}"
        )

        g = (
            df[
                df["Ticker"] == ticker
            ]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if g.empty:
            continue

        try:
            z = v4.process_ticker(
                g
            )

        except Exception as exc:
            print(
                f"  ERRORE: {exc}"
            )
            continue

        z["Ticker"] = ticker

        all_rows.append(z)

    if not all_rows:
        raise RuntimeError(
            "Nessun ticker processato."
        )

    out = pd.concat(
        all_rows,
        ignore_index=True,
    )

    required = [
        "Ticker",
        "Date",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_DYNAMICS_V31",
        "PHASE_DYNAMICS_V4_PRE_V4010",
        "PHASE_DYNAMICS_V4",
        "V4_INVALID_BUY_SELL",
        "V4_INVALID_FINAL_BUY",
        "V4_INVALID_FINAL_SELL",
        "PHASE_DYNAMICS_V4_REASON",
    ]

    missing = [
        c
        for c in required
        if c not in out.columns
    ]

    if missing:
        raise RuntimeError(
            "Colonne mancanti: "
            + ", ".join(missing)
        )

    # ============================================================
    # RISULTATO GENERALE
    # ============================================================

    banner(
        "RISULTATO GENERALE"
    )

    total_rows = len(out)

    changed_mask = (
        out["PHASE_DYNAMICS_V31"]
        !=
        out["PHASE_DYNAMICS_V4"]
    )

    reprocessed_mask = (
        pd.to_numeric(
            out["V4_INVALID_BUY_SELL"],
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
    )

    invalid_final_buy = int(
        pd.to_numeric(
            out[
                "V4_INVALID_FINAL_BUY"
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    invalid_final_sell = int(
        pd.to_numeric(
            out[
                "V4_INVALID_FINAL_SELL"
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    changed_rows = int(
        changed_mask.sum()
    )

    reprocessed_rows = int(
        reprocessed_mask.sum()
    )

    print()
    print(
        f"Righe totali: "
        f"{total_rows:,}"
    )

    print(
        f"BUY/SELL non qualificati rielaborati: "
        f"{reprocessed_rows}"
    )

    print(
        f"Righe cambiate V3.1 -> V4: "
        f"{changed_rows} "
        f"({changed_rows / total_rows * 100:.2f}%)"
    )

    print()
    print(
        f"BUY finali non validi: "
        f"{invalid_final_buy}"
    )

    print(
        f"SELL finali non validi: "
        f"{invalid_final_sell}"
    )

    # ============================================================
    # SAFETY BUY / SELL
    # ============================================================

    banner(
        "SAFETY BUY / SELL"
    )

    sar_side = pd.to_numeric(
        out["SAR_SIDE"],
        errors="coerce",
    )

    sar_age = pd.to_numeric(
        out["SAR_AGE"],
        errors="coerce",
    )

    qualified = pd.to_numeric(
        out["V4010_RUN_QUALIFIED"],
        errors="coerce",
    ).fillna(0)

    valid_buy = (
        qualified.eq(1)
        &
        sar_side.eq(1)
        &
        sar_age.isin([1, 2, 3])
    )

    valid_sell = (
        qualified.eq(1)
        &
        sar_side.eq(-1)
        &
        sar_age.isin([1, 2, 3])
    )

    final_buy = (
        out[
            "PHASE_DYNAMICS_V4"
        ].eq("BUY")
    )

    final_sell = (
        out[
            "PHASE_DYNAMICS_V4"
        ].eq("SELL")
    )

    print()
    print(
        f"BUY finali totali: "
        f"{int(final_buy.sum())}"
    )

    print(
        f"BUY finali validi: "
        f"{int((final_buy & valid_buy).sum())}"
    )

    print(
        f"BUY finali non validi: "
        f"{int((final_buy & ~valid_buy).sum())}"
    )

    print()
    print(
        f"SELL finali totali: "
        f"{int(final_sell.sum())}"
    )

    print(
        f"SELL finali validi: "
        f"{int((final_sell & valid_sell).sum())}"
    )

    print(
        f"SELL finali non validi: "
        f"{int((final_sell & ~valid_sell).sum())}"
    )

    # ============================================================
    # DISTRIBUZIONE
    # ============================================================

    banner(
        "DISTRIBUZIONE PHASE V3.1 VS V4"
    )

    distribution = pd.concat(
        [
            out[
                "PHASE_DYNAMICS_V31"
            ]
            .value_counts()
            .rename("V31"),

            out[
                "PHASE_DYNAMICS_V4"
            ]
            .value_counts()
            .rename("V4"),
        ],
        axis=1,
    ).fillna(0).astype(int)

    print()
    print(
        distribution.to_string()
    )

    # ============================================================
    # TRANSIZIONI
    # ============================================================

    banner(
        "TRANSIZIONI V3.1 -> V4"
    )

    transitions = (
        out.loc[
            changed_mask,
            [
                "PHASE_DYNAMICS_V31",
                "PHASE_DYNAMICS_V4",
            ],
        ]
        .value_counts()
        .reset_index(
            name="COUNT"
        )
        .sort_values(
            "COUNT",
            ascending=False,
        )
    )

    if transitions.empty:
        print()
        print(
            "Nessuna modifica."
        )
    else:
        print()
        print(
            transitions.to_string(
                index=False
            )
        )

    # ============================================================
    # ESITO DEI BUY / SELL NON QUALIFICATI
    # ============================================================

    banner(
        "ESITO BUY / SELL NON QUALIFICATI RIELABORATI"
    )

    reclassified = (
        out.loc[
            reprocessed_mask,
            [
                "PHASE_DYNAMICS_V31",
                "PHASE_DYNAMICS_V4_PRE_V4010",
                "PHASE_DYNAMICS_V4",
            ],
        ]
        .value_counts()
        .reset_index(
            name="COUNT"
        )
        .sort_values(
            "COUNT",
            ascending=False,
        )
    )

    print()
    print(
        reclassified.to_string(
            index=False
        )
    )

    # ============================================================
    # MOTIVI
    # ============================================================

    banner(
        "MOTIVI V4"
    )

    reasons = (
        out.loc[
            reprocessed_mask,
            "PHASE_DYNAMICS_V4_REASON",
        ]
        .value_counts()
        .reset_index()
    )

    reasons.columns = [
        "REASON",
        "COUNT",
    ]

    print()
    print(
        reasons.to_string(
            index=False
        )
    )

    # ============================================================
    # PRIMI 80 CASI RIELABORATI
    # ============================================================

    banner(
        "PRIMI 80 BUY / SELL NON QUALIFICATI RIELABORATI"
    )

    detail_cols = [
        "Ticker",
        "Date",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_DYNAMICS_V31",
        "PHASE_DYNAMICS_V4_PRE_V4010",
        "PHASE_DYNAMICS_V4",
        "PHASE_DYNAMICS_V4_REASON",
    ]

    optional_cols = [
        "PHASE_DYNAMICS_REASON",
        "LATERAL_SIGNAL",
        "WEAK_UP_TRIGGER",
        "WEAK_DOWN_TRIGGER",
        "BULL_RECOVERY_RAW",
        "BEAR_RECOVERY_RAW",
        "BULL_RECOVERY_CONFIRMED",
        "BEAR_RECOVERY_CONFIRMED",
    ]

    for c in optional_cols:
        if c in out.columns:
            detail_cols.append(c)

    print()
    print(
        out.loc[
            reprocessed_mask,
            detail_cols,
        ]
        .head(80)
        .to_string(
            index=False
        )
    )

    # ============================================================
    # ULTIMA SETTIMANA
    # ============================================================

    banner(
        "ULTIMA SETTIMANA PER TICKER"
    )

    latest = (
        out.sort_values(
            [
                "Ticker",
                "Date",
            ]
        )
        .groupby(
            "Ticker",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    latest_cols = [
        "Ticker",
        "Date",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_RUN_QUALIFIED",
        "PHASE_DYNAMICS_V31",
        "PHASE_DYNAMICS_V4",
        "PHASE_DYNAMICS_V4_REASON",
    ]

    print()
    print(
        latest[
            latest_cols
        ]
        .sort_values(
            "Ticker"
        )
        .to_string(
            index=False
        )
    )

    # ============================================================
    # FOCUS
    # ============================================================

    for ticker in FOCUS_TICKERS:

        banner(
            ticker
        )

        z = (
            out[
                out["Ticker"] == ticker
            ]
            .copy()
            .sort_values("Date")
        )

        if z.empty:
            print(
                "Ticker non trovato."
            )
            continue

        if ticker == "FBK.MI":
            z = z[
                z["Date"]
                >=
                pd.Timestamp(
                    "2026-05-01",
                    tz="UTC",
                )
            ]

        elif ticker == "UCG.MI":
            z = z[
                (
                    z["Date"]
                    >=
                    pd.Timestamp(
                        "2023-11-01",
                        tz="UTC",
                    )
                )
                &
                (
                    z["Date"]
                    <=
                    pd.Timestamp(
                        "2024-08-31",
                        tz="UTC",
                    )
                )
            ]

        elif ticker == "ENEL.MI":
            z = z[
                z["Date"]
                >=
                pd.Timestamp(
                    "2026-03-01",
                    tz="UTC",
                )
            ]

        focus_cols = [
            "Date",
            "Close",
            "SAR_SIDE",
            "SAR_AGE",
            "V4010_RUN_QUALIFIED",
            "PHASE_DYNAMICS_V31",
            "PHASE_DYNAMICS_V4_PRE_V4010",
            "PHASE_DYNAMICS_V4",
            "PHASE_DYNAMICS_V4_REASON",
        ]

        focus_cols = [
            c
            for c in focus_cols
            if c in z.columns
        ]

        print()
        print(
            z[
                focus_cols
            ]
            .to_string(
                index=False
            )
        )

    # ============================================================
    # SALVATAGGI
    # ============================================================

    banner(
        "SALVATAGGIO"
    )

    changed_df = (
        out.loc[
            changed_mask
        ]
        .copy()
    )

    reprocessed_df = (
        out.loc[
            reprocessed_mask
        ]
        .copy()
    )

    summary = pd.DataFrame(
        [
            {
                "ROWS": total_rows,
                "TICKERS": out[
                    "Ticker"
                ].nunique(),
                "REPROCESSED_INVALID_BUY_SELL":
                    reprocessed_rows,
                "CHANGED_V31_TO_V4":
                    changed_rows,
                "CHANGED_PCT":
                    (
                        changed_rows
                        /
                        total_rows
                        *
                        100
                    ),
                "INVALID_FINAL_BUY":
                    invalid_final_buy,
                "INVALID_FINAL_SELL":
                    invalid_final_sell,
            }
        ]
    )

    changed_df.to_csv(
        OUT_CHANGED,
        index=False,
    )

    reprocessed_df.to_csv(
        OUT_REPROCESSED,
        index=False,
    )

    latest.to_csv(
        OUT_LATEST,
        index=False,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    print()
    print(
        f"Salvato: "
        f"{OUT_CHANGED}"
    )

    print(
        f"Salvato: "
        f"{OUT_REPROCESSED}"
    )

    print(
        f"Salvato: "
        f"{OUT_LATEST}"
    )

    print(
        f"Salvato: "
        f"{OUT_SUMMARY}"
    )

    banner(
        "FINE VALIDAZIONE"
    )

    if (
        invalid_final_buy == 0
        and
        invalid_final_sell == 0
    ):
        print()
        print(
            "SAFETY BUY/SELL: OK"
        )
    else:
        print()
        print(
            "ATTENZIONE: restano BUY/SELL "
            "fuori dalla tassonomia."
        )

    print()
    print(
        "Nessun file production modificato."
    )


if __name__ == "__main__":
    main()