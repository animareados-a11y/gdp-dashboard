"""
MarketSentinel
VALIDAZIONE PHASE DYNAMICS V3.1 - ITALIA

Scopo:
- testare la shadow V3.1 su un gruppo ampio di titoli italiani;
- non modificare alcun file production;
- confrontare PHASE base vs PHASE_DYNAMICS_SHADOW;
- evidenziare distribuzioni, cambiamenti e casi sospetti.

Input:
    data/v40_35_full200_weekly.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow as dynamics


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")


# ============================================================
# HELPERS
# ============================================================

def pct(n, d):
    if d == 0:
        return 0.0
    return 100.0 * n / d


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print("VALIDAZIONE PHASE DYNAMICS V3.1 - TITOLI ITALIANI")
    print("=" * 120)

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

    # --------------------------------------------------------
    # TITOLI ITALIANI
    # --------------------------------------------------------

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

    print(
        ", ".join(
            italy_tickers
        )
    )

    print()
    print("=" * 120)

    all_results = []

    ticker_summary = []

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
            z = dynamics.process_ticker(g)

        except Exception as exc:
            print(
                f"  ERRORE: {exc}"
            )
            continue

        z["Ticker"] = ticker

        all_results.append(z)

        base = z["PHASE_DYNAMICS_BASE"].astype(str)
        shadow = z["PHASE_DYNAMICS_SHADOW"].astype(str)

        changed = (
            base != shadow
        )

        n_rows = len(z)
        n_changed = int(changed.sum())

        n_ind_base = int(
            (base == "INDECISIONE").sum()
        )

        n_ind_shadow = int(
            (shadow == "INDECISIONE").sum()
        )

        n_buy_base = int(
            (base == "BUY").sum()
        )

        n_buy_shadow = int(
            (shadow == "BUY").sum()
        )

        n_sell_base = int(
            (base == "SELL").sum()
        )

        n_sell_shadow = int(
            (shadow == "SELL").sum()
        )

        ticker_summary.append(
            {
                "Ticker": ticker,
                "Rows": n_rows,
                "Changed": n_changed,
                "ChangedPct": pct(
                    n_changed,
                    n_rows,
                ),
                "IndecisionBase": n_ind_base,
                "IndecisionShadow": n_ind_shadow,
                "BuyBase": n_buy_base,
                "BuyShadow": n_buy_shadow,
                "SellBase": n_sell_base,
                "SellShadow": n_sell_shadow,
            }
        )

    if not all_results:
        raise RuntimeError(
            "Nessun ticker processato."
        )

    out = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        ticker_summary
    )

    # ========================================================
    # GLOBAL
    # ========================================================

    print()
    print("=" * 120)
    print("RISULTATO AGGREGATO ITALIA")
    print("=" * 120)

    total_rows = len(out)

    changed = (
        out["PHASE_DYNAMICS_BASE"].astype(str)
        !=
        out["PHASE_DYNAMICS_SHADOW"].astype(str)
    )

    total_changed = int(
        changed.sum()
    )

    print()
    print(
        f"Righe totali: {total_rows:,}"
    )

    print(
        f"Righe modificate: "
        f"{total_changed:,} "
        f"({pct(total_changed, total_rows):.2f}%)"
    )

    print()

    # ========================================================
    # DISTRIBUTION BASE VS SHADOW
    # ========================================================

    print("=" * 120)
    print("DISTRIBUZIONE PHASE")
    print("=" * 120)

    base_counts = (
        out["PHASE_DYNAMICS_BASE"]
        .value_counts()
    )

    shadow_counts = (
        out["PHASE_DYNAMICS_SHADOW"]
        .value_counts()
    )

    labels = sorted(
        set(base_counts.index)
        |
        set(shadow_counts.index)
    )

    rows = []

    for phase in labels:

        b = int(
            base_counts.get(
                phase,
                0,
            )
        )

        s = int(
            shadow_counts.get(
                phase,
                0,
            )
        )

        rows.append(
            {
                "PHASE": phase,
                "BASE": b,
                "SHADOW": s,
                "DELTA": s - b,
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
    # TRANSITIONS
    # ========================================================

    print()
    print("=" * 120)
    print("CAMBIAMENTI BASE -> SHADOW")
    print("=" * 120)

    changes = (
        out.loc[
            changed,
            [
                "PHASE_DYNAMICS_BASE",
                "PHASE_DYNAMICS_SHADOW",
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
    # REASONS
    # ========================================================

    print()
    print("=" * 120)
    print("REASONS")
    print("=" * 120)

    reasons = (
        out["PHASE_DYNAMICS_REASON"]
        .value_counts()
        .reset_index()
    )

    reasons.columns = [
        "Reason",
        "Count",
    ]

    print()
    print(
        reasons.to_string(
            index=False
        )
    )

    # ========================================================
    # TICKER CON PIU' CAMBIAMENTI
    # ========================================================

    print()
    print("=" * 120)
    print("TICKER CON PIU' CAMBIAMENTI")
    print("=" * 120)

    summary = summary.sort_values(
        [
            "ChangedPct",
            "Changed",
        ],
        ascending=False,
    )

    print()
    print(
        summary.head(40).to_string(
            index=False
        )
    )

    # ========================================================
    # BUY / SELL SAFETY
    # ========================================================

    print()
    print("=" * 120)
    print("BUY / SELL SAFETY")
    print("=" * 120)

    base = (
        out["PHASE_DYNAMICS_BASE"]
        .astype(str)
    )

    shadow = (
        out["PHASE_DYNAMICS_SHADOW"]
        .astype(str)
    )

    invented_buy = int(
        (
            (shadow == "BUY")
            &
            (base != "BUY")
        ).sum()
    )

    invented_sell = int(
        (
            (shadow == "SELL")
            &
            (base != "SELL")
        ).sum()
    )

    modified_buy = int(
        (
            (base == "BUY")
            &
            (shadow != "BUY")
        ).sum()
    )

    modified_sell = int(
        (
            (base == "SELL")
            &
            (shadow != "SELL")
        ).sum()
    )

    print()
    print(
        f"BUY inventati:      {invented_buy}"
    )
    print(
        f"SELL inventati:     {invented_sell}"
    )
    print(
        f"BUY modificati:     {modified_buy}"
    )
    print(
        f"SELL modificati:    {modified_sell}"
    )

    # ========================================================
    # LATEST WEEK
    # ========================================================

    print()
    print("=" * 120)
    print("ULTIMA SETTIMANA PER TICKER")
    print("=" * 120)

    latest = (
        out.sort_values("Date")
        .groupby("Ticker")
        .tail(1)
        [
            [
                "Ticker",
                "Date",
                "PHASE_DYNAMICS_BASE",
                "PHASE_DYNAMICS_SHADOW",
                "PHASE_DYNAMICS_REASON",
            ]
        ]
        .sort_values("Ticker")
    )

    print()
    print(
        latest.to_string(
            index=False
        )
    )

    latest_changed = (
        latest[
            latest["PHASE_DYNAMICS_BASE"]
            !=
            latest["PHASE_DYNAMICS_SHADOW"]
        ]
    )

    print()
    print(
        f"Ultime settimane modificate: "
        f"{len(latest_changed)}/{len(latest)}"
    )


if __name__ == "__main__":
    main()