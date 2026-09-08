"""
MarketSentinel
DIAGNOSI V40.5 - FALSE LATERALITA / INDECISIONE

OBIETTIVO
---------
Analizzare gli episodi che V40.5 classifica come INDECISIONE.

INDECISIONE deve rappresentare LATERALITA'.

Il diagnostico NON modifica alcuna logica e NON decide nuove soglie.
Misura semplicemente cosa fa realmente il prezzo durante gli episodi
di INDECISIONE, per individuare i casi nei quali la lateralita'
potrebbe nascondere un movimento direzionale.

Ticker focus:
    FBK.MI   Fineco
    UCG.MI   UniCredit
    BBVA.MC  BBVA
    TIT.MI   Telecom Italia

OUTPUT
------
Solo terminale.
Nessun file di produzione modificato.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import weekly_v40_5_core_phase_engine as v405


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

TICKERS = [
    "FBK.MI",
    "UCG.MI",
    "BBVA.MC",
    "TIT.MI",
]


def banner(text):
    print("\n" + "=" * 145)
    print(text)
    print("=" * 145)


def safe_num(x):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return np.nan


def pct_change(a, b):
    a = safe_num(a)
    b = safe_num(b)

    if not np.isfinite(a) or not np.isfinite(b) or a == 0:
        return np.nan

    return (b / a - 1.0) * 100.0


def linear_slope_pct(values):
    """
    Regressione lineare semplice del Close sul numero di settimane.
    Restituisce la pendenza settimanale come % del primo prezzo.
    E' solo una misura descrittiva, NON una soglia PHASE.
    """
    y = np.asarray(values, dtype=float)

    mask = np.isfinite(y)
    y = y[mask]

    if len(y) < 2:
        return np.nan

    x = np.arange(len(y), dtype=float)

    slope = np.polyfit(x, y, 1)[0]

    if y[0] == 0:
        return np.nan

    return slope / y[0] * 100.0


def find_close_column(df):
    for col in [
        "Close",
        "close",
        "Adj Close",
        "Adj_Close",
        "PRICE",
        "Price",
    ]:
        if col in df.columns:
            return col

    raise RuntimeError(
        "Non trovo una colonna prezzo Close nel dataset."
    )


def main():

    banner("CARICAMENTO DATI")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    close_col = find_close_column(df)

    print(f"File: {INPUT_FILE}")
    print(f"Colonna prezzo utilizzata: {close_col}")

    all_episodes = []

    for ticker in TICKERS:

        raw = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if raw.empty:
            print(f"\nATTENZIONE: {ticker} non trovato.")
            continue

        g = v405.process_ticker(raw)

        g = (
            g.sort_values("Date")
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Identificazione episodi contigui di INDECISIONE V40.5
        # --------------------------------------------------------

        in_episode = False
        start_pos = None

        episodes = []

        for i, row in g.iterrows():

            phase = str(row.get("PHASE", ""))

            if phase == "INDECISIONE" and not in_episode:
                in_episode = True
                start_pos = i

            if phase != "INDECISIONE" and in_episode:

                end_pos = i - 1

                episodes.append(
                    (start_pos, end_pos)
                )

                in_episode = False
                start_pos = None

        if in_episode:
            episodes.append(
                (start_pos, len(g) - 1)
            )

        # --------------------------------------------------------
        # Metriche episodio
        # --------------------------------------------------------

        ticker_records = []

        for ep_num, (start, end) in enumerate(
            episodes,
            start=1,
        ):

            ep = g.iloc[start:end + 1].copy()

            prices = pd.to_numeric(
                ep[close_col],
                errors="coerce",
            )

            valid_prices = prices.dropna()

            if valid_prices.empty:
                continue

            start_price = valid_prices.iloc[0]
            end_price = valid_prices.iloc[-1]

            ret_pct = pct_change(
                start_price,
                end_price,
            )

            high_price = valid_prices.max()
            low_price = valid_prices.min()

            max_up_from_start = pct_change(
                start_price,
                high_price,
            )

            max_down_from_start = pct_change(
                start_price,
                low_price,
            )

            price_range_pct = (
                (high_price - low_price)
                / start_price
                * 100.0
                if start_price != 0
                else np.nan
            )

            slope_pct_week = linear_slope_pct(
                valid_prices.values
            )

            positive_weeks = 0
            negative_weeks = 0
            flat_weeks = 0

            weekly_changes = (
                valid_prices
                .pct_change()
                .dropna()
            )

            for x in weekly_changes:

                if x > 0:
                    positive_weeks += 1
                elif x < 0:
                    negative_weeks += 1
                else:
                    flat_weeks += 1

            sar_side_start = (
                ep.iloc[0].get(
                    "SAR_SIDE",
                    np.nan,
                )
            )

            sar_side_end = (
                ep.iloc[-1].get(
                    "SAR_SIDE",
                    np.nan,
                )
            )

            weak_up_count = int(
                pd.to_numeric(
                    ep.get(
                        "WEAK_UP_TRIGGER",
                        pd.Series(
                            0,
                            index=ep.index,
                        ),
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )

            weak_down_count = int(
                pd.to_numeric(
                    ep.get(
                        "WEAK_DOWN_TRIGGER",
                        pd.Series(
                            0,
                            index=ep.index,
                        ),
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )

            lateral_count = int(
                pd.to_numeric(
                    ep.get(
                        "LATERAL_SIGNAL",
                        pd.Series(
                            0,
                            index=ep.index,
                        ),
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )

            bull_recovery_raw_count = int(
                pd.to_numeric(
                    ep.get(
                        "BULL_RECOVERY_RAW",
                        pd.Series(
                            0,
                            index=ep.index,
                        ),
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )

            bear_recovery_raw_count = int(
                pd.to_numeric(
                    ep.get(
                        "BEAR_RECOVERY_RAW",
                        pd.Series(
                            0,
                            index=ep.index,
                        ),
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )

            prev_phase = (
                str(g.iloc[start - 1]["PHASE"])
                if start > 0
                else "START_DATA"
            )

            next_phase = (
                str(g.iloc[end + 1]["PHASE"])
                if end + 1 < len(g)
                else "END_DATA"
            )

            record = {
                "Ticker": ticker,
                "Episode": ep_num,
                "Start": ep.iloc[0]["Date"],
                "End": ep.iloc[-1]["Date"],
                "Weeks": len(ep),

                "PrevPhase": prev_phase,
                "NextPhase": next_phase,

                "StartPrice": start_price,
                "EndPrice": end_price,

                "ReturnPct": ret_pct,
                "SlopePctWeek": slope_pct_week,
                "RangePct": price_range_pct,

                "MaxUpPct": max_up_from_start,
                "MaxDownPct": max_down_from_start,

                "PositiveWeeks": positive_weeks,
                "NegativeWeeks": negative_weeks,
                "FlatWeeks": flat_weeks,

                "SARSideStart": sar_side_start,
                "SARSideEnd": sar_side_end,

                "LateralSignalWeeks": lateral_count,
                "WeakUpWeeks": weak_up_count,
                "WeakDownWeeks": weak_down_count,

                "BullRecoveryRawWeeks":
                    bull_recovery_raw_count,

                "BearRecoveryRawWeeks":
                    bear_recovery_raw_count,
            }

            ticker_records.append(record)
            all_episodes.append(record)

        ticker_df = pd.DataFrame(
            ticker_records
        )

        banner(
            f"{ticker} - EPISODI INDECISIONE V40.5"
        )

        if ticker_df.empty:
            print("Nessun episodio.")
            continue

        display_cols = [
            "Episode",
            "Start",
            "End",
            "Weeks",
            "PrevPhase",
            "NextPhase",
            "ReturnPct",
            "SlopePctWeek",
            "RangePct",
            "PositiveWeeks",
            "NegativeWeeks",
            "LateralSignalWeeks",
            "WeakUpWeeks",
            "WeakDownWeeks",
            "BullRecoveryRawWeeks",
            "BearRecoveryRawWeeks",
        ]

        print(
            ticker_df[
                display_cols
            ]
            .round(3)
            .to_string(
                index=False
            )
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    result = pd.DataFrame(
        all_episodes
    )

    banner(
        "SUMMARY COMPLESSIVO"
    )

    if result.empty:
        print(
            "Nessun episodio INDECISIONE trovato."
        )
        return

    print(
        f"Episodi INDECISIONE totali: "
        f"{len(result):,}"
    )

    print(
        f"Durata media: "
        f"{result['Weeks'].mean():.2f} settimane"
    )

    print(
        f"Durata mediana: "
        f"{result['Weeks'].median():.2f} settimane"
    )

    print(
        "\nDistribuzione rendimento durante INDECISIONE:"
    )

    print(
        result["ReturnPct"]
        .describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
        .round(3)
        .to_string()
    )

    # ------------------------------------------------------------
    # Casi più direzionali
    #
    # NON imponiamo una soglia per cambiare PHASE.
    # Ordiniamo semplicemente gli episodi per movimento assoluto.
    # ------------------------------------------------------------

    result[
        "AbsReturnPct"
    ] = result[
        "ReturnPct"
    ].abs()

    banner(
        "TOP 20 INDECISIONE CON MAGGIORE MOVIMENTO ASSOLUTO"
    )

    top = (
        result
        .sort_values(
            [
                "AbsReturnPct",
                "Weeks",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(20)
    )

    cols = [
        "Ticker",
        "Start",
        "End",
        "Weeks",
        "PrevPhase",
        "NextPhase",
        "ReturnPct",
        "SlopePctWeek",
        "RangePct",
        "PositiveWeeks",
        "NegativeWeeks",
        "LateralSignalWeeks",
        "WeakUpWeeks",
        "WeakDownWeeks",
        "BullRecoveryRawWeeks",
        "BearRecoveryRawWeeks",
    ]

    print(
        top[
            cols
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # FINECO 2026
    # ------------------------------------------------------------

    banner(
        "FINECO 2026 - DETTAGLIO SETTIMANALE V40.5"
    )

    raw = (
        df[
            df["Ticker"] == "FBK.MI"
        ]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    fineco = v405.process_ticker(
        raw
    )

    fineco = fineco[
        fineco["Date"]
        >= pd.Timestamp(
            "2026-05-01",
            tz="UTC",
        )
    ].copy()

    detail_cols = [
        "Date",
        close_col,
        "PHASE",
        "SAR_SIDE",
        "SAR_AGE",
        "LATERAL_SIGNAL",
        "WEAK_UP_TRIGGER",
        "WEAK_DOWN_TRIGGER",
        "BULL_RECOVERY_RAW",
        "BEAR_RECOVERY_RAW",
    ]

    detail_cols = [
        c
        for c in detail_cols
        if c in fineco.columns
    ]

    print(
        fineco[
            detail_cols
        ]
        .to_string(
            index=False
        )
    )

    banner(
        "FINE DIAGNOSI"
    )

    print(
        "Nessuna soglia PHASE introdotta."
    )

    print(
        "Nessun file di produzione modificato."
    )


if __name__ == "__main__":
    main()
    