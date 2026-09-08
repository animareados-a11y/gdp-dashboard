import os

import numpy as np
import pandas as pd

from engine import analyze

from backtest import (
    universe_tickers,
    download_backtest_frames,
    daily_price_series,
    evaluation_positions,
    truncate_frame,
    timeframe_available,
    neutral_breadth,
    future_return,
    future_excursions,
    ticker_market,
)

from ml_dataset import safe_float

# NUOVO FEATURE ENGINE V2.1
from ml_features_v2 import extract_features_v2


# ============================================================
# MARKET SENTINEL
# MACHINE LEARNING DATASET BUILDER V2.1
# ============================================================
#
# OBIETTIVO
#
# Costruire il dataset storico per il Machine Learning usando
# il nuovo ADVANCED FEATURE ENGINE V2.1.
#
# Per ogni data storica:
#
# 1. tronchiamo tutti i dati alla data di osservazione
# 2. Market Sentinel analizza SOLO ciò che era noto allora
# 3. ml_features_v2 traduce il grafico in feature numeriche
# 4. SOLO DOPO guardiamo cosa è successo nel futuro
#
# TARGET PRINCIPALE:
#
# BUY:
#   nelle successive 15 sedute il titolo raggiunge
#   +4% PRIMA di raggiungere -2%
#
# SELL:
#   nelle successive 15 sedute il titolo raggiunge
#   -4% PRIMA di raggiungere +2%
#
# ============================================================


# ============================================================
# CONFIGURAZIONE
# ============================================================

PERIOD_LABEL = "2 anni"

EVENT_HORIZON = 15


# ============================================================
# TARGET BUY
# ============================================================

BUY_PROFIT_TARGET = 4.0
BUY_ADVERSE_LIMIT = -2.0


# ============================================================
# TARGET SELL
# ============================================================

SELL_PROFIT_TARGET = -4.0
SELL_ADVERSE_LIMIT = 2.0


# ============================================================
# TARGET SECONDARI
# ============================================================

RETURN_HORIZONS = [
    5,
    10,
    15,
    20,
]


OUTPUT_FILE = os.path.join(
    "data",
    "ml_dataset_v2.csv"
)


# ============================================================
# DATAFRAME DAILY
# ============================================================

def normalized_daily_frame(frames):

    daily = frames.get("Daily")

    if daily is None or daily.empty:
        return None

    data = daily.copy()

    try:

        data.index = pd.to_datetime(
            data.index,
            utc=True
        )

        data = data.sort_index()

        data = data[
            ~data.index.duplicated(
                keep="last"
            )
        ]

    except Exception:
        return None

    required = [
        "Close",
        "High",
        "Low",
    ]

    for column in required:

        if column not in data.columns:
            return None

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    return data


# ============================================================
# FIRST PASSAGE EVENT
# ============================================================

def first_passage_event(
    daily,
    position,
    horizon,
    profit_threshold_pct,
    adverse_threshold_pct,
    direction
):
    """
    Cerca quale livello viene raggiunto PER PRIMO.

    BUY:
        profit  = +4%
        adverse = -2%

    SELL:
        profit  = -4%
        adverse = +2%

    Ritorna:

        target = 1
            movimento corretto raggiunto per primo

        target = 0
            barriera contraria raggiunta per prima

        target = NaN
            nessuna barriera raggiunta
            oppure caso ambiguo
    """

    if daily is None or daily.empty:

        return {
            "target": np.nan,
            "days_to_event": np.nan,
            "event": "NO_DATA",
        }

    if position >= len(daily):

        return {
            "target": np.nan,
            "days_to_event": np.nan,
            "event": "NO_DATA",
        }

    try:

        start_price = float(
            daily["Close"].iloc[position]
        )

    except Exception:

        return {
            "target": np.nan,
            "days_to_event": np.nan,
            "event": "NO_DATA",
        }

    if (
        not np.isfinite(start_price)
        or start_price <= 0
    ):

        return {
            "target": np.nan,
            "days_to_event": np.nan,
            "event": "NO_DATA",
        }

    final_position = min(
        position + horizon,
        len(daily) - 1
    )

    profit_price = (
        start_price
        *
        (
            1
            +
            profit_threshold_pct / 100
        )
    )

    adverse_price = (
        start_price
        *
        (
            1
            +
            adverse_threshold_pct / 100
        )
    )

    # ========================================================
    # ANALISI DELLE SEDUTE FUTURE
    # ========================================================

    for future_position in range(
        position + 1,
        final_position + 1
    ):

        try:

            high = float(
                daily["High"].iloc[
                    future_position
                ]
            )

            low = float(
                daily["Low"].iloc[
                    future_position
                ]
            )

        except Exception:
            continue

        if (
            not np.isfinite(high)
            or not np.isfinite(low)
        ):
            continue

        days_elapsed = (
            future_position
            -
            position
        )

        # ====================================================
        # BUY
        # ====================================================

        if direction == "buy":

            profit_hit = (
                high >= profit_price
            )

            adverse_hit = (
                low <= adverse_price
            )

            # Entrambi toccati nella stessa seduta:
            # con dati Daily non sappiamo quale sia arrivato prima.

            if profit_hit and adverse_hit:

                return {
                    "target": np.nan,
                    "days_to_event": days_elapsed,
                    "event": "AMBIGUOUS",
                }

            if profit_hit:

                return {
                    "target": 1,
                    "days_to_event": days_elapsed,
                    "event": "BUY_SUCCESS",
                }

            if adverse_hit:

                return {
                    "target": 0,
                    "days_to_event": days_elapsed,
                    "event": "BUY_FAILURE",
                }

        # ====================================================
        # SELL
        # ====================================================

        else:

            profit_hit = (
                low <= profit_price
            )

            adverse_hit = (
                high >= adverse_price
            )

            if profit_hit and adverse_hit:

                return {
                    "target": np.nan,
                    "days_to_event": days_elapsed,
                    "event": "AMBIGUOUS",
                }

            if profit_hit:

                return {
                    "target": 1,
                    "days_to_event": days_elapsed,
                    "event": "SELL_SUCCESS",
                }

            if adverse_hit:

                return {
                    "target": 0,
                    "days_to_event": days_elapsed,
                    "event": "SELL_FAILURE",
                }

    return {
        "target": np.nan,
        "days_to_event": np.nan,
        "event": "NO_EVENT",
    }


# ============================================================
# FUTURE TARGETS
# ============================================================

def add_v2_targets(
    row,
    daily,
    prices,
    position
):

    # ========================================================
    # BUY
    # ========================================================

    buy_result = first_passage_event(
        daily=daily,
        position=position,
        horizon=EVENT_HORIZON,
        profit_threshold_pct=BUY_PROFIT_TARGET,
        adverse_threshold_pct=BUY_ADVERSE_LIMIT,
        direction="buy"
    )

    row["target_buy_v2"] = buy_result["target"]
    row["target_buy_days"] = buy_result["days_to_event"]
    row["target_buy_event"] = buy_result["event"]

    # ========================================================
    # SELL
    # ========================================================

    sell_result = first_passage_event(
        daily=daily,
        position=position,
        horizon=EVENT_HORIZON,
        profit_threshold_pct=SELL_PROFIT_TARGET,
        adverse_threshold_pct=SELL_ADVERSE_LIMIT,
        direction="sell"
    )

    row["target_sell_v2"] = sell_result["target"]
    row["target_sell_days"] = sell_result["days_to_event"]
    row["target_sell_event"] = sell_result["event"]

    # ========================================================
    # RENDIMENTI FUTURI SECONDARI
    # ========================================================

    for horizon in RETURN_HORIZONS:

        future_ret = future_return(
            prices,
            position,
            horizon
        )

        (
            max_up,
            max_down
        ) = future_excursions(
            prices,
            position,
            horizon
        )

        row[
            f"future_return_{horizon}d"
        ] = future_ret

        row[
            f"future_max_up_{horizon}d"
        ] = max_up

        row[
            f"future_max_down_{horizon}d"
        ] = max_down

    return row


# ============================================================
# DATASET DI UN TITOLO
# ============================================================

def build_ticker_dataset_v2(
    ticker,
    period_label=PERIOD_LABEL,
    progress_callback=None
):

    ticker = (
        str(ticker)
        .upper()
        .strip()
    )

    frames = download_backtest_frames(
        ticker,
        progress_callback=None
    )

    prices = daily_price_series(
        frames
    )

    daily = normalized_daily_frame(
        frames
    )

    if (
        prices.empty
        or daily is None
        or daily.empty
    ):

        raise RuntimeError(
            f"Dati Daily insufficienti per {ticker}"
        )

    # ========================================================
    # ALLINEAMENTO
    # ========================================================

    common_index = (
        prices.index
        .intersection(
            daily.index
        )
    )

    prices = prices.loc[
        common_index
    ]

    daily = daily.loc[
        common_index
    ]

    if prices.empty:

        raise RuntimeError(
            f"Nessun dato Daily allineato per {ticker}"
        )

    positions = evaluation_positions(
        prices,
        period_label
    )

    max_future = max(
        max(RETURN_HORIZONS),
        EVENT_HORIZON
    )

    positions = [
        position
        for position in positions
        if (
            position
            +
            max_future
            <
            len(prices)
        )
    ]

    if not positions:

        raise RuntimeError(
            f"Nessuna osservazione valida per {ticker}"
        )

    breadth = neutral_breadth()

    rows = []

    total = len(positions)

    # ========================================================
    # WALK FORWARD
    # ========================================================

    for number, position in enumerate(
        positions,
        start=1
    ):

        cutoff = prices.index[position]

        # ====================================================
        # 1. TRONCHIAMO IL FUTURO
        # ====================================================

        truncated = {}

        for timeframe, dataframe in frames.items():

            cut = truncate_frame(
                dataframe,
                cutoff
            )

            if (
                cut is not None
                and not cut.empty
            ):

                truncated[
                    timeframe
                ] = cut

        if not timeframe_available(
            truncated,
            "Daily"
        ):
            continue

        # ====================================================
        # 2. ANALISI MARKET SENTINEL
        # ====================================================

        try:

            analysis = analyze(
                ticker,
                truncated,
                breadth
            )

        except Exception:
            analysis = None

        if not analysis:
            continue

        # ====================================================
        # 3. FOTOGRAFIA TECNICA
        #
        # QUI È LA MODIFICA FONDAMENTALE:
        # utilizziamo ADVANCED FEATURE ENGINE V2.1
        # ====================================================

        row = {

            "ticker":
                ticker,

            "market":
                ticker_market(
                    ticker
                ),

            "date":
                cutoff,

            "price":
                safe_float(
                    prices.iloc[
                        position
                    ]
                ),
        }

        try:

            features = extract_features_v2(
                analysis
            )

        except Exception as exc:

            print(
                f"\nERRORE FEATURE "
                f"{ticker} "
                f"{cutoff}: "
                f"{exc}"
            )

            continue

        if not features:
            continue

        row.update(
            features
        )

        # ====================================================
        # 4. SOLO ADESSO GUARDIAMO IL FUTURO
        # ====================================================

        row = add_v2_targets(
            row=row,
            daily=daily,
            prices=prices,
            position=position
        )

        rows.append(
            row
        )

        if progress_callback:

            progress_callback(
                number
                /
                max(
                    total,
                    1
                ),
                (
                    f"{ticker} "
                    f"{number}/{total} "
                    f"{cutoff.strftime('%d/%m/%Y')}"
                )
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        raise RuntimeError(
            f"Dataset vuoto per {ticker}"
        )

    return result


# ============================================================
# TUTTO IL PANIERE
# ============================================================

def get_full_universe():

    items = universe_tickers()

    tickers = [
        item["ticker"]
        for item in items
    ]

    tickers = list(
        dict.fromkeys(
            tickers
        )
    )

    return tickers


# ============================================================
# COSTRUZIONE DATASET COMPLETO
# ============================================================

def build_full_dataset_v2(
    tickers=None,
    period_label=PERIOD_LABEL,
    progress_callback=None
):

    if tickers is None:
        tickers = get_full_universe()

    tickers = [
        str(ticker)
        .upper()
        .strip()

        for ticker in tickers

        if str(ticker).strip()
    ]

    tickers = list(
        dict.fromkeys(
            tickers
        )
    )

    if not tickers:

        raise RuntimeError(
            "Paniere vuoto."
        )

    datasets = []
    failures = []

    total_tickers = len(
        tickers
    )

    for ticker_number, ticker in enumerate(
        tickers,
        start=1
    ):

        def current_progress(
            value,
            text
        ):

            if progress_callback is None:
                return

            start = (
                ticker_number - 1
            ) / total_tickers

            width = (
                1.0
                /
                total_tickers
            )

            overall = (
                start
                +
                float(value)
                *
                width
            )

            progress_callback(
                min(
                    overall,
                    0.999
                ),
                (
                    f"{ticker_number}/"
                    f"{total_tickers} — "
                    f"{text}"
                )
            )

        try:

            current = build_ticker_dataset_v2(
                ticker=ticker,
                period_label=period_label,
                progress_callback=current_progress
            )

            datasets.append(
                current
            )

        except Exception as exc:

            failures.append(
                {
                    "Ticker":
                        ticker,

                    "Market":
                        ticker_market(
                            ticker
                        ),

                    "Error":
                        str(
                            exc
                        ),
                }
            )

    if not datasets:

        raise RuntimeError(
            "Nessun dataset prodotto."
        )

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    combined["date"] = pd.to_datetime(
        combined["date"],
        utc=True
    )

    combined = (
        combined
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
        .drop_duplicates(
            subset=[
                "ticker",
                "date",
            ],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    failures_df = pd.DataFrame(
        failures
    )

    if progress_callback:

        progress_callback(
            1.0,
            (
                f"Dataset V2.1 completato: "
                f"{combined['ticker'].nunique()} titoli, "
                f"{len(combined)} osservazioni."
            )
        )

    return (
        combined,
        failures_df
    )


# ============================================================
# SAVE
# ============================================================

def save_dataset_v2(
    dataframe,
    path=OUTPUT_FILE
):

    directory = os.path.dirname(
        path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    dataframe.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# SUMMARY
# ============================================================

def dataset_v2_summary(
    dataframe
):

    if (
        dataframe is None
        or dataframe.empty
    ):
        return {}

    feature_columns = [

        column

        for column in dataframe.columns

        if (
            not column.startswith(
                "target_"
            )
            and
            not column.startswith(
                "future_"
            )
            and
            column
            not in [
                "ticker",
                "market",
                "date",
            ]
        )
    ]

    buy_valid = dataframe[
        "target_buy_v2"
    ].dropna()

    sell_valid = dataframe[
        "target_sell_v2"
    ].dropna()

    return {

        "rows":
            len(dataframe),

        "tickers":
            dataframe[
                "ticker"
            ].nunique(),

        "features":
            len(feature_columns),

        "first_date":
            dataframe[
                "date"
            ].min(),

        "last_date":
            dataframe[
                "date"
            ].max(),

        "buy_valid":
            len(buy_valid),

        "buy_success_rate":
            (
                buy_valid.mean()
                *
                100

                if len(buy_valid)

                else np.nan
            ),

        "sell_valid":
            len(sell_valid),

        "sell_success_rate":
            (
                sell_valid.mean()
                *
                100

                if len(sell_valid)

                else np.nan
            ),
    }


# ============================================================
# TERMINAL
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "============================================"
    )
    print(
        " MARKET SENTINEL"
    )
    print(
        " ML DATASET V2.1"
    )
    print(
        " ADVANCED FEATURE ENGINE V2.1"
    )
    print(
        "============================================"
    )
    print()

    tickers = get_full_universe()

    print(
        f"Titoli nel paniere: "
        f"{len(tickers)}"
    )

    print(
        f"Periodo: "
        f"{PERIOD_LABEL}"
    )

    print()

    print(
        "BUY:"
    )

    print(
        f"  +{BUY_PROFIT_TARGET:.1f}% "
        f"prima di "
        f"{BUY_ADVERSE_LIMIT:.1f}%"
    )

    print()

    print(
        "SELL:"
    )

    print(
        f"  {SELL_PROFIT_TARGET:.1f}% "
        f"prima di "
        f"+{SELL_ADVERSE_LIMIT:.1f}%"
    )

    print()

    print(
        "Feature Engine:"
    )

    print(
        "  ml_features_v2.extract_features_v2"
    )

    print()

    print(
        "Costruzione dataset..."
    )

    print()

    # ========================================================
    # PROGRESS TERMINALE
    # ========================================================

    last_percentage = {
        "value": -1
    }

    def terminal_progress(
        value,
        text
    ):

        percentage = int(
            float(value)
            *
            100
        )

        if (
            percentage
            !=
            last_percentage[
                "value"
            ]
        ):

            last_percentage[
                "value"
            ] = percentage

            print(
                f"[{percentage:3d}%] "
                f"{text}"
            )

    # ========================================================
    # BUILD
    # ========================================================

    (
        dataset,
        failures
    ) = build_full_dataset_v2(
        tickers=tickers,
        period_label=PERIOD_LABEL,
        progress_callback=terminal_progress
    )

    output_path = save_dataset_v2(
        dataset
    )

    summary = dataset_v2_summary(
        dataset
    )

    # ========================================================
    # RISULTATO
    # ========================================================

    print()
    print(
        "============================================"
    )
    print(
        " DATASET V2.1 COMPLETATO"
    )
    print(
        "============================================"
    )
    print()

    print(
        f"Righe:          "
        f"{summary['rows']}"
    )

    print(
        f"Titoli:         "
        f"{summary['tickers']}"
    )

    print(
        f"Features:       "
        f"{summary['features']}"
    )

    print(
        f"Dal:            "
        f"{summary['first_date']}"
    )

    print(
        f"Al:             "
        f"{summary['last_date']}"
    )

    print()

    print(
        f"BUY valutabili: "
        f"{summary['buy_valid']}"
    )

    print(
        f"BUY successi:   "
        f"{summary['buy_success_rate']:.1f}%"
    )

    print()

    print(
        f"SELL valutabili:"
        f" {summary['sell_valid']}"
    )

    print(
        f"SELL successi:  "
        f"{summary['sell_success_rate']:.1f}%"
    )

    print()

    print(
        "File salvato:"
    )

    print(
        output_path
    )

    if (
        failures is not None
        and not failures.empty
    ):

        print()
        print(
            "============================================"
        )
        print(
            " TITOLI NON ELABORATI"
        )
        print(
            "============================================"
        )
        print()

        print(
            failures.to_string(
                index=False
            )
        )

    print()