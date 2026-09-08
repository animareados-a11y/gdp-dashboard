import random

import numpy as np
import pandas as pd

from config import MARKETS

from data import (
    clean_universe,
    download_market_timeframes,
)

from engine import analyze


# ============================================================
# MARKET SENTINEL
# WALK-FORWARD PREDICTIVE BACKTEST V5
#
# SINGLE STOCK + MULTI STOCK
# ============================================================
#
# OBIETTIVO
#
# Per ogni data simulata:
#
# 1. MarketSentinel vede SOLO il passato.
# 2. Calcola BUY / SELL / Trend / Reversal.
# 3. Il futuro viene mantenuto completamente nascosto.
# 4. Solo DOPO la previsione controlliamo cosa succede
#    a 10 e 15 sedute.
#
# Ogni simulazione è indipendente.
#
# La modalità MULTI-TITOLO applica esattamente la stessa
# procedura a molti titoli e poi aggrega tutte le osservazioni.
#
# ============================================================


# ============================================================
# CONFIGURAZIONE
# ============================================================

BACKTEST_PERIODS = {
    "1 anno": 252,
    "2 anni": 504,
    "3 anni": 756,
}


# Almeno un anno iniziale di storico
# prima delle osservazioni fuori campione.
WARMUP_DAILY_BARS = 252


# Nuova fotografia del modello ogni 5 sedute.
STEP_SESSIONS = 5


# Orizzonti predittivi.
FORECAST_HORIZONS = [
    10,
    15,
]


# ============================================================
# VECCHIE SOGLIE OPERATIVE
#
# Restano per compatibilità con eventuali
# parti dell'app, ma NON servono per valutare
# la capacità predittiva principale.
# ============================================================

BUY_DECISION_THRESHOLD = 7.0

SELL_DECISION_THRESHOLD = 6.5

MAX_SELL_FOR_BUY = 5.0

REAL_MOVE_THRESHOLD = 2.0


# ============================================================
# FASCE SCORE
# ============================================================

SCORE_BINS = [
    0,
    4,
    5,
    6,
    7,
    8,
    10.0001,
]

SCORE_LABELS = [
    "0–4",
    "4–5",
    "5–6",
    "6–7",
    "7–8",
    "8–10",
]


# ============================================================
# CAMPIONE MULTI-TITOLO
# ============================================================
#
# Il campione da 20 è volutamente diversificato:
#
# Italia:
# banche, industria, energia, tecnologia
#
# Europa:
# industria, tecnologia, consumer, pharma
#
# USA:
# tech, banche, consumer, healthcare
#
# ============================================================

DEFAULT_MULTI_SAMPLE_20 = [

    # ITALIA
    "UCG.MI",
    "ISP.MI",
    "LDO.MI",
    "PRY.MI",
    "ENI.MI",
    "SPM.MI",
    "STMMI.MI",

    # EUROPA
    "SAP.DE",
    "SIE.DE",
    "ASML.AS",
    "BAYN.DE",
    "OR.PA",
    "SAN.PA",

    # USA
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "JPM",
    "TSLA",
    "MRNA",
]


# ============================================================
# UNIVERSO
# ============================================================

def universe_tickers():
    """
    Restituisce tutti i ticker presenti
    nei tre mercati configurati.

    Elimina eventuali duplicati.
    """

    seen = set()

    items = []

    for market_name, config in MARKETS.items():

        tickers = clean_universe(
            config.get(
                "tickers",
                []
            )
        )

        for ticker in tickers:

            ticker = str(
                ticker
            ).strip()

            if not ticker:
                continue

            if ticker in seen:
                continue

            seen.add(
                ticker
            )

            items.append(
                {
                    "ticker":
                        ticker,

                    "market":
                        market_name,
                }
            )

    return items


def all_tickers():
    """
    Solo ticker.
    """

    return [
        item[
            "ticker"
        ]
        for item
        in universe_tickers()
    ]


def ticker_market(
    ticker
):
    """
    Mercato di appartenenza del ticker.
    """

    ticker = (
        str(
            ticker
        )
        .upper()
        .strip()
    )

    for item in universe_tickers():

        if (
            item[
                "ticker"
            ]
            .upper()
            ==
            ticker
        ):

            return item[
                "market"
            ]

    return "World"


def random_ticker(
    exclude=None
):
    """
    Estrae un titolo casuale dall'universo.
    """

    items = universe_tickers()

    if exclude:

        alternatives = [

            item

            for item
            in items

            if item[
                "ticker"
            ]
            !=
            exclude
        ]

        if alternatives:

            items = alternatives

    if not items:

        return None

    return random.choice(
        items
    )[
        "ticker"
    ]


# ============================================================
# CAMPIONI MULTI-TITOLO
# ============================================================

def multi_sample_tickers(
    sample_size=20,
    random_seed=42
):
    """
    Restituisce il campione da utilizzare
    nel test multi-titolo.

    20:
        usa il campione rappresentativo fisso.

    40:
        parte dal campione fisso da 20
        e aggiunge altri titoli casuali.

    valore >= universo:
        usa tutto l'universo.
    """

    universe = all_tickers()

    if not universe:

        return []

    sample_size = int(
        sample_size
    )

    if sample_size >= len(
        universe
    ):

        return universe.copy()

    # --------------------------------------------------------
    # CAMPIONE 20 FISSO
    # --------------------------------------------------------

    base = [
        ticker
        for ticker
        in DEFAULT_MULTI_SAMPLE_20
        if ticker in universe
    ]

    # Se alcuni ticker non sono presenti
    # nella configurazione, integriamo.
    if len(base) < min(
        20,
        sample_size
    ):

        missing = [
            ticker
            for ticker
            in universe
            if ticker not in base
        ]

        needed = (
            min(
                20,
                sample_size
            )
            -
            len(
                base
            )
        )

        base.extend(
            missing[
                :needed
            ]
        )

    if sample_size <= 20:

        return base[
            :sample_size
        ]

    # --------------------------------------------------------
    # CAMPIONE > 20
    # --------------------------------------------------------

    remaining = [
        ticker
        for ticker
        in universe
        if ticker not in base
    ]

    rng = random.Random(
        random_seed
    )

    rng.shuffle(
        remaining
    )

    extra_needed = (
        sample_size
        -
        len(
            base
        )
    )

    return (
        base
        +
        remaining[
            :extra_needed
        ]
    )


# ============================================================
# BREADTH NEUTRO
# ============================================================

def neutral_breadth():
    """
    Nel backtest storico non utilizziamo
    il breadth corrente.

    Finché non ricostruiremo il breadth
    storico per ciascuna data,
    lo manteniamo neutro a 5.
    """

    return {
        "Monthly": 5.0,
        "Weekly": 5.0,
        "Daily": 5.0,
        "4H": 5.0,
        "2H": 5.0,
        "1H": 5.0,
    }


# ============================================================
# DATA UTILITIES
# ============================================================

def normalize_dataframe_index(
    df
):
    """
    Normalizza indice datetime.
    """

    if (
        df is None
        or
        df.empty
    ):

        return None

    data = df.copy()

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

        return data

    except Exception:

        return None


def truncate_frame(
    df,
    cutoff_timestamp
):
    """
    Elimina completamente qualsiasi dato
    successivo alla data simulata.

    Questa è la protezione fondamentale
    contro il look-ahead bias.
    """

    data = normalize_dataframe_index(
        df
    )

    if (
        data is None
        or
        data.empty
    ):

        return None

    try:

        cutoff = pd.Timestamp(
            cutoff_timestamp
        )

        if cutoff.tzinfo is None:

            cutoff = cutoff.tz_localize(
                "UTC"
            )

        else:

            cutoff = cutoff.tz_convert(
                "UTC"
            )

        cutoff = (
            cutoff.normalize()
            +
            pd.Timedelta(
                days=1
            )
            -
            pd.Timedelta(
                microseconds=1
            )
        )

        data = data.loc[
            data.index
            <=
            cutoff
        ]

    except Exception:

        return None

    if data.empty:

        return None

    return data


def daily_price_series(
    frames
):
    """
    Serie Daily usata per:
    - date delle simulazioni
    - prezzo iniziale
    - rendimenti futuri
    """

    daily = frames.get(
        "Daily"
    )

    daily = normalize_dataframe_index(
        daily
    )

    if (
        daily is None
        or
        daily.empty
        or
        "Close"
        not in daily.columns
    ):

        return pd.Series(
            dtype=float
        )

    prices = pd.to_numeric(
        daily[
            "Close"
        ],
        errors="coerce"
    )

    return prices.dropna()


# ============================================================
# TIMEFRAME COVERAGE
# ============================================================

def timeframe_available(
    frames,
    timeframe,
    minimum_rows=60
):

    df = frames.get(
        timeframe
    )

    return (
        df is not None
        and
        not df.empty
        and
        len(
            df
        )
        >=
        minimum_rows
    )


def coverage_string(
    frames
):

    available = []

    for timeframe in [
        "Monthly",
        "Weekly",
        "Daily",
        "4H",
        "2H",
        "1H",
    ]:

        if timeframe_available(
            frames,
            timeframe
        ):

            available.append(
                timeframe
            )

    return " | ".join(
        available
    )


# ============================================================
# SAFE VALUE
# ============================================================

def safe_analysis_tf_value(
    analysis,
    timeframe,
    key,
    default=np.nan
):

    try:

        value = float(
            analysis[
                "frames"
            ][
                timeframe
            ][
                key
            ]
        )

        if np.isfinite(
            value
        ):

            return value

    except Exception:

        pass

    return default


# ============================================================
# FUTURE RETURN
# ============================================================

def future_return(
    prices,
    current_position,
    horizon
):
    """
    Rendimento esatto dopo N
    vere sedute di Borsa.
    """

    future_position = (
        current_position
        +
        horizon
    )

    if future_position >= len(
        prices
    ):

        return np.nan

    try:

        current_price = float(
            prices.iloc[
                current_position
            ]
        )

        future_price = float(
            prices.iloc[
                future_position
            ]
        )

        if (
            not np.isfinite(
                current_price
            )
            or
            not np.isfinite(
                future_price
            )
            or
            current_price == 0
        ):

            return np.nan

        return (
            (
                future_price
                /
                current_price
            )
            -
            1
        ) * 100

    except Exception:

        return np.nan


# ============================================================
# FUTURE EXCURSIONS
# ============================================================

def future_excursions(
    prices,
    current_position,
    horizon
):
    """
    Misura quanto il titolo è salito/scese
    AL MASSIMO dentro la finestra futura.

    Esempio:
    un BUY può essere corretto anche se
    al giorno 15 il prezzo è tornato indietro,
    ma nel frattempo aveva fatto +10%.
    """

    start = (
        current_position
        +
        1
    )

    end = min(
        current_position
        +
        horizon
        +
        1,
        len(
            prices
        )
    )

    if start >= end:

        return (
            np.nan,
            np.nan
        )

    try:

        current_price = float(
            prices.iloc[
                current_position
            ]
        )

        future_window = (
            prices.iloc[
                start:end
            ]
            .astype(float)
        )

        if (
            current_price == 0
            or
            future_window.empty
        ):

            return (
                np.nan,
                np.nan
            )

        variation = (
            (
                future_window
                /
                current_price
            )
            -
            1
        ) * 100

        return (
            float(
                variation.max()
            ),

            float(
                variation.min()
            ),
        )

    except Exception:

        return (
            np.nan,
            np.nan
        )


# ============================================================
# WALK-FORWARD POSITIONS
# ============================================================

def evaluation_positions(
    prices,
    period_label
):

    total_rows = len(
        prices
    )

    maximum_future = max(
        FORECAST_HORIZONS
    )

    if total_rows <= (
        WARMUP_DAILY_BARS
        +
        maximum_future
    ):

        return []

    requested_bars = (
        BACKTEST_PERIODS.get(
            period_label,
            504
        )
    )

    natural_start = max(
        WARMUP_DAILY_BARS,
        total_rows
        -
        requested_bars
    )

    final_position = (
        total_rows
        -
        maximum_future
    )

    return list(
        range(
            natural_start,
            final_position,
            STEP_SESSIONS
        )
    )


# ============================================================
# DOWNLOAD
# ============================================================

def download_backtest_frames(
    ticker,
    progress_callback=None
):

    if progress_callback:

        progress_callback(
            0.01,
            f"Scarico storico {ticker}..."
        )

    payload = (
        download_market_timeframes(
            [
                ticker
            ],
            progress_callback=None
        )
    )

    frames = payload.get(
        ticker
    )

    if not frames:

        raise RuntimeError(
            f"Nessun dato disponibile "
            f"per {ticker}."
        )

    normalized = {}

    for timeframe, df in frames.items():

        current = (
            normalize_dataframe_index(
                df
            )
        )

        if (
            current is not None
            and
            not current.empty
        ):

            normalized[
                timeframe
            ] = current

    return normalized


# ============================================================
# SINGLE STOCK WALK-FORWARD
# ============================================================

def run_backtest(
    ticker,
    period_label="2 anni",
    progress_callback=None,
):
    """
    Walk-forward su UN singolo titolo.

    Ogni riga = una previsione indipendente.
    """

    ticker = (
        str(
            ticker
        )
        .upper()
        .strip()
    )

    frames = download_backtest_frames(
        ticker,
        progress_callback=
            progress_callback
    )

    prices = daily_price_series(
        frames
    )

    if prices.empty:

        raise RuntimeError(
            f"Manca storico Daily "
            f"per {ticker}."
        )

    positions = evaluation_positions(
        prices,
        period_label
    )

    if not positions:

        raise RuntimeError(
            "Storico insufficiente "
            "per il walk-forward."
        )

    breadth = neutral_breadth()

    rows = []

    total_tests = len(
        positions
    )

    for test_number, position in enumerate(
        positions,
        start=1
    ):

        cutoff = prices.index[
            position
        ]

        # ====================================================
        # 1. NASCONDIAMO IL FUTURO
        # ====================================================

        truncated = {}

        for timeframe, df in frames.items():

            cut = truncate_frame(
                df,
                cutoff
            )

            if (
                cut is not None
                and
                not cut.empty
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
        # 2. ANALISI
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

        try:

            price = float(
                prices.iloc[
                    position
                ]
            )

        except Exception:

            continue

        buy_score = float(
            analysis[
                "technical_buy_score"
            ]
        )

        sell_score = float(
            analysis[
                "sell_score"
            ]
        )

        row = {

            "Ticker":
                ticker,

            "Market":
                ticker_market(
                    ticker
                ),

            "Date":
                cutoff,

            "Price":
                price,

            "Buy Score":
                buy_score,

            "Sell Score":
                sell_score,

            "Trend Score":
                float(
                    analysis[
                        "trend_score"
                    ]
                ),

            "Entry Score":
                float(
                    analysis[
                        "entry_score"
                    ]
                ),

            "Reversal Up":
                float(
                    analysis[
                        "bullish_reversal"
                    ]
                ),

            "Reversal Down":
                float(
                    analysis[
                        "bearish_risk"
                    ]
                ),

            "Confidence":
                float(
                    analysis[
                        "confidence"
                    ]
                ),

            "Entry":
                analysis[
                    "entry_text"
                ],

            "Sell Action":
                analysis[
                    "sell_text"
                ],

            "Phase":
                analysis[
                    "phase"
                ],

            "Weekly":
                safe_analysis_tf_value(
                    analysis,
                    "Weekly",
                    "score"
                ),

            "Daily":
                safe_analysis_tf_value(
                    analysis,
                    "Daily",
                    "score"
                ),

            "4H":
                safe_analysis_tf_value(
                    analysis,
                    "4H",
                    "score"
                ),

            "2H":
                safe_analysis_tf_value(
                    analysis,
                    "2H",
                    "score"
                ),

            "1H":
                safe_analysis_tf_value(
                    analysis,
                    "1H",
                    "score"
                ),

            "TF Coverage":
                coverage_string(
                    truncated
                ),
        }

        # ====================================================
        # 3. APRIAMO IL FUTURO
        # ====================================================

        for horizon in (
            FORECAST_HORIZONS
        ):

            row[
                f"Return {horizon}d %"
            ] = future_return(
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
                f"Max Up {horizon}d %"
            ] = max_up

            row[
                f"Max Down {horizon}d %"
            ] = max_down

        # ====================================================
        # COMPATIBILITÀ VECCHIA UI
        # ====================================================

        decision = model_decision(
            buy_score,
            sell_score
        )

        row[
            "Decision"
        ] = decision

        return_15 = row.get(
            "Return 15d %",
            np.nan
        )

        realized_state = (
            realized_market_state(
                return_15
            )
        )

        row[
            "Realized State"
        ] = realized_state

        row[
            "Correct"
        ] = decision_correct(
            decision,
            realized_state
        )

        row[
            "Quality"
        ] = decision_quality(
            decision,
            row.get(
                "Return 10d %",
                np.nan
            ),
            row.get(
                "Return 15d %",
                np.nan
            ),
            row.get(
                "Max Up 15d %",
                np.nan
            ),
            row.get(
                "Max Down 15d %",
                np.nan
            ),
        )

        rows.append(
            row
        )

        if progress_callback:

            fraction = (
                test_number
                /
                max(
                    total_tests,
                    1
                )
            )

            progress_callback(
                0.05
                +
                fraction
                *
                0.93,

                (
                    f"{ticker}: "
                    f"{test_number}/"
                    f"{total_tests} — "
                    f"{cutoff.strftime('%d/%m/%Y')}"
                )
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        raise RuntimeError(
            "Nessuna simulazione valida."
        )

    result = (
        result
        .sort_values(
            "Date"
        )
        .drop_duplicates(
            subset=[
                "Ticker",
                "Date",
            ],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    if progress_callback:

        progress_callback(
            1.0,
            f"{ticker}: completato."
        )

    return result


# ============================================================
# MULTI STOCK WALK-FORWARD
# ============================================================

def run_multi_backtest(
    tickers=None,
    sample_size=20,
    period_label="2 anni",
    progress_callback=None,
):
    """
    Esegue il medesimo walk-forward
    su MOLTI TITOLI.

    Restituisce:

        combined_df
        ticker_summary
        failures

    combined_df:
        tutte le osservazioni aggregate.

    ticker_summary:
        statistiche sintetiche per titolo.

    failures:
        eventuali titoli non processabili.
    """

    if tickers is None:

        tickers = multi_sample_tickers(
            sample_size=
                sample_size
        )

    tickers = [
        str(
            ticker
        )
        .upper()
        .strip()

        for ticker
        in tickers

        if str(
            ticker
        ).strip()
    ]

    # elimina duplicati mantenendo l'ordine
    tickers = list(
        dict.fromkeys(
            tickers
        )
    )

    if not tickers:

        raise RuntimeError(
            "Nessun ticker disponibile "
            "per il test multi-titolo."
        )

    all_results = []

    failures = []

    total_tickers = len(
        tickers
    )

    for ticker_number, ticker in enumerate(
        tickers,
        start=1
    ):

        if progress_callback:

            base_fraction = (
                (
                    ticker_number - 1
                )
                /
                max(
                    total_tickers,
                    1
                )
            )

            progress_callback(
                base_fraction,
                (
                    f"Titolo "
                    f"{ticker_number}/"
                    f"{total_tickers}: "
                    f"{ticker}"
                )
            )

        # progress interno del singolo titolo
        def single_progress(
            value,
            text
        ):

            if progress_callback is None:
                return

            ticker_start = (
                ticker_number - 1
            ) / max(
                total_tickers,
                1
            )

            ticker_width = (
                1
                /
                max(
                    total_tickers,
                    1
                )
            )

            combined_value = (
                ticker_start
                +
                float(
                    value
                )
                *
                ticker_width
            )

            progress_callback(
                min(
                    combined_value,
                    0.999
                ),
                (
                    f"{ticker_number}/"
                    f"{total_tickers} "
                    f"{text}"
                )
            )

        try:

            current = run_backtest(
                ticker,
                period_label=
                    period_label,
                progress_callback=
                    single_progress
            )

            if (
                current is not None
                and
                not current.empty
            ):

                all_results.append(
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

    if not all_results:

        raise RuntimeError(
            "Nessun titolo ha prodotto "
            "un backtest valido."
        )

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined = (
        combined
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ticker_summary = (
        build_ticker_summary(
            combined
        )
    )

    failure_df = pd.DataFrame(
        failures
    )

    if progress_callback:

        progress_callback(
            1.0,
            (
                f"Multi-backtest completato: "
                f"{combined['Ticker'].nunique()} "
                f"titoli, "
                f"{len(combined)} osservazioni."
            )
        )

    return (
        combined,
        ticker_summary,
        failure_df,
    )


# ============================================================
# TICKER SUMMARY
# ============================================================

def build_ticker_summary(
    df,
    horizon=15
):
    """
    Sintesi predittiva per ogni titolo.
    """

    if (
        df is None
        or
        df.empty
    ):

        return pd.DataFrame()

    return_column = (
        f"Return {horizon}d %"
    )

    rows = []

    for ticker, subset in (
        df.groupby(
            "Ticker"
        )
    ):

        working = subset.dropna(
            subset=[
                "Buy Score",
                "Sell Score",
                return_column,
            ]
        )

        if len(
            working
        ) < 5:

            continue

        try:

            buy_corr = (
                working[
                    "Buy Score"
                ]
                .corr(
                    working[
                        return_column
                    ],
                    method="spearman"
                )
            )

        except Exception:

            buy_corr = np.nan

        try:

            raw_sell_corr = (
                working[
                    "Sell Score"
                ]
                .corr(
                    working[
                        return_column
                    ],
                    method="spearman"
                )
            )

            sell_corr = (
                -raw_sell_corr
                if np.isfinite(
                    raw_sell_corr
                )
                else
                np.nan
            )

        except Exception:

            sell_corr = np.nan

        rows.append(
            {

                "Ticker":
                    ticker,

                "Market":
                    (
                        working[
                            "Market"
                        ].iloc[0]
                        if
                        "Market"
                        in working.columns
                        else
                        ticker_market(
                            ticker
                        )
                    ),

                "Osservazioni":
                    len(
                        working
                    ),

                "BUY Corr":
                    buy_corr,

                "SELL Corr":
                    sell_corr,

                "BUY medio":
                    working[
                        "Buy Score"
                    ].mean(),

                "SELL medio":
                    working[
                        "Sell Score"
                    ].mean(),

                f"Rend. medio {horizon}g %":
                    working[
                        return_column
                    ].mean(),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    result[
        "Predictive Avg"
    ] = (
        result[
            [
                "BUY Corr",
                "SELL Corr",
            ]
        ]
        .mean(
            axis=1
        )
    )

    return (
        result
        .sort_values(
            "Predictive Avg",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SCORE CALIBRATION
# ============================================================

def score_calibration(
    df,
    score_column,
    horizon=15,
    direction="buy",
):
    """
    Divide BUY o SELL in fasce
    e misura ciò che accade dopo.
    """

    return_column = (
        f"Return {horizon}d %"
    )

    max_up_column = (
        f"Max Up {horizon}d %"
    )

    max_down_column = (
        f"Max Down {horizon}d %"
    )

    required = [
        score_column,
        return_column,
        max_up_column,
        max_down_column,
    ]

    working = (
        df[
            required
        ]
        .dropna()
        .copy()
    )

    if working.empty:

        return pd.DataFrame()

    working[
        "Fascia"
    ] = pd.cut(
        working[
            score_column
        ],
        bins=SCORE_BINS,
        labels=SCORE_LABELS,
        right=False,
        include_lowest=True
    )

    rows = []

    for fascia in (
        SCORE_LABELS
    ):

        subset = working[
            working[
                "Fascia"
            ]
            .astype(str)
            ==
            fascia
        ]

        cases = len(
            subset
        )

        if cases == 0:

            rows.append(
                {
                    "Fascia":
                        fascia,

                    "N. casi":
                        0,

                    "Direzione corretta %":
                        np.nan,

                    f"Rend. medio {horizon}g %":
                        np.nan,

                    f"Rend. mediano {horizon}g %":
                        np.nan,

                    f"Max rialzo medio {horizon}g %":
                        np.nan,

                    f"Max ribasso medio {horizon}g %":
                        np.nan,
                }
            )

            continue

        future_returns = (
            subset[
                return_column
            ]
        )

        if direction == "buy":

            correct_rate = (
                future_returns
                >
                0
            ).mean() * 100

        else:

            correct_rate = (
                future_returns
                <
                0
            ).mean() * 100

        rows.append(
            {

                "Fascia":
                    fascia,

                "N. casi":
                    cases,

                "Direzione corretta %":
                    float(
                        correct_rate
                    ),

                f"Rend. medio {horizon}g %":
                    float(
                        future_returns.mean()
                    ),

                f"Rend. mediano {horizon}g %":
                    float(
                        future_returns.median()
                    ),

                f"Max rialzo medio {horizon}g %":
                    float(
                        subset[
                            max_up_column
                        ].mean()
                    ),

                f"Max ribasso medio {horizon}g %":
                    float(
                        subset[
                            max_down_column
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PREDICTIVE CORRELATIONS
# ============================================================

def predictive_correlations(
    df
):
    """
    BUY:
        relazione positiva con rendimento futuro.

    SELL:
        relazione negativa con rendimento futuro,
        quindi invertiamo il segno.
    """

    rows = []

    for horizon in (
        FORECAST_HORIZONS
    ):

        return_column = (
            f"Return {horizon}d %"
        )

        working = (
            df[
                [
                    "Buy Score",
                    "Sell Score",
                    return_column,
                ]
            ]
            .dropna()
        )

        if len(
            working
        ) < 5:

            continue

        try:

            buy_corr = (
                working[
                    "Buy Score"
                ]
                .corr(
                    working[
                        return_column
                    ],
                    method="spearman"
                )
            )

        except Exception:

            buy_corr = np.nan

        try:

            sell_raw_corr = (
                working[
                    "Sell Score"
                ]
                .corr(
                    working[
                        return_column
                    ],
                    method="spearman"
                )
            )

            sell_corr = (
                -sell_raw_corr
                if np.isfinite(
                    sell_raw_corr
                )
                else
                np.nan
            )

        except Exception:

            sell_corr = np.nan

        rows.append(
            {

                "Orizzonte":
                    f"{horizon} sedute",

                "Osservazioni":
                    len(
                        working
                    ),

                "Correlazione BUY":
                    buy_corr,

                "Correlazione SELL":
                    sell_corr,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MARKET SUMMARY
# ============================================================

def market_predictive_summary(
    df,
    horizon=15
):
    """
    Confronta Italia / Europa / USA.
    """

    if (
        df is None
        or
        df.empty
        or
        "Market"
        not in df.columns
    ):

        return pd.DataFrame()

    return_column = (
        f"Return {horizon}d %"
    )

    rows = []

    for market, subset in (
        df.groupby(
            "Market"
        )
    ):

        working = subset.dropna(
            subset=[
                "Buy Score",
                "Sell Score",
                return_column,
            ]
        )

        if len(
            working
        ) < 5:

            continue

        try:

            buy_corr = (
                working[
                    "Buy Score"
                ]
                .corr(
                    working[
                        return_column
                    ],
                    method="spearman"
                )
            )

        except Exception:

            buy_corr = np.nan

        try:

            raw_sell = (
                working[
                    "Sell Score"
                ]
                .corr(
                    working[
                        return_column
                    ],
                    method="spearman"
                )
            )

            sell_corr = (
                -raw_sell
                if np.isfinite(
                    raw_sell
                )
                else
                np.nan
            )

        except Exception:

            sell_corr = np.nan

        rows.append(
            {

                "Mercato":
                    market,

                "Titoli":
                    working[
                        "Ticker"
                    ].nunique(),

                "Osservazioni":
                    len(
                        working
                    ),

                "BUY Corr":
                    buy_corr,

                "SELL Corr":
                    sell_corr,

                f"Rend. medio {horizon}g %":
                    working[
                        return_column
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SCORE EXTREMES
# ============================================================

def score_extremes_analysis(
    df,
    horizon=15
):
    """
    Confronta 25% score più alti
    con 25% score più bassi.
    """

    return_column = (
        f"Return {horizon}d %"
    )

    working = (
        df[
            [
                "Buy Score",
                "Sell Score",
                return_column,
            ]
        ]
        .dropna()
        .copy()
    )

    if len(
        working
    ) < 10:

        return pd.DataFrame()

    rows = []

    # ========================================================
    # BUY
    # ========================================================

    buy_high_limit = (
        working[
            "Buy Score"
        ]
        .quantile(
            0.75
        )
    )

    buy_low_limit = (
        working[
            "Buy Score"
        ]
        .quantile(
            0.25
        )
    )

    buy_high = working[
        working[
            "Buy Score"
        ]
        >=
        buy_high_limit
    ]

    buy_low = working[
        working[
            "Buy Score"
        ]
        <=
        buy_low_limit
    ]

    rows.append(
        {

            "Score":
                "BUY",

            "Gruppo alto - soglia":
                buy_high_limit,

            "Rend. futuro gruppo alto %":
                buy_high[
                    return_column
                ].mean(),

            "% rialzi gruppo alto":
                (
                    buy_high[
                        return_column
                    ]
                    >
                    0
                ).mean()
                *
                100,

            "Gruppo basso - soglia":
                buy_low_limit,

            "Rend. futuro gruppo basso %":
                buy_low[
                    return_column
                ].mean(),

            "% rialzi gruppo basso":
                (
                    buy_low[
                        return_column
                    ]
                    >
                    0
                ).mean()
                *
                100,

            "Differenza rendimento %":
                (
                    buy_high[
                        return_column
                    ].mean()
                    -
                    buy_low[
                        return_column
                    ].mean()
                ),
        }
    )

    # ========================================================
    # SELL
    # ========================================================

    sell_high_limit = (
        working[
            "Sell Score"
        ]
        .quantile(
            0.75
        )
    )

    sell_low_limit = (
        working[
            "Sell Score"
        ]
        .quantile(
            0.25
        )
    )

    sell_high = working[
        working[
            "Sell Score"
        ]
        >=
        sell_high_limit
    ]

    sell_low = working[
        working[
            "Sell Score"
        ]
        <=
        sell_low_limit
    ]

    rows.append(
        {

            "Score":
                "SELL",

            "Gruppo alto - soglia":
                sell_high_limit,

            "Rend. futuro gruppo alto %":
                sell_high[
                    return_column
                ].mean(),

            "% rialzi gruppo alto":
                (
                    sell_high[
                        return_column
                    ]
                    >
                    0
                ).mean()
                *
                100,

            "Gruppo basso - soglia":
                sell_low_limit,

            "Rend. futuro gruppo basso %":
                sell_low[
                    return_column
                ].mean(),

            "% rialzi gruppo basso":
                (
                    sell_low[
                        return_column
                    ]
                    >
                    0
                ).mean()
                *
                100,

            "Differenza rendimento %":
                (
                    sell_low[
                        return_column
                    ].mean()
                    -
                    sell_high[
                        return_column
                    ].mean()
                ),
        }
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PREDICTIVE DIAGNOSIS
# ============================================================

def predictive_diagnosis(
    df,
    horizon=15
):
    """
    Giudizio descrittivo senza creare
    un voto 0-10 artificiale.
    """

    return_column = (
        f"Return {horizon}d %"
    )

    working = (
        df[
            [
                "Buy Score",
                "Sell Score",
                return_column,
            ]
        ]
        .dropna()
        .copy()
    )

    if len(
        working
    ) < 20:

        return {
            "samples":
                len(
                    working
                ),

            "buy_corr":
                np.nan,

            "sell_corr":
                np.nan,

            "buy_label":
                "CAMPIONE INSUFFICIENTE",

            "sell_label":
                "CAMPIONE INSUFFICIENTE",
        }

    try:

        buy_corr = (
            working[
                "Buy Score"
            ]
            .corr(
                working[
                    return_column
                ],
                method="spearman"
            )
        )

    except Exception:

        buy_corr = np.nan

    try:

        sell_raw = (
            working[
                "Sell Score"
            ]
            .corr(
                working[
                    return_column
                ],
                method="spearman"
            )
        )

        sell_corr = (
            -sell_raw
            if np.isfinite(
                sell_raw
            )
            else
            np.nan
        )

    except Exception:

        sell_corr = np.nan

    def label_from_corr(
        value
    ):

        if not np.isfinite(
            value
        ):

            return "NON VALUTABILE"

        if value >= 0.35:

            return "FORTE"

        if value >= 0.20:

            return "INTERESSANTE"

        if value >= 0.10:

            return "DEBOLE MA PRESENTE"

        if value > -0.10:

            return "NESSUNA EVIDENZA"

        return "RELAZIONE INVERSA / PROBLEMATICA"

    return {

        "samples":
            len(
                working
            ),

        "buy_corr":
            buy_corr,

        "sell_corr":
            sell_corr,

        "buy_label":
            label_from_corr(
                buy_corr
            ),

        "sell_label":
            label_from_corr(
                sell_corr
            ),
    }


# ============================================================
# MULTI BACKTEST SUMMARY
# ============================================================

def multi_predictive_report(
    df,
    horizon=15
):
    """
    Pacchetto completo di statistiche
    per il backtest aggregato.

    Restituisce un dizionario pronto
    per essere usato da Streamlit.
    """

    diagnosis = predictive_diagnosis(
        df,
        horizon=
            horizon
    )

    buy_calibration = score_calibration(
        df,
        "Buy Score",
        horizon=
            horizon,
        direction="buy"
    )

    sell_calibration = score_calibration(
        df,
        "Sell Score",
        horizon=
            horizon,
        direction="sell"
    )

    extremes = score_extremes_analysis(
        df,
        horizon=
            horizon
    )

    ticker_summary = build_ticker_summary(
        df,
        horizon=
            horizon
    )

    market_summary = (
        market_predictive_summary(
            df,
            horizon=
                horizon
        )
    )

    correlations = (
        predictive_correlations(
            df
        )
    )

    return {

        "diagnosis":
            diagnosis,

        "buy_calibration":
            buy_calibration,

        "sell_calibration":
            sell_calibration,

        "extremes":
            extremes,

        "ticker_summary":
            ticker_summary,

        "market_summary":
            market_summary,

        "correlations":
            correlations,

        "observations":
            len(
                df
            ),

        "tickers":
            (
                df[
                    "Ticker"
                ].nunique()
                if
                "Ticker"
                in df.columns
                else
                0
            ),
    }


# ============================================================
# ============================================================
# COMPATIBILITÀ CON UI PRECEDENTE
# ============================================================
# ============================================================

def model_decision(
    buy_score,
    sell_score
):

    try:

        buy_score = float(
            buy_score
        )

        sell_score = float(
            sell_score
        )

    except Exception:

        return "FERMO"

    if (
        sell_score
        >=
        SELL_DECISION_THRESHOLD
    ):

        return "ESCI"

    if (
        buy_score
        >=
        BUY_DECISION_THRESHOLD
        and
        sell_score
        <
        MAX_SELL_FOR_BUY
    ):

        return "ENTRA"

    return "FERMO"


def realized_market_state(
    future_return_15
):

    try:

        future_return_15 = float(
            future_return_15
        )

    except Exception:

        return "NON VALUTABILE"

    if not np.isfinite(
        future_return_15
    ):

        return "NON VALUTABILE"

    if (
        future_return_15
        >=
        REAL_MOVE_THRESHOLD
    ):

        return "RIALZO"

    if (
        future_return_15
        <=
        -REAL_MOVE_THRESHOLD
    ):

        return "RIBASSO"

    return "NEUTRO"


def decision_correct(
    decision,
    realized_state
):

    if (
        realized_state
        ==
        "NON VALUTABILE"
    ):

        return np.nan

    if (
        decision == "ENTRA"
        and
        realized_state == "RIALZO"
    ):

        return True

    if (
        decision == "ESCI"
        and
        realized_state == "RIBASSO"
    ):

        return True

    if (
        decision == "FERMO"
        and
        realized_state == "NEUTRO"
    ):

        return True

    return False


def decision_quality(
    decision,
    return_10,
    return_15,
    max_up_15,
    max_down_15
):

    values = [
        return_10,
        return_15,
        max_up_15,
        max_down_15,
    ]

    try:

        values = [
            float(
                value
            )
            for value
            in values
        ]

    except Exception:

        return "NON VALUTABILE"

    if not all(
        np.isfinite(
            value
        )
        for value
        in values
    ):

        return "NON VALUTABILE"

    if decision == "ENTRA":

        if (
            return_15 >= 5
            or
            max_up_15 >= 7
        ):

            return "OTTIMA PREVISIONE"

        if return_15 >= 2:

            return "PREVISIONE CORRETTA"

        if return_15 > -2:

            return "POCO SIGNIFICATIVA"

        return "PREVISIONE ERRATA"

    if decision == "ESCI":

        if (
            return_15 <= -5
            or
            max_down_15 <= -7
        ):

            return "OTTIMA PREVISIONE"

        if return_15 <= -2:

            return "PREVISIONE CORRETTA"

        if return_15 < 2:

            return "POCO SIGNIFICATIVA"

        return "PREVISIONE ERRATA"

    if abs(
        return_15
    ) < 2:

        return "CORRETTO RESTARE FERMI"

    if abs(
        return_15
    ) < 4:

        return "MOVIMENTO MODERATO"

    return "MOVIMENTO NON INTERCETTATO"


def decision_statistics(
    df
):

    rows = []

    for decision in [
        "ENTRA",
        "ESCI",
        "FERMO",
    ]:

        subset = (
            df[
                df[
                    "Decision"
                ]
                ==
                decision
            ]
            .copy()
        )

        evaluable = subset.dropna(
            subset=[
                "Correct"
            ]
        )

        cases = len(
            evaluable
        )

        if cases:

            accuracy = (
                evaluable[
                    "Correct"
                ]
                .astype(bool)
                .mean()
                *
                100
            )

            avg_10 = (
                evaluable[
                    "Return 10d %"
                ]
                .mean()
            )

            avg_15 = (
                evaluable[
                    "Return 15d %"
                ]
                .mean()
            )

            median_15 = (
                evaluable[
                    "Return 15d %"
                ]
                .median()
            )

            max_up = (
                evaluable[
                    "Max Up 15d %"
                ]
                .mean()
            )

            max_down = (
                evaluable[
                    "Max Down 15d %"
                ]
                .mean()
            )

        else:

            accuracy = np.nan
            avg_10 = np.nan
            avg_15 = np.nan
            median_15 = np.nan
            max_up = np.nan
            max_down = np.nan

        rows.append(
            {

                "Decisione":
                    decision,

                "N. casi":
                    cases,

                "Previsioni corrette %":
                    accuracy,

                "Rend. medio 10g %":
                    avg_10,

                "Rend. medio 15g %":
                    avg_15,

                "Rend. mediano 15g %":
                    median_15,

                "Max rialzo medio 15g %":
                    max_up,

                "Max ribasso medio 15g %":
                    max_down,
            }
        )

    return pd.DataFrame(
        rows
    )


def decision_matrix(
    df
):

    if df.empty:

        return pd.DataFrame()

    return pd.crosstab(
        df[
            "Decision"
        ],
        df[
            "Realized State"
        ],
        margins=True
    )


def overall_accuracy(
    df
):

    evaluable = df.dropna(
        subset=[
            "Correct"
        ]
    )

    if evaluable.empty:

        return {
            "accuracy": np.nan,
            "correct": 0,
            "wrong": 0,
            "total": 0,
        }

    correct = int(
        evaluable[
            "Correct"
        ]
        .astype(bool)
        .sum()
    )

    total = len(
        evaluable
    )

    wrong = (
        total
        -
        correct
    )

    accuracy = (
        correct
        /
        total
        *
        100
    )

    return {
        "accuracy":
            accuracy,

        "correct":
            correct,

        "wrong":
            wrong,

        "total":
            total,
    }


def active_signal_accuracy(
    df
):

    active = df[
        df[
            "Decision"
        ]
        .isin(
            [
                "ENTRA",
                "ESCI",
            ]
        )
    ].copy()

    active = active.dropna(
        subset=[
            "Correct"
        ]
    )

    if active.empty:

        return {
            "accuracy": np.nan,
            "signals": 0,
            "correct": 0,
        }

    correct = int(
        active[
            "Correct"
        ]
        .astype(bool)
        .sum()
    )

    signals = len(
        active
    )

    return {
        "accuracy":
            (
                correct
                /
                signals
                *
                100
            ),

        "signals":
            signals,

        "correct":
            correct,
    }


def key_moments(
    df
):

    result = {}

    if (
        df is None
        or
        df.empty
    ):

        return result

    try:

        buy_index = (
            df[
                "Buy Score"
            ]
            .idxmax()
        )

        result[
            "max_buy"
        ] = (
            df.loc[
                buy_index
            ]
            .to_dict()
        )

    except Exception:

        pass

    try:

        sell_index = (
            df[
                "Sell Score"
            ]
            .idxmax()
        )

        result[
            "max_sell"
        ] = (
            df.loc[
                sell_index
            ]
            .to_dict()
        )

    except Exception:

        pass

    return result