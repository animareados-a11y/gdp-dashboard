import os

import numpy as np
import pandas as pd

from engine import analyze

from backtest import (
    download_backtest_frames,
    daily_price_series,
    evaluation_positions,
    truncate_frame,
    timeframe_available,
    neutral_breadth,
    future_return,
    future_excursions,
    ticker_market,
    multi_sample_tickers,
)


# ============================================================
# MARKET SENTINEL
# MACHINE LEARNING DATASET BUILDER V1
# ============================================================
#
# SCOPO
#
# Per ogni data storica:
#
# 1. nascondiamo completamente il futuro
# 2. calcoliamo tutti gli indicatori disponibili
# 3. salviamo la fotografia tecnica
# 4. SOLO DOPO aggiungiamo ciò che è successo
#    nelle 5 / 10 / 15 / 20 sedute successive
#
# Questo file NON addestra ancora il Machine Learning.
#
# Costruisce la "scuola" sulla quale
# il Machine Learning imparerà.
#
# ============================================================


# ============================================================
# CONFIGURAZIONE
# ============================================================

ML_HORIZONS = [
    5,
    10,
    15,
    20,
]


ML_TIMEFRAMES = [
    "Monthly",
    "Weekly",
    "Daily",
    "4H",
    "2H",
    "1H",
]


# Movimento minimo che consideriamo
# significativo per classificare
# rialzo / ribasso.

TARGET_THRESHOLD = 2.0


# Massimo numero di barre usato
# per misurare la "freschezza"
# di un segnale.

MAX_EVENT_AGE = 30


DEFAULT_OUTPUT_FILE = os.path.join(
    "data",
    "ml_dataset.csv"
)


# ============================================================
# UTILITY
# ============================================================

def safe_float(
    value,
    default=np.nan
):

    try:

        value = float(
            value
        )

        if np.isfinite(
            value
        ):

            return value

    except Exception:

        pass

    return default


def safe_series_value(
    df,
    column,
    index=-1,
    default=np.nan
):

    try:

        if (
            df is None
            or
            df.empty
            or
            column not in df.columns
        ):

            return default

        value = float(
            df[
                column
            ].iloc[
                index
            ]
        )

        if np.isfinite(
            value
        ):

            return value

    except Exception:

        pass

    return default


def safe_change(
    df,
    column,
    periods=1,
    default=np.nan
):

    try:

        if (
            df is None
            or
            df.empty
            or
            column not in df.columns
            or
            len(
                df
            )
            <=
            periods
        ):

            return default

        current = float(
            df[
                column
            ].iloc[
                -1
            ]
        )

        previous = float(
            df[
                column
            ].iloc[
                -1 - periods
            ]
        )

        if (
            np.isfinite(
                current
            )
            and
            np.isfinite(
                previous
            )
        ):

            return (
                current
                -
                previous
            )

    except Exception:

        pass

    return default


def safe_pct_change(
    df,
    column,
    periods=1,
    default=np.nan
):

    try:

        if (
            df is None
            or
            df.empty
            or
            column not in df.columns
            or
            len(
                df
            )
            <=
            periods
        ):

            return default

        current = float(
            df[
                column
            ].iloc[
                -1
            ]
        )

        previous = float(
            df[
                column
            ].iloc[
                -1 - periods
            ]
        )

        if (
            not np.isfinite(
                current
            )
            or
            not np.isfinite(
                previous
            )
            or
            previous == 0
        ):

            return default

        return (
            (
                current
                /
                previous
            )
            -
            1
        ) * 100

    except Exception:

        return default


# ============================================================
# EVENT AGE
# ============================================================

def bars_since_cross(
    series_a,
    series_b,
    direction="up",
    max_bars=MAX_EVENT_AGE
):
    """
    Quante barre sono trascorse
    dall'ultimo incrocio.

    0 = incrocio appena avvenuto
    1 = una barra fa
    ecc.

    Se non troviamo un incrocio recente:
    restituiamo max_bars + 1.
    """

    try:

        a = pd.to_numeric(
            series_a,
            errors="coerce"
        )

        b = pd.to_numeric(
            series_b,
            errors="coerce"
        )

        valid = pd.DataFrame(
            {
                "a": a,
                "b": b,
            }
        ).dropna()

        if len(
            valid
        ) < 3:

            return max_bars + 1

        start = max(
            1,
            len(
                valid
            )
            -
            max_bars
            -
            1
        )

        for i in range(
            len(
                valid
            )
            -
            1,
            start - 1,
            -1
        ):

            current_a = float(
                valid[
                    "a"
                ].iloc[
                    i
                ]
            )

            current_b = float(
                valid[
                    "b"
                ].iloc[
                    i
                ]
            )

            previous_a = float(
                valid[
                    "a"
                ].iloc[
                    i - 1
                ]
            )

            previous_b = float(
                valid[
                    "b"
                ].iloc[
                    i - 1
                ]
            )

            if direction == "up":

                crossed = (
                    current_a
                    >
                    current_b
                    and
                    previous_a
                    <=
                    previous_b
                )

            else:

                crossed = (
                    current_a
                    <
                    current_b
                    and
                    previous_a
                    >=
                    previous_b
                )

            if crossed:

                return (
                    len(
                        valid
                    )
                    -
                    1
                    -
                    i
                )

    except Exception:

        pass

    return max_bars + 1


def bars_since_sar_flip(
    df,
    direction="bullish",
    max_bars=MAX_EVENT_AGE
):
    """
    Freschezza del cambio Parabolic SAR.

    Questa feature è particolarmente importante
    perché vogliamo distinguere:

    SAR bullish appena girato

    da

    SAR bullish da 20 barre.
    """

    try:

        if (
            df is None
            or
            df.empty
            or
            "Close" not in df.columns
            or
            "SAR" not in df.columns
        ):

            return max_bars + 1

        close = pd.to_numeric(
            df[
                "Close"
            ],
            errors="coerce"
        )

        sar = pd.to_numeric(
            df[
                "SAR"
            ],
            errors="coerce"
        )

        valid = pd.DataFrame(
            {
                "close": close,
                "sar": sar,
            }
        ).dropna()

        if len(
            valid
        ) < 3:

            return max_bars + 1

        bullish = (
            valid[
                "close"
            ]
            >
            valid[
                "sar"
            ]
        )

        start = max(
            1,
            len(
                bullish
            )
            -
            max_bars
            -
            1
        )

        for i in range(
            len(
                bullish
            )
            -
            1,
            start - 1,
            -1
        ):

            current = bool(
                bullish.iloc[
                    i
                ]
            )

            previous = bool(
                bullish.iloc[
                    i - 1
                ]
            )

            if direction == "bullish":

                flipped = (
                    current
                    and
                    not previous
                )

            else:

                flipped = (
                    not current
                    and
                    previous
                )

            if flipped:

                return (
                    len(
                        bullish
                    )
                    -
                    1
                    -
                    i
                )

    except Exception:

        pass

    return max_bars + 1


# ============================================================
# RAW TIMEFRAME FEATURES
# ============================================================

def extract_raw_features(
    data,
    prefix
):

    features = {}

    if (
        data is None
        or
        data.empty
    ):

        return features

    # --------------------------------------------------------
    # PREZZO / HEIKIN ASHI
    # --------------------------------------------------------

    close = safe_series_value(
        data,
        "Close"
    )

    ha_close = safe_series_value(
        data,
        "HA_Close"
    )

    ha_open = safe_series_value(
        data,
        "HA_Open"
    )

    atr = safe_series_value(
        data,
        "ATR"
    )

    sar = safe_series_value(
        data,
        "SAR"
    )

    bb_mid = safe_series_value(
        data,
        "BB_Mid"
    )

    bb_upper = safe_series_value(
        data,
        "BB_Upper"
    )

    bb_lower = safe_series_value(
        data,
        "BB_Lower"
    )

    features[
        f"{prefix}_close"
    ] = close

    features[
        f"{prefix}_price_change_1"
    ] = safe_pct_change(
        data,
        "Close",
        1
    )

    features[
        f"{prefix}_price_change_3"
    ] = safe_pct_change(
        data,
        "Close",
        3
    )

    features[
        f"{prefix}_price_change_5"
    ] = safe_pct_change(
        data,
        "Close",
        5
    )

    features[
        f"{prefix}_ha_green"
    ] = (
        1
        if (
            np.isfinite(
                ha_close
            )
            and
            np.isfinite(
                ha_open
            )
            and
            ha_close > ha_open
        )
        else
        0
    )

    # --------------------------------------------------------
    # PARABOLIC SAR
    # --------------------------------------------------------

    features[
        f"{prefix}_sar_bullish"
    ] = (
        1
        if (
            np.isfinite(
                close
            )
            and
            np.isfinite(
                sar
            )
            and
            close > sar
        )
        else
        0
    )

    if (
        np.isfinite(
            close
        )
        and
        np.isfinite(
            sar
        )
        and
        np.isfinite(
            atr
        )
        and
        atr != 0
    ):

        features[
            f"{prefix}_price_sar_atr"
        ] = (
            close
            -
            sar
        ) / atr

    else:

        features[
            f"{prefix}_price_sar_atr"
        ] = np.nan

    features[
        f"{prefix}_sar_flip_up_age"
    ] = bars_since_sar_flip(
        data,
        "bullish"
    )

    features[
        f"{prefix}_sar_flip_down_age"
    ] = bars_since_sar_flip(
        data,
        "bearish"
    )

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    features[
        f"{prefix}_above_bb_mid"
    ] = (
        1
        if (
            np.isfinite(
                ha_close
            )
            and
            np.isfinite(
                bb_mid
            )
            and
            ha_close > bb_mid
        )
        else
        0
    )

    if (
        np.isfinite(
            ha_close
        )
        and
        np.isfinite(
            bb_mid
        )
        and
        np.isfinite(
            atr
        )
        and
        atr != 0
    ):

        features[
            f"{prefix}_ha_bbmid_atr"
        ] = (
            ha_close
            -
            bb_mid
        ) / atr

    else:

        features[
            f"{prefix}_ha_bbmid_atr"
        ] = np.nan

    if (
        np.isfinite(
            ha_close
        )
        and
        np.isfinite(
            bb_upper
        )
        and
        np.isfinite(
            bb_lower
        )
        and
        bb_upper != bb_lower
    ):

        features[
            f"{prefix}_bollinger_position"
        ] = (
            ha_close
            -
            bb_lower
        ) / (
            bb_upper
            -
            bb_lower
        )

    else:

        features[
            f"{prefix}_bollinger_position"
        ] = np.nan

    if (
        "HA_Close"
        in data.columns
        and
        "BB_Mid"
        in data.columns
    ):

        features[
            f"{prefix}_bb_mid_cross_up_age"
        ] = bars_since_cross(
            data[
                "HA_Close"
            ],
            data[
                "BB_Mid"
            ],
            "up"
        )

        features[
            f"{prefix}_bb_mid_cross_down_age"
        ] = bars_since_cross(
            data[
                "HA_Close"
            ],
            data[
                "BB_Mid"
            ],
            "down"
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd = safe_series_value(
        data,
        "MACD"
    )

    macd_signal = safe_series_value(
        data,
        "MACD_Signal"
    )

    macd_hist = safe_series_value(
        data,
        "MACD_Hist"
    )

    features[
        f"{prefix}_macd"
    ] = macd

    features[
        f"{prefix}_macd_signal"
    ] = macd_signal

    features[
        f"{prefix}_macd_hist"
    ] = macd_hist

    features[
        f"{prefix}_macd_above_signal"
    ] = (
        1
        if (
            np.isfinite(
                macd
            )
            and
            np.isfinite(
                macd_signal
            )
            and
            macd > macd_signal
        )
        else
        0
    )

    features[
        f"{prefix}_macd_hist_change_1"
    ] = safe_change(
        data,
        "MACD_Hist",
        1
    )

    features[
        f"{prefix}_macd_hist_change_3"
    ] = safe_change(
        data,
        "MACD_Hist",
        3
    )

    if (
        "MACD"
        in data.columns
        and
        "MACD_Signal"
        in data.columns
    ):

        features[
            f"{prefix}_macd_cross_up_age"
        ] = bars_since_cross(
            data[
                "MACD"
            ],
            data[
                "MACD_Signal"
            ],
            "up"
        )

        features[
            f"{prefix}_macd_cross_down_age"
        ] = bars_since_cross(
            data[
                "MACD"
            ],
            data[
                "MACD_Signal"
            ],
            "down"
        )

    # --------------------------------------------------------
    # RSI / STOCH RSI
    # --------------------------------------------------------

    features[
        f"{prefix}_rsi"
    ] = safe_series_value(
        data,
        "RSI"
    )

    features[
        f"{prefix}_rsi_change_1"
    ] = safe_change(
        data,
        "RSI",
        1
    )

    features[
        f"{prefix}_rsi_change_3"
    ] = safe_change(
        data,
        "RSI",
        3
    )

    stoch = safe_series_value(
        data,
        "StochRSI"
    )

    stoch_signal = safe_series_value(
        data,
        "StochRSI_Signal"
    )

    features[
        f"{prefix}_stoch_rsi"
    ] = stoch

    features[
        f"{prefix}_stoch_signal"
    ] = stoch_signal

    features[
        f"{prefix}_stoch_above_signal"
    ] = (
        1
        if (
            np.isfinite(
                stoch
            )
            and
            np.isfinite(
                stoch_signal
            )
            and
            stoch > stoch_signal
        )
        else
        0
    )

    if (
        "StochRSI"
        in data.columns
        and
        "StochRSI_Signal"
        in data.columns
    ):

        features[
            f"{prefix}_stoch_cross_up_age"
        ] = bars_since_cross(
            data[
                "StochRSI"
            ],
            data[
                "StochRSI_Signal"
            ],
            "up"
        )

        features[
            f"{prefix}_stoch_cross_down_age"
        ] = bars_since_cross(
            data[
                "StochRSI"
            ],
            data[
                "StochRSI_Signal"
            ],
            "down"
        )

    # --------------------------------------------------------
    # FLOWS
    # --------------------------------------------------------

    features[
        f"{prefix}_chaikin"
    ] = safe_series_value(
        data,
        "Chaikin"
    )

    features[
        f"{prefix}_chaikin_change_1"
    ] = safe_change(
        data,
        "Chaikin",
        1
    )

    features[
        f"{prefix}_chaikin_change_3"
    ] = safe_change(
        data,
        "Chaikin",
        3
    )

    features[
        f"{prefix}_cmf"
    ] = safe_series_value(
        data,
        "CMF"
    )

    features[
        f"{prefix}_cmf_change_3"
    ] = safe_change(
        data,
        "CMF",
        3
    )

    # --------------------------------------------------------
    # AROON
    # --------------------------------------------------------

    features[
        f"{prefix}_aroon_up"
    ] = safe_series_value(
        data,
        "AroonUp"
    )

    features[
        f"{prefix}_aroon_down"
    ] = safe_series_value(
        data,
        "AroonDown"
    )

    aroon_up = features[
        f"{prefix}_aroon_up"
    ]

    aroon_down = features[
        f"{prefix}_aroon_down"
    ]

    if (
        np.isfinite(
            aroon_up
        )
        and
        np.isfinite(
            aroon_down
        )
    ):

        features[
            f"{prefix}_aroon_diff"
        ] = (
            aroon_up
            -
            aroon_down
        )

    else:

        features[
            f"{prefix}_aroon_diff"
        ] = np.nan

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    features[
        f"{prefix}_adx"
    ] = safe_series_value(
        data,
        "ADX"
    )

    features[
        f"{prefix}_di_plus"
    ] = safe_series_value(
        data,
        "DIPlus"
    )

    features[
        f"{prefix}_di_minus"
    ] = safe_series_value(
        data,
        "DIMinus"
    )

    features[
        f"{prefix}_adx_change_3"
    ] = safe_change(
        data,
        "ADX",
        3
    )

    di_plus = features[
        f"{prefix}_di_plus"
    ]

    di_minus = features[
        f"{prefix}_di_minus"
    ]

    if (
        np.isfinite(
            di_plus
        )
        and
        np.isfinite(
            di_minus
        )
    ):

        features[
            f"{prefix}_di_diff"
        ] = (
            di_plus
            -
            di_minus
        )

    else:

        features[
            f"{prefix}_di_diff"
        ] = np.nan

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    features[
        f"{prefix}_volume_osc"
    ] = safe_series_value(
        data,
        "VolumeOsc"
    )

    features[
        f"{prefix}_volume_osc_change_3"
    ] = safe_change(
        data,
        "VolumeOsc",
        3
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    features[
        f"{prefix}_momentum"
    ] = safe_series_value(
        data,
        "Momentum"
    )

    features[
        f"{prefix}_momentum_change_1"
    ] = safe_change(
        data,
        "Momentum",
        1
    )

    features[
        f"{prefix}_momentum_change_3"
    ] = safe_change(
        data,
        "Momentum",
        3
    )

    return features


# ============================================================
# ENGINE FEATURES
# ============================================================

def extract_engine_features(
    analysis
):

    features = {}

    # --------------------------------------------------------
    # SCORE GLOBALI ATTUALI
    #
    # Li conserviamo perché il ML potrà stabilire
    # se hanno valore oppure no.
    # --------------------------------------------------------

    features[
        "engine_buy_score"
    ] = safe_float(
        analysis.get(
            "technical_buy_score"
        )
    )

    features[
        "engine_sell_score"
    ] = safe_float(
        analysis.get(
            "sell_score"
        )
    )

    features[
        "engine_trend_score"
    ] = safe_float(
        analysis.get(
            "trend_score"
        )
    )

    features[
        "engine_entry_score"
    ] = safe_float(
        analysis.get(
            "entry_score"
        )
    )

    features[
        "engine_reversal_up"
    ] = safe_float(
        analysis.get(
            "bullish_reversal"
        )
    )

    features[
        "engine_reversal_down"
    ] = safe_float(
        analysis.get(
            "bearish_risk"
        )
    )

    features[
        "engine_confidence"
    ] = safe_float(
        analysis.get(
            "confidence"
        )
    )

    # --------------------------------------------------------
    # TIMEFRAME
    # --------------------------------------------------------

    frame_results = analysis.get(
        "frames",
        {}
    )

    for timeframe in ML_TIMEFRAMES:

        if timeframe not in frame_results:

            continue

        tf_result = frame_results[
            timeframe
        ]

        prefix = (
            timeframe
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        features[
            f"{prefix}_trend_score"
        ] = safe_float(
            tf_result.get(
                "score"
            )
        )

        features[
            f"{prefix}_entry_score"
        ] = safe_float(
            tf_result.get(
                "entry_score"
            )
        )

        features[
            f"{prefix}_reversal_up"
        ] = safe_float(
            tf_result.get(
                "bullish_reversal"
            )
        )

        features[
            f"{prefix}_reversal_down"
        ] = safe_float(
            tf_result.get(
                "bearish_reversal"
            )
        )

        features[
            f"{prefix}_overextension"
        ] = safe_float(
            tf_result.get(
                "overextension"
            )
        )

        features[
            f"{prefix}_trend_age"
        ] = safe_float(
            tf_result.get(
                "trend_age"
            )
        )

        # ----------------------------------------------------
        # COMPONENT SCORES
        # ----------------------------------------------------

        indicator_scores = (
            tf_result.get(
                "indicators",
                {}
            )
        )

        for (
            indicator_name,
            indicator_value
        ) in indicator_scores.items():

            clean_name = (
                str(
                    indicator_name
                )
                .lower()
                .replace(
                    "/",
                    "_"
                )
                .replace(
                    " ",
                    "_"
                )
            )

            features[
                f"{prefix}_component_{clean_name}"
            ] = safe_float(
                indicator_value
            )

        # ----------------------------------------------------
        # BOLLINGER EVENTS
        # ----------------------------------------------------

        events = tf_result.get(
            "events",
            {}
        )

        for (
            event_name,
            event_value
        ) in events.items():

            features[
                f"{prefix}_event_{event_name}"
            ] = (
                1
                if bool(
                    event_value
                )
                else
                0
            )

        # ----------------------------------------------------
        # DATI RAW DEGLI INDICATORI
        # ----------------------------------------------------

        indicator_data = (
            tf_result.get(
                "data"
            )
        )

        raw_features = (
            extract_raw_features(
                indicator_data,
                prefix
            )
        )

        features.update(
            raw_features
        )

    # --------------------------------------------------------
    # RELAZIONI MULTI-TIMEFRAME
    # --------------------------------------------------------

    weekly = features.get(
        "weekly_trend_score",
        np.nan
    )

    daily = features.get(
        "daily_trend_score",
        np.nan
    )

    h4 = features.get(
        "4h_trend_score",
        np.nan
    )

    h2 = features.get(
        "2h_trend_score",
        np.nan
    )

    h1 = features.get(
        "1h_trend_score",
        np.nan
    )

    if (
        np.isfinite(
            h4
        )
        and
        np.isfinite(
            daily
        )
    ):

        features[
            "alignment_4h_minus_daily"
        ] = (
            h4
            -
            daily
        )

    else:

        features[
            "alignment_4h_minus_daily"
        ] = np.nan

    if (
        np.isfinite(
            daily
        )
        and
        np.isfinite(
            weekly
        )
    ):

        features[
            "alignment_daily_minus_weekly"
        ] = (
            daily
            -
            weekly
        )

    else:

        features[
            "alignment_daily_minus_weekly"
        ] = np.nan

    # 4H anticipa il Daily al rialzo

    features[
        "early_bullish_4h_vs_daily"
    ] = (
        1
        if (
            np.isfinite(
                h4
            )
            and
            np.isfinite(
                daily
            )
            and
            h4 >= 6
            and
            daily < 5.7
        )
        else
        0
    )

    # 4H anticipa il Daily al ribasso

    features[
        "early_bearish_4h_vs_daily"
    ] = (
        1
        if (
            np.isfinite(
                h4
            )
            and
            np.isfinite(
                daily
            )
            and
            h4 <= 4
            and
            daily > 4.3
        )
        else
        0
    )

    # Allineamento forte rialzista

    features[
        "aligned_bullish_weekly_daily_4h"
    ] = (
        1
        if (
            np.isfinite(
                weekly
            )
            and
            np.isfinite(
                daily
            )
            and
            np.isfinite(
                h4
            )
            and
            weekly > 5.5
            and
            daily > 5.5
            and
            h4 > 5.5
        )
        else
        0
    )

    # Allineamento forte ribassista

    features[
        "aligned_bearish_weekly_daily_4h"
    ] = (
        1
        if (
            np.isfinite(
                weekly
            )
            and
            np.isfinite(
                daily
            )
            and
            np.isfinite(
                h4
            )
            and
            weekly < 4.5
            and
            daily < 4.5
            and
            h4 < 4.5
        )
        else
        0
    )

    return features


# ============================================================
# TARGETS
# ============================================================

def add_future_targets(
    row,
    prices,
    position
):
    """
    Questa funzione guarda il futuro.

    IMPORTANTISSIMO:
    viene chiamata SOLO dopo che tutte
    le feature tecniche sono già state calcolate.
    """

    for horizon in ML_HORIZONS:

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
            f"target_return_{horizon}d"
        ] = future_ret

        row[
            f"target_max_up_{horizon}d"
        ] = max_up

        row[
            f"target_max_down_{horizon}d"
        ] = max_down

        # -----------------------------------------------
        # CLASSIFICAZIONE BUY
        # -----------------------------------------------

        if np.isfinite(
            future_ret
        ):

            row[
                f"target_up_{horizon}d"
            ] = (
                1
                if future_ret
                >=
                TARGET_THRESHOLD
                else
                0
            )

            # -------------------------------------------
            # CLASSIFICAZIONE SELL
            # -------------------------------------------

            row[
                f"target_down_{horizon}d"
            ] = (
                1
                if future_ret
                <=
                -TARGET_THRESHOLD
                else
                0
            )

            # -------------------------------------------
            # 3 CLASSI
            #
            # +1 = rialzo
            #  0 = neutro
            # -1 = ribasso
            # -------------------------------------------

            if future_ret >= TARGET_THRESHOLD:

                move_class = 1

            elif future_ret <= -TARGET_THRESHOLD:

                move_class = -1

            else:

                move_class = 0

            row[
                f"target_direction_{horizon}d"
            ] = move_class

        else:

            row[
                f"target_up_{horizon}d"
            ] = np.nan

            row[
                f"target_down_{horizon}d"
            ] = np.nan

            row[
                f"target_direction_{horizon}d"
            ] = np.nan

    return row


# ============================================================
# DATASET SINGOLO TITOLO
# ============================================================

def build_ticker_ml_dataset(
    ticker,
    period_label="2 anni",
    progress_callback=None
):

    ticker = (
        str(
            ticker
        )
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

    if prices.empty:

        raise RuntimeError(
            f"Storico Daily mancante: "
            f"{ticker}"
        )

    positions = evaluation_positions(
        prices,
        period_label
    )

    # Servono anche 20 sedute future.
    positions = [
        position
        for position
        in positions
        if (
            position
            +
            max(
                ML_HORIZONS
            )
            <
            len(
                prices
            )
        )
    ]

    if not positions:

        raise RuntimeError(
            f"Nessuna osservazione ML valida "
            f"per {ticker}."
        )

    breadth = neutral_breadth()

    rows = []

    total = len(
        positions
    )

    for number, position in enumerate(
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

        for (
            timeframe,
            dataframe
        ) in frames.items():

            cut = truncate_frame(
                dataframe,
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
        # 2. ANALISI SOLO CON DATI PASSATI
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
        # 3. COSTRUIAMO LE FEATURE
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

        engine_features = (
            extract_engine_features(
                analysis
            )
        )

        row.update(
            engine_features
        )

        # ====================================================
        # 4. SOLO ORA APRIAMO IL FUTURO
        # ====================================================

        row = add_future_targets(
            row,
            prices,
            position
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
                    f"{ticker} — "
                    f"{number}/{total} — "
                    f"{cutoff.strftime('%d/%m/%Y')}"
                )
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        raise RuntimeError(
            f"Dataset ML vuoto per "
            f"{ticker}."
        )

    return result


# ============================================================
# DATASET MULTI-TITOLO
# ============================================================

def build_ml_dataset(
    tickers=None,
    sample_size=20,
    period_label="2 anni",
    progress_callback=None
):

    if tickers is None:

        tickers = multi_sample_tickers(
            sample_size=
                sample_size
        )

    tickers = list(
        dict.fromkeys(
            [
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
        )
    )

    if not tickers:

        raise RuntimeError(
            "Nessun ticker selezionato."
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

        def ticker_progress(
            value,
            text
        ):

            if progress_callback is None:

                return

            start = (
                ticker_number
                -
                1
            ) / total_tickers

            width = (
                1
                /
                total_tickers
            )

            global_value = (
                start
                +
                float(
                    value
                )
                *
                width
            )

            progress_callback(
                min(
                    global_value,
                    0.999
                ),
                (
                    f"{ticker_number}/"
                    f"{total_tickers} — "
                    f"{text}"
                )
            )

        try:

            current = (
                build_ticker_ml_dataset(
                    ticker,
                    period_label=
                        period_label,
                    progress_callback=
                        ticker_progress
                )
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
            "Nessun dataset ML "
            "è stato prodotto."
        )

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    combined[
        "date"
    ] = pd.to_datetime(
        combined[
            "date"
        ],
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
                f"Dataset ML completato — "
                f"{combined['ticker'].nunique()} titoli — "
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

def save_ml_dataset(
    dataframe,
    path=DEFAULT_OUTPUT_FILE
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
# DATASET SUMMARY
# ============================================================

def dataset_summary(
    dataframe
):

    if (
        dataframe is None
        or
        dataframe.empty
    ):

        return {}

    feature_columns = [
        column
        for column
        in dataframe.columns
        if not column.startswith(
            "target_"
        )
        and
        column not in [
            "ticker",
            "market",
            "date",
        ]
    ]

    target_columns = [
        column
        for column
        in dataframe.columns
        if column.startswith(
            "target_"
        )
    ]

    return {

        "rows":
            len(
                dataframe
            ),

        "tickers":
            dataframe[
                "ticker"
            ].nunique(),

        "features":
            len(
                feature_columns
            ),

        "targets":
            len(
                target_columns
            ),

        "first_date":
            dataframe[
                "date"
            ].min(),

        "last_date":
            dataframe[
                "date"
            ].max(),
    }


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "============================================"
    )
    print(
        " MARKET SENTINEL - ML DATASET BUILDER"
    )
    print(
        "============================================"
    )
    print()

    print(
        "Creo dataset su 20 titoli / 2 anni..."
    )
    print()

    def terminal_progress(
        value,
        text
    ):

        percentage = int(
            float(
                value
            )
            *
            100
        )

        print(
            f"[{percentage:3d}%] {text}"
        )

    dataset, failures = (
        build_ml_dataset(
            sample_size=20,
            period_label="2 anni",
            progress_callback=
                terminal_progress
        )
    )

    output_file = save_ml_dataset(
        dataset
    )

    summary = dataset_summary(
        dataset
    )

    print()
    print(
        "============================================"
    )
    print(
        " DATASET COMPLETATO"
    )
    print(
        "============================================"
    )

    print(
        f"Righe:    "
        f"{summary['rows']}"
    )

    print(
        f"Titoli:   "
        f"{summary['tickers']}"
    )

    print(
        f"Features: "
        f"{summary['features']}"
    )

    print(
        f"Targets:  "
        f"{summary['targets']}"
    )

    print(
        f"Dal:      "
        f"{summary['first_date']}"
    )

    print(
        f"Al:       "
        f"{summary['last_date']}"
    )

    print()
    print(
        f"File salvato in:"
    )
    print(
        output_file
    )

    if (
        failures is not None
        and
        not failures.empty
    ):

        print()
        print(
            "Titoli non elaborati:"
        )

        print(
            failures.to_string(
                index=False
            )
        )

    print()