import numpy as np

from indicators import add_all, clip


# ============================================================
# MARKET SENTINEL V3.6
# BUY + EARLY EXIT / SELL ENGINE
# ============================================================

COMPONENT_WEIGHTS = {
    "Structure": 0.22,
    "MACD": 0.18,
    "SAR": 0.12,
    "Flows": 0.12,
    "Aroon": 0.10,
    "ADX": 0.08,
    "Volume": 0.08,
    "RSI/Stoch": 0.06,
    "Momentum": 0.04,
}


TREND_TIMEFRAME_WEIGHTS = {
    "Monthly": 0.08,
    "Weekly": 0.27,
    "Daily": 0.35,
    "4H": 0.24,
    "2H": 0.04,
    "1H": 0.02,
}


ENTRY_TIMEFRAME_WEIGHTS = {
    "Monthly": 0.02,
    "Weekly": 0.10,
    "Daily": 0.31,
    "4H": 0.37,
    "2H": 0.12,
    "1H": 0.08,
}


REVERSAL_TIMEFRAME_WEIGHTS = {
    "Monthly": 0.03,
    "Weekly": 0.15,
    "Daily": 0.35,
    "4H": 0.32,
    "2H": 0.10,
    "1H": 0.05,
}


# ============================================================
# UTILITY
# ============================================================

def safe(df, col, index=-2, default=0.0):
    try:
        value = float(df[col].iloc[index])

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def weighted_average(
    result,
    key,
    weights,
    default=5.0
):
    valid = []

    for timeframe, weight in weights.items():

        if timeframe not in result:
            continue

        value = result[
            timeframe
        ].get(
            key
        )

        if value is None:
            continue

        try:
            value = float(
                value
            )

        except Exception:
            continue

        if np.isfinite(
            value
        ):
            valid.append(
                (
                    value,
                    weight
                )
            )

    if not valid:
        return default

    total_weight = sum(
        weight
        for _, weight
        in valid
    )

    if total_weight <= 0:
        return default

    return (
        sum(
            value * weight
            for value, weight
            in valid
        )
        /
        total_weight
    )


def recent_cross_up(
    series_a,
    series_b,
    lookback=3
):
    try:

        if len(series_a) < (
            lookback + 3
        ):
            return False

        a = series_a.iloc[
            -lookback - 2:-1
        ]

        b = series_b.iloc[
            -lookback - 2:-1
        ]

        for i in range(
            1,
            len(a)
        ):

            if (
                a.iloc[i]
                >
                b.iloc[i]
                and
                a.iloc[i - 1]
                <=
                b.iloc[i - 1]
            ):
                return True

    except Exception:
        pass

    return False


def recent_cross_down(
    series_a,
    series_b,
    lookback=3
):
    try:

        if len(series_a) < (
            lookback + 3
        ):
            return False

        a = series_a.iloc[
            -lookback - 2:-1
        ]

        b = series_b.iloc[
            -lookback - 2:-1
        ]

        for i in range(
            1,
            len(a)
        ):

            if (
                a.iloc[i]
                <
                b.iloc[i]
                and
                a.iloc[i - 1]
                >=
                b.iloc[i - 1]
            ):
                return True

    except Exception:
        pass

    return False


def get_tf_value(
    result,
    timeframe,
    key,
    default=5.0
):
    try:
        return float(
            result[
                timeframe
            ][
                key
            ]
        )

    except Exception:
        return default


def get_tf_object(
    result,
    timeframe
):
    try:
        return result[
            timeframe
        ]

    except Exception:
        return None


# ============================================================
# MARKET BREADTH
# ============================================================

def breadth_score(
    up,
    down
):
    total = max(
        up + down,
        1
    )

    ratio = (
        up - down
    ) / total

    return clip(
        5
        +
        ratio * 5
    )


# ============================================================
# INDICATOR SCORES
# ============================================================

def structure_score(df):

    close = safe(
        df,
        "HA_Close"
    )

    ha_open = safe(
        df,
        "HA_Open"
    )

    middle = safe(
        df,
        "BB_Mid"
    )

    upper = safe(
        df,
        "BB_Upper"
    )

    lower = safe(
        df,
        "BB_Lower"
    )

    prev_middle = safe(
        df,
        "BB_Mid",
        -5,
        middle
    )

    atr_value = max(
        safe(
            df,
            "ATR",
            default=1.0
        ),
        1e-9
    )

    score = 5.0

    score += (
        1.8
        if close > middle
        else -1.8
    )

    if close > ha_open:
        score += 1.0

    elif close < ha_open:
        score -= 1.0

    if middle > prev_middle:
        score += 0.8

    elif middle < prev_middle:
        score -= 0.8

    distance = (
        close - middle
    ) / atr_value

    if distance > 0:

        score += min(
            distance * 0.30,
            1.2
        )

    else:

        score -= min(
            abs(distance)
            * 0.30,
            1.2
        )

    if close > upper:
        score -= 0.5

    if close < lower:
        score += 0.4

    return clip(
        score
    )


def sar_score(df):

    close = safe(
        df,
        "Close"
    )

    sar = safe(
        df,
        "SAR"
    )

    atr_value = max(
        safe(
            df,
            "ATR",
            default=1.0
        ),
        1e-9
    )

    distance = (
        close - sar
    ) / atr_value

    if close > sar:

        return clip(
            7.2
            +
            min(
                max(
                    distance,
                    0
                )
                * 0.45,
                2.0
            )
        )

    return clip(
        2.8
        -
        min(
            max(
                -distance,
                0
            )
            * 0.45,
            2.0
        )
    )


def macd_score(df):

    macd = safe(
        df,
        "MACD"
    )

    signal = safe(
        df,
        "MACD_Signal"
    )

    hist = safe(
        df,
        "MACD_Hist"
    )

    hist_prev = safe(
        df,
        "MACD_Hist",
        -3
    )

    hist_prev2 = safe(
        df,
        "MACD_Hist",
        -4
    )

    score = 5.0

    score += (
        1.7
        if macd > signal
        else -1.7
    )

    score += (
        1.0
        if hist > hist_prev
        else -1.0
    )

    if hist > hist_prev > hist_prev2:
        score += 0.5

    cross_up = recent_cross_up(
        df["MACD"],
        df["MACD_Signal"],
        3
    )

    cross_down = recent_cross_down(
        df["MACD"],
        df["MACD_Signal"],
        3
    )

    if cross_up and macd < 0:

        score += 1.5

    elif cross_up:

        score += 0.7

    if cross_down and macd > 0:

        score -= 1.5

    elif cross_down:

        score -= 0.7

    return clip(
        score
    )


def flow_score(df):

    chaikin = safe(
        df,
        "Chaikin"
    )

    chaikin_prev = safe(
        df,
        "Chaikin",
        -4,
        chaikin
    )

    cmf = safe(
        df,
        "CMF"
    )

    score = 5.0

    score += (
        0.8
        if chaikin > 0
        else -0.8
    )

    score += (
        1.3
        if chaikin > chaikin_prev
        else -1.3
    )

    score += float(
        np.clip(
            cmf * 9,
            -1.8,
            1.8
        )
    )

    return clip(
        score
    )


def aroon_score(df):

    up = safe(
        df,
        "AroonUp",
        default=50
    )

    down = safe(
        df,
        "AroonDown",
        default=50
    )

    return clip(
        5
        +
        (
            (
                up - down
            )
            /
            100
        )
        * 5
    )


def adx_score(df):

    adx_value = safe(
        df,
        "ADX"
    )

    plus = safe(
        df,
        "DIPlus"
    )

    minus = safe(
        df,
        "DIMinus"
    )

    if adx_value < 15:
        return 5.0

    total = max(
        plus + minus,
        1e-9
    )

    direction = (
        plus - minus
    ) / total

    strength = min(
        adx_value / 40,
        1
    )

    return clip(
        5
        +
        direction
        *
        strength
        *
        5
    )


def volume_score(df):

    volume_osc = safe(
        df,
        "VolumeOsc"
    )

    volume_prev = safe(
        df,
        "VolumeOsc",
        -4
    )

    close = safe(
        df,
        "Close"
    )

    close_prev = safe(
        df,
        "Close",
        -6,
        close
    )

    price_direction = np.sign(
        close - close_prev
    )

    score = 5.0

    if price_direction > 0:

        score += (
            1.4
            if volume_osc > 0
            else -0.7
        )

    elif price_direction < 0:

        score += (
            -1.4
            if volume_osc > 0
            else 0.4
        )

    score += (
        0.6
        if volume_osc > volume_prev
        else -0.6
    )

    return clip(
        score
    )


def rsi_stoch_score(df):

    rsi = safe(
        df,
        "RSI",
        default=50
    )

    stoch = safe(
        df,
        "StochRSI",
        default=50
    )

    stoch_signal = safe(
        df,
        "StochRSI_Signal",
        default=50
    )

    score = 5.0

    if 50 <= rsi <= 67:

        score += 1.0

    elif 40 <= rsi < 50:

        score += 0.2

    elif 30 <= rsi < 40:

        score -= 0.4

    elif rsi < 30:

        score += 0.5

    elif 67 < rsi <= 75:

        score += 0.4

    elif rsi > 75:

        score -= 0.8

    if stoch > stoch_signal:

        score += 0.8

    else:

        score -= 0.8

    if stoch < 20:
        score += 0.6

    if stoch > 90:
        score -= 0.4

    return clip(
        score
    )


def momentum_score(df):

    close = safe(
        df,
        "Close"
    )

    momentum = safe(
        df,
        "Momentum"
    )

    momentum_prev = safe(
        df,
        "Momentum",
        -4
    )

    score = 5.0

    if momentum > 0:

        score += 1.0

    else:

        score -= 1.0

    if momentum > momentum_prev:

        score += 0.8

    else:

        score -= 0.8

    if close > safe(
        df,
        "BB_Mid"
    ):
        score += 0.3

    return clip(
        score
    )


# ============================================================
# BOLLINGER EVENTS
# ============================================================

def bollinger_events(df):

    close = safe(
        df,
        "HA_Close"
    )

    previous_close = safe(
        df,
        "HA_Close",
        -3,
        close
    )

    middle = safe(
        df,
        "BB_Mid"
    )

    previous_middle = safe(
        df,
        "BB_Mid",
        -3,
        middle
    )

    lower = safe(
        df,
        "BB_Lower"
    )

    upper = safe(
        df,
        "BB_Upper"
    )

    atr_value = max(
        safe(
            df,
            "ATR",
            default=1
        ),
        1e-9
    )

    lower_distance = (
        close - lower
    ) / atr_value

    upper_distance = (
        upper - close
    ) / atr_value

    return {

        "middle_break_up":
            (
                close > middle
                and
                previous_close
                <=
                previous_middle
            ),

        "middle_break_down":
            (
                close < middle
                and
                previous_close
                >=
                previous_middle
            ),

        "detach_lower":
            (
                previous_close <= lower
                and
                close > lower
                and
                lower_distance > 0.20
            ),

        "detach_upper":
            (
                previous_close >= upper
                and
                close < upper
                and
                upper_distance > 0.20
            ),
    }


# ============================================================
# TREND AGE
# ============================================================

def trend_age(df):

    try:

        close = df[
            "Close"
        ].iloc[:-1]

        sar = df[
            "SAR"
        ].iloc[:-1]

        bullish = (
            close > sar
        )

        current = bool(
            bullish.iloc[-1]
        )

        age = 0

        for value in reversed(
            bullish.tolist()
        ):

            if bool(value) == current:

                age += 1

            else:

                break

        return age

    except Exception:

        return 0


# ============================================================
# OVEREXTENSION
# ============================================================

def overextension_score(df):

    close = safe(
        df,
        "Close"
    )

    upper = safe(
        df,
        "BB_Upper"
    )

    middle = safe(
        df,
        "BB_Mid"
    )

    atr_value = max(
        safe(
            df,
            "ATR",
            default=1
        ),
        1e-9
    )

    rsi = safe(
        df,
        "RSI",
        default=50
    )

    stoch = safe(
        df,
        "StochRSI",
        default=50
    )

    score = 0.0

    if close > upper:

        score += 2.2

    distance = (
        close - middle
    ) / atr_value

    if distance > 2:

        score += min(
            (
                distance - 2
            )
            * 0.8,
            3
        )

    if rsi > 72:

        score += (
            rsi - 72
        ) * 0.12

    if stoch > 90:

        score += 1

    return clip(
        score
    )


# ============================================================
# REVERSAL
# ============================================================

def reversal_scores(
    df,
    events
):

    bullish = 0.0
    bearish = 0.0

    bullish_reasons = []
    bearish_reasons = []

    close = safe(
        df,
        "Close"
    )

    previous_close = safe(
        df,
        "Close",
        -3,
        close
    )

    rsi = safe(
        df,
        "RSI",
        default=50
    )

    rsi_previous = safe(
        df,
        "RSI",
        -4,
        rsi
    )

    stoch = safe(
        df,
        "StochRSI",
        default=50
    )

    stoch_signal = safe(
        df,
        "StochRSI_Signal",
        default=50
    )

    stoch_previous = safe(
        df,
        "StochRSI",
        -4,
        stoch
    )

    stoch_signal_previous = safe(
        df,
        "StochRSI_Signal",
        -4,
        stoch_signal
    )

    macd = safe(
        df,
        "MACD"
    )

    hist = safe(
        df,
        "MACD_Hist"
    )

    hist_previous = safe(
        df,
        "MACD_Hist",
        -4,
        hist
    )

    chaikin = safe(
        df,
        "Chaikin"
    )

    chaikin_previous = safe(
        df,
        "Chaikin",
        -5,
        chaikin
    )

    cmf = safe(
        df,
        "CMF"
    )

    sar = safe(
        df,
        "SAR"
    )

    previous_sar = safe(
        df,
        "SAR",
        -3,
        sar
    )

    # --------------------------------------------------------
    # BULLISH
    # --------------------------------------------------------

    if events[
        "detach_lower"
    ]:

        bullish += 1.4

        bullish_reasons.append(
            (
                "MEDIO",
                "distacco dalla Bollinger inferiore"
            )
        )

    if events[
        "middle_break_up"
    ]:

        bullish += 1.3

        bullish_reasons.append(
            (
                "FORTE",
                "recupero della banda centrale"
            )
        )

    if recent_cross_up(
        df["MACD"],
        df["MACD_Signal"],
        3
    ):

        if macd < 0:

            bullish += 1.8

            bullish_reasons.append(
                (
                    "FORTE",
                    "MACD bullish cross sotto zero"
                )
            )

        else:

            bullish += 1

            bullish_reasons.append(
                (
                    "MEDIO",
                    "MACD bullish cross"
                )
            )

    if (
        hist > hist_previous
        and
        hist_previous < 0
    ):

        bullish += 0.7

        bullish_reasons.append(
            (
                "DEBOLE",
                "istogramma MACD in recupero"
            )
        )

    if (
        rsi < 40
        and
        rsi > rsi_previous
    ):

        bullish += 1

        bullish_reasons.append(
            (
                "MEDIO",
                "RSI risale da zona debole"
            )
        )

    if (
        stoch < 25
        and
        stoch > stoch_signal
        and
        stoch_previous
        <=
        stoch_signal_previous
    ):

        bullish += 1.3

        bullish_reasons.append(
            (
                "FORTE",
                "Stoch RSI gira dall'ipervenduto"
            )
        )

    if (
        chaikin > chaikin_previous
        and
        chaikin_previous < 0
    ):

        bullish += 0.8

        bullish_reasons.append(
            (
                "MEDIO",
                "flussi in miglioramento"
            )
        )

    if cmf > 0:

        bullish += min(
            cmf * 3,
            0.7
        )

    if (
        close > sar
        and
        previous_close
        <=
        previous_sar
    ):

        bullish += 1.5

        bullish_reasons.append(
            (
                "FORTE",
                "SAR appena girato rialzista"
            )
        )

    # --------------------------------------------------------
    # BEARISH
    # --------------------------------------------------------

    if events[
        "detach_upper"
    ]:

        bearish += 1.4

        bearish_reasons.append(
            (
                "MEDIO",
                "distacco dalla Bollinger superiore"
            )
        )

    if events[
        "middle_break_down"
    ]:

        bearish += 1.3

        bearish_reasons.append(
            (
                "FORTE",
                "rottura della banda centrale"
            )
        )

    if recent_cross_down(
        df["MACD"],
        df["MACD_Signal"],
        3
    ):

        if macd > 0:

            bearish += 1.8

            bearish_reasons.append(
                (
                    "FORTE",
                    "MACD bearish cross sopra zero"
                )
            )

        else:

            bearish += 1

            bearish_reasons.append(
                (
                    "MEDIO",
                    "MACD bearish cross"
                )
            )

    if (
        hist < hist_previous
        and
        hist_previous > 0
    ):

        bearish += 0.7

        bearish_reasons.append(
            (
                "DEBOLE",
                "istogramma MACD in deterioramento"
            )
        )

    if (
        rsi > 70
        and
        rsi < rsi_previous
    ):

        bearish += 1

        bearish_reasons.append(
            (
                "MEDIO",
                "RSI gira dall'ipercomprato"
            )
        )

    if (
        stoch > 80
        and
        stoch < stoch_signal
        and
        stoch_previous
        >=
        stoch_signal_previous
    ):

        bearish += 1.3

        bearish_reasons.append(
            (
                "FORTE",
                "Stoch RSI gira dall'ipercomprato"
            )
        )

    if (
        chaikin < chaikin_previous
        and
        chaikin_previous > 0
    ):

        bearish += 0.8

        bearish_reasons.append(
            (
                "MEDIO",
                "flussi in deterioramento"
            )
        )

    if cmf < 0:

        bearish += min(
            abs(cmf) * 3,
            0.7
        )

    if (
        close < sar
        and
        previous_close
        >=
        previous_sar
    ):

        bearish += 1.5

        bearish_reasons.append(
            (
                "FORTE",
                "SAR appena girato ribassista"
            )
        )

    return (
        clip(
            bullish
        ),

        clip(
            bearish
        ),

        bullish_reasons,

        bearish_reasons,
    )


# ============================================================
# ENTRY
# ============================================================

def entry_score(
    trend_score_value,
    bullish_reversal,
    bearish_reversal,
    age,
    overextension,
    events,
):

    score = 5.0

    score += (
        trend_score_value - 5
    ) * 0.55

    score += (
        bullish_reversal - 4
    ) * 0.30

    score -= max(
        bearish_reversal - 3.5,
        0
    ) * 0.40

    if age <= 4:

        score += 1.3

    elif age <= 9:

        score += 0.7

    elif age >= 18:

        score -= min(
            (
                age - 15
            )
            * 0.08,
            2.2
        )

    score -= (
        overextension
        * 0.35
    )

    if events[
        "middle_break_up"
    ]:

        score += 0.8

    if events[
        "middle_break_down"
    ]:

        score -= 0.9

    return clip(
        score
    )


def entry_label(
    entry_numeric,
    trend_score_value,
    phase
):

    if entry_numeric >= 8:

        return (
            "OTTIMO MOMENTO PER ENTRARE"
        )

    if entry_numeric >= 6.8:

        return (
            "BUON MOMENTO PER ENTRARE"
        )

    if entry_numeric >= 5.6:

        return (
            "INGRESSO DISCRETO"
        )

    if entry_numeric >= 4.5:

        return (
            "MEGLIO ATTENDERE"
        )

    if (
        trend_score_value >= 6
        or
        "MATURE" in phase
    ):

        return (
            "TARDI PER ENTRARE"
        )

    return (
        "PESSIMO MOMENTO PER ENTRARE"
    )


# ============================================================
# PHASE
# ============================================================

def trend_phase(
    trend_score_value,
    entry,
    bullish_reversal,
    bearish_reversal,
    age,
    overextension,
):

    if (
        bullish_reversal >= 7
        and
        trend_score_value < 6
    ):

        return (
            "EARLY REVERSAL ↑"
        )

    if (
        bearish_reversal >= 7
        and
        trend_score_value > 5
    ):

        return (
            "EXHAUSTION ↓"
        )

    if (
        4.5
        <=
        trend_score_value
        <=
        5.5
    ):

        return (
            "SIDEWAYS"
        )

    if trend_score_value > 5.5:

        if (
            age <= 5
            and
            entry >= 6.3
        ):

            return (
                "EARLY TREND ↑"
            )

        if age <= 15:

            return (
                "DEVELOPING ↑"
            )

        if (
            overextension >= 5.5
            or
            bearish_reversal >= 5.5
        ):

            return (
                "MATURE / WATCH ↑"
            )

        return (
            "MATURE ↑"
        )

    if trend_score_value < 4.5:

        if age <= 5:

            return (
                "EARLY TREND ↓"
            )

        if age <= 15:

            return (
                "DEVELOPING ↓"
            )

        if bullish_reversal >= 5.5:

            return (
                "MATURE / WATCH ↓"
            )

        return (
            "MATURE ↓"
        )

    return (
        "NEUTRAL"
    )


# ============================================================
# SINGLE TIMEFRAME
# ============================================================

def timeframe_scores(
    df,
    breadth=5.0
):

    if (
        df is None
        or
        len(df) < 60
    ):

        return None

    data = add_all(
        df
    )

    events = bollinger_events(
        data
    )

    indicator_scores = {

        "Structure":
            structure_score(
                data
            ),

        "MACD":
            macd_score(
                data
            ),

        "SAR":
            sar_score(
                data
            ),

        "Flows":
            flow_score(
                data
            ),

        "Aroon":
            aroon_score(
                data
            ),

        "ADX":
            adx_score(
                data
            ),

        "Volume":
            volume_score(
                data
            ),

        "RSI/Stoch":
            rsi_stoch_score(
                data
            ),

        "Momentum":
            momentum_score(
                data
            ),
    }

    trend = sum(
        indicator_scores[key]
        *
        COMPONENT_WEIGHTS[key]
        for key
        in COMPONENT_WEIGHTS
    )

    trend += (
        breadth - 5
    ) * 0.08

    trend = clip(
        trend
    )

    (
        bullish_reversal,
        bearish_reversal,
        bullish_reasons,
        bearish_reasons,
    ) = reversal_scores(
        data,
        events
    )

    age = trend_age(
        data
    )

    overextension = (
        overextension_score(
            data
        )
    )

    entry = entry_score(
        trend,
        bullish_reversal,
        bearish_reversal,
        age,
        overextension,
        events,
    )

    phase = trend_phase(
        trend,
        entry,
        bullish_reversal,
        bearish_reversal,
        age,
        overextension,
    )

    return {

        "score":
            round(
                trend,
                2
            ),

        "entry_score":
            round(
                entry,
                2
            ),

        "bullish_reversal":
            round(
                bullish_reversal,
                2
            ),

        "bearish_reversal":
            round(
                bearish_reversal,
                2
            ),

        "overextension":
            round(
                overextension,
                2
            ),

        "trend_age":
            int(
                age
            ),

        "phase":
            phase,

        "indicators":
            indicator_scores,

        "events":
            events,

        "data":
            data,

        "bullish_reasons":
            bullish_reasons,

        "bearish_reasons":
            bearish_reasons,
    }


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_score(
    result,
    global_trend,
    bullish_reversal,
    bearish_reversal
):

    values = []
    weights = []

    for timeframe, weight in (
        TREND_TIMEFRAME_WEIGHTS.items()
    ):

        if timeframe not in result:
            continue

        values.append(
            result[
                timeframe
            ][
                "score"
            ]
        )

        weights.append(
            weight
        )

    if len(values) <= 1:

        base = 6.0

    else:

        values = np.array(
            values,
            dtype=float
        )

        weights = np.array(
            weights,
            dtype=float
        )

        weights /= (
            weights.sum()
        )

        mean = np.sum(
            values
            *
            weights
        )

        variance = np.sum(
            weights
            *
            (
                values - mean
            )
            ** 2
        )

        std = np.sqrt(
            variance
        )

        base = (
            10
            -
            std * 1.45
        )

    contradiction = (
        bearish_reversal
        if global_trend >= 5
        else bullish_reversal
    )

    base -= max(
        contradiction - 5,
        0
    ) * 0.30

    return clip(
        base
    )


# ============================================================
# GLOBAL PHASE
# ============================================================

def global_phase(
    result,
    global_trend,
    entry,
    rev_up,
    rev_down
):

    weekly = get_tf_value(
        result,
        "Weekly",
        "score"
    )

    daily = get_tf_value(
        result,
        "Daily",
        "score"
    )

    h4 = get_tf_value(
        result,
        "4H",
        "score"
    )

    daily_rev_up = get_tf_value(
        result,
        "Daily",
        "bullish_reversal",
        0
    )

    daily_rev_down = get_tf_value(
        result,
        "Daily",
        "bearish_reversal",
        0
    )

    h4_rev_down = get_tf_value(
        result,
        "4H",
        "bearish_reversal",
        0
    )

    if (
        daily_rev_up >= 6.5
        and
        h4 >= 5.8
        and
        weekly < 5.5
    ):

        return (
            "EARLY REVERSAL ↑"
        )

    if (
        daily >= 5.8
        and
        h4 >= 6.3
        and
        entry >= 6.2
        and
        weekly < 6
    ):

        return (
            "EARLY TREND ↑"
        )

    if (
        rev_down >= 6.5
        or
        (
            daily_rev_down >= 6
            and
            h4_rev_down >= 5
        )
    ):

        if global_trend >= 5.4:

            return (
                "EXHAUSTION / WATCH ↓"
            )

    if global_trend >= 7.2:

        if entry >= 6.5:

            return (
                "STRONG TREND ↑"
            )

        return (
            "MATURE TREND ↑"
        )

    if global_trend >= 5.6:

        if entry >= 6.2:

            return (
                "DEVELOPING TREND ↑"
            )

        return (
            "TREND ↑"
        )

    if (
        4.5
        <=
        global_trend
        <=
        5.5
    ):

        if rev_up >= 6:

            return (
                "WATCH REVERSAL ↑"
            )

        if rev_down >= 6:

            return (
                "WATCH REVERSAL ↓"
            )

        return (
            "SIDEWAYS"
        )

    if global_trend < 4.5:

        if rev_up >= 6.5:

            return (
                "POSSIBLE BOTTOM ↑"
            )

        return (
            "TREND ↓"
        )

    return (
        "NEUTRAL"
    )


# ============================================================
# TECHNICAL BUY SCORE
# ============================================================

def calculate_buy_score(
    trend_score_value,
    entry_numeric,
    rev_up,
    rev_down,
    confidence,
    overextension,
    phase,
    result,
):

    score = (

        trend_score_value
        * 0.30

        +

        entry_numeric
        * 0.38

        +

        rev_up
        * 0.14

        +

        confidence
        * 0.12

        +

        5.0
        * 0.06
    )

    daily = get_tf_value(
        result,
        "Daily",
        "score"
    )

    h4 = get_tf_value(
        result,
        "4H",
        "score"
    )

    weekly = get_tf_value(
        result,
        "Weekly",
        "score"
    )

    daily_rev_up = get_tf_value(
        result,
        "Daily",
        "bullish_reversal",
        0
    )

    if (
        h4 >= 6.5
        and
        daily >= 5.5
        and
        daily_rev_up >= 5.5
    ):

        score += 0.55

    if (
        "EARLY REVERSAL ↑"
        in phase
    ):

        score += 0.55

    elif (
        "EARLY TREND ↑"
        in phase
    ):

        score += 0.50

    elif (
        "DEVELOPING TREND ↑"
        in phase
    ):

        score += 0.25

    if (
        weekly >= 6.5
        and
        daily >= 6.5
        and
        h4 >= 6.5
    ):

        score += 0.35

    score -= max(
        rev_down - 3.5,
        0
    ) * 0.32

    score -= max(
        overextension - 3,
        0
    ) * 0.18

    if (
        "MATURE TREND"
        in phase
    ):

        score -= 0.65

    if (
        "EXHAUSTION"
        in phase
    ):

        score -= 1.30

    if (
        "TREND ↓"
        in phase
    ):

        score -= 1.15

    if (
        "SIDEWAYS"
        in phase
    ):

        score -= 0.35

    if entry_numeric < 4.0:

        score = min(
            score,
            5.6
        )

    elif entry_numeric < 4.5:

        score = min(
            score,
            6.0
        )

    elif entry_numeric < 5.6:

        score = min(
            score,
            6.7
        )

    elif entry_numeric < 6.8:

        score = min(
            score,
            7.6
        )

    if rev_down >= 7:

        score = min(
            score,
            5.5
        )

    return clip(
        score
    )


# ============================================================
# SELL ENGINE HELPERS
# ============================================================

def bearish_event_strength(
    result,
    timeframe
):
    """
    Quanto sono recenti e significativi gli eventi
    ribassisti sul timeframe.

    È volutamente event-driven:
    vogliamo premiare l'INIZIO del deterioramento.
    """

    obj = get_tf_object(
        result,
        timeframe
    )

    if obj is None:
        return 0.0

    data = obj.get(
        "data"
    )

    events = obj.get(
        "events",
        {}
    )

    if data is None:
        return 0.0

    strength = 0.0

    # Rottura della Bollinger centrale.
    if events.get(
        "middle_break_down",
        False
    ):

        strength += 1.7

    # Distacco dalla banda superiore:
    # possibile esaurimento del rialzo.
    if events.get(
        "detach_upper",
        False
    ):

        strength += 0.9

    # MACD bearish cross recente.
    try:

        if recent_cross_down(
            data["MACD"],
            data["MACD_Signal"],
            3
        ):

            macd = safe(
                data,
                "MACD"
            )

            if macd > 0:

                strength += 1.8

            else:

                strength += 1.1

    except Exception:
        pass

    # SAR appena girato.
    try:

        close = safe(
            data,
            "Close"
        )

        previous_close = safe(
            data,
            "Close",
            -3,
            close
        )

        sar = safe(
            data,
            "SAR"
        )

        previous_sar = safe(
            data,
            "SAR",
            -3,
            sar
        )

        if (
            close < sar
            and
            previous_close
            >=
            previous_sar
        ):

            strength += 1.8

    except Exception:
        pass

    # Flussi in peggioramento.
    try:

        chaikin = safe(
            data,
            "Chaikin"
        )

        chaikin_previous = safe(
            data,
            "Chaikin",
            -5,
            chaikin
        )

        if (
            chaikin < chaikin_previous
            and
            chaikin_previous > 0
        ):

            strength += 0.8

    except Exception:
        pass

    # RSI gira dall'ipercomprato.
    try:

        rsi = safe(
            data,
            "RSI",
            default=50
        )

        rsi_previous = safe(
            data,
            "RSI",
            -4,
            rsi
        )

        if (
            rsi > 65
            and
            rsi < rsi_previous
        ):

            strength += 0.6

    except Exception:
        pass

    # Stoch RSI gira dall'ipercomprato.
    try:

        stoch = safe(
            data,
            "StochRSI",
            default=50
        )

        signal = safe(
            data,
            "StochRSI_Signal",
            default=50
        )

        previous_stoch = safe(
            data,
            "StochRSI",
            -4,
            stoch
        )

        previous_signal = safe(
            data,
            "StochRSI_Signal",
            -4,
            signal
        )

        if (
            previous_stoch >= 80
            and
            previous_stoch >= previous_signal
            and
            stoch < signal
        ):

            strength += 0.7

    except Exception:
        pass

    return min(
        strength,
        6.0
    )


def bearish_trend_age(
    result,
    timeframe
):
    """
    Età del trend SAR ribassista.

    Se il SAR non è ribassista, restituisce 0.
    """

    obj = get_tf_object(
        result,
        timeframe
    )

    if obj is None:
        return 0

    data = obj.get(
        "data"
    )

    if data is None:
        return 0

    try:

        close = safe(
            data,
            "Close"
        )

        sar = safe(
            data,
            "SAR"
        )

        if close >= sar:
            return 0

        return int(
            obj.get(
                "trend_age",
                0
            )
        )

    except Exception:

        return 0


def oversold_recovery_strength(
    result,
    timeframe
):
    """
    Cerca condizioni in cui continuare a vendere
    diventa progressivamente meno interessante:

    - RSI basso
    - Stoch RSI basso
    - recupero da Bollinger inferiore
    - MACD histogram in miglioramento
    """

    obj = get_tf_object(
        result,
        timeframe
    )

    if obj is None:
        return 0.0

    data = obj.get(
        "data"
    )

    events = obj.get(
        "events",
        {}
    )

    if data is None:
        return 0.0

    recovery = 0.0

    rsi = safe(
        data,
        "RSI",
        default=50
    )

    rsi_previous = safe(
        data,
        "RSI",
        -4,
        rsi
    )

    stoch = safe(
        data,
        "StochRSI",
        default=50
    )

    stoch_signal = safe(
        data,
        "StochRSI_Signal",
        default=50
    )

    hist = safe(
        data,
        "MACD_Hist"
    )

    hist_previous = safe(
        data,
        "MACD_Hist",
        -4,
        hist
    )

    close = safe(
        data,
        "HA_Close"
    )

    lower = safe(
        data,
        "BB_Lower"
    )

    if rsi < 35:

        recovery += 0.7

    if (
        rsi < 40
        and
        rsi > rsi_previous
    ):

        recovery += 0.8

    if stoch < 20:

        recovery += 0.6

    if (
        stoch < 30
        and
        stoch > stoch_signal
    ):

        recovery += 0.8

    if events.get(
        "detach_lower",
        False
    ):

        recovery += 1.1

    if (
        close <= lower * 1.01
    ):

        recovery += 0.5

    if (
        hist < 0
        and
        hist > hist_previous
    ):

        recovery += 0.7

    return min(
        recovery,
        4.0
    )


# ============================================================
# SELL SCORE
# ============================================================

def calculate_sell_score(
    trend_score_value,
    entry_numeric,
    rev_up,
    rev_down,
    confidence,
    overextension,
    phase,
    result,
):
    """
    MARKET SENTINEL V3.6 SELL SCORE

    PRINCIPIO:

    Il SELL SCORE NON misura:
        "quanto è brutto il grafico adesso?"

    Misura:
        "quanto è forte ADESSO il segnale di uscita?"

    Quindi:

    1) sale molto vicino all'inversione;
    2) 4H può anticipare Daily;
    3) Daily dà conferma importante;
    4) Weekly rafforza ma non è necessario per l'alert;
    5) se il ribasso è già maturo, il SELL decade;
    6) se compare ipervenduto / reversal up,
       il SELL diminuisce.
    """

    daily = get_tf_value(
        result,
        "Daily",
        "score",
        5
    )

    h4 = get_tf_value(
        result,
        "4H",
        "score",
        5
    )

    weekly = get_tf_value(
        result,
        "Weekly",
        "score",
        5
    )

    daily_rev_down = get_tf_value(
        result,
        "Daily",
        "bearish_reversal",
        0
    )

    h4_rev_down = get_tf_value(
        result,
        "4H",
        "bearish_reversal",
        0
    )

    weekly_rev_down = get_tf_value(
        result,
        "Weekly",
        "bearish_reversal",
        0
    )

    daily_rev_up = get_tf_value(
        result,
        "Daily",
        "bullish_reversal",
        0
    )

    h4_rev_up = get_tf_value(
        result,
        "4H",
        "bullish_reversal",
        0
    )

    # ========================================================
    # EVENTI RECENTI
    # ========================================================

    h4_event = bearish_event_strength(
        result,
        "4H"
    )

    daily_event = bearish_event_strength(
        result,
        "Daily"
    )

    weekly_event = bearish_event_strength(
        result,
        "Weekly"
    )

    # ========================================================
    # ETÀ DEL RIBASSO
    # ========================================================

    h4_bear_age = bearish_trend_age(
        result,
        "4H"
    )

    daily_bear_age = bearish_trend_age(
        result,
        "Daily"
    )

    weekly_bear_age = bearish_trend_age(
        result,
        "Weekly"
    )

    # ========================================================
    # RECUPERO / IPERVENDUTO
    # ========================================================

    h4_recovery = oversold_recovery_strength(
        result,
        "4H"
    )

    daily_recovery = oversold_recovery_strength(
        result,
        "Daily"
    )

    # ========================================================
    # BASE
    #
    # Rev Down globale conta, ma meno della V3.5:
    # non vogliamo che il semplice deterioramento già avvenuto
    # tenga il SELL artificialmente alto.
    # ========================================================

    score = 0.0

    score += (
        rev_down
        * 0.28
    )

    # ========================================================
    # 4H = EARLY WARNING
    # ========================================================

    score += (
        h4_event
        * 0.42
    )

    # ========================================================
    # DAILY = CONFERMA PRINCIPALE
    # ========================================================

    score += (
        daily_event
        * 0.58
    )

    # ========================================================
    # WEEKLY = CONFERMA, NON REQUISITO
    # ========================================================

    score += (
        weekly_event
        * 0.15
    )

    # ========================================================
    # REVERSAL 4H / DAILY
    # ========================================================

    score += (
        h4_rev_down
        * 0.12
    )

    score += (
        daily_rev_down
        * 0.18
    )

    # ========================================================
    # SINCRONIZZAZIONE 4H + DAILY
    #
    # Qui il SELL può diventare davvero forte.
    # ========================================================

    if (
        h4_event >= 2.0
        and
        daily_event >= 2.0
    ):

        score += 1.30

    elif (
        h4_event >= 1.5
        and
        daily_event >= 1.0
    ):

        score += 0.75

    # 4H anticipa Daily.
    if (
        h4_event >= 2.5
        and
        daily_event < 1.5
        and
        daily >= 5.0
    ):

        score += 0.55

    # ========================================================
    # APPENA GIRATO RIBASSISTA
    #
    # Il segnale è particolarmente utile quando il cambio è
    # recente, non quando dura da settimane.
    # ========================================================

    if (
        1 <= h4_bear_age <= 4
    ):

        score += 0.70

    elif (
        5 <= h4_bear_age <= 8
    ):

        score += 0.25

    if (
        1 <= daily_bear_age <= 4
    ):

        score += 1.00

    elif (
        5 <= daily_bear_age <= 8
    ):

        score += 0.40

    if (
        1 <= weekly_bear_age <= 3
    ):

        score += 0.30

    # ========================================================
    # BREAKDOWN DI TREND PRIMA DEL CROLLO
    # ========================================================

    if (
        h4 < 4.8
        and
        daily >= 5.0
    ):

        score += 0.45

    if (
        h4 < 4.6
        and
        daily < 5.0
        and
        daily >= 4.2
    ):

        score += 0.55

    if (
        daily < 4.6
        and
        weekly >= 5.0
    ):

        score += 0.65

    # ========================================================
    # FASE
    # ========================================================

    phase_upper = str(
        phase
    ).upper()

    if (
        "EXHAUSTION"
        in phase_upper
    ):

        score += 0.90

    if (
        "WATCH ↓"
        in phase_upper
    ):

        score += 0.45

    # Importante:
    # un TREND DOWN già conclamato NON riceve un enorme bonus.
    if (
        "TREND ↓"
        in phase_upper
    ):

        score += 0.15

    # ========================================================
    # CONFIDENCE
    #
    # Confidence aiuta solo se il deterioramento è recente.
    # ========================================================

    if (
        confidence >= 7.5
        and
        (
            h4_event >= 1.5
            or
            daily_event >= 1.5
        )
    ):

        score += 0.35

    # ========================================================
    # DECAY: RIBASSO GIÀ SVILUPPATO
    #
    # Questo è il cambiamento più importante.
    # ========================================================

    if daily_bear_age >= 9:

        score -= min(
            (
                daily_bear_age - 8
            )
            * 0.16,
            2.10
        )

    if h4_bear_age >= 14:

        score -= min(
            (
                h4_bear_age - 13
            )
            * 0.07,
            1.10
        )

    # Ribasso maturo contemporaneamente su 4H e Daily.
    if (
        daily_bear_age >= 12
        and
        h4_bear_age >= 16
    ):

        score -= 0.65

    # ========================================================
    # SE IL TITOLO È GIÀ MOLTO DEBOLE,
    # NON VOGLIAMO DIRE "VENDI ORA" SOLO PERCHÉ È DEBOLE.
    # ========================================================

    if (
        daily < 3.8
        and
        h4 < 3.8
        and
        daily_event < 1.5
    ):

        score -= 0.70

    if (
        trend_score_value < 4.0
        and
        daily_bear_age >= 10
    ):

        score -= 0.55

    # ========================================================
    # IPERVENDUTO / POSSIBILE BOTTOM
    # ========================================================

    score -= (
        h4_recovery
        * 0.25
    )

    score -= (
        daily_recovery
        * 0.40
    )

    # ========================================================
    # REVERSAL UP COME CONTRAPPESO
    # ========================================================

    score -= (
        rev_up
        * 0.10
    )

    score -= (
        h4_rev_up
        * 0.08
    )

    score -= (
        daily_rev_up
        * 0.12
    )

    if (
        daily_rev_up >= 5.5
        and
        h4_rev_up >= 4.5
    ):

        score -= 0.75

    # ========================================================
    # POSSIBLE BOTTOM
    # ========================================================

    if (
        "POSSIBLE BOTTOM"
        in phase_upper
    ):

        score -= 1.20

    if (
        "EARLY REVERSAL ↑"
        in phase_upper
    ):

        score -= 1.10

    # ========================================================
    # LIMITI DI COERENZA
    # ========================================================

    # Un SELL 9-10 deve richiedere davvero eventi recenti.
    if (
        daily_event < 1.0
        and
        h4_event < 1.0
    ):

        score = min(
            score,
            5.5
        )

    # Per superare 8 vogliamo almeno una vera conferma Daily
    # oppure sincronizzazione fortissima 4H + Daily.
    if (
        score > 8.0
        and
        daily_event < 1.5
        and
        not (
            h4_event >= 2.5
            and
            daily_rev_down >= 4.5
        )
    ):

        score = 8.0

    return clip(
        score
    )


def sell_label(
    sell_score
):

    if sell_score >= 8.0:

        return (
            "FORTE SEGNALE DI USCITA"
        )

    if sell_score >= 6.5:

        return (
            "SEGNALE DI USCITA"
        )

    if sell_score >= 5.0:

        return (
            "VALUTA ALLEGGERIMENTO"
        )

    if sell_score >= 3.5:

        return (
            "ATTENZIONE"
        )

    if sell_score >= 2.5:

        return (
            "MONITORA"
        )

    return (
        "MANTIENI"
    )


# ============================================================
# PRICE TARGET ADJUSTMENT
# ============================================================

def price_target_adjustment(
    expected_return,
    technical_buy_score
):

    try:

        expected_return = float(
            expected_return
        )

        technical_buy_score = float(
            technical_buy_score
        )

    except Exception:

        return 0.0

    if (
        not np.isfinite(
            expected_return
        )
        or
        not np.isfinite(
            technical_buy_score
        )
    ):

        return 0.0

    if expected_return > 30:

        adjustment = 0.30

    elif expected_return > 20:

        adjustment = 0.25

    elif expected_return > 10:

        adjustment = 0.15

    elif expected_return > 5:

        adjustment = 0.05

    elif expected_return >= -5:

        adjustment = 0.0

    elif expected_return >= -10:

        adjustment = -0.05

    elif expected_return >= -20:

        adjustment = -0.15

    elif expected_return >= -30:

        adjustment = -0.25

    else:

        adjustment = -0.30

    if adjustment > 0:

        if technical_buy_score < 5:

            return 0.0

        if technical_buy_score < 6:

            adjustment *= 0.5

    return round(
        adjustment,
        2
    )


def final_buy_score(
    technical_buy_score,
    expected_return
):

    adjustment = (
        price_target_adjustment(
            expected_return,
            technical_buy_score
        )
    )

    final_score = clip(
        float(
            technical_buy_score
        )
        +
        adjustment
    )

    return (
        round(
            final_score,
            2
        ),

        adjustment,
    )


# ============================================================
# DIAGNOSIS
# ============================================================

def build_diagnosis(
    phase,
    buy_score,
    trend_score_value,
    entry_text,
    result,
    rev_up,
    rev_down,
):

    weekly = get_tf_value(
        result,
        "Weekly",
        "score"
    )

    daily = get_tf_value(
        result,
        "Daily",
        "score"
    )

    h4 = get_tf_value(
        result,
        "4H",
        "score"
    )

    parts = []

    if buy_score >= 8:

        parts.append(
            "Configurazione tecnica di acquisto molto interessante."
        )

    elif buy_score >= 7:

        parts.append(
            "Configurazione tecnica di acquisto interessante."
        )

    elif buy_score >= 6:

        parts.append(
            "Configurazione tecnica discreta, ma non ancora ideale."
        )

    else:

        parts.append(
            "Il rapporto fra trend, timing e rischio non è ancora particolarmente favorevole."
        )

    if (
        h4 >= 6.5
        and
        daily >= 5.5
        and
        weekly < 5.5
    ):

        parts.append(
            "4H e Daily stanno anticipando il Weekly."
        )

    elif (
        weekly >= 6.5
        and
        daily >= 6.5
        and
        h4 >= 6
    ):

        parts.append(
            "Weekly, Daily e 4H sono ben allineati."
        )

    if rev_up >= 6:

        parts.append(
            "Sono presenti segnali di reversal rialzista."
        )

    if rev_down >= 6:

        parts.append(
            "Sono presenti segnali di deterioramento da monitorare."
        )

    parts.append(
        f"Timing: {entry_text.lower()}."
    )

    return " ".join(
        parts
    )


# ============================================================
# ANALYZE
# ============================================================

def analyze(
    ticker,
    frames,
    breadth
):

    result = {}

    for timeframe, df in (
        frames.items()
    ):

        if (
            df is None
            or
            df.empty
        ):

            continue

        current = timeframe_scores(
            df,
            breadth.get(
                timeframe,
                5
            )
        )

        if current is not None:

            result[
                timeframe
            ] = current

    if not result:

        return None

    global_trend = (
        weighted_average(
            result,
            "score",
            TREND_TIMEFRAME_WEIGHTS,
        )
    )

    entry_numeric = (
        weighted_average(
            result,
            "entry_score",
            ENTRY_TIMEFRAME_WEIGHTS,
        )
    )

    rev_up = (
        weighted_average(
            result,
            "bullish_reversal",
            REVERSAL_TIMEFRAME_WEIGHTS,
            0,
        )
    )

    rev_down = (
        weighted_average(
            result,
            "bearish_reversal",
            REVERSAL_TIMEFRAME_WEIGHTS,
            0,
        )
    )

    overextension = (
        weighted_average(
            result,
            "overextension",
            ENTRY_TIMEFRAME_WEIGHTS,
            0,
        )
    )

    confidence = (
        confidence_score(
            result,
            global_trend,
            rev_up,
            rev_down,
        )
    )

    phase = global_phase(
        result,
        global_trend,
        entry_numeric,
        rev_up,
        rev_down,
    )

    entry_text = entry_label(
        entry_numeric,
        global_trend,
        phase,
    )

    technical_buy_score = (
        calculate_buy_score(
            global_trend,
            entry_numeric,
            rev_up,
            rev_down,
            confidence,
            overextension,
            phase,
            result,
        )
    )

    sell_score_value = (
        calculate_sell_score(
            global_trend,
            entry_numeric,
            rev_up,
            rev_down,
            confidence,
            overextension,
            phase,
            result,
        )
    )

    sell_text = (
        sell_label(
            sell_score_value
        )
    )

    portfolio_alert = (
        sell_score_value
    )

    bullish_reasons = []
    bearish_reasons = []

    for timeframe in [
        "1H",
        "2H",
        "4H",
        "Daily",
        "Weekly",
        "Monthly",
    ]:

        if timeframe not in result:

            continue

        for (
            strength,
            reason
        ) in result[
            timeframe
        ][
            "bullish_reasons"
        ]:

            bullish_reasons.append(
                {

                    "timeframe":
                        timeframe,

                    "strength":
                        strength,

                    "reason":
                        reason,
                }
            )

        for (
            strength,
            reason
        ) in result[
            timeframe
        ][
            "bearish_reasons"
        ]:

            bearish_reasons.append(
                {

                    "timeframe":
                        timeframe,

                    "strength":
                        strength,

                    "reason":
                        reason,
                }
            )

    diagnosis = (
        build_diagnosis(
            phase,
            technical_buy_score,
            global_trend,
            entry_text,
            result,
            rev_up,
            rev_down,
        )
    )

    return {

        "ticker":
            ticker,

        "buy_score":
            round(
                technical_buy_score,
                2
            ),

        "technical_buy_score":
            round(
                technical_buy_score,
                2
            ),

        "sell_score":
            round(
                sell_score_value,
                2
            ),

        "sell_text":
            sell_text,

        "trend_score":
            round(
                global_trend,
                2
            ),

        "global_score":
            round(
                global_trend,
                2
            ),

        "entry_score":
            round(
                entry_numeric,
                2
            ),

        "entry_text":
            entry_text,

        "bullish_reversal":
            round(
                rev_up,
                2
            ),

        "bearish_risk":
            round(
                rev_down,
                2
            ),

        "confidence":
            round(
                confidence,
                2
            ),

        "phase":
            phase,

        "portfolio_alert":
            round(
                portfolio_alert,
                2
            ),

        "diagnosis":
            diagnosis,

        "frames":
            result,

        "bullish_reasons":
            bullish_reasons,

        "bearish_reasons":
            bearish_reasons,
    }