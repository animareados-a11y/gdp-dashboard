import numpy as np
import pandas as pd


# ============================================================
# MARKET SENTINEL
# ADVANCED FEATURE ENGINE V2.2
# ============================================================
#
# OBIETTIVO
#
# Tradurre la lettura VISIVA dei grafici in variabili numeriche
# utilizzabili dal Machine Learning, con particolare attenzione a:
#
# - stato dell'indicatore
# - estremo relativo
# - direzione / pendenza
# - accelerazione / decelerazione
# - evento appena avvenuto
# - ETA' dell'evento
# - FRESHNESS: forza che decade con il passare delle barre
# - sequenza temporale fra Bollinger, Heikin-Ashi, MACD e SAR
# - relazioni fra 4H, Daily e Weekly
#
# V2.2:
# - separa PRE-ALERT da TREND ATTIVO e CONFERMA
# - BUY e SELL possono coesistere come pre-alert
# - BUY e SELL NON possono essere contemporaneamente confermati
# - la conferma richiede lo STATO ATTUALE coerente di SAR e MACD
# - mantiene la memoria degli eventi tramite freshness
#
# Nessuna funzione usa dati futuri: riceve solo il dataframe già
# troncato alla data di simulazione.
# ============================================================


TIMEFRAMES = [
    "Monthly",
    "Weekly",
    "Daily",
    "4H",
    "2H",
    "1H",
]

MAX_EVENT_AGE = 30
FRESHNESS_DECAY = 0.22


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return default


def safe_value(df, column, index=-1, default=np.nan):
    try:
        if df is None or df.empty or column not in df.columns:
            return default

        value = float(df[column].iloc[index])

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def safe_change(df, column, periods=1, default=np.nan):
    try:
        if (
            df is None
            or df.empty
            or column not in df.columns
            or len(df) <= periods
        ):
            return default

        current = float(df[column].iloc[-1])
        previous = float(df[column].iloc[-1 - periods])

        if np.isfinite(current) and np.isfinite(previous):
            return current - previous

    except Exception:
        pass

    return default


def safe_pct_change(df, column, periods=1, default=np.nan):
    try:
        current = safe_value(df, column, -1)
        previous = safe_value(df, column, -1 - periods)

        if (
            not np.isfinite(current)
            or not np.isfinite(previous)
            or previous == 0
        ):
            return default

        return ((current / previous) - 1) * 100

    except Exception:
        return default


def clipped(value, lower=0.0, upper=1.0):
    try:
        value = float(value)

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, lower, upper))

    except Exception:
        return 0.0


# ============================================================
# ROLLING STATISTICS
# ============================================================

def rolling_percentile(series, lookback=100):
    try:
        values = pd.to_numeric(series, errors="coerce").dropna()

        if len(values) < 10:
            return np.nan

        window = values.iloc[-min(lookback, len(values)):]
        current = float(window.iloc[-1])

        return float((window <= current).mean())

    except Exception:
        return np.nan


def rolling_zscore(series, lookback=100):
    try:
        values = pd.to_numeric(series, errors="coerce").dropna()

        if len(values) < 10:
            return np.nan

        window = values.iloc[-min(lookback, len(values)):]

        mean = float(window.mean())
        std = float(window.std())
        current = float(window.iloc[-1])

        if not np.isfinite(std) or std == 0:
            return 0.0

        return (current - mean) / std

    except Exception:
        return np.nan


def slope(series, periods=3):
    try:
        values = pd.to_numeric(series, errors="coerce").dropna()

        if len(values) < 2:
            return np.nan

        periods = int(max(2, min(periods, len(values))))

        y = values.iloc[-periods:].to_numpy(dtype=float)

        if len(y) < 2:
            return np.nan

        x = np.arange(len(y), dtype=float)

        return float(np.polyfit(x, y, 1)[0])

    except Exception:
        return np.nan


def acceleration(series, short=3, long=6):
    try:
        values = pd.to_numeric(series, errors="coerce").dropna()

        short = int(max(2, short))
        long = int(max(2, long))

        if len(values) < short + long:
            return np.nan

        recent = values.iloc[-short:]
        previous = values.iloc[-(short + long):-short]

        recent_slope = slope(recent, short)
        previous_slope = slope(previous, long)

        if (
            np.isfinite(recent_slope)
            and np.isfinite(previous_slope)
        ):
            return recent_slope - previous_slope

    except Exception:
        pass

    return np.nan


# ============================================================
# EVENT AGE + FRESHNESS
# ============================================================

def bars_since_true(condition, max_bars=MAX_EVENT_AGE):
    try:
        values = pd.Series(condition).fillna(False).astype(bool)

        if values.empty:
            return max_bars + 1

        start = max(
            0,
            len(values) - max_bars - 1,
        )

        for i in range(
            len(values) - 1,
            start - 1,
            -1,
        ):
            if bool(values.iloc[i]):
                return len(values) - 1 - i

    except Exception:
        pass

    return max_bars + 1


def freshness_from_age(
    age,
    decay=FRESHNESS_DECAY,
    max_age=MAX_EVENT_AGE,
):
    try:
        age = float(age)

        if (
            not np.isfinite(age)
            or age < 0
            or age > max_age
        ):
            return 0.0

        return float(
            np.exp(-decay * age)
        )

    except Exception:
        return 0.0


def recent_true(series, lookback=3):
    try:
        values = pd.Series(series).fillna(False)

        return int(
            values.iloc[-lookback:]
            .astype(bool)
            .any()
        )

    except Exception:
        return 0


def bars_since_cross(
    series_a,
    series_b,
    direction="up",
    max_bars=MAX_EVENT_AGE,
):
    try:
        frame = pd.DataFrame(
            {
                "a": pd.to_numeric(
                    series_a,
                    errors="coerce",
                ),
                "b": pd.to_numeric(
                    series_b,
                    errors="coerce",
                ),
            }
        ).dropna()

        if len(frame) < 2:
            return max_bars + 1

        if direction == "up":
            events = (
                (frame["a"] > frame["b"])
                & (
                    frame["a"].shift(1)
                    <= frame["b"].shift(1)
                )
            )

        else:
            events = (
                (frame["a"] < frame["b"])
                & (
                    frame["a"].shift(1)
                    >= frame["b"].shift(1)
                )
            )

        return bars_since_true(
            events,
            max_bars,
        )

    except Exception:
        return max_bars + 1


# ============================================================
# HISTORICAL PRESSURE HELPERS
# ============================================================

def historical_pressure_event_age(
    series,
    mode,
    max_bars=MAX_EVENT_AGE,
    lookback=100,
):
    try:
        values = pd.to_numeric(
            series,
            errors="coerce",
        )

        if values.dropna().shape[0] < 12:
            return max_bars + 1

        final = len(values) - 1
        start = max(
            2,
            final - max_bars,
        )

        for i in range(
            final,
            start - 1,
            -1,
        ):
            history = (
                values.iloc[: i + 1]
                .dropna()
            )

            if len(history) < 10:
                continue

            window = history.iloc[
                -min(lookback, len(history)):
            ]

            current = float(
                window.iloc[-1]
            )

            pct = float(
                (window <= current).mean()
            )

            std = float(window.std())
            mean = float(window.mean())

            z = (
                0.0
                if (
                    not np.isfinite(std)
                    or std == 0
                )
                else (current - mean) / std
            )

            local_slope = slope(
                history,
                3,
            )

            inflated = (
                pct >= 0.90
                or z >= 1.5
            )

            deflated = (
                pct <= 0.10
                or z <= -1.5
            )

            if mode == "decompression":
                event = (
                    inflated
                    and np.isfinite(local_slope)
                    and local_slope < 0
                )

            else:
                event = (
                    deflated
                    and np.isfinite(local_slope)
                    and local_slope > 0
                )

            if event:
                return final - i

    except Exception:
        pass

    return max_bars + 1


# ============================================================
# SAR
# ============================================================

def sar_features(df, prefix):
    result = {}

    close = safe_value(
        df,
        "Close",
    )

    sar = safe_value(
        df,
        "SAR",
    )

    atr = safe_value(
        df,
        "ATR",
    )

    bullish = (
        np.isfinite(close)
        and np.isfinite(sar)
        and close > sar
    )

    result[f"{prefix}_sar_bullish"] = int(
        bullish
    )

    result[f"{prefix}_sar_bearish"] = int(
        not bullish
    )

    try:
        close_series = pd.to_numeric(
            df["Close"],
            errors="coerce",
        )

        sar_series = pd.to_numeric(
            df["SAR"],
            errors="coerce",
        )

        condition = (
            close_series > sar_series
        )

        flip_up = (
            condition
            & (
                ~condition
                .shift(1)
                .fillna(False)
            )
        )

        flip_down = (
            (~condition)
            & condition
            .shift(1)
            .fillna(False)
        )

        up_age = bars_since_true(
            flip_up
        )

        down_age = bars_since_true(
            flip_down
        )

        result[f"{prefix}_sar_flip_up_now"] = int(
            up_age == 0
        )

        result[f"{prefix}_sar_flip_down_now"] = int(
            down_age == 0
        )

        result[f"{prefix}_sar_flip_up_age"] = up_age
        result[f"{prefix}_sar_flip_down_age"] = down_age

        result[f"{prefix}_sar_flip_up_freshness"] = (
            freshness_from_age(up_age)
        )

        result[f"{prefix}_sar_flip_down_freshness"] = (
            freshness_from_age(down_age)
        )

    except Exception:
        result[f"{prefix}_sar_flip_up_now"] = 0
        result[f"{prefix}_sar_flip_down_now"] = 0

        result[f"{prefix}_sar_flip_up_age"] = (
            MAX_EVENT_AGE + 1
        )

        result[f"{prefix}_sar_flip_down_age"] = (
            MAX_EVENT_AGE + 1
        )

        result[f"{prefix}_sar_flip_up_freshness"] = 0.0
        result[f"{prefix}_sar_flip_down_freshness"] = 0.0

    if (
        np.isfinite(close)
        and np.isfinite(sar)
        and np.isfinite(atr)
        and atr != 0
    ):
        distance = (
            (close - sar) / atr
        )

    else:
        distance = np.nan

    result[f"{prefix}_sar_distance_atr"] = distance

    try:
        distance_series = (
            pd.to_numeric(
                df["Close"],
                errors="coerce",
            )
            - pd.to_numeric(
                df["SAR"],
                errors="coerce",
            )
        ) / (
            pd.to_numeric(
                df["ATR"],
                errors="coerce",
            )
            .replace(0, np.nan)
        )

        result[f"{prefix}_sar_distance_change_1"] = safe_float(
            distance_series.iloc[-1]
            - distance_series.iloc[-2]
        )

        result[f"{prefix}_sar_distance_change_3"] = safe_float(
            distance_series.iloc[-1]
            - distance_series.iloc[-4]
        )

        abs_distance = (
            distance_series.abs()
        )

        result[f"{prefix}_sar_approaching_price_1"] = int(
            len(abs_distance) >= 2
            and np.isfinite(abs_distance.iloc[-1])
            and np.isfinite(abs_distance.iloc[-2])
            and abs_distance.iloc[-1]
            < abs_distance.iloc[-2]
        )

        result[f"{prefix}_sar_approaching_price_3"] = int(
            len(abs_distance) >= 4
            and np.isfinite(abs_distance.iloc[-1])
            and np.isfinite(abs_distance.iloc[-4])
            and abs_distance.iloc[-1]
            < abs_distance.iloc[-4]
        )

    except Exception:
        result[f"{prefix}_sar_distance_change_1"] = np.nan
        result[f"{prefix}_sar_distance_change_3"] = np.nan

        result[f"{prefix}_sar_approaching_price_1"] = 0
        result[f"{prefix}_sar_approaching_price_3"] = 0

    return result


# ============================================================
# HEIKIN ASHI
# ============================================================

def heikin_ashi_features(df, prefix):
    result = {}

    ha_open = safe_value(
        df,
        "HA_Open",
    )

    ha_close = safe_value(
        df,
        "HA_Close",
    )

    ha_high = safe_value(
        df,
        "HA_High",
        default=safe_value(
            df,
            "High",
        ),
    )

    ha_low = safe_value(
        df,
        "HA_Low",
        default=safe_value(
            df,
            "Low",
        ),
    )

    if not all(
        np.isfinite(v)
        for v in [
            ha_open,
            ha_close,
            ha_high,
            ha_low,
        ]
    ):
        return result

    candle_range = max(
        ha_high - ha_low,
        1e-9,
    )

    body = abs(
        ha_close - ha_open
    )

    upper_wick = max(
        ha_high
        - max(
            ha_open,
            ha_close,
        ),
        0.0,
    )

    lower_wick = max(
        min(
            ha_open,
            ha_close,
        )
        - ha_low,
        0.0,
    )

    green = (
        ha_close > ha_open
    )

    red = (
        ha_close < ha_open
    )

    result[f"{prefix}_ha_green"] = int(green)
    result[f"{prefix}_ha_red"] = int(red)

    result[f"{prefix}_ha_body_ratio"] = (
        body / candle_range
    )

    result[f"{prefix}_ha_upper_wick_ratio"] = (
        upper_wick / candle_range
    )

    result[f"{prefix}_ha_lower_wick_ratio"] = (
        lower_wick / candle_range
    )

    uncertainty = (
        (body / candle_range) <= 0.40
        and (upper_wick / candle_range) >= 0.15
        and (lower_wick / candle_range) >= 0.15
    )

    result[f"{prefix}_ha_uncertainty"] = int(
        uncertainty
    )

    try:
        opens = pd.to_numeric(
            df["HA_Open"],
            errors="coerce",
        )

        closes = pd.to_numeric(
            df["HA_Close"],
            errors="coerce",
        )

        highs = pd.to_numeric(
            (
                df["HA_High"]
                if "HA_High" in df.columns
                else df["High"]
            ),
            errors="coerce",
        )

        lows = pd.to_numeric(
            (
                df["HA_Low"]
                if "HA_Low" in df.columns
                else df["Low"]
            ),
            errors="coerce",
        )

        ranges = (
            highs - lows
        ).replace(
            0,
            np.nan,
        )

        bodies = (
            closes - opens
        ).abs()

        upper_wicks = (
            highs
            - pd.concat(
                [opens, closes],
                axis=1,
            ).max(axis=1)
        )

        lower_wicks = (
            pd.concat(
                [opens, closes],
                axis=1,
            ).min(axis=1)
            - lows
        )

        uncertainty_series = (
            (bodies / ranges <= 0.40)
            & (upper_wicks / ranges >= 0.15)
            & (lower_wicks / ranges >= 0.15)
        ).fillna(False)

        colors = np.sign(
            closes - opens
        )

        flip_green = (
            (colors > 0)
            & (
                colors.shift(1)
                <= 0
            )
        )

        flip_red = (
            (colors < 0)
            & (
                colors.shift(1)
                >= 0
            )
        )

        uncertainty_age = bars_since_true(
            uncertainty_series
        )

        green_age = bars_since_true(
            flip_green
        )

        red_age = bars_since_true(
            flip_red
        )

        result[f"{prefix}_ha_uncertainty_age"] = (
            uncertainty_age
        )

        result[f"{prefix}_ha_uncertainty_freshness"] = (
            freshness_from_age(
                uncertainty_age
            )
        )

        result[f"{prefix}_ha_flip_green_now"] = int(
            green_age == 0
        )

        result[f"{prefix}_ha_flip_red_now"] = int(
            red_age == 0
        )

        result[f"{prefix}_ha_flip_green_age"] = (
            green_age
        )

        result[f"{prefix}_ha_flip_red_age"] = (
            red_age
        )

        result[f"{prefix}_ha_flip_green_freshness"] = (
            freshness_from_age(
                green_age
            )
        )

        result[f"{prefix}_ha_flip_red_freshness"] = (
            freshness_from_age(
                red_age
            )
        )

        current_color = (
            colors.iloc[-1]
        )

        streak = 0

        for value in reversed(
            colors.tolist()
        ):
            if (
                value == current_color
                and current_color != 0
            ):
                streak += 1
            else:
                break

        result[f"{prefix}_ha_color_streak"] = (
            streak
        )

        body_ratio_series = (
            bodies / ranges
        )

        result[f"{prefix}_ha_body_change_1"] = safe_float(
            body_ratio_series.iloc[-1]
            - body_ratio_series.iloc[-2]
        )

        result[f"{prefix}_ha_body_shrinking"] = int(
            len(body_ratio_series) >= 2
            and np.isfinite(
                body_ratio_series.iloc[-1]
            )
            and np.isfinite(
                body_ratio_series.iloc[-2]
            )
            and body_ratio_series.iloc[-1]
            < body_ratio_series.iloc[-2]
        )

    except Exception:
        result[f"{prefix}_ha_flip_green_now"] = 0
        result[f"{prefix}_ha_flip_red_now"] = 0

        result[f"{prefix}_ha_flip_green_age"] = (
            MAX_EVENT_AGE + 1
        )

        result[f"{prefix}_ha_flip_red_age"] = (
            MAX_EVENT_AGE + 1
        )

        result[f"{prefix}_ha_flip_green_freshness"] = 0.0
        result[f"{prefix}_ha_flip_red_freshness"] = 0.0

        result[f"{prefix}_ha_uncertainty_age"] = (
            MAX_EVENT_AGE + 1
        )

        result[f"{prefix}_ha_uncertainty_freshness"] = 0.0
        result[f"{prefix}_ha_color_streak"] = 0
        result[f"{prefix}_ha_body_change_1"] = np.nan
        result[f"{prefix}_ha_body_shrinking"] = 0

    return result


# ============================================================
# BOLLINGER
# ============================================================

def bollinger_features(df, prefix):
    result = {}

    price_col = (
        "HA_Close"
        if "HA_Close" in df.columns
        else "Close"
    )

    close = safe_value(
        df,
        price_col,
    )

    upper = safe_value(
        df,
        "BB_Upper",
    )

    middle = safe_value(
        df,
        "BB_Mid",
    )

    lower = safe_value(
        df,
        "BB_Lower",
    )

    atr = safe_value(
        df,
        "ATR",
    )

    if not all(
        np.isfinite(v)
        for v in [
            close,
            upper,
            middle,
            lower,
        ]
    ):
        return result

    width = max(
        upper - lower,
        1e-9,
    )

    result[f"{prefix}_bb_position"] = (
        (close - lower) / width
    )

    result[f"{prefix}_bb_above_upper"] = int(
        close > upper
    )

    result[f"{prefix}_bb_below_lower"] = int(
        close < lower
    )

    result[f"{prefix}_bb_above_mid"] = int(
        close > middle
    )

    if (
        np.isfinite(atr)
        and atr != 0
    ):
        result[f"{prefix}_bb_upper_distance_atr"] = (
            (close - upper) / atr
        )

        result[f"{prefix}_bb_lower_distance_atr"] = (
            (lower - close) / atr
        )

        result[f"{prefix}_bb_mid_distance_atr"] = (
            (close - middle) / atr
        )

    else:
        result[f"{prefix}_bb_upper_distance_atr"] = np.nan
        result[f"{prefix}_bb_lower_distance_atr"] = np.nan
        result[f"{prefix}_bb_mid_distance_atr"] = np.nan

    try:
        prices = pd.to_numeric(
            df[price_col],
            errors="coerce",
        )

        uppers = pd.to_numeric(
            df["BB_Upper"],
            errors="coerce",
        )

        lowers = pd.to_numeric(
            df["BB_Lower"],
            errors="coerce",
        )

        mids = pd.to_numeric(
            df["BB_Mid"],
            errors="coerce",
        )

        atrs = pd.to_numeric(
            df["ATR"],
            errors="coerce",
        ).replace(
            0,
            np.nan,
        )

        upper_distance = (
            prices - uppers
        ) / atrs

        lower_distance = (
            lowers - prices
        ) / atrs

        result[f"{prefix}_bb_max_upper_detach_5"] = safe_float(
            upper_distance
            .iloc[-5:]
            .max()
        )

        result[f"{prefix}_bb_max_lower_detach_5"] = safe_float(
            lower_distance
            .iloc[-5:]
            .max()
        )

        outside_upper = (
            upper_distance > 0
        ).fillna(False)

        outside_lower = (
            lower_distance > 0
        ).fillna(False)

        reentry_upper = (
            (~outside_upper)
            & outside_upper
            .shift(1)
            .fillna(False)
        )

        reentry_lower = (
            (~outside_lower)
            & outside_lower
            .shift(1)
            .fillna(False)
        )

        upper_detach_age = bars_since_true(
            outside_upper
        )

        lower_detach_age = bars_since_true(
            outside_lower
        )

        reentry_upper_age = bars_since_true(
            reentry_upper
        )

        reentry_lower_age = bars_since_true(
            reentry_lower
        )

        result[f"{prefix}_bb_upper_detach_age"] = (
            upper_detach_age
        )

        result[f"{prefix}_bb_lower_detach_age"] = (
            lower_detach_age
        )

        result[f"{prefix}_bb_upper_detach_freshness"] = (
            freshness_from_age(
                upper_detach_age
            )
        )

        result[f"{prefix}_bb_lower_detach_freshness"] = (
            freshness_from_age(
                lower_detach_age
            )
        )

        result[f"{prefix}_bb_reentry_from_upper"] = int(
            reentry_upper_age == 0
        )

        result[f"{prefix}_bb_reentry_from_lower"] = int(
            reentry_lower_age == 0
        )

        result[f"{prefix}_bb_reentry_upper_age"] = (
            reentry_upper_age
        )

        result[f"{prefix}_bb_reentry_lower_age"] = (
            reentry_lower_age
        )

        result[f"{prefix}_bb_reentry_upper_freshness"] = (
            freshness_from_age(
                reentry_upper_age
            )
        )

        result[f"{prefix}_bb_reentry_lower_freshness"] = (
            freshness_from_age(
                reentry_lower_age
            )
        )

        mid_up_age = bars_since_cross(
            prices,
            mids,
            "up",
        )

        mid_down_age = bars_since_cross(
            prices,
            mids,
            "down",
        )

        result[f"{prefix}_bb_mid_cross_up_age"] = (
            mid_up_age
        )

        result[f"{prefix}_bb_mid_cross_down_age"] = (
            mid_down_age
        )

        result[f"{prefix}_bb_mid_cross_up_freshness"] = (
            freshness_from_age(
                mid_up_age
            )
        )

        result[f"{prefix}_bb_mid_cross_down_freshness"] = (
            freshness_from_age(
                mid_down_age
            )
        )

        result[f"{prefix}_bb_upper_detach_change_1"] = safe_float(
            upper_distance.iloc[-1]
            - upper_distance.iloc[-2]
        )

        result[f"{prefix}_bb_lower_detach_change_1"] = safe_float(
            lower_distance.iloc[-1]
            - lower_distance.iloc[-2]
        )

    except Exception:
        pass

    return result


# ============================================================
# MACD
# ============================================================

def macd_features(df, prefix):
    result = {}

    required = [
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
    ]

    if any(
        column not in df.columns
        for column in required
    ):
        return result

    macd = safe_value(
        df,
        "MACD",
    )

    signal = safe_value(
        df,
        "MACD_Signal",
    )

    hist = safe_value(
        df,
        "MACD_Hist",
    )

    result[f"{prefix}_macd"] = macd
    result[f"{prefix}_macd_signal"] = signal
    result[f"{prefix}_macd_hist"] = hist

    spread = (
        macd - signal
        if (
            np.isfinite(macd)
            and np.isfinite(signal)
        )
        else np.nan
    )

    result[f"{prefix}_macd_spread"] = spread

    result[f"{prefix}_macd_above_signal"] = int(
        np.isfinite(spread)
        and spread > 0
    )

    result[f"{prefix}_macd_below_signal"] = int(
        np.isfinite(spread)
        and spread < 0
    )

    result[f"{prefix}_macd_above_zero"] = int(
        np.isfinite(macd)
        and macd > 0
    )

    result[f"{prefix}_macd_below_zero"] = int(
        np.isfinite(macd)
        and macd < 0
    )

    up_age = bars_since_cross(
        df["MACD"],
        df["MACD_Signal"],
        "up",
    )

    down_age = bars_since_cross(
        df["MACD"],
        df["MACD_Signal"],
        "down",
    )

    result[f"{prefix}_macd_cross_up_age"] = up_age
    result[f"{prefix}_macd_cross_down_age"] = down_age

    result[f"{prefix}_macd_cross_up_now"] = int(
        up_age == 0
    )

    result[f"{prefix}_macd_cross_down_now"] = int(
        down_age == 0
    )

    result[f"{prefix}_macd_cross_up_freshness"] = (
        freshness_from_age(
            up_age
        )
    )

    result[f"{prefix}_macd_cross_down_freshness"] = (
        freshness_from_age(
            down_age
        )
    )

    result[f"{prefix}_macd_recent_bull_cross_below_zero"] = int(
        up_age <= 3
        and np.isfinite(macd)
        and macd < 0
    )

    result[f"{prefix}_macd_recent_bear_cross_above_zero"] = int(
        down_age <= 3
        and np.isfinite(macd)
        and macd > 0
    )

    result[f"{prefix}_macd_bull_cross_below_zero"] = int(
        up_age == 0
        and np.isfinite(macd)
        and macd < 0
    )

    result[f"{prefix}_macd_bear_cross_above_zero"] = int(
        down_age == 0
        and np.isfinite(macd)
        and macd > 0
    )

    result[f"{prefix}_macd_hist_slope_3"] = slope(
        df["MACD_Hist"],
        3,
    )

    result[f"{prefix}_macd_hist_slope_5"] = slope(
        df["MACD_Hist"],
        5,
    )

    result[f"{prefix}_macd_hist_acceleration"] = acceleration(
        df["MACD_Hist"],
        3,
        6,
    )

    spread_df = pd.DataFrame(
        {
            "spread": (
                pd.to_numeric(
                    df["MACD"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    df["MACD_Signal"],
                    errors="coerce",
                )
            )
        }
    )

    result[f"{prefix}_macd_spread_change_1"] = safe_change(
        spread_df,
        "spread",
        1,
    )

    result[f"{prefix}_macd_spread_change_3"] = safe_change(
        spread_df,
        "spread",
        3,
    )

    try:
        spread_series = (
            spread_df["spread"]
        )

        abs_spread = (
            spread_series.abs()
        )

        result[f"{prefix}_macd_converging"] = int(
            len(abs_spread) >= 2
            and np.isfinite(
                abs_spread.iloc[-1]
            )
            and np.isfinite(
                abs_spread.iloc[-2]
            )
            and abs_spread.iloc[-1]
            < abs_spread.iloc[-2]
        )

    except Exception:
        result[f"{prefix}_macd_converging"] = 0

    return result


# ============================================================
# RSI
# ============================================================

def rsi_features(df, prefix):
    result = {}

    if "RSI" not in df.columns:
        return result

    rsi_series = pd.to_numeric(
        df["RSI"],
        errors="coerce",
    )

    rsi = safe_value(
        df,
        "RSI",
        default=50,
    )

    percentile = rolling_percentile(
        rsi_series,
        100,
    )

    zscore = rolling_zscore(
        rsi_series,
        100,
    )

    result[f"{prefix}_rsi"] = rsi

    result[f"{prefix}_rsi_overbought"] = int(
        rsi >= 70
    )

    result[f"{prefix}_rsi_very_overbought"] = int(
        rsi >= 75
    )

    result[f"{prefix}_rsi_oversold"] = int(
        rsi <= 30
    )

    result[f"{prefix}_rsi_very_oversold"] = int(
        rsi <= 25
    )

    result[f"{prefix}_rsi_percentile"] = percentile
    result[f"{prefix}_rsi_zscore"] = zscore

    result[f"{prefix}_rsi_slope_3"] = slope(
        rsi_series,
        3,
    )

    result[f"{prefix}_rsi_slope_5"] = slope(
        rsi_series,
        5,
    )

    result[f"{prefix}_rsi_acceleration"] = acceleration(
        rsi_series,
        3,
        6,
    )

    try:
        turn_down_series = (
            (
                rsi_series.shift(2)
                < rsi_series.shift(1)
            )
            & (
                rsi_series
                < rsi_series.shift(1)
            )
        )

        turn_up_series = (
            (
                rsi_series.shift(2)
                > rsi_series.shift(1)
            )
            & (
                rsi_series
                > rsi_series.shift(1)
            )
        )

        overbought_context = (
            (
                rsi_series.shift(1)
                >= 65
            )
            | (
                rsi_series.shift(2)
                >= 65
            )
        )

        oversold_context = (
            (
                rsi_series.shift(1)
                <= 35
            )
            | (
                rsi_series.shift(2)
                <= 35
            )
        )

        overbought_turn_down = (
            turn_down_series
            & overbought_context
        )

        oversold_turn_up = (
            turn_up_series
            & oversold_context
        )

        turn_down_age = bars_since_true(
            turn_down_series
        )

        turn_up_age = bars_since_true(
            turn_up_series
        )

        ob_down_age = bars_since_true(
            overbought_turn_down
        )

        os_up_age = bars_since_true(
            oversold_turn_up
        )

        result[f"{prefix}_rsi_turn_down"] = int(
            turn_down_age == 0
        )

        result[f"{prefix}_rsi_turn_up"] = int(
            turn_up_age == 0
        )

        result[f"{prefix}_rsi_turn_down_age"] = (
            turn_down_age
        )

        result[f"{prefix}_rsi_turn_up_age"] = (
            turn_up_age
        )

        result[f"{prefix}_rsi_turn_down_freshness"] = (
            freshness_from_age(
                turn_down_age
            )
        )

        result[f"{prefix}_rsi_turn_up_freshness"] = (
            freshness_from_age(
                turn_up_age
            )
        )

        current_ob_turn = (
            (
                rsi >= 65
                or (
                    np.isfinite(percentile)
                    and percentile >= 0.90
                )
            )
            and turn_down_age == 0
        )

        current_os_turn = (
            (
                rsi <= 35
                or (
                    np.isfinite(percentile)
                    and percentile <= 0.10
                )
            )
            and turn_up_age == 0
        )

        result[f"{prefix}_rsi_overbought_turn_down"] = int(
            current_ob_turn
        )

        result[f"{prefix}_rsi_oversold_turn_up"] = int(
            current_os_turn
        )

        result[f"{prefix}_rsi_overbought_turn_down_age"] = (
            ob_down_age
        )

        result[f"{prefix}_rsi_oversold_turn_up_age"] = (
            os_up_age
        )

        result[f"{prefix}_rsi_overbought_turn_down_freshness"] = (
            freshness_from_age(
                ob_down_age
            )
        )

        result[f"{prefix}_rsi_oversold_turn_up_freshness"] = (
            freshness_from_age(
                os_up_age
            )
        )

    except Exception:
        result[f"{prefix}_rsi_turn_down"] = 0
        result[f"{prefix}_rsi_turn_up"] = 0

        result[f"{prefix}_rsi_overbought_turn_down"] = 0
        result[f"{prefix}_rsi_oversold_turn_up"] = 0

    return result


# ============================================================
# CHAIKIN / A-D
# ============================================================

def find_ad_series(df):
    possible_names = [
        "AD",
        "A_D",
        "ADL",
        "AccumDist",
        "AccumulationDistribution",
        "Accumulation_Distribution",
    ]

    for name in possible_names:
        if name in df.columns:
            return pd.to_numeric(
                df[name],
                errors="coerce",
            )

    required = [
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return None

    try:
        high = pd.to_numeric(
            df["High"],
            errors="coerce",
        )

        low = pd.to_numeric(
            df["Low"],
            errors="coerce",
        )

        close = pd.to_numeric(
            df["Close"],
            errors="coerce",
        )

        volume = pd.to_numeric(
            df["Volume"],
            errors="coerce",
        )

        denominator = (
            high - low
        ).replace(
            0,
            np.nan,
        )

        multiplier = (
            (
                (close - low)
                - (high - close)
            )
            / denominator
        )

        money_flow_volume = (
            multiplier * volume
        ).fillna(0)

        return (
            money_flow_volume.cumsum()
        )

    except Exception:
        return None


def pressure_features(series, prefix, name):
    result = {}

    if series is None:
        return result

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    current = safe_float(
        series.iloc[-1]
    )

    percentile = rolling_percentile(
        series,
        100,
    )

    zscore = rolling_zscore(
        series,
        100,
    )

    slope_3 = slope(
        series,
        3,
    )

    slope_5 = slope(
        series,
        5,
    )

    accel = acceleration(
        series,
        3,
        6,
    )

    result[f"{prefix}_{name}"] = current
    result[f"{prefix}_{name}_percentile"] = percentile
    result[f"{prefix}_{name}_zscore"] = zscore
    result[f"{prefix}_{name}_slope_3"] = slope_3
    result[f"{prefix}_{name}_slope_5"] = slope_5
    result[f"{prefix}_{name}_acceleration"] = accel

    inflated = (
        (
            np.isfinite(percentile)
            and percentile >= 0.90
        )
        or (
            np.isfinite(zscore)
            and zscore >= 1.5
        )
    )

    very_inflated = (
        (
            np.isfinite(percentile)
            and percentile >= 0.97
        )
        or (
            np.isfinite(zscore)
            and zscore >= 2.0
        )
    )

    deflated = (
        (
            np.isfinite(percentile)
            and percentile <= 0.10
        )
        or (
            np.isfinite(zscore)
            and zscore <= -1.5
        )
    )

    very_deflated = (
        (
            np.isfinite(percentile)
            and percentile <= 0.03
        )
        or (
            np.isfinite(zscore)
            and zscore <= -2.0
        )
    )

    result[f"{prefix}_{name}_inflated"] = int(
        inflated
    )

    result[f"{prefix}_{name}_very_inflated"] = int(
        very_inflated
    )

    result[f"{prefix}_{name}_deflated"] = int(
        deflated
    )

    result[f"{prefix}_{name}_very_deflated"] = int(
        very_deflated
    )

    result[f"{prefix}_{name}_inflation_score"] = clipped(
        max(
            (
                0.0
                if not np.isfinite(percentile)
                else (percentile - 0.50) / 0.50
            ),
            (
                0.0
                if not np.isfinite(zscore)
                else zscore / 2.5
            ),
        )
    )

    result[f"{prefix}_{name}_deflation_score"] = clipped(
        max(
            (
                0.0
                if not np.isfinite(percentile)
                else (0.50 - percentile) / 0.50
            ),
            (
                0.0
                if not np.isfinite(zscore)
                else (-zscore) / 2.5
            ),
        )
    )

    result[f"{prefix}_{name}_decompression"] = int(
        inflated
        and np.isfinite(slope_3)
        and slope_3 < 0
    )

    result[f"{prefix}_{name}_pressure_loss"] = int(
        inflated
        and np.isfinite(accel)
        and accel < 0
    )

    result[f"{prefix}_{name}_reinflation"] = int(
        deflated
        and np.isfinite(slope_3)
        and slope_3 > 0
    )

    decompression_age = historical_pressure_event_age(
        series,
        "decompression",
    )

    reinflation_age = historical_pressure_event_age(
        series,
        "reinflation",
    )

    result[f"{prefix}_{name}_decompression_age"] = (
        decompression_age
    )

    result[f"{prefix}_{name}_reinflation_age"] = (
        reinflation_age
    )

    result[f"{prefix}_{name}_decompression_freshness"] = (
        freshness_from_age(
            decompression_age
        )
    )

    result[f"{prefix}_{name}_reinflation_freshness"] = (
        freshness_from_age(
            reinflation_age
        )
    )

    return result


def flow_pressure_features(df, prefix):
    result = {}

    if "Chaikin" in df.columns:
        result.update(
            pressure_features(
                df["Chaikin"],
                prefix,
                "chaikin",
            )
        )

    if "CMF" in df.columns:
        result[f"{prefix}_cmf"] = safe_value(
            df,
            "CMF",
        )

        result[f"{prefix}_cmf_percentile"] = rolling_percentile(
            df["CMF"],
            100,
        )

        result[f"{prefix}_cmf_slope_3"] = slope(
            df["CMF"],
            3,
        )

        result[f"{prefix}_cmf_acceleration"] = acceleration(
            df["CMF"],
            3,
            6,
        )

    ad_series = find_ad_series(
        df
    )

    if ad_series is not None:
        result.update(
            pressure_features(
                ad_series,
                prefix,
                "ad",
            )
        )

    return result


# ============================================================
# OTHER INDICATORS
# ============================================================

def other_features(df, prefix):
    result = {}

    columns = [
        "ADX",
        "DIPlus",
        "DIMinus",
        "AroonUp",
        "AroonDown",
        "VolumeOsc",
        "Momentum",
        "StochRSI",
        "StochRSI_Signal",
    ]

    for column in columns:
        if column in df.columns:
            clean = (
                column
                .lower()
                .replace(
                    " ",
                    "_",
                )
            )

            result[f"{prefix}_{clean}"] = safe_value(
                df,
                column,
            )

            result[f"{prefix}_{clean}_change_3"] = safe_change(
                df,
                column,
                3,
            )

    if "ADX" in df.columns:
        result[f"{prefix}_adx_slope_3"] = slope(
            df["ADX"],
            3,
        )

        result[f"{prefix}_adx_acceleration"] = acceleration(
            df["ADX"],
            3,
            6,
        )

    return result


# ============================================================
# SINGLE TIMEFRAME
# ============================================================

def timeframe_features_v2(df, prefix):
    result = {}

    if (
        df is None
        or df.empty
    ):
        return result

    result.update(
        sar_features(
            df,
            prefix,
        )
    )

    result.update(
        heikin_ashi_features(
            df,
            prefix,
        )
    )

    result.update(
        bollinger_features(
            df,
            prefix,
        )
    )

    result.update(
        macd_features(
            df,
            prefix,
        )
    )

    result.update(
        rsi_features(
            df,
            prefix,
        )
    )

    result.update(
        flow_pressure_features(
            df,
            prefix,
        )
    )

    result.update(
        other_features(
            df,
            prefix,
        )
    )

    result[f"{prefix}_price_change_1"] = safe_pct_change(
        df,
        "Close",
        1,
    )

    result[f"{prefix}_price_change_3"] = safe_pct_change(
        df,
        "Close",
        3,
    )

    result[f"{prefix}_price_change_5"] = safe_pct_change(
        df,
        "Close",
        5,
    )

    return result


# ============================================================
# ENGINE SCORES
# ============================================================

def add_existing_engine_scores(analysis):
    return {
        "engine_buy_score": safe_float(
            analysis.get(
                "technical_buy_score"
            )
        ),
        "engine_sell_score": safe_float(
            analysis.get(
                "sell_score"
            )
        ),
        "engine_trend_score": safe_float(
            analysis.get(
                "trend_score"
            )
        ),
        "engine_entry_score": safe_float(
            analysis.get(
                "entry_score"
            )
        ),
        "engine_reversal_up": safe_float(
            analysis.get(
                "bullish_reversal"
            )
        ),
        "engine_reversal_down": safe_float(
            analysis.get(
                "bearish_risk"
            )
        ),
        "engine_confidence": safe_float(
            analysis.get(
                "confidence"
            )
        ),
    }


# ============================================================
# COMPOSITE / MULTI-TIMEFRAME
# ============================================================

def _fresh(features, key):
    return clipped(
        features.get(
            key,
            0.0,
        )
    )


def _age(
    features,
    key,
    default=MAX_EVENT_AGE + 1,
):
    try:
        value = float(
            features.get(
                key,
                default,
            )
        )

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return float(default)


def _sequence_score(
    ages,
    max_gap=5,
):
    try:
        ages = [
            float(a)
            for a in ages
        ]

        if any(
            (
                not np.isfinite(a)
                or a > MAX_EVENT_AGE
            )
            for a in ages
        ):
            return 0.0

        comparisons = []

        for older, newer in zip(
            ages[:-1],
            ages[1:],
        ):
            if (
                older >= newer
                and (
                    older - newer
                ) <= max_gap
            ):
                comparisons.append(
                    1.0
                )

            elif older >= newer:
                comparisons.append(
                    0.5
                )

            else:
                comparisons.append(
                    0.0
                )

        if not comparisons:
            return 0.0

        chronology = float(
            np.mean(
                comparisons
            )
        )

        recent_age = min(
            ages
        )

        return (
            chronology
            * freshness_from_age(
                recent_age
            )
        )

    except Exception:
        return 0.0


def multi_timeframe_features(features):
    result = {}

    weekly = features.get(
        "weekly_trend_score",
        np.nan,
    )

    daily = features.get(
        "daily_trend_score",
        np.nan,
    )

    h4 = features.get(
        "4h_trend_score",
        np.nan,
    )

    if (
        np.isfinite(h4)
        and np.isfinite(daily)
    ):
        result["mtf_4h_minus_daily"] = (
            h4 - daily
        )

    if (
        np.isfinite(daily)
        and np.isfinite(weekly)
    ):
        result["mtf_daily_minus_weekly"] = (
            daily - weekly
        )

    # ========================================================
    # STATO ATTUALE
    # ========================================================

    sar_4h_bull = bool(
        features.get(
            "4h_sar_bullish",
            0,
        )
    )

    sar_daily_bull = bool(
        features.get(
            "daily_sar_bullish",
            0,
        )
    )

    sar_weekly_bull = bool(
        features.get(
            "weekly_sar_bullish",
            0,
        )
    )

    macd_daily_bull = bool(
        features.get(
            "daily_macd_above_signal",
            0,
        )
    )

    macd_daily_bear = bool(
        features.get(
            "daily_macd_below_signal",
            0,
        )
    )

    ha_daily_green = bool(
        features.get(
            "daily_ha_green",
            0,
        )
    )

    ha_daily_red = bool(
        features.get(
            "daily_ha_red",
            0,
        )
    )

    bb_daily_above_mid = bool(
        features.get(
            "daily_bb_above_mid",
            0,
        )
    )

    # ========================================================
    # SAR LEAD / LAG
    # ========================================================

    result["mtf_sar_4h_bull_daily_bear"] = int(
        sar_4h_bull
        and not sar_daily_bull
    )

    result["mtf_sar_4h_bear_daily_bull"] = int(
        (not sar_4h_bull)
        and sar_daily_bull
    )

    result["mtf_sar_daily_bull_weekly_bear"] = int(
        sar_daily_bull
        and not sar_weekly_bull
    )

    result["mtf_sar_daily_bear_weekly_bull"] = int(
        (not sar_daily_bull)
        and sar_weekly_bull
    )

    result["mtf_sar_4h_flip_up_freshness"] = _fresh(
        features,
        "4h_sar_flip_up_freshness",
    )

    result["mtf_sar_4h_flip_down_freshness"] = _fresh(
        features,
        "4h_sar_flip_down_freshness",
    )

    result["mtf_sar_daily_flip_up_freshness"] = _fresh(
        features,
        "daily_sar_flip_up_freshness",
    )

    result["mtf_sar_daily_flip_down_freshness"] = _fresh(
        features,
        "daily_sar_flip_down_freshness",
    )

    # ========================================================
    # MACD LEAD / LAG
    # ========================================================

    result["mtf_macd_4h_cross_up_before_daily"] = int(
        _age(
            features,
            "4h_macd_cross_up_age",
        ) <= 2
        and _age(
            features,
            "daily_macd_cross_up_age",
        ) > 2
    )

    result["mtf_macd_4h_cross_down_before_daily"] = int(
        _age(
            features,
            "4h_macd_cross_down_age",
        ) <= 2
        and _age(
            features,
            "daily_macd_cross_down_age",
        ) > 2
    )

    # ========================================================
    # SELL PRE-ALERT
    # ========================================================

    sell_structure = max(
        _fresh(
            features,
            "daily_bb_upper_detach_freshness",
        ),
        _fresh(
            features,
            "daily_bb_reentry_upper_freshness",
        ),
        _fresh(
            features,
            "daily_ha_uncertainty_freshness",
        ),
        _fresh(
            features,
            "daily_ha_flip_red_freshness",
        ),
    )

    sell_pressure = max(
        _fresh(
            features,
            "daily_rsi_overbought_turn_down_freshness",
        ),
        _fresh(
            features,
            "daily_chaikin_decompression_freshness",
        ),
        _fresh(
            features,
            "daily_ad_decompression_freshness",
        ),
    )

    sell_momentum_event = _fresh(
        features,
        "daily_macd_cross_down_freshness",
    )

    sell_sar_event = _fresh(
        features,
        "daily_sar_flip_down_freshness",
    )

    sell_prealert_strength = clipped(
        0.30 * sell_structure
        + 0.30 * sell_pressure
        + 0.25 * sell_momentum_event
        + 0.15 * sell_sar_event
    )

    result["setup_early_sell_strength_daily"] = (
        sell_prealert_strength
    )

    result["setup_early_sell_daily"] = int(
        sell_prealert_strength >= 0.35
    )

    # ========================================================
    # BUY PRE-ALERT
    # ========================================================

    buy_structure = max(
        _fresh(
            features,
            "daily_bb_lower_detach_freshness",
        ),
        _fresh(
            features,
            "daily_bb_reentry_lower_freshness",
        ),
        _fresh(
            features,
            "daily_ha_uncertainty_freshness",
        ),
        _fresh(
            features,
            "daily_ha_flip_green_freshness",
        ),
    )

    buy_pressure = max(
        _fresh(
            features,
            "daily_rsi_oversold_turn_up_freshness",
        ),
        _fresh(
            features,
            "daily_chaikin_reinflation_freshness",
        ),
        _fresh(
            features,
            "daily_ad_reinflation_freshness",
        ),
    )

    buy_momentum_event = _fresh(
        features,
        "daily_macd_cross_up_freshness",
    )

    buy_sar_event = _fresh(
        features,
        "daily_sar_flip_up_freshness",
    )

    buy_prealert_strength = clipped(
        0.30 * buy_structure
        + 0.30 * buy_pressure
        + 0.25 * buy_momentum_event
        + 0.15 * buy_sar_event
    )

    result["setup_early_buy_strength_daily"] = (
        buy_prealert_strength
    )

    result["setup_early_buy_daily"] = int(
        buy_prealert_strength >= 0.35
    )

    # ========================================================
    # TREND ATTIVO
    #
    # Qui guardiamo lo STATO ATTUALE.
    # La memoria dell'evento non basta.
    # ========================================================

    sell_current_state = (
        0.40 * int(not sar_daily_bull)
        + 0.30 * int(macd_daily_bear)
        + 0.20 * int(ha_daily_red)
        + 0.10 * int(not bb_daily_above_mid)
    )

    buy_current_state = (
        0.40 * int(sar_daily_bull)
        + 0.30 * int(macd_daily_bull)
        + 0.20 * int(ha_daily_green)
        + 0.10 * int(bb_daily_above_mid)
    )

    result["setup_sell_current_state_strength"] = clipped(
        sell_current_state
    )

    result["setup_buy_current_state_strength"] = clipped(
        buy_current_state
    )

    result["setup_sell_trend_active"] = int(
        sell_current_state >= 0.60
    )

    result["setup_buy_trend_active"] = int(
        buy_current_state >= 0.60
    )

    # ========================================================
    # TREND DECAY
    #
    # Segnale importante:
    # trend ancora ribassista ma iniziano segnali opposti.
    # ========================================================

    sell_decay_pressure = max(
        _fresh(
            features,
            "daily_chaikin_reinflation_freshness",
        ),
        _fresh(
            features,
            "daily_ad_reinflation_freshness",
        ),
        _fresh(
            features,
            "daily_rsi_turn_up_freshness",
        ),
    )

    buy_decay_pressure = max(
        _fresh(
            features,
            "daily_chaikin_decompression_freshness",
        ),
        _fresh(
            features,
            "daily_ad_decompression_freshness",
        ),
        _fresh(
            features,
            "daily_rsi_turn_down_freshness",
        ),
    )

    result["setup_sell_decay_strength"] = clipped(
        sell_current_state
        * sell_decay_pressure
    )

    result["setup_buy_decay_strength"] = clipped(
        buy_current_state
        * buy_decay_pressure
    )

    result["setup_sell_trend_decaying"] = int(
        sell_current_state >= 0.60
        and sell_decay_pressure >= 0.30
    )

    result["setup_buy_trend_decaying"] = int(
        buy_current_state >= 0.60
        and buy_decay_pressure >= 0.30
    )

    # ========================================================
    # CONFERMA BUY / SELL
    #
    # REGOLA FONDAMENTALE V2.2:
    #
    # SELL confermato:
    # - SAR attualmente bearish
    # - MACD attualmente bearish
    #
    # BUY confermato:
    # - SAR attualmente bullish
    # - MACD attualmente bullish
    #
    # Quindi non possono essere contemporaneamente confermati.
    # ========================================================

    sell_confirmation_strength = clipped(
        0.45 * sell_current_state
        + 0.35 * sell_prealert_strength
        + 0.20 * sell_sar_event
    )

    buy_confirmation_strength = clipped(
        0.45 * buy_current_state
        + 0.35 * buy_prealert_strength
        + 0.20 * buy_sar_event
    )

    sell_confirmed = (
        (not sar_daily_bull)
        and macd_daily_bear
        and sell_current_state >= 0.60
    )

    buy_confirmed = (
        sar_daily_bull
        and macd_daily_bull
        and buy_current_state >= 0.60
    )

    # Sicurezza logica supplementare.
    if sell_confirmed:
        buy_confirmed = False

    elif buy_confirmed:
        sell_confirmed = False

    result["setup_sell_with_sar_confirmation_strength"] = (
        sell_confirmation_strength
        if sell_confirmed
        else 0.0
    )

    result["setup_buy_with_sar_confirmation_strength"] = (
        buy_confirmation_strength
        if buy_confirmed
        else 0.0
    )

    result["setup_sell_with_sar_confirmation"] = int(
        sell_confirmed
    )

    result["setup_buy_with_sar_confirmation"] = int(
        buy_confirmed
    )

    # ========================================================
    # SEQUENZA TEMPORALE DAILY
    # Bollinger -> HA -> MACD -> SAR
    # ========================================================

    sell_bb_age = min(
        _age(
            features,
            "daily_bb_upper_detach_age",
        ),
        _age(
            features,
            "daily_bb_reentry_upper_age",
        ),
    )

    sell_ha_age = min(
        _age(
            features,
            "daily_ha_uncertainty_age",
        ),
        _age(
            features,
            "daily_ha_flip_red_age",
        ),
    )

    sell_macd_age = _age(
        features,
        "daily_macd_cross_down_age",
    )

    sell_sar_age = _age(
        features,
        "daily_sar_flip_down_age",
    )

    buy_bb_age = min(
        _age(
            features,
            "daily_bb_lower_detach_age",
        ),
        _age(
            features,
            "daily_bb_reentry_lower_age",
        ),
    )

    buy_ha_age = min(
        _age(
            features,
            "daily_ha_uncertainty_age",
        ),
        _age(
            features,
            "daily_ha_flip_green_age",
        ),
    )

    buy_macd_age = _age(
        features,
        "daily_macd_cross_up_age",
    )

    buy_sar_age = _age(
        features,
        "daily_sar_flip_up_age",
    )

    result["setup_sell_sequence_boll_ha_macd_sar"] = (
        _sequence_score(
            [
                sell_bb_age,
                sell_ha_age,
                sell_macd_age,
                sell_sar_age,
            ]
        )
    )

    result["setup_buy_sequence_boll_ha_macd_sar"] = (
        _sequence_score(
            [
                buy_bb_age,
                buy_ha_age,
                buy_macd_age,
                buy_sar_age,
            ]
        )
    )

    # ========================================================
    # 4H -> DAILY EARLY WARNING
    # ========================================================

    result["setup_4h_early_sell_vs_daily"] = clipped(
        0.35
        * _fresh(
            features,
            "4h_sar_flip_down_freshness",
        )
        + 0.35
        * _fresh(
            features,
            "4h_macd_cross_down_freshness",
        )
        + 0.15
        * _fresh(
            features,
            "4h_ha_flip_red_freshness",
        )
        + 0.15
        * _fresh(
            features,
            "4h_bb_reentry_upper_freshness",
        )
    )

    result["setup_4h_early_buy_vs_daily"] = clipped(
        0.35
        * _fresh(
            features,
            "4h_sar_flip_up_freshness",
        )
        + 0.35
        * _fresh(
            features,
            "4h_macd_cross_up_freshness",
        )
        + 0.15
        * _fresh(
            features,
            "4h_ha_flip_green_freshness",
        )
        + 0.15
        * _fresh(
            features,
            "4h_bb_reentry_lower_freshness",
        )
    )

    return result


# ============================================================
# MAIN EXTRACTOR
# ============================================================

def extract_features_v2(analysis):
    result = {}

    if not analysis:
        return result

    result.update(
        add_existing_engine_scores(
            analysis
        )
    )

    frame_results = analysis.get(
        "frames",
        {},
    )

    for timeframe in TIMEFRAMES:
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
                "_",
            )
        )

        result[f"{prefix}_trend_score"] = safe_float(
            tf_result.get(
                "score"
            )
        )

        result[f"{prefix}_entry_score"] = safe_float(
            tf_result.get(
                "entry_score"
            )
        )

        result[f"{prefix}_reversal_up"] = safe_float(
            tf_result.get(
                "bullish_reversal"
            )
        )

        result[f"{prefix}_reversal_down"] = safe_float(
            tf_result.get(
                "bearish_reversal"
            )
        )

        result[f"{prefix}_overextension"] = safe_float(
            tf_result.get(
                "overextension"
            )
        )

        result[f"{prefix}_trend_age"] = safe_float(
            tf_result.get(
                "trend_age"
            )
        )

        indicators = tf_result.get(
            "indicators",
            {},
        )

        for name, value in indicators.items():
            clean_name = (
                str(name)
                .lower()
                .replace(
                    "/",
                    "_",
                )
                .replace(
                    " ",
                    "_",
                )
            )

            result[
                f"{prefix}_component_{clean_name}"
            ] = safe_float(
                value
            )

        events = tf_result.get(
            "events",
            {},
        )

        for event_name, event_value in events.items():
            result[
                f"{prefix}_event_{event_name}"
            ] = int(
                bool(event_value)
            )

        data = tf_result.get(
            "data"
        )

        if (
            data is not None
            and not data.empty
        ):
            result.update(
                timeframe_features_v2(
                    data,
                    prefix,
                )
            )

    result.update(
        multi_timeframe_features(
            result
        )
    )

    return result


# ============================================================
# FEATURE SUMMARY
# ============================================================

def feature_summary(features):
    if not features:
        return {}

    categories = {
        "SAR": 0,
        "MACD": 0,
        "RSI": 0,
        "Chaikin": 0,
        "AD": 0,
        "HeikinAshi": 0,
        "Bollinger": 0,
        "MultiTimeframe": 0,
        "Other": 0,
    }

    for feature in features.keys():
        name = feature.lower()

        if "sar" in name:
            categories["SAR"] += 1

        elif "macd" in name:
            categories["MACD"] += 1

        elif "rsi" in name:
            categories["RSI"] += 1

        elif "chaikin" in name:
            categories["Chaikin"] += 1

        elif "_ad_" in name:
            categories["AD"] += 1

        elif "_ha_" in name:
            categories["HeikinAshi"] += 1

        elif "_bb_" in name:
            categories["Bollinger"] += 1

        elif (
            "mtf_" in name
            or "setup_" in name
        ):
            categories["MultiTimeframe"] += 1

        else:
            categories["Other"] += 1

    return categories