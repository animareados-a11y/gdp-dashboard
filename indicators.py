import numpy as np
import pandas as pd


# ============================================================
# UTILITY
# ============================================================

def clip(x, lo=0.0, hi=10.0):
    try:
        return float(np.clip(x, lo, hi))
    except Exception:
        return 5.0


# ============================================================
# HEIKIN ASHI
# ============================================================

def heikin_ashi(df):

    ha = pd.DataFrame(index=df.index)

    ha["HA_Close"] = (
        df["Open"] +
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 4.0

    ha["HA_Open"] = 0.0

    if len(df) > 0:

        ha.iloc[
            0,
            ha.columns.get_loc("HA_Open")
        ] = (
            df["Open"].iloc[0] +
            df["Close"].iloc[0]
        ) / 2.0

        for i in range(1, len(df)):

            ha.iloc[
                i,
                ha.columns.get_loc("HA_Open")
            ] = (
                ha["HA_Open"].iloc[i - 1] +
                ha["HA_Close"].iloc[i - 1]
            ) / 2.0

    ha["HA_High"] = pd.concat(
        [
            df["High"],
            ha["HA_Open"],
            ha["HA_Close"],
        ],
        axis=1
    ).max(axis=1)

    ha["HA_Low"] = pd.concat(
        [
            df["Low"],
            ha["HA_Open"],
            ha["HA_Close"],
        ],
        axis=1
    ).min(axis=1)

    rng = (
        ha["HA_High"] -
        ha["HA_Low"]
    ).replace(0, np.nan)

    body = (
        ha["HA_Close"] -
        ha["HA_Open"]
    ).abs()

    ha["HA_BodyPct"] = (
        body / rng
    ).fillna(0)

    ha["HA_Indecision"] = (
        ha["HA_BodyPct"] < 0.25
    )

    ha["HA_Bull"] = (
        ha["HA_Close"] >
        ha["HA_Open"]
    )

    ha["HA_Bear"] = (
        ha["HA_Close"] <
        ha["HA_Open"]
    )

    return ha


# ============================================================
# PARABOLIC SAR
# ============================================================

def psar(df, step=0.02, max_step=0.20):

    if len(df) < 3:
        return pd.Series(
            index=df.index,
            dtype=float
        )

    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)

    sar = np.zeros(len(df))

    sar[0] = low[0]

    ep = high[0]
    af = step
    bull = True

    for i in range(1, len(df)):

        sar[i] = (
            sar[i - 1] +
            af * (ep - sar[i - 1])
        )

        if bull:

            sar[i] = min(
                sar[i],
                low[i - 1],
                low[i - 2] if i > 1 else low[i - 1],
            )

            if low[i] < sar[i]:

                bull = False
                sar[i] = ep
                ep = low[i]
                af = step

            elif high[i] > ep:

                ep = high[i]
                af = min(
                    max_step,
                    af + step
                )

        else:

            sar[i] = max(
                sar[i],
                high[i - 1],
                high[i - 2] if i > 1 else high[i - 1],
            )

            if high[i] > sar[i]:

                bull = True
                sar[i] = ep
                ep = high[i]
                af = step

            elif low[i] < ep:

                ep = low[i]
                af = min(
                    max_step,
                    af + step
                )

    return pd.Series(
        sar,
        index=df.index
    )


# ============================================================
# RSI
# ============================================================

def rsi(close, period=14):

    delta = close.diff()

    gain = (
        delta.clip(lower=0)
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    loss = (
        (-delta.clip(upper=0))
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    rs = (
        gain /
        loss.replace(0, np.nan)
    )

    return (
        100 -
        100 / (1 + rs)
    ).fillna(50)


# ============================================================
# ATR
# ============================================================

def atr(df, period=14):

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (
                df["High"] -
                df["Close"].shift()
            ).abs(),
            (
                df["Low"] -
                df["Close"].shift()
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# MACD
# ============================================================

def macd(close):

    fast = close.ewm(
        span=12,
        adjust=False
    ).mean()

    slow = close.ewm(
        span=26,
        adjust=False
    ).mean()

    line = fast - slow

    signal = line.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = line - signal

    return line, signal, histogram


# ============================================================
# BOLLINGER SU HEIKIN ASHI
# ============================================================

def bollinger(
    ha_close,
    period=20,
    mult=2
):

    middle = (
        ha_close
        .rolling(period)
        .mean()
    )

    std = (
        ha_close
        .rolling(period)
        .std()
    )

    upper = middle + mult * std
    lower = middle - mult * std

    return upper, middle, lower


# ============================================================
# STOCHASTIC RSI
# ============================================================

def stochastic_rsi(
    rsi_series,
    period=14
):

    low = (
        rsi_series
        .rolling(period)
        .min()
    )

    high = (
        rsi_series
        .rolling(period)
        .max()
    )

    raw = (
        100 *
        (
            (rsi_series - low) /
            (high - low).replace(
                0,
                np.nan
            )
        )
    ).fillna(50)

    return raw


# ============================================================
# CHAIKIN OSCILLATOR
# ============================================================

def chaikin_oscillator(df):

    denominator = (
        df["High"] -
        df["Low"]
    ).replace(0, np.nan)

    money_flow_multiplier = (
        (
            (df["Close"] - df["Low"]) -
            (df["High"] - df["Close"])
        )
        /
        denominator
    )

    money_flow_volume = (
        money_flow_multiplier *
        df["Volume"]
    )

    adl = (
        money_flow_volume
        .fillna(0)
        .cumsum()
    )

    fast = adl.ewm(
        span=3,
        adjust=False
    ).mean()

    slow = adl.ewm(
        span=10,
        adjust=False
    ).mean()

    return fast - slow


# ============================================================
# CHAIKIN MONEY FLOW
# ============================================================

def cmf(df, period=20):

    denominator = (
        df["High"] -
        df["Low"]
    ).replace(0, np.nan)

    multiplier = (
        (
            (df["Close"] - df["Low"]) -
            (df["High"] - df["Close"])
        )
        /
        denominator
    )

    mf_volume = (
        multiplier.fillna(0) *
        df["Volume"]
    )

    volume_sum = (
        df["Volume"]
        .rolling(period)
        .sum()
        .replace(0, np.nan)
    )

    result = (
        mf_volume
        .rolling(period)
        .sum()
        /
        volume_sum
    )

    return result.fillna(0)


# ============================================================
# VOLUME OSCILLATOR
# ============================================================

def volume_oscillator(volume):

    fast = (
        volume
        .ewm(
            span=5,
            adjust=False
        )
        .mean()
    )

    slow = (
        volume
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    return (
        100 *
        (fast - slow) /
        slow.replace(0, np.nan)
    ).fillna(0)


# ============================================================
# MOMENTUM
# ============================================================

def momentum(close, period=10):

    return (
        close -
        close.shift(period)
    ).fillna(0)


# ============================================================
# AROON
# ============================================================

def aroon(df, period=14):

    highs = df["High"]
    lows = df["Low"]

    def aroon_up_window(x):

        if len(x) < period + 1:
            return np.nan

        periods_since_high = (
            len(x) - 1 -
            int(np.argmax(x))
        )

        return (
            100 *
            (period - periods_since_high)
            /
            period
        )

    def aroon_down_window(x):

        if len(x) < period + 1:
            return np.nan

        periods_since_low = (
            len(x) - 1 -
            int(np.argmin(x))
        )

        return (
            100 *
            (period - periods_since_low)
            /
            period
        )

    up = highs.rolling(
        period + 1
    ).apply(
        aroon_up_window,
        raw=True
    )

    down = lows.rolling(
        period + 1
    ).apply(
        aroon_down_window,
        raw=True
    )

    return (
        up.fillna(50),
        down.fillna(50)
    )


# ============================================================
# ADX
# ============================================================

def adx(df, period=14):

    up = df["High"].diff()
    down = -df["Low"].diff()

    plus = up.where(
        (up > down) &
        (up > 0),
        0.0
    )

    minus = down.where(
        (down > up) &
        (down > 0),
        0.0
    )

    a = atr(df, period)

    plus_di = (
        100 *
        plus.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        /
        a.replace(0, np.nan)
    )

    minus_di = (
        100 *
        minus.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()
        /
        a.replace(0, np.nan)
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        /
        (
            plus_di +
            minus_di
        ).replace(
            0,
            np.nan
        )
    )

    adx_value = (
        dx
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
        .fillna(0)
    )

    return (
        adx_value,
        plus_di.fillna(0),
        minus_di.fillna(0)
    )


# ============================================================
# TUTTI GLI INDICATORI
# ============================================================

def add_all(df):

    out = df.copy()

    ha = heikin_ashi(df)

    out = out.join(ha)

    out["SAR"] = psar(df)

    out["ATR"] = atr(df)

    (
        out["BB_Upper"],
        out["BB_Mid"],
        out["BB_Lower"]
    ) = bollinger(
        out["HA_Close"]
    )

    (
        out["MACD"],
        out["MACD_Signal"],
        out["MACD_Hist"]
    ) = macd(
        out["Close"]
    )

    out["RSI"] = rsi(
        out["Close"]
    )

    out["StochRSI"] = stochastic_rsi(
        out["RSI"]
    )

    out["Chaikin"] = chaikin_oscillator(
        df
    )

    out["CMF"] = cmf(
        df
    )

    out["VolumeOsc"] = volume_oscillator(
        df["Volume"]
    )

    out["Momentum"] = momentum(
        out["Close"]
    )

    (
        out["AroonUp"],
        out["AroonDown"]
    ) = aroon(
        df
    )

    (
        out["ADX"],
        out["DIPlus"],
        out["DIMinus"]
    ) = adx(
        df
    )

    return out