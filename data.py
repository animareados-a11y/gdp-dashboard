import warnings
warnings.filterwarnings("ignore")

from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# MARKET SENTINEL - DATA ENGINE V3.2
# ============================================================


# ------------------------------------------------------------
# CORREZIONI TICKER
# ------------------------------------------------------------

TICKER_ALIASES = {
    # Italia
    "PIR.MI": "PIRC.MI",

    # Europa
    "ADYEN.AMS": "ADYEN.AS",
    "PRX.AMS": "PRX.AS",

    # Flutter è ora quotata anche NYSE.
    # Se nel tuo config c'è ancora FLTR.ID, usiamo FLUT.
    "FLTR.ID": "FLUT",
}


# Ticker che dai test precedenti risultavano non validi
# e rallentavano Yahoo Finance.
BLOCKED_TICKERS = {
    "BCA.MI",
    "BPSO.MI",
}


# ============================================================
# TICKER
# ============================================================

def normalize_ticker(ticker):
    ticker = str(ticker).strip().upper()

    if ticker in BLOCKED_TICKERS:
        return None

    return TICKER_ALIASES.get(
        ticker,
        ticker
    )


def clean_universe(tickers):
    result = []

    for ticker in tickers:
        fixed = normalize_ticker(ticker)

        if fixed is None:
            continue

        if fixed not in result:
            result.append(fixed)

    return result


# ============================================================
# CLEAN OHLCV
# ============================================================

def _clean_ohlcv(df):
    if df is None or df.empty:
        return None

    df = df.copy()

    try:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for col in required:
        if col not in df.columns:
            return None

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    if df.empty:
        return None

    df["Volume"] = (
        df["Volume"]
        .fillna(0)
    )

    return df.sort_index()


# ============================================================
# ESTRAZIONE DA DOWNLOAD MULTI-TICKER
# ============================================================

def _extract_ticker(downloaded, ticker):
    if downloaded is None or downloaded.empty:
        return None

    try:
        if isinstance(
            downloaded.columns,
            pd.MultiIndex
        ):
            level0 = (
                downloaded.columns
                .get_level_values(0)
            )

            level1 = (
                downloaded.columns
                .get_level_values(1)
            )

            # group_by="ticker"
            if ticker in level0:
                return _clean_ohlcv(
                    downloaded[
                        ticker
                    ].copy()
                )

            # Compatibilità con struttura Price/Ticker
            if ticker in level1:
                extracted = (
                    downloaded
                    .xs(
                        ticker,
                        axis=1,
                        level=1
                    )
                    .copy()
                )

                return _clean_ohlcv(
                    extracted
                )

        # Singolo ticker
        return _clean_ohlcv(
            downloaded.copy()
        )

    except Exception:
        return None


# ============================================================
# RESAMPLING DAILY
# ============================================================

def _resample_calendar(df, rule):
    if df is None or df.empty:
        return None

    try:
        result = (
            df
            .resample(rule)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna(
                subset=[
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
            )
        )

        return result

    except Exception:
        return None


# ============================================================
# RESAMPLING INTRADAY
# ============================================================

def _intraday_chunks(df, bars_per_candle):
    """
    Trasforma:
      1H -> 2H
      1H -> 4H

    senza mischiare due giornate diverse.
    """

    if df is None or df.empty:
        return None

    pieces = []

    try:
        grouped = df.groupby(
            df.index.date
        )

        for _, day in grouped:
            day = day.sort_index()

            if day.empty:
                continue

            group_number = (
                np.arange(
                    len(day)
                )
                //
                bars_per_candle
            )

            aggregated = (
                day
                .groupby(
                    group_number
                )
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
            )

            new_index = []

            for group_id in sorted(
                set(group_number)
            ):
                mask = (
                    group_number
                    ==
                    group_id
                )

                original_index = (
                    day.index[
                        mask
                    ]
                )

                new_index.append(
                    original_index[-1]
                )

            aggregated.index = (
                pd.DatetimeIndex(
                    new_index
                )
            )

            pieces.append(
                aggregated
            )

        if not pieces:
            return None

        return (
            pd.concat(
                pieces
            )
            .sort_index()
        )

    except Exception:
        return None


# ============================================================
# DOWNLOAD DI UN CHUNK
# ============================================================

def _download_chunk(
    tickers,
    period,
    interval,
):
    if not tickers:
        return pd.DataFrame()

    ticker_string = " ".join(
        tickers
    )

    try:
        return yf.download(
            tickers=ticker_string,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=False,
            progress=False,
            timeout=15,
        )

    except Exception:
        return pd.DataFrame()


# ============================================================
# DOWNLOAD MERCATO
# ============================================================

def download_market_timeframes(
    tickers,
    progress_callback=None,
):
    """
    Scarica i ticker a piccoli blocchi.

    Questo evita:
    - centinaia di chiamate individuali
    - sovraccarico Yahoo
    - blocchi lunghi con threads multipli
    - eccessivi HTTP 401 / crumb errors

    Daily:
        Daily
        Weekly
        Monthly

    1H:
        1H
        2H
        4H
    """

    tickers = clean_universe(
        tickers
    )

    if not tickers:
        return {}

    # Blocchi piccoli e prevedibili.
    chunk_size = 8

    chunks = [
        tickers[i:i + chunk_size]
        for i in range(
            0,
            len(tickers),
            chunk_size
        )
    ]

    daily_frames = {}
    hourly_frames = {}

    total_steps = max(
        len(chunks) * 2,
        1
    )

    step = 0

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    for chunk in chunks:
        step += 1

        if progress_callback:
            progress_callback(
                step,
                total_steps,
                "Dati Daily"
            )

        downloaded = _download_chunk(
            chunk,
            period="10y",
            interval="1d",
        )

        for ticker in chunk:
            df = _extract_ticker(
                downloaded,
                ticker
            )

            if df is not None:
                daily_frames[
                    ticker
                ] = df

    # --------------------------------------------------------
    # HOURLY
    #
    # 60 giorni sono più che sufficienti per avere
    # > 60 barre anche su 4H.
    # --------------------------------------------------------

    for chunk in chunks:
        step += 1

        if progress_callback:
            progress_callback(
                step,
                total_steps,
                "Dati intraday"
            )

        downloaded = _download_chunk(
            chunk,
            period="60d",
            interval="1h",
        )

        for ticker in chunk:
            df = _extract_ticker(
                downloaded,
                ticker
            )

            if df is not None:
                hourly_frames[
                    ticker
                ] = df

    # --------------------------------------------------------
    # COSTRUZIONE TIMEFRAME
    # --------------------------------------------------------

    result = {}

    for ticker in tickers:
        frames = {}

        daily = daily_frames.get(
            ticker
        )

        hourly = hourly_frames.get(
            ticker
        )

        if daily is not None:
            frames[
                "Daily"
            ] = daily

            weekly = _resample_calendar(
                daily,
                "W-FRI"
            )

            if weekly is not None:
                frames[
                    "Weekly"
                ] = weekly

            monthly = _resample_calendar(
                daily,
                "ME"
            )

            if monthly is None:
                monthly = _resample_calendar(
                    daily,
                    "M"
                )

            if monthly is not None:
                frames[
                    "Monthly"
                ] = monthly

        if hourly is not None:
            frames[
                "1H"
            ] = hourly

            h2 = _intraday_chunks(
                hourly,
                2
            )

            h4 = _intraday_chunks(
                hourly,
                4
            )

            if h2 is not None:
                frames[
                    "2H"
                ] = h2

            if h4 is not None:
                frames[
                    "4H"
                ] = h4

        if frames:
            result[
                ticker
            ] = frames

    return result


# ============================================================
# COMPATIBILITÀ get_timeframe
# ============================================================

@lru_cache(maxsize=512)
def get_timeframe(
    ticker,
    timeframe
):
    ticker = normalize_ticker(
        ticker
    )

    if ticker is None:
        return None

    result = (
        download_market_timeframes(
            [ticker]
        )
    )

    if ticker not in result:
        return None

    return (
        result[
            ticker
        ].get(
            timeframe
        )
    )


# ============================================================
# INFO FONDAMENTALI
# ============================================================

@lru_cache(maxsize=512)
def get_info(ticker):
    ticker = normalize_ticker(
        ticker
    )

    output = {
        "price": np.nan,
        "target": np.nan,
        "beta": np.nan,
    }

    if ticker is None:
        return output

    try:
        obj = yf.Ticker(
            ticker
        )

        # ----------------------------------------------------
        # FAST INFO
        # ----------------------------------------------------

        try:
            fast = obj.fast_info

            if fast:
                value = fast.get(
                    "last_price"
                )

                if value is not None:
                    output[
                        "price"
                    ] = float(
                        value
                    )

        except Exception:
            pass

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        try:
            info = obj.info or {}

            current = info.get(
                "currentPrice"
            )

            if current is not None:
                output[
                    "price"
                ] = float(
                    current
                )

            target = info.get(
                "targetMeanPrice"
            )

            if target is not None:
                output[
                    "target"
                ] = float(
                    target
                )

            beta = info.get(
                "beta"
            )

            if beta is not None:
                output[
                    "beta"
                ] = float(
                    beta
                )

        except Exception:
            pass

    except Exception:
        pass

    return output