"""Filtro SMA20 subordinato alla media lunga DAILY/WEEKLY.

Qualifica soltanto BUY1, BUY2 e BUY3.
Preserva i PR già attribuiti dal filtro SMA365/52.
Non modifica il motore scientifico.
"""
import pandas as pd

BUY = {"BUY1", "BUY2", "BUY3"}


def apply_sma20_pr_filter(df, timeframe="DAILY",
                          phase_col="PHASE_NUOVA"):
    timeframe = timeframe.upper()
    if timeframe not in ("DAILY", "WEEKLY"):
        raise ValueError("Timeframe non valido")

    window = 200 if timeframe == "DAILY" else 40

    required = {
        "Ticker", "Date", "Close",
        "V4015_HA_CLOSE", "V4015_BB_MID", phase_col
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {sorted(missing)}")

    out = df.copy()
    out["Date"] = pd.to_datetime(
        out["Date"], utc=True, format="mixed"
    )
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    close = pd.to_numeric(out["Close"], errors="coerce")
    ha = pd.to_numeric(out["V4015_HA_CLOSE"], errors="coerce")
    sma20 = pd.to_numeric(out["V4015_BB_MID"], errors="coerce")

    out["SMA_PR_LONG"] = close.groupby(out["Ticker"]).transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )

    phase = out[phase_col].astype("string")
    eligible = phase.isin(BUY)

    condition = (
        eligible
        & close.lt(out["SMA_PR_LONG"])
        & ha.le(sma20)
    )

    out["SMA20_PR_APPLIED"] = condition.fillna(False)

    out.loc[condition, phase_col] = (
        phase.loc[condition] + "(PR)"
    )

    return out
