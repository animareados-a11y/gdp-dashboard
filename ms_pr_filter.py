"""MarketSentinel - filtro PR DAILY/WEEKLY.

Il filtro qualifica esclusivamente BUY scientificamente già esistenti.
Non modifica SAR, Heikin-Ashi, rating V160 o fasi canoniche.
"""
import pandas as pd

BUY = ("BUY1", "BUY2", "BUY3")


def apply_pr_filter(df, timeframe="DAILY", phase_col="PHASE_NUOVA"):
    timeframe = timeframe.upper()
    if timeframe not in ("DAILY", "WEEKLY"):
        raise ValueError("Timeframe non valido")

    window = 365 if timeframe == "DAILY" else 52
    required = {"Ticker", "Date", "Close", phase_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {sorted(missing)}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], utc=True)
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    close = pd.to_numeric(out["Close"], errors="coerce")
    out["PR_MA"] = (
        close.groupby(out["Ticker"])
        .transform(lambda s: s.rolling(window, min_periods=window).mean())
    )

    previous_ma = out.groupby("Ticker")["PR_MA"].shift(1)
    out["PR_MA_ON"] = (
        out["PR_MA"].notna()
        & previous_ma.notna()
        & out["PR_MA"].lt(previous_ma)
    )

    out["PHASE_PRE_PR"] = out[phase_col].copy()
    out["PR_APPLIED"] = False
    out["PR_CYCLE_COMPLETED"] = False

    for _, positions in out.groupby("Ticker", sort=False).groups.items():
        completed = False
        previous_pr_stage = 0

        for pos in positions:
            phase = str(out.at[pos, "PHASE_PRE_PR"])
            ma_on = bool(out.at[pos, "PR_MA_ON"])

            if phase in ("SELL1", "SELL2", "SELL3", "TREND_RIBASSISTA", "DEBOLEZZA_RIBASSISTA"):
                completed = False
                previous_pr_stage = 0

            if not ma_on:
                previous_pr_stage = 0
                continue

            if completed:
                out.at[pos, "PR_CYCLE_COMPLETED"] = True
                continue

            if phase not in BUY:
                previous_pr_stage = 0
                continue

            stage = BUY.index(phase) + 1

            if stage == 1:
                previous_pr_stage = 1
                out.at[pos, phase_col] = "BUY1(PR)"
                out.at[pos, "PR_APPLIED"] = True
            elif stage == previous_pr_stage + 1:
                previous_pr_stage = stage
                out.at[pos, phase_col] = f"BUY{stage}(PR)"
                out.at[pos, "PR_APPLIED"] = True
                if stage == 3:
                    completed = True
                    out.at[pos, "PR_CYCLE_COMPLETED"] = True
            else:
                previous_pr_stage = 0

    return out
