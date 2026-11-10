"""MarketSentinel - qualificazione PR unificata DAILY/WEEKLY.

Non modifica le fasi scientifiche, gli indicatori o i rating.
Le sequenze PR sono attribuite soltanto a BUY scientifici consecutivi.
"""
import pandas as pd

BUY = ("BUY1", "BUY2", "BUY3")
BEARISH = {
    "SELL1", "SELL2", "SELL3",
    "TREND_RIBASSISTA", "DEBOLEZZA_RIBASSISTA",
}


def apply_pr_unified(df, timeframe="DAILY", phase_col="PHASE_NUOVA"):
    timeframe = timeframe.upper()
    if timeframe not in ("DAILY", "WEEKLY"):
        raise ValueError("Timeframe non valido")

    required = {
        "Ticker", "Date", "Close", phase_col,
        "V4015_HA_OPEN", "V4015_HA_CLOSE", "V4015_BB_MID",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {sorted(missing)}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], utc=True, format="mixed")
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    close = pd.to_numeric(out["Close"], errors="coerce")
    ha = pd.to_numeric(out["V4015_HA_CLOSE"], errors="coerce")
    ha_open = pd.to_numeric(out["V4015_HA_OPEN"], errors="coerce")
    bb_mid = pd.to_numeric(out["V4015_BB_MID"], errors="coerce")

    long_window = 365 if timeframe == "DAILY" else 52
    secondary_window = 200 if timeframe == "DAILY" else 40

    out["PR_MA"] = close.groupby(out["Ticker"]).transform(
        lambda s: s.rolling(long_window, min_periods=long_window).mean()
    )
    previous_ma = out.groupby("Ticker")["PR_MA"].shift(1)

    out["PR_MA_ON"] = (
        out["PR_MA"].notna()
        & previous_ma.notna()
        & out["PR_MA"].lt(previous_ma)
    )

    out["SMA_PR_LONG"] = close.groupby(out["Ticker"]).transform(
        lambda s: s.rolling(
            secondary_window, min_periods=secondary_window
        ).mean()
    )

    # Pendenza significativa SMA200 DAILY / SMA40 WEEKLY.
    # Oscillazioni entro +/-1% nel periodo sono considerate fisiologiche.
    slope_period = 10 if timeframe == "DAILY" else 4
    previous_secondary = out.groupby("Ticker")["SMA_PR_LONG"].shift(slope_period)
    out["SMA_PR_SLOPE_PCT"] = (
        (out["SMA_PR_LONG"] / previous_secondary - 1.0) * 100.0
    )
    out["SMA_PR_DECLINING"] = (
        previous_secondary.gt(0)
        & out["SMA_PR_SLOPE_PCT"].lt(-1.0)
    ).fillna(False)

    previous_mid = bb_mid.groupby(out["Ticker"]).shift(1)
    out["SMA20_DECLINING"] = (
        bb_mid.notna()
        & previous_mid.notna()
        & bb_mid.lt(previous_mid)
    )
    out["SMA20_BODY_CROSS"] = (
        ha_open.notna()
        & ha.notna()
        & bb_mid.notna()
        & ha.gt(ha_open)
        & ha_open.le(bb_mid)
        & bb_mid.lt(ha)
    )
    out["SMA20_PR_ON"] = (
        close.lt(out["SMA_PR_LONG"])
        & out["SMA20_DECLINING"]
        & out["SMA_PR_DECLINING"]
    ).fillna(False)

    out["PR_CONDITION"] = (
        out["PR_MA_ON"] | out["SMA20_PR_ON"]
    )

    out["PHASE_PRE_PR"] = out[phase_col].copy()
    out["PR_APPLIED"] = False
    out["PR_CYCLE_COMPLETED"] = False
    out["PR_SEQUENCE_BLOCKED"] = False

    for _, positions in out.groupby("Ticker", sort=False).groups.items():
        stage = 0
        clean_stage = 0
        for pos in positions:
            scientific = str(out.at[pos, "PHASE_PRE_PR"])
            sar = pd.to_numeric(out.at[pos, "SAR_SIDE"], errors="coerce")
            ha_dir = pd.to_numeric(
                out.at[pos, "V4015_HA_DIRECTION"], errors="coerce"
            )
            indecision = pd.to_numeric(
                out.at[pos, "V4015_HA_INDECISION"], errors="coerce"
            )
            bullish = sar == 1 and ha_dir == 1 and indecision == 0
            ma_pr = bool(out.at[pos, "PR_MA_ON"])
            falling = bool(out.at[pos, "SMA20_DECLINING"])
            secondary = out.at[pos, "SMA_PR_LONG"]
            below = pd.notna(secondary) and close.iloc[pos] < secondary

            # SMA365/52 resta una condizione PR indipendente.
            # Sotto SMA200/40, SMA20 decrescente impedisce BUY puliti.
            force_pr = bool(
                below and falling and out.at[pos, "SMA_PR_DECLINING"]
            )
            condition = ma_pr or force_pr

            # Uscita PR: prezzo sopra entrambe le medie.
            # DAILY: SMA20 + SMA200
            # WEEKLY: SMA20 + SMA40
            above_both = (
                pd.notna(secondary)
                and pd.notna(bb_mid.iloc[pos])
                and close.iloc[pos] > secondary
                and close.iloc[pos] > bb_mid.iloc[pos]
            )
            if above_both:
                if stage > 0 and bullish and scientific in (
                    "BUY1", "BUY2", "BUY3", "TREND_RIALZISTA"
                ):
                    clean_stage = 1
                    out.at[pos, phase_col] = "BUY1"
                    out.at[pos, "PHASE_NUOVA_REASON"] = "PR_EXIT_RESTART_BUY1"
                stage = 0
                if clean_stage == 0:
                    continue

            if clean_stage:
                compatible = scientific in (
                    "BUY1", "BUY2", "BUY3",
                    "TREND_RIALZISTA", "DEBOLEZZA_RIALZISTA"
                )
                if not bullish or not compatible:
                    clean_stage = 0
                elif out.at[pos, "PHASE_NUOVA_REASON"] == "PR_EXIT_RESTART_BUY1":
                    continue
                elif clean_stage < 3:
                    clean_stage += 1
                    out.at[pos, phase_col] = f"BUY{clean_stage}"
                    out.at[pos, "PHASE_NUOVA_REASON"] = "PR_EXIT_BUY_PROGRESS"
                    continue
                elif scientific == "TREND_RIALZISTA":
                    clean_stage = 0
                    out.at[pos, phase_col] = "TREND_RIALZISTA"
                    continue
                else:
                    out.at[pos, phase_col] = "BUY3"
                    continue

            if scientific in BEARISH or scientific in (
                "DEBOLEZZA_RIALZISTA", "INDECISIONE"
            ):
                stage = 0
                continue

            if scientific == "BUY1":
                if condition:
                    stage = 1
                    out.at[pos, phase_col] = "BUY1(PR)"
                    out.at[pos, "PR_APPLIED"] = True
                else:
                    stage = 0
                continue

            if stage == 0:
                continue

            # La progressione PR non dipende dal superamento
            # della SMA20: conta la sequenza scientifica.
            if stage == 1 and scientific == "BUY2":
                stage = 2
                out.at[pos, phase_col] = "BUY2(PR)"
            elif stage == 2 and scientific == "BUY3":
                stage = 3
                out.at[pos, phase_col] = "BUY3(PR)"
            elif stage >= 3 and scientific == "TREND_RIALZISTA":
                stage = 4
                out.at[pos, phase_col] = "TREND_RIALZISTA(PR)"
            else:
                out.at[pos, "PR_SEQUENCE_BLOCKED"] = True
                stage = 0
                continue

            out.at[pos, "PR_APPLIED"] = True
            if stage >= 3:
                out.at[pos, "PR_CYCLE_COMPLETED"] = True

    return out
