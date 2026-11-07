"""
MARKETSENTINEL - PHASE NUOVA V1

Early BUY:
- SAR bullish flip
- Heikin-Ashi bullish
- Rating BUY1 V160 >= 7

Non modifica la PHASE canonica.
"""

import pandas as pd


VERSION = "PHASE_NUOVA_V1"
MIN_RATING = 7.0


def calculate_phase_nuova(
    df,
    phase_col="PHASE",
    sar_col="SAR_BULLISH_FLIP",
    ha_col="HA_DIRECTION",
    rating_col="RATING_V160_BUY1",
):
    required = [
        "Ticker",
        "Date",
        phase_col,
        sar_col,
        ha_col,
        rating_col,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Colonne mancanti: {missing}"
        )

    result = df.copy()
    result["Date"] = pd.to_datetime(result["Date"], utc=True)
    result = result.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    result["PHASE_NUOVA"] = result[phase_col]
    result["EARLY_BUY"] = False
    result["EARLY_BUY_RATING"] = float("nan")
    result["PHASE_NUOVA_REASON"] = "CANONICAL"

    for ticker, idx in result.groupby("Ticker").groups.items():

        positions = list(idx)
        early_stage = 0

        for pos in positions:

            phase = result.at[pos, phase_col]

            flip = result.at[pos, sar_col]
            ha = pd.to_numeric(
                result.at[pos, ha_col],
                errors="coerce"
            )
            rating = pd.to_numeric(
                result.at[pos, rating_col],
                errors="coerce"
            )

            # Progressione di un Early BUY già attivato
            if early_stage in (1, 2):
                compatible = phase in (
                    "BUY1", "BUY2", "BUY3",
                    "TREND_RIALZISTA", "DEBOLEZZA_RIALZISTA"
                )
                sar_side = pd.to_numeric(
                    result.at[pos, "SAR_SIDE"], errors="coerce"
                )
                ha_indecision = pd.to_numeric(
                    result.at[pos, "V4015_HA_INDECISION"],
                    errors="coerce"
                )
                confirmed = (
                    compatible
                    and sar_side == 1
                    and ha == 1
                    and ha_indecision == 0
                )
                if not confirmed:
                    early_stage = 0
                    continue
                early_stage += 1

                result.at[pos, "PHASE_NUOVA"] = (
                    f"BUY{early_stage}"
                )
                result.at[pos, "EARLY_BUY"] = True
                result.at[
                    pos, "PHASE_NUOVA_REASON"
                ] = "EARLY_BUY_PROGRESS"

                if early_stage == 3:
                    early_stage = 0

                continue

            bullish_flip = (
                pd.notna(flip)
                and str(flip).lower()
                in ("true", "1", "1.0")
            )

            bullish_ha = ha == 1

            valid_rating = (
                pd.notna(rating)
                and rating >= MIN_RATING
            )

            if (
                bullish_flip
                and bullish_ha
                and valid_rating
                and phase not in ("BUY1", "BUY2", "BUY3")
            ):

                early_stage = 1

                result.at[pos, "PHASE_NUOVA"] = "BUY1"
                result.at[pos, "EARLY_BUY"] = True
                result.at[
                    pos, "EARLY_BUY_RATING"
                ] = float(rating)
                result.at[
                    pos, "PHASE_NUOVA_REASON"
                ] = "SAR_HA_RATING_CONFIRM"

    return result
