"""
MarketSentinel
PHASE DYNAMICS SHADOW V5
========================

OBIETTIVO
---------
Estendere V4 con una protezione BUY-only dopo un lungo regime bearish.

REGOLA V5
---------
Dopo almeno N=6 settimane del precedente regime SAR bearish,
un nuovo flip SAR bullish NON genera subito BUY.

Servono K=3 HA bullish decisive.

Conferme:
- HA bullish decisiva:
    +1
- HA di indecisione:
    mantiene il conteggio
- HA bearish decisiva:
    RESET del conteggio a 0

Quindi K=3 significa:
3 HA bullish decisive senza che tra loro compaia
una HA bearish decisiva.

Quando viene raggiunta la terza conferma:

    settimana conferma -> BUY1
    settimana successiva -> BUY2
    settimana successiva -> BUY3

Dalla quarta settimana dopo la conferma:
si torna alla normale classificazione dinamica V4.

IMPORTANTE
----------
- BUY protection only
- SELL invariato
- V4 invariata
- V40.10 invariata
- production invariata
- engine.py invariato

Questa è una versione SHADOW sperimentale.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v4 as v4

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

VERSION = "V5"

N_PREV_BEAR_MIN = 6
K_BULL_HA_CONFIRMATIONS = 3


# ============================================================
# UTILS
# ============================================================

def _num_value(value, default=0):
    try:
        if pd.isna(value):
            return default

        if isinstance(value, (bool, np.bool_)):
            return np.bool_(value)

        if isinstance(value, (int, np.integer)):
            return np.int64(value)

        if isinstance(value, (float, np.floating)):
            return np.float64(value)

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
            if value.lstrip("+-").isdigit():
                return np.int64(value)

        number = pd.to_numeric(value, errors="coerce")
        if pd.isna(number):
            return default
        if isinstance(number, (bool, np.bool_)):
            return np.bool_(number)
        if isinstance(number, (int, np.integer)):
            return np.int64(number)
        return np.float64(number)

    except Exception:
        return default


def _find_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col

    return None


def _safe_test_phase(phase):
    """
    Durante l'attesa BUY:
    evita di mantenere BUY proveniente da V40.10.

    Le altre fasi V4 vengono conservate.
    """

    if pd.isna(phase):
        return "INDECISIONE"

    phase = str(phase)

    if phase == "BUY":
        return "INDECISIONE"

    return phase


# ============================================================
# BUY LONG-BEAR CONFIRMATION
# ============================================================

def apply_buy_long_bear_confirmation(df: pd.DataFrame) -> pd.DataFrame:
    """Protezione BUY mediante attraversamento HA della SMA20."""
    g = df.copy().reset_index(drop=True)

    required = [
        "SAR_SIDE", "SAR_AGE", "HA_Open", "HA_Close",
        "HA_DIRECTION", "HA_INDECISION", "BB_MIDDLE",
        "PHASE_DYNAMICS_V4", "PHASE_DYNAMICS_V4_PRE_V4010",
    ]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise KeyError(f"Colonne mancanti SMA20: {missing}")

    g["PHASE_DYNAMICS_V5"] = g["PHASE_DYNAMICS_V4"].copy()
    g["V5_BUY_LONG_BEAR_PROTECTED"] = 0
    g["V5_BUY_CONFIRM_COUNT"] = 0
    g["V5_BUY_CONFIRMATION_REACHED"] = 0
    g["V5_BUY_CONFIRMATION_AGE"] = 0
    g["V5_BUY_STAGE"] = ""
    g["V5_BUY_REASON"] = "V4_UNCHANGED"
    g["V5_BUY_CONFIRM_RESET"] = 0
    g["V5_BUY_RESET_COUNT"] = 0

    def number(value):
        return pd.to_numeric(value, errors="coerce")

    i = 0
    while i < len(g):
        side = number(g.at[i, "SAR_SIDE"])
        age = number(g.at[i, "SAR_AGE"])
        if side != 1 or age != 1:
            i += 1
            continue

        start = i
        while i < len(g) and number(g.at[i, "SAR_SIDE"]) == 1:
            i += 1
        positions = range(start, i)

        sma = number(g.at[start, "BB_MIDDLE"])
        ha_open = number(g.at[start, "HA_Open"])
        if pd.isna(sma) or pd.isna(ha_open) or ha_open >= sma:
            continue

        stage = 0
        interrupted = False
        trend_seen = False

        for j in positions:
            g.at[j, "V5_BUY_LONG_BEAR_PROTECTED"] = 1
            base = str(g.at[j, "PHASE_DYNAMICS_V4_PRE_V4010"])
            if base == "TREND_RIALZISTA":
                trend_seen = True

            ha_o = number(g.at[j, "HA_Open"])
            ha_c = number(g.at[j, "HA_Close"])
            mid = number(g.at[j, "BB_MIDDLE"])
            bullish = (
                number(g.at[j, "HA_DIRECTION"]) == 1
                and number(g.at[j, "HA_INDECISION"]) == 0
                and pd.notna(ha_o) and pd.notna(ha_c)
                and ha_c > ha_o
            )
            crossing = (
                bullish and pd.notna(mid)
                and ha_o <= mid < ha_c
            )

            if stage == 0:
                if crossing and not trend_seen:
                    stage = 1
                    g.at[j, "PHASE_DYNAMICS_V5"] = "BUY"
                    g.at[j, "V5_BUY_STAGE"] = "BUY1"
                    g.at[j, "V5_BUY_REASON"] = "SMA20_CROSS_BUY1"
                    g.at[j, "V5_BUY_CONFIRMATION_REACHED"] = 1
                else:
                    g.at[j, "PHASE_DYNAMICS_V5"] = _safe_test_phase(base)
                    g.at[j, "V5_BUY_STAGE"] = "WAIT_SMA20"
                    g.at[j, "V5_BUY_REASON"] = "WAIT_HA_SMA20_CROSS"
                continue

            if stage in (1, 2):
                incompatible = base in (
                    "INDECISIONE",
                    "DEBOLEZZA_RIBASSISTA",
                    "TREND_RIBASSISTA",
                    "SELL1", "SELL2", "SELL3",
                )
                if not interrupted and not incompatible and (bullish or base == "TREND_RIALZISTA"):
                    stage += 1
                    g.at[j, "PHASE_DYNAMICS_V5"] = "BUY"
                    g.at[j, "V5_BUY_STAGE"] = f"BUY{stage}"
                    g.at[j, "V5_BUY_REASON"] = f"SMA20_CONFIRMED_BUY{stage}"
                else:
                    interrupted = True
                    actual_phase = _safe_test_phase(base)
                    if number(g.at[j, "HA_INDECISION"]) == 1 and actual_phase == "TREND_RIALZISTA":
                        actual_phase = "INDECISIONE"
                    g.at[j, "PHASE_DYNAMICS_V5"] = actual_phase
                    g.at[j, "V5_BUY_STAGE"] = "INTERRUPTED"
                    g.at[j, "V5_BUY_REASON"] = "BUY_SEQUENCE_INTERRUPTED"
                continue

            g.at[j, "PHASE_DYNAMICS_V5"] = _safe_test_phase(base)
            g.at[j, "V5_BUY_STAGE"] = "POST_BUY3"
            g.at[j, "V5_BUY_REASON"] = "SMA20_POST_BUY3"

    return g


# ============================================================
# PROCESS TICKER
# ============================================================

def process_ticker(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # V4 resta completamente intatta
    base = v4.process_ticker(
        df.copy()
    )

    # V5 applica solo la nuova protezione BUY
    out = apply_buy_long_bear_confirmation(
        base
    )

    return out


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def apply_phase_dynamics_shadow(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return process_ticker(df)