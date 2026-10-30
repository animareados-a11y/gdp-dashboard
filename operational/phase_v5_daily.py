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

N_PREV_BEAR_MIN = 12
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

def apply_buy_long_bear_confirmation(
    df: pd.DataFrame,
) -> pd.DataFrame:

    g = df.copy().reset_index(drop=True)

    # --------------------------------------------------------
    # Individuazione colonne V4 / V40.10
    # --------------------------------------------------------

    phase_v4_col = _find_col(
        g,
        [
            "PHASE_DYNAMICS_V4",
            "PHASE",
        ],
    )

    if phase_v4_col is None:
        raise KeyError(
            "Impossibile trovare PHASE_DYNAMICS_V4 / PHASE."
        )

    pre_v4010_col = _find_col(
        g,
        [
            "PHASE_DYNAMICS_V4_PRE_V4010",
            "PHASE_DYNAMICS_V31",
            "PHASE",
        ],
    )

    if pre_v4010_col is None:
        raise KeyError(
            "Impossibile trovare la phase pre-V40.10."
        )

    prev_sar_age_col = _find_col(
        g,
        [
            "V4010_PREV_SAR_AGE",
            "PREV_SAR_AGE",
            "V4010_PREVIOUS_SAR_AGE",
        ],
    )

    if prev_sar_age_col is None:
        raise KeyError(
            "Impossibile trovare la colonna PREV SAR AGE."
        )

    run_qualified_col = _find_col(
        g,
        [
            "V4010_RUN_QUALIFIED",
            "RUN_QUALIFIED",
        ],
    )

    if run_qualified_col is None:
        raise KeyError(
            "Impossibile trovare V4010_RUN_QUALIFIED."
        )

    run_id_col = _find_col(
        g,
        [
            "V4010_SAR_RUN_ID",
            "SAR_RUN_ID",
        ],
    )

    if run_id_col is None:
        raise KeyError(
            "Impossibile trovare V4010_SAR_RUN_ID."
        )

    required = [
        "SAR_SIDE",
        "SAR_AGE",
        "HA_DIRECTION",
        "HA_INDECISION",
    ]

    missing = [
        c for c in required
        if c not in g.columns
    ]

    if missing:
        raise KeyError(
            f"Colonne mancanti V5: {missing}"
        )

    # --------------------------------------------------------
    # Output V5
    # --------------------------------------------------------

    g["PHASE_DYNAMICS_V5"] = (
        g[phase_v4_col].copy()
    )

    g["V5_BUY_LONG_BEAR_PROTECTED"] = 0
    g["V5_BUY_CONFIRM_COUNT"] = 0
    g["V5_BUY_CONFIRMATION_REACHED"] = 0
    g["V5_BUY_CONFIRMATION_AGE"] = 0
    g["V5_BUY_STAGE"] = ""
    g["V5_BUY_REASON"] = ""

    # nuove colonne diagnostiche reset
    g["V5_BUY_CONFIRM_RESET"] = 0
    g["V5_BUY_RESET_COUNT"] = 0

    # --------------------------------------------------------
    # Identificazione dei flip candidati
    # --------------------------------------------------------

    candidate_positions = []

    for i in range(len(g)):

        sar_side = int(
            _num_value(
                g.at[i, "SAR_SIDE"],
                0,
            )
        )

        sar_age = int(
            _num_value(
                g.at[i, "SAR_AGE"],
                0,
            )
        )

        qualified = int(
            _num_value(
                g.at[i, run_qualified_col],
                0,
            )
        )

        prev_age = float(
            _num_value(
                g.at[i, prev_sar_age_col],
                0,
            )
        )

        if (
            sar_side == 1
            and sar_age == 1
            and qualified == 1
            and prev_age >= N_PREV_BEAR_MIN
        ):
            candidate_positions.append(i)

    # --------------------------------------------------------
    # Process dei run protetti
    # --------------------------------------------------------

    processed_run_ids = set()

    for start_pos in candidate_positions:

        run_id = g.at[
            start_pos,
            run_id_col,
        ]

        run_key = str(run_id)

        if run_key in processed_run_ids:
            continue

        processed_run_ids.add(run_key)

        # righe appartenenti allo stesso run SAR
        run_positions = []

        for j in range(
            start_pos,
            len(g),
        ):

            current_side = int(
                _num_value(
                    g.at[j, "SAR_SIDE"],
                    0,
                )
            )

            current_run_id = str(
                g.at[j, run_id_col]
            )

            if (
                current_side != 1
                or current_run_id != run_key
            ):
                break

            run_positions.append(j)

        if not run_positions:
            continue

        confirm_count = 0
        confirmation_position = None
        reset_count = 0

        # ====================================================
        # CAUSAL SCAN DEL RUN BULLISH
        # ====================================================

        for j in run_positions:

            g.at[
                j,
                "V5_BUY_LONG_BEAR_PROTECTED",
            ] = 1

            ha_direction = int(
                _num_value(
                    g.at[
                        j,
                        "HA_DIRECTION",
                    ],
                    0,
                )
            )

            ha_indecision = int(
                _num_value(
                    g.at[
                        j,
                        "HA_INDECISION",
                    ],
                    0,
                )
            )

            decisive_bull = (
                ha_direction == 1
                and ha_indecision == 0
            )

            decisive_bear = (
                ha_direction == -1
                and ha_indecision == 0
            )

            # =================================================
            # PRIMA DELLA CONFERMA
            # =================================================

            if confirmation_position is None:

                # ---------------------------------------------
                # RESET:
                # una HA bearish decisiva invalida tutte
                # le precedenti conferme bullish.
                # ---------------------------------------------

                if decisive_bear:

                    if confirm_count > 0:
                        reset_count += 1

                        g.at[
                            j,
                            "V5_BUY_CONFIRM_RESET",
                        ] = 1

                    confirm_count = 0

                # ---------------------------------------------
                # bullish decisiva -> +1
                # ---------------------------------------------

                elif decisive_bull:

                    confirm_count += 1

                    if (
                        confirm_count
                        >= K_BULL_HA_CONFIRMATIONS
                    ):
                        confirmation_position = j

                        g.at[
                            j,
                            "V5_BUY_CONFIRMATION_REACHED",
                        ] = 1

            # =================================================
            # DIAGNOSTICA
            # =================================================

            g.at[
                j,
                "V5_BUY_CONFIRM_COUNT",
            ] = confirm_count

            g.at[
                j,
                "V5_BUY_RESET_COUNT",
            ] = reset_count

            # =================================================
            # PHASE PRIMA DELLA CONFERMA
            # =================================================

            if confirmation_position is None:

                base_phase = g.at[
                    j,
                    pre_v4010_col,
                ]

                g.at[
                    j,
                    "PHASE_DYNAMICS_V5",
                ] = _safe_test_phase(
                    base_phase
                )

                g.at[
                    j,
                    "V5_BUY_STAGE",
                ] = f"TEST_{confirm_count}"

                if (
                    g.at[
                        j,
                        "V5_BUY_CONFIRM_RESET",
                    ]
                    == 1
                ):
                    reason = (
                        "LONG_BEAR_BUY_RESET_"
                        "DECISIVE_BEAR"
                    )

                else:
                    reason = (
                        "LONG_BEAR_BUY_"
                        "WAIT_CONFIRMATION"
                    )

                g.at[
                    j,
                    "V5_BUY_REASON",
                ] = reason

                continue

            # =================================================
            # DOPO LA CONFERMA
            # =================================================

            confirmation_age = (
                run_positions.index(j)
                - run_positions.index(
                    confirmation_position
                )
                + 1
            )

            g.at[
                j,
                "V5_BUY_CONFIRMATION_AGE",
            ] = confirmation_age

            # BUY consecutivi: INDECISIONE interrompe la sequenza.
            if confirmation_age in (2, 3):
                previous_j = run_positions[run_positions.index(j) - 1]
                if str(g.at[previous_j, "PHASE_DYNAMICS_V5"]) != "BUY":
                    confirmation_age = 4

            # Allinea il diagnostico alla progressione effettiva.
            g.at[j, "V5_BUY_CONFIRMATION_AGE"] = confirmation_age

            # -------------------------------------------------
            # BUY1
            # -------------------------------------------------

            if confirmation_age == 1:

                g.at[
                    j,
                    "PHASE_DYNAMICS_V5",
                ] = "BUY"

                g.at[
                    j,
                    "V5_BUY_STAGE",
                ] = "BUY1"

                g.at[
                    j,
                    "V5_BUY_REASON",
                ] = (
                    "LONG_BEAR_BUY_"
                    "CONFIRMED_BUY1"
                )

            # -------------------------------------------------
            # BUY2
            # -------------------------------------------------

            elif confirmation_age == 2:
                if not decisive_bull:
                    g.at[j, "PHASE_DYNAMICS_V5"] = "INDECISIONE"
                    g.at[j, "V5_BUY_STAGE"] = "INDECISIONE"
                    g.at[j, "V5_BUY_REASON"] = (
                        "BUY2_NOT_CONFIRMED_HA"
                    )
                    continue


                g.at[
                    j,
                    "PHASE_DYNAMICS_V5",
                ] = "BUY"

                g.at[
                    j,
                    "V5_BUY_STAGE",
                ] = "BUY2"

                g.at[
                    j,
                    "V5_BUY_REASON",
                ] = (
                    "LONG_BEAR_BUY_"
                    "CONFIRMED_BUY2"
                )

            # -------------------------------------------------
            # BUY3
            # -------------------------------------------------

            elif confirmation_age == 3:
                if not decisive_bull:
                    g.at[j, "PHASE_DYNAMICS_V5"] = "INDECISIONE"
                    g.at[j, "V5_BUY_STAGE"] = "INDECISIONE"
                    g.at[j, "V5_BUY_REASON"] = (
                        "BUY3_NOT_CONFIRMED_HA"
                    )
                    continue


                g.at[
                    j,
                    "PHASE_DYNAMICS_V5",
                ] = "BUY"

                g.at[
                    j,
                    "V5_BUY_STAGE",
                ] = "BUY3"

                g.at[
                    j,
                    "V5_BUY_REASON",
                ] = (
                    "LONG_BEAR_BUY_"
                    "CONFIRMED_BUY3"
                )

            # -------------------------------------------------
            # DAL QUARTO CONFIRMED WEEK IN POI:
            # torna alla dinamica V4 pre-V40.10
            # -------------------------------------------------

            else:

                base_phase = g.at[
                    j,
                    pre_v4010_col,
                ]

                g.at[
                    j,
                    "PHASE_DYNAMICS_V5",
                ] = _safe_test_phase(
                    base_phase
                )

                g.at[
                    j,
                    "V5_BUY_STAGE",
                ] = "POST_BUY3"

                g.at[
                    j,
                    "V5_BUY_REASON",
                ] = (
                    "LONG_BEAR_POST_"
                    "CONFIRMED_BUY3"
                )

    # --------------------------------------------------------
    # Tutte le righe non protette restano identiche a V4
    # --------------------------------------------------------

    non_protected = (
        g[
            "V5_BUY_LONG_BEAR_PROTECTED"
        ] != 1
    )

    g.loc[
        non_protected,
        "PHASE_DYNAMICS_V5",
    ] = g.loc[
        non_protected,
        phase_v4_col,
    ]

    g.loc[
        non_protected,
        "V5_BUY_REASON",
    ] = "V4_UNCHANGED"

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