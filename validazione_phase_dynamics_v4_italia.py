"""
MarketSentinel
V40.5 PHASE DYNAMICS SHADOW V4

OBIETTIVO
---------
Correggere esclusivamente i BUY / SELL presenti nei primi 3 AGE
di run SAR che V40.10 considera NON qualificati.

IMPORTANTE
----------
- NON modifica production
- NON modifica V3.1
- NON modifica V40.10
- NON modifica engine.py
- nessuna nuova soglia tecnica
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import weekly_v40_5_phase_dynamics_shadow as v31
import weekly_v40_10_sar_phase_fix as v4010


VERSION = "V4"


def _run_v4010_components(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()

    g = v4010.add_previous_sar_run_information(g)
    g = v4010.qualify_sar_flips(g)
    g = v4010.propagate_qualified_signal(g)

    return g


def _invalid_buy_sell_mask(g: pd.DataFrame) -> pd.Series:

    age = pd.to_numeric(
        g["SAR_AGE"],
        errors="coerce",
    )

    side = pd.to_numeric(
        g["SAR_SIDE"],
        errors="coerce",
    )

    qualified = pd.to_numeric(
        g["V4010_RUN_QUALIFIED"],
        errors="coerce",
    ).fillna(0)

    phase = g["PHASE_DYNAMICS_SHADOW"]

    invalid_buy = (
        phase.eq("BUY")
        & side.eq(1)
        & age.isin([1, 2, 3])
        & qualified.eq(0)
    )

    invalid_sell = (
        phase.eq("SELL")
        & side.eq(-1)
        & age.isin([1, 2, 3])
        & qualified.eq(0)
    )

    return invalid_buy | invalid_sell


def _classify_from_neutralized_base(
    first: pd.DataFrame,
    invalid: pd.Series,
) -> pd.DataFrame:
    """
    Riesegue SOLO la logica dinamica V3.1 usando come base
    la PHASE_DYNAMICS_BASE già calcolata nel primo passaggio.

    I BUY/SELL non qualificati vengono neutralizzati
    a INDECISIONE.

    NOTA:
    non richiamiamo v31.process_ticker(), perché quello
    ricostruirebbe nuovamente la base a monte.
    """

    g = first.copy().reset_index(drop=True)

    g["V4_BASE_ORIGINAL"] = (
        g["PHASE_DYNAMICS_BASE"].copy()
    )

    g["V4_INVALID_BUY_SELL"] = (
        invalid.astype(int).to_numpy()
    )

    g["PHASE_DYNAMICS_BASE"] = (
        g["PHASE_DYNAMICS_BASE"].copy()
    )

    g.loc[
        invalid.to_numpy(),
        "PHASE_DYNAMICS_BASE",
    ] = "INDECISIONE"

    # ---------------------------------------------------------
    # La V3.1 espone già tutte le feature tecniche necessarie.
    # Dobbiamo quindi richiamare la funzione che produce
    # la PHASE dinamica partendo dalla PHASE_DYNAMICS_BASE.
    #
    # Cerchiamo esplicitamente la funzione disponibile,
    # senza inventare nomi di output o nuove regole.
    # ---------------------------------------------------------

    candidate_functions = [
        "apply_phase_dynamics",
        "apply_dynamics",
        "apply_phase_dynamics_shadow",
        "apply_dynamics_shadow",
    ]

    applied = False

    for function_name in candidate_functions:

        if hasattr(v31, function_name):

            func = getattr(
                v31,
                function_name,
            )

            g = func(g)

            applied = True
            break

    if not applied:
        raise RuntimeError(
            "Non trovo nella V3.1 la funzione che applica "
            "la logica Dynamics alla PHASE_DYNAMICS_BASE. "
            "Serve leggere il nome reale della funzione "
            "prima di procedere."
        )

    return g


def process_ticker(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    # =========================================================
    # 1. V3.1 NORMALE
    # =========================================================

    first = v31.process_ticker(
        df.copy()
    ).reset_index(drop=True)

    required = [
        "PHASE_DYNAMICS_BASE",
        "PHASE_DYNAMICS_SHADOW",
        "SAR_SIDE",
        "SAR_AGE",
    ]

    missing = [
        c
        for c in required
        if c not in first.columns
    ]

    if missing:
        raise RuntimeError(
            "Colonne mancanti dopo V3.1: "
            + ", ".join(missing)
        )

    phase_v31 = (
        first["PHASE_DYNAMICS_SHADOW"]
        .copy()
    )

    # =========================================================
    # 2. QUALIFICAZIONE V40.10
    # =========================================================

    q = first.copy()

    q["PHASE"] = (
        q["PHASE_DYNAMICS_SHADOW"]
        .copy()
    )

    q = _run_v4010_components(q)

    invalid = _invalid_buy_sell_mask(q)

    # =========================================================
    # 3. SE NON CI SONO CASI INVALIDI
    # =========================================================

    if int(invalid.sum()) == 0:

        final = q.copy()

        final["PHASE"] = (
            final["PHASE_DYNAMICS_SHADOW"]
            .copy()
        )

        final = v4010.apply_v4010_fix(
            final
        )

        final["PHASE_DYNAMICS_V31"] = (
            phase_v31.to_numpy()
        )

        final[
            "PHASE_DYNAMICS_V4_PRE_V4010"
        ] = final[
            "PHASE_DYNAMICS_SHADOW"
        ].copy()

        final["PHASE_DYNAMICS_V4"] = (
            final["PHASE"].copy()
        )

        final["V4_INVALID_BUY_SELL"] = 0

        final[
            "V4_FIRST_PASS_RUN_QUALIFIED"
        ] = pd.to_numeric(
            final["V4010_RUN_QUALIFIED"],
            errors="coerce",
        ).fillna(0).astype(int)

        final[
            "PHASE_DYNAMICS_V4_REASON"
        ] = "UNCHANGED_V31"

    else:

        # =====================================================
        # 4. NEUTRALIZZAZIONE DEI SOLI BUY/SELL NON QUALIFICATI
        # =====================================================

        second = _classify_from_neutralized_base(
            q,
            invalid,
        )

        if "PHASE_DYNAMICS_SHADOW" not in second.columns:
            raise RuntimeError(
                "La riclassificazione V3.1 non ha prodotto "
                "PHASE_DYNAMICS_SHADOW."
            )

        second[
            "PHASE_DYNAMICS_V4_PRE_V4010"
        ] = second[
            "PHASE_DYNAMICS_SHADOW"
        ].copy()

        # =====================================================
        # 5. V40.10 FINALE
        # =====================================================

        second["PHASE"] = second[
            "PHASE_DYNAMICS_V4_PRE_V4010"
        ].copy()

        # Ricalcoliamo V40.10 sulla nuova PHASE
        for col in [
            "V4010_CORRECTION_REASON",
            "V4010_PHASE_CORRECTED",
            "V4010_PREVIOUS_STRUCTURE_CONFIRMED",
            "V4010_PREV_PHASE",
            "V4010_PREV_SAR_AGE",
            "V4010_PREV_SAR_SIDE",
            "V4010_QUALIFIED_FLIP",
            "V4010_RUN_QUALIFIED",
            "V4010_SAR_FLIP",
            "V4010_SAR_RUN_ID",
        ]:
            if col in second.columns:
                second = second.drop(
                    columns=[col]
                )

        second = _run_v4010_components(
            second
        )

        second = v4010.apply_v4010_fix(
            second
        )

        final = second

        final["PHASE_DYNAMICS_V31"] = (
            phase_v31.to_numpy()
        )

        final["PHASE_DYNAMICS_V4"] = (
            final["PHASE"].copy()
        )

        final["V4_INVALID_BUY_SELL"] = (
            invalid.astype(int).to_numpy()
        )

        final[
            "V4_FIRST_PASS_RUN_QUALIFIED"
        ] = pd.to_numeric(
            q["V4010_RUN_QUALIFIED"],
            errors="coerce",
        ).fillna(0).astype(int).to_numpy()

        reasons = np.full(
            len(final),
            "UNCHANGED_V31",
            dtype=object,
        )

        invalid_np = invalid.to_numpy()

        for i in range(len(final)):

            if not invalid_np[i]:
                continue

            old_phase = str(
                phase_v31.iloc[i]
            )

            new_phase = str(
                final[
                    "PHASE_DYNAMICS_V4_PRE_V4010"
                ].iloc[i]
            )

            if old_phase == "BUY":
                reasons[i] = (
                    "UNQUALIFIED_BUY_RECLASSIFIED_TO_"
                    + new_phase
                )

            else:
                reasons[i] = (
                    "UNQUALIFIED_SELL_RECLASSIFIED_TO_"
                    + new_phase
                )

        final[
            "PHASE_DYNAMICS_V4_REASON"
        ] = reasons

    # =========================================================
    # 6. SAFETY FINALE
    # =========================================================

    side = pd.to_numeric(
        final["SAR_SIDE"],
        errors="coerce",
    )

    age = pd.to_numeric(
        final["SAR_AGE"],
        errors="coerce",
    )

    qualified = pd.to_numeric(
        final["V4010_RUN_QUALIFIED"],
        errors="coerce",
    ).fillna(0)

    valid_buy = (
        qualified.eq(1)
        & side.eq(1)
        & age.isin([1, 2, 3])
    )

    valid_sell = (
        qualified.eq(1)
        & side.eq(-1)
        & age.isin([1, 2, 3])
    )

    final["V4_INVALID_FINAL_BUY"] = (
        final[
            "PHASE_DYNAMICS_V4"
        ].eq("BUY")
        & ~valid_buy
    ).astype(int)

    final["V4_INVALID_FINAL_SELL"] = (
        final[
            "PHASE_DYNAMICS_V4"
        ].eq("SELL")
        & ~valid_sell
    ).astype(int)

    return final