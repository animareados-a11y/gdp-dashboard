"""
MarketSentinel
V40.5 PHASE DYNAMICS SHADOW V4

VERSIONE CHIRURGICA

OBIETTIVO
---------
Correggere esclusivamente i BUY / SELL appartenenti a run SAR
che V40.10 considera NON qualificati.

Principio fondamentale:

- V3.1 rimane completamente invariata.
- V40.10 rimane completamente invariata.
- I run BUY/SELL non qualificati vengono neutralizzati
  solo durante una simulazione locale del relativo run.
- Della simulazione vengono recuperate SOLO le settimane
  appartenenti al run non qualificato.
- Tutte le settimane successive mantengono esattamente
  la classificazione V3.1 originale.
- Infine V40.10 viene applicata normalmente ai run qualificati.

NON modifica:
- production
- V3.1
- V40.10
- engine.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import weekly_v40_5_phase_dynamics_shadow as v31
import weekly_v40_10_sar_phase_fix as v4010


VERSION = "V4_SURGICAL"


# ============================================================
# V40.10 QUALIFICATION
# ============================================================

def _run_v4010_qualification(df):

    g = df.copy()

    g = v4010.add_previous_sar_run_information(g)
    g = v4010.qualify_sar_flips(g)
    g = v4010.propagate_qualified_signal(g)

    return g


# ============================================================
# INVALID BUY / SELL
# ============================================================

def _invalid_buy_sell_mask(g):

    phase = (
        g["PHASE_DYNAMICS_SHADOW"]
        .astype(str)
    )

    side = pd.to_numeric(
        g["SAR_SIDE"],
        errors="coerce",
    )

    age = pd.to_numeric(
        g["SAR_AGE"],
        errors="coerce",
    )

    qualified = pd.to_numeric(
        g["V4010_RUN_QUALIFIED"],
        errors="coerce",
    ).fillna(0)

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


# ============================================================
# SURGICAL RECLASSIFICATION
# ============================================================

def _surgical_reclassification(
    first,
    q,
    invalid,
):

    """
    Rielabora ogni run SAR non qualificato separatamente.

    Per ciascun run:

    1. riparte dalla BASE ORIGINALE V3.1;
    2. neutralizza SOLO BUY/SELL non qualificati di quel run;
    3. esegue V3.1;
    4. recupera SOLO il risultato delle righe del run;
    5. scarta qualsiasi modifica prodotta sulle settimane successive.

    In questo modo non esiste propagazione indesiderata.
    """

    base_original = (
        first["PHASE_DYNAMICS_BASE"]
        .copy()
        .reset_index(drop=True)
    )

    phase_v31 = (
        first["PHASE_DYNAMICS_SHADOW"]
        .copy()
        .reset_index(drop=True)
    )

    surgical = (
        phase_v31.copy()
    )

    # --------------------------------------------------------
    # Identificazione run
    # --------------------------------------------------------

    if "V4010_SAR_RUN_ID" not in q.columns:
        raise RuntimeError(
            "V4010_SAR_RUN_ID mancante."
        )

    invalid_positions = np.flatnonzero(
        invalid.to_numpy()
    )

    if len(invalid_positions) == 0:
        return surgical

    run_ids = (
        q.loc[
            invalid,
            "V4010_SAR_RUN_ID",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # Un run alla volta
    # --------------------------------------------------------

    for run_id in run_ids:

        run_invalid = (
            invalid
            &
            q[
                "V4010_SAR_RUN_ID"
            ].eq(run_id)
        )

        positions = np.flatnonzero(
            run_invalid.to_numpy()
        )

        if len(positions) == 0:
            continue

        # Base originale completa
        temp = first.copy()

        temp["PHASE"] = (
            base_original.copy()
        )

        # Neutralizziamo SOLO il run corrente
        temp.loc[
            run_invalid.to_numpy(),
            "PHASE",
        ] = "INDECISIONE"

        # V3.1 completa per mantenere il corretto
        # stato storico precedente al run
        # Ottimizzazione: V3.1 fino all'ultima settimana del run.
        # Le settimane successive non vengono mai utilizzate.
        end = int(positions.max()) + 1
        temp = (
            v31.apply_phase_dynamics_shadow(
                temp.iloc[:end].copy()
            )
            .reset_index(drop=True)
        )

        # Recuperiamo SOLO le righe del run.
        # Tutte le altre modifiche vengono ignorate.
        surgical.iloc[
            positions
        ] = (
            temp[
                "PHASE_DYNAMICS_SHADOW"
            ]
            .iloc[positions]
            .to_numpy()
        )

    # --------------------------------------------------------
    # Eventuali invalid senza RUN_ID
    #
    # Può succedere solo nelle righe iniziali del dataset.
    # Le trattiamo singolarmente, sempre senza propagazione.
    # --------------------------------------------------------

    no_run_id = (
        invalid
        &
        q[
            "V4010_SAR_RUN_ID"
        ].isna()
    )

    for pos in np.flatnonzero(
        no_run_id.to_numpy()
    ):

        temp = first.copy()

        temp["PHASE"] = (
            base_original.copy()
        )

        temp.loc[
            pos,
            "PHASE",
        ] = "INDECISIONE"

        # Ottimizzazione: V3.1 fino alla settimana interessata.
        temp = (
            v31.apply_phase_dynamics_shadow(
                temp.iloc[:int(pos) + 1].copy()
            )
            .reset_index(drop=True)
        )

        surgical.iloc[pos] = (
            temp[
                "PHASE_DYNAMICS_SHADOW"
            ].iloc[pos]
        )

    return surgical


# ============================================================
# PROCESS TICKER
# ============================================================

def process_ticker(df):

    if df.empty:
        return df.copy()

    # ========================================================
    # 1. V3.1 ORIGINALE
    # ========================================================

    first = (
        v31.process_ticker(
            df.copy()
        )
        .reset_index(drop=True)
    )

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
        first[
            "PHASE_DYNAMICS_SHADOW"
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # 2. QUALIFICAZIONE V40.10
    # ========================================================

    q = first.copy()

    q["PHASE"] = (
        phase_v31.copy()
    )

    q = (
        _run_v4010_qualification(
            q
        )
        .reset_index(drop=True)
    )

    invalid = (
        _invalid_buy_sell_mask(
            q
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 3. RICLASSIFICAZIONE CHIRURGICA
    # ========================================================

    surgical_phase = (
        _surgical_reclassification(
            first=first,
            q=q,
            invalid=invalid,
        )
        .reset_index(drop=True)
    )

    second = first.copy()

    second[
        "PHASE_DYNAMICS_V31"
    ] = phase_v31.to_numpy()

    second[
        "PHASE_DYNAMICS_V4_PRE_V4010"
    ] = surgical_phase.to_numpy()

    second[
        "V4_INVALID_BUY_SELL"
    ] = (
        invalid
        .astype(int)
        .to_numpy()
    )

    second[
        "V4_FIRST_PASS_RUN_QUALIFIED"
    ] = (
        pd.to_numeric(
            q[
                "V4010_RUN_QUALIFIED"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    # ========================================================
    # 4. RIPRISTINO COLONNE V40.10
    # ========================================================

    v4010_cols = [
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
    ]

    for col in v4010_cols:

        if col in q.columns:

            second[col] = (
                q[col]
                .to_numpy()
            )

    # ========================================================
    # 5. V40.10 FINALE
    #
    # Qui BUY / SELL qualificati tornano normalmente
    # sulle prime 3 settimane del vero nuovo regime.
    # ========================================================

    second["PHASE"] = (
        second[
            "PHASE_DYNAMICS_V4_PRE_V4010"
        ]
        .copy()
    )

    final = (
        v4010.apply_v4010_fix(
            second
        )
        .reset_index(drop=True)
    )

    final[
        "PHASE_DYNAMICS_V31"
    ] = phase_v31.to_numpy()

    final[
        "PHASE_DYNAMICS_V4"
    ] = final[
        "PHASE"
    ].copy()

    # ========================================================
    # 6. REASON
    # ========================================================

    reasons = np.full(
        len(final),
        "UNCHANGED_V31",
        dtype=object,
    )

    for i in np.flatnonzero(
        invalid.to_numpy()
    ):

        old_phase = str(
            phase_v31.iloc[i]
        )

        new_phase = str(
            surgical_phase.iloc[i]
        )

        if old_phase == "BUY":

            reasons[i] = (
                "UNQUALIFIED_BUY_RECLASSIFIED_TO_"
                + new_phase
            )

        elif old_phase == "SELL":

            reasons[i] = (
                "UNQUALIFIED_SELL_RECLASSIFIED_TO_"
                + new_phase
            )

    final[
        "PHASE_DYNAMICS_V4_REASON"
    ] = reasons

    # ========================================================
    # 7. SAFETY BUY / SELL
    # ========================================================

    side = pd.to_numeric(
        final["SAR_SIDE"],
        errors="coerce",
    )

    age = pd.to_numeric(
        final["SAR_AGE"],
        errors="coerce",
    )

    qualified = pd.to_numeric(
        final[
            "V4010_RUN_QUALIFIED"
        ],
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

    final[
        "V4_INVALID_FINAL_BUY"
    ] = (
        final[
            "PHASE_DYNAMICS_V4"
        ].eq("BUY")
        &
        ~valid_buy
    ).astype(int)

    final[
        "V4_INVALID_FINAL_SELL"
    ] = (
        final[
            "PHASE_DYNAMICS_V4"
        ].eq("SELL")
        &
        ~valid_sell
    ).astype(int)

    return final