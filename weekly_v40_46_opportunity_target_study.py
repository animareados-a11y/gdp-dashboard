#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MarketSentinel
V40.46 - OPPORTUNITY TARGET STUDY
Corrected forward-only Reversal calibration version.

SCOPO
-----
Costruire il dataset storico OOS necessario per progettare Opportunity,
incrociando:

1. PHASE V40.39
2. REVERSAL V40.40a OOS
3. calibrazione forward-only congelata V40.40b
4. STRENGTH V40.43 OOS - COMPACT_ROLLING
5. target prospettici V40.45

REGOLE METODOLOGICHE
--------------------
- PHASE determina la direzione corrente.
- STRENGTH misura la qualità del trend corrente.
- REVERSAL misura il rischio/probabilità di inversione.
- Opportunity NON viene ancora modellata.
- Nessuno score 1-10.
- Nessun peso arbitrario.
- Nessuna soglia arbitraria.

REVERSAL:
V40.40a implementa calibrazione forward-only:
    Fold 1 = RAW only, nessun OOS precedente
    Fold 2 = calibrabile con OOS precedente
    Fold 3 = calibrabile con OOS precedente

V40.46 utilizza SOLO probabilità calibrate.
NON usa RAW_PROB come fallback.

Per una PHASE BULL sono richiesti:
    BULL_TO_BEAR 1W / 2W / 3W calibrati

Per una PHASE BEAR sono richiesti:
    BEAR_TO_BULL 1W / 2W / 3W calibrati

Le probabilità dell'altra direzione non sono richieste.

NON vengono utilizzati come feature:
- Y_TRUE Reversal
- y_num Strength
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

VERSION = "V40.46"

TARGET_FILE = Path(
    "data/v40_45_opportunity_target_audit/"
    "v40_45_opportunity_targets.csv"
)

STRENGTH_FILE = Path(
    "data/v40_43_trend_strength_oos/"
    "v40_43_oos_predictions.csv"
)

REVERSAL_FILE = Path(
    "data/v40_40a_oos_predictions.csv"
)

CALIBRATION_MANIFEST = Path(
    "data/v40_40b_calibration_manifest.csv"
)

OUTPUT_DIR = Path(
    "data/v40_46_opportunity_target_study"
)

OUTPUT_COMMON = (
    OUTPUT_DIR /
    "v40_46_common_oos_targets.csv"
)

OUTPUT_PHASE_SUMMARY = (
    OUTPUT_DIR /
    "v40_46_phase_target_summary.csv"
)

OUTPUT_STRENGTH_SUMMARY = (
    OUTPUT_DIR /
    "v40_46_strength_target_summary.csv"
)

OUTPUT_REVERSAL_DECILES = (
    OUTPUT_DIR /
    "v40_46_reversal_decile_summary.csv"
)

OUTPUT_CORRELATIONS = (
    OUTPUT_DIR /
    "v40_46_target_correlations.csv"
)

OUTPUT_METADATA = (
    OUTPUT_DIR /
    "v40_46_metadata.json"
)

OUTPUT_MANIFEST_USED = (
    OUTPUT_DIR /
    "v40_46_reversal_manifest_used.csv"
)


TARGET_METRICS = [
    "MFE_PCT",
    "MAE_PCT",
    "SPEED_PCT_PER_WEEK",
    "FINAL_DIRECTIONAL_RETURN_PCT",
    "FAVORABLE_WEEKS",
    "ADVERSE_WEEKS",
    "PATH_EFFICIENCY",
    "HAS_FAVORABLE_MOVE",
]

BULL_PHASES = {
    "BUY",
    "TREND_RIALZISTA",
    "DEBOLEZZA_RIALZISTA",
}

BEAR_PHASES = {
    "SELL",
    "TREND_RIBASSISTA",
    "DEBOLEZZA_RIBASSISTA",
}


# =============================================================================
# HELPERS
# =============================================================================

def banner(text: str) -> None:
    print()
    print("=" * 79)
    print(text)
    print("=" * 79)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path}"
        )


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label}: colonne mancanti: {missing}"
        )


def spearman_safe(
    x: pd.Series,
    y: pd.Series,
) -> float:

    tmp = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(tmp) < 3:
        return np.nan

    if (
        tmp["x"].nunique() < 2
        or
        tmp["y"].nunique() < 2
    ):
        return np.nan

    return float(
        tmp["x"].corr(
            tmp["y"],
            method="spearman",
        )
    )


def expected_phase_direction(
    phase,
) -> str:

    p = str(
        phase
    ).strip().upper()

    if p in BULL_PHASES:
        return "BULL"

    if p in BEAR_PHASES:
        return "BEAR"

    return "OTHER"


# =============================================================================
# INPUT CHECK
# =============================================================================

banner(
    f"{VERSION} - OPPORTUNITY TARGET STUDY"
)

for path in [
    TARGET_FILE,
    STRENGTH_FILE,
    REVERSAL_FILE,
    CALIBRATION_MANIFEST,
]:
    require_file(path)

    print(
        f"OK: {path}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CALIBRATION MANIFEST
# =============================================================================

banner(
    "CARICAMENTO MANIFEST V40.40b"
)

manifest = pd.read_csv(
    CALIBRATION_MANIFEST
)

require_columns(
    manifest,
    [
        "TARGET",
        "METHOD",
    ],
    "Manifest V40.40b",
)

manifest["TARGET"] = (
    manifest["TARGET"]
    .astype(str)
    .str.strip()
    .str.upper()
)

manifest["METHOD"] = (
    manifest["METHOD"]
    .astype(str)
    .str.strip()
    .str.upper()
)

expected_reversal_targets = {
    "BEAR_TO_BULL_1W",
    "BEAR_TO_BULL_2W",
    "BEAR_TO_BULL_3W",
    "BULL_TO_BEAR_1W",
    "BULL_TO_BEAR_2W",
    "BULL_TO_BEAR_3W",
}

actual_reversal_targets = set(
    manifest["TARGET"]
)

if actual_reversal_targets != expected_reversal_targets:
    raise RuntimeError(
        "Manifest V40.40b diverso dai 6 target attesi."
    )

allowed_methods = {
    "PLATT",
    "ISOTONIC",
}

bad_methods = (
    set(
        manifest["METHOD"]
    )
    -
    allowed_methods
)

if bad_methods:
    raise RuntimeError(
        f"Metodi calibrazione inattesi: {bad_methods}"
    )

method_map = dict(
    zip(
        manifest["TARGET"],
        manifest["METHOD"],
    )
)

for target in sorted(
    method_map
):
    print(
        f"{target}: {method_map[target]}"
    )

manifest.to_csv(
    OUTPUT_MANIFEST_USED,
    index=False,
)


# =============================================================================
# STRENGTH V40.43 OOS
# =============================================================================

banner(
    "CARICAMENTO STRENGTH V40.43 OOS"
)

strength = pd.read_csv(
    STRENGTH_FILE,
    low_memory=False,
)

require_columns(
    strength,
    [
        "direction",
        "age_bucket",
        "variant",
        "fold",
        "heldout_ticker",
        "Ticker",
        "week_date",
        "strength_class",
        "pred_expected",
        "pred_class",
    ],
    "STRENGTH V40.43",
)

strength = strength.loc[
    strength["variant"]
    .astype(str)
    .eq(
        "COMPACT_ROLLING"
    )
].copy()

strength["Ticker"] = (
    strength["Ticker"]
    .astype(str)
)

strength["Date"] = pd.to_datetime(
    strength["week_date"],
    errors="coerce",
)

strength["STRENGTH_DIRECTION"] = (
    strength["direction"]
    .astype(str)
    .str.strip()
    .str.upper()
)

strength["STRENGTH_EXPECTED"] = pd.to_numeric(
    strength["pred_expected"],
    errors="coerce",
)

strength["STRENGTH_CLASS"] = (
    strength["pred_class"]
    .astype(str)
)

strength["STRENGTH_AGE_BUCKET"] = (
    strength["age_bucket"]
    .astype(str)
)

strength["STRENGTH_FOLD"] = (
    strength["fold"]
)

strength["STRENGTH_HELDOUT_TICKER"] = (
    strength["heldout_ticker"]
    .astype(str)
)

strength = strength[
    [
        "Ticker",
        "Date",
        "STRENGTH_DIRECTION",
        "STRENGTH_EXPECTED",
        "STRENGTH_CLASS",
        "STRENGTH_AGE_BUCKET",
        "STRENGTH_FOLD",
        "STRENGTH_HELDOUT_TICKER",
    ]
].copy()

bad_strength_dates = int(
    strength["Date"]
    .isna()
    .sum()
)

dup_strength = int(
    strength.duplicated(
        [
            "Ticker",
            "Date",
        ]
    ).sum()
)

nonfinite_strength = int(
    (
        ~np.isfinite(
            strength[
                "STRENGTH_EXPECTED"
            ].to_numpy(
                dtype=float
            )
        )
    ).sum()
)

print(
    f"Righe COMPACT_ROLLING: {len(strength):,}"
)

print(
    f"Ticker: {strength['Ticker'].nunique():,}"
)

print(
    f"Date non valide: {bad_strength_dates:,}"
)

print(
    f"Duplicati Ticker/Date: {dup_strength:,}"
)

print(
    f"Strength expected non-finite: "
    f"{nonfinite_strength:,}"
)

if bad_strength_dates:
    raise RuntimeError(
        "Date STRENGTH non valide."
    )

if dup_strength:
    raise RuntimeError(
        "Duplicati STRENGTH."
    )

if nonfinite_strength:
    raise RuntimeError(
        "STRENGTH_EXPECTED non finito."
    )


# =============================================================================
# REVERSAL V40.40a
# =============================================================================

banner(
    "CARICAMENTO REVERSAL V40.40a OOS"
)

reversal = pd.read_csv(
    REVERSAL_FILE,
    low_memory=False,
)

require_columns(
    reversal,
    [
        "Ticker",
        "Date",
        "PHASE",
        "PHASE_FAMILY",
        "TARGET",
        "HORIZON_W",
        "FOLD",
        "Y_TRUE",
        "RAW_PROB",
        "PLATT_PROB",
        "ISOTONIC_PROB",
    ],
    "REVERSAL V40.40a",
)

reversal["Ticker"] = (
    reversal["Ticker"]
    .astype(str)
)

reversal["Date"] = pd.to_datetime(
    reversal["Date"],
    errors="coerce",
)

reversal["TARGET"] = (
    reversal["TARGET"]
    .astype(str)
    .str.strip()
    .str.upper()
)

reversal["FOLD"] = pd.to_numeric(
    reversal["FOLD"],
    errors="coerce",
)

bad_reversal_dates = int(
    reversal["Date"]
    .isna()
    .sum()
)

dup_reversal = int(
    reversal.duplicated(
        [
            "Ticker",
            "Date",
            "TARGET",
        ]
    ).sum()
)

print(
    f"Righe Reversal OOS: {len(reversal):,}"
)

print(
    f"Ticker: {reversal['Ticker'].nunique():,}"
)

print(
    f"Date non valide: {bad_reversal_dates:,}"
)

print(
    f"Duplicati Ticker/Date/TARGET: "
    f"{dup_reversal:,}"
)

if bad_reversal_dates:
    raise RuntimeError(
        "Date REVERSAL non valide."
    )

if dup_reversal:
    raise RuntimeError(
        "Duplicati REVERSAL."
    )


# =============================================================================
# EXACT SELECTED CALIBRATION
# =============================================================================

banner(
    "CALIBRAZIONE FORWARD-ONLY V40.40a/V40.40b"
)

reversal["REVERSAL_PROB"] = np.nan

for target, method in method_map.items():

    mask = (
        reversal["TARGET"]
        ==
        target
    )

    if method == "PLATT":
        source_col = "PLATT_PROB"

    elif method == "ISOTONIC":
        source_col = "ISOTONIC_PROB"

    else:
        raise RuntimeError(
            f"Metodo non gestito: {method}"
        )

    reversal.loc[
        mask,
        "REVERSAL_PROB",
    ] = pd.to_numeric(
        reversal.loc[
            mask,
            source_col,
        ],
        errors="coerce",
    )

    g = reversal.loc[
        mask
    ]

    print()
    print(
        f"{target} | {method}"
    )

    for fold in sorted(
        g["FOLD"]
        .dropna()
        .unique()
    ):

        gf = g.loc[
            g["FOLD"]
            ==
            fold
        ]

        valid = int(
            gf[
                "REVERSAL_PROB"
            ]
            .notna()
            .sum()
        )

        missing = int(
            gf[
                "REVERSAL_PROB"
            ]
            .isna()
            .sum()
        )

        print(
            f"  Fold {int(fold)}: "
            f"validi={valid:,} | "
            f"mancanti={missing:,}"
        )


# =============================================================================
# VERIFY FORWARD-ONLY STRUCTURE
# =============================================================================

banner(
    "AUDIT STRUTTURA FORWARD-ONLY"
)

fold1_bad_valid = 0
later_bad_missing = 0

for target in sorted(
    expected_reversal_targets
):

    g = reversal.loc[
        reversal["TARGET"]
        ==
        target
    ]

    fold1 = g.loc[
        g["FOLD"]
        ==
        1
    ]

    later = g.loc[
        g["FOLD"]
        .isin(
            [
                2,
                3,
            ]
        )
    ]

    fold1_valid = int(
        fold1[
            "REVERSAL_PROB"
        ]
        .notna()
        .sum()
    )

    later_missing = int(
        later[
            "REVERSAL_PROB"
        ]
        .isna()
        .sum()
    )

    fold1_bad_valid += (
        fold1_valid
    )

    later_bad_missing += (
        later_missing
    )

print(
    "Probabilità calibrate presenti "
    f"nel Fold 1: {fold1_bad_valid:,}"
)

print(
    "Probabilità calibrate mancanti "
    f"nei Fold 2-3: {later_bad_missing:,}"
)

if fold1_bad_valid != 0:
    raise RuntimeError(
        "Struttura Fold 1 diversa da quella "
        "documentata in V40.40a."
    )

if later_bad_missing != 0:
    raise RuntimeError(
        "Probabilità calibrate mancanti "
        "nei Fold 2-3."
    )

print()
print(
    "Forward-only calibration structure: OK"
)

print(
    "Fold 1 escluso implicitamente dal campione "
    "calibrato: SI"
)

print(
    "RAW_PROB usato come fallback: NO"
)


# =============================================================================
# PHASE REFERENCE
# =============================================================================

phase_ref = (
    reversal[
        [
            "Ticker",
            "Date",
            "PHASE",
            "PHASE_FAMILY",
        ]
    ]
    .drop_duplicates()
)

phase_ref_dup = int(
    phase_ref.duplicated(
        [
            "Ticker",
            "Date",
        ]
    ).sum()
)

if phase_ref_dup:
    raise RuntimeError(
        "Più PHASE per la stessa Ticker/Date "
        "nel Reversal OOS."
    )


# =============================================================================
# REVERSAL LONG -> WIDE
# =============================================================================

banner(
    "REVERSAL CALIBRATO LONG -> WIDE"
)

reversal_wide = (
    reversal.pivot(
        index=[
            "Ticker",
            "Date",
        ],
        columns="TARGET",
        values="REVERSAL_PROB",
    )
    .reset_index()
)

reversal_wide.columns.name = None

for col in sorted(
    expected_reversal_targets
):
    if col not in reversal_wide.columns:
        reversal_wide[col] = np.nan

reversal_wide = reversal_wide.merge(
    phase_ref,
    on=[
        "Ticker",
        "Date",
    ],
    how="left",
    validate="one_to_one",
)


# =============================================================================
# SELECT ONLY PHASE-RELEVANT REVERSAL DIRECTION
# =============================================================================

banner(
    "SELEZIONE REVERSAL COERENTE CON PHASE"
)

phase_expected = (
    reversal_wide[
        "PHASE"
    ]
    .apply(
        expected_phase_direction
    )
)

reversal_wide[
    "PHASE_EXPECTED_DIRECTION"
] = phase_expected

bull_mask = (
    phase_expected
    ==
    "BULL"
)

bear_mask = (
    phase_expected
    ==
    "BEAR"
)

reversal_wide[
    "REVERSAL_DIRECTION"
] = np.where(
    bull_mask,
    "BULL_TO_BEAR",
    np.where(
        bear_mask,
        "BEAR_TO_BULL",
        None,
    ),
)

reversal_wide[
    "REVERSAL_1W"
] = np.where(
    bull_mask,
    reversal_wide[
        "BULL_TO_BEAR_1W"
    ],
    np.where(
        bear_mask,
        reversal_wide[
            "BEAR_TO_BULL_1W"
        ],
        np.nan,
    ),
)

reversal_wide[
    "REVERSAL_2W"
] = np.where(
    bull_mask,
    reversal_wide[
        "BULL_TO_BEAR_2W"
    ],
    np.where(
        bear_mask,
        reversal_wide[
            "BEAR_TO_BULL_2W"
        ],
        np.nan,
    ),
)

reversal_wide[
    "REVERSAL_3W"
] = np.where(
    bull_mask,
    reversal_wide[
        "BULL_TO_BEAR_3W"
    ],
    np.where(
        bear_mask,
        reversal_wide[
            "BEAR_TO_BULL_3W"
        ],
        np.nan,
    ),
)

directional_rows = reversal_wide.loc[
    reversal_wide[
        "PHASE_EXPECTED_DIRECTION"
    ].isin(
        [
            "BULL",
            "BEAR",
        ]
    )
].copy()

complete_reversal = (
    directional_rows[
        [
            "REVERSAL_1W",
            "REVERSAL_2W",
            "REVERSAL_3W",
        ]
    ]
    .notna()
    .all(
        axis=1
    )
)

print(
    f"Ticker/Date direzionali: "
    f"{len(directional_rows):,}"
)

print(
    "Ticker/Date con 3 orizzonti calibrati "
    f"della direzione corrente: "
    f"{int(complete_reversal.sum()):,}"
)

reversal_ready = (
    directional_rows.loc[
        complete_reversal
    ]
    .copy()
)

for col in [
    "REVERSAL_1W",
    "REVERSAL_2W",
    "REVERSAL_3W",
]:

    arr = reversal_ready[
        col
    ].to_numpy(
        dtype=float
    )

    bad = int(
        (
            ~np.isfinite(
                arr
            )
        ).sum()
    )

    outside = int(
        (
            (arr < 0.0)
            |
            (arr > 1.0)
        ).sum()
    )

    print(
        f"{col}: non-finite={bad:,} | "
        f"fuori [0,1]={outside:,}"
    )

    if bad or outside:
        raise RuntimeError(
            f"Audit fallito: {col}"
        )


# =============================================================================
# JOIN STRENGTH + REVERSAL
# =============================================================================

banner(
    "JOIN STRENGTH OOS + REVERSAL CALIBRATO"
)

pillars = strength.merge(
    reversal_ready,
    on=[
        "Ticker",
        "Date",
    ],
    how="inner",
    validate="one_to_one",
)

pillars = (
    pillars.sort_values(
        [
            "Ticker",
            "Date",
        ]
    )
    .reset_index(
        drop=True
    )
)

print(
    f"Ticker/Date comuni: {len(pillars):,}"
)

print(
    f"Ticker comuni: "
    f"{pillars['Ticker'].nunique():,}"
)

if len(pillars):

    print(
        "Periodo:",
        pillars["Date"].min().date(),
        "->",
        pillars["Date"].max().date(),
    )


# =============================================================================
# PHASE / STRENGTH CONSISTENCY
# =============================================================================

banner(
    "AUDIT COERENZA PHASE / STRENGTH"
)

phase_strength_conflicts = int(
    (
        pillars[
            "PHASE_EXPECTED_DIRECTION"
        ]
        !=
        pillars[
            "STRENGTH_DIRECTION"
        ]
    ).sum()
)

print(
    f"Conflitti: "
    f"{phase_strength_conflicts:,}"
)

if phase_strength_conflicts:

    print(
        pillars.loc[
            pillars[
                "PHASE_EXPECTED_DIRECTION"
            ]
            !=
            pillars[
                "STRENGTH_DIRECTION"
            ],
            [
                "Ticker",
                "Date",
                "PHASE",
                "PHASE_FAMILY",
                "PHASE_EXPECTED_DIRECTION",
                "STRENGTH_DIRECTION",
            ],
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Conflitto PHASE/STRENGTH."
    )

print()
print(
    pillars[
        "STRENGTH_DIRECTION"
    ]
    .value_counts()
    .to_string()
)


# =============================================================================
# LOAD V40.45 TARGETS
# =============================================================================

banner(
    "CARICAMENTO TARGET V40.45"
)

target_usecols = [
    "Ticker",
    "Date",
    "PHASE",
    "PHASE_DIRECTION",
    "PHASE_AGE",
    "PHASE_CHANGED",
    "SIDE",
    "HORIZON_W",
] + TARGET_METRICS

targets = pd.read_csv(
    TARGET_FILE,
    usecols=target_usecols,
    low_memory=False,
)

targets["Ticker"] = (
    targets["Ticker"]
    .astype(str)
)

targets["Date"] = pd.to_datetime(
    targets["Date"],
    errors="coerce",
)

targets["SIDE"] = (
    targets["SIDE"]
    .astype(str)
    .str.strip()
    .str.upper()
)

targets["HORIZON_W"] = pd.to_numeric(
    targets["HORIZON_W"],
    errors="coerce",
)

bad_target_dates = int(
    targets["Date"]
    .isna()
    .sum()
)

dup_targets = int(
    targets.duplicated(
        [
            "Ticker",
            "Date",
            "SIDE",
            "HORIZON_W",
        ]
    ).sum()
)

print(
    f"Righe target: {len(targets):,}"
)

print(
    f"Date non valide: "
    f"{bad_target_dates:,}"
)

print(
    f"Duplicati chiave target: "
    f"{dup_targets:,}"
)

if bad_target_dates:
    raise RuntimeError(
        "Date V40.45 non valide."
    )

if dup_targets:
    raise RuntimeError(
        "Duplicati V40.45."
    )


# =============================================================================
# JOIN THREE PILLARS + V40.45
# =============================================================================

banner(
    "JOIN TRE PILASTRI + GROUND TRUTH V40.45"
)

common = targets.merge(
    pillars,
    on=[
        "Ticker",
        "Date",
    ],
    how="inner",
    suffixes=(
        "_TARGET",
        "_PILLAR",
    ),
    validate="many_to_one",
)

print(
    f"Righe dopo join: "
    f"{len(common):,}"
)

print(
    f"Ticker: "
    f"{common['Ticker'].nunique():,}"
)

n_target_dates = (
    common[
        [
            "Ticker",
            "Date",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    f"Ticker/Date con target futuro disponibile: "
    f"{n_target_dates:,}"
)


# =============================================================================
# PHASE CONSISTENCY V40.45 vs PILLARS
# =============================================================================

banner(
    "AUDIT PHASE V40.45 vs OOS"
)

if (
    "PHASE_TARGET" not in common.columns
    or
    "PHASE_PILLAR" not in common.columns
):
    raise RuntimeError(
        "Colonne PHASE post-join non trovate."
    )

phase_target_conflicts = int(
    (
        common[
            "PHASE_TARGET"
        ].astype(str)
        !=
        common[
            "PHASE_PILLAR"
        ].astype(str)
    ).sum()
)

print(
    f"Conflitti PHASE: "
    f"{phase_target_conflicts:,}"
)

if phase_target_conflicts:
    raise RuntimeError(
        "PHASE V40.45 diversa da PHASE OOS."
    )

common["PHASE"] = (
    common[
        "PHASE_PILLAR"
    ]
)


# =============================================================================
# TARGET COMPLETENESS
# =============================================================================

banner(
    "AUDIT COMPLETEZZA BUY/SELL 5W/8W"
)

counts = (
    common.groupby(
        [
            "Ticker",
            "Date",
        ],
        observed=True,
    )
    .size()
)

count_distribution = (
    counts
    .value_counts()
    .sort_index()
)

print(
    count_distribution.to_string()
)

complete_four_dates = counts[
    counts == 4
].index

complete_keys = pd.DataFrame(
    complete_four_dates.tolist(),
    columns=[
        "Ticker",
        "Date",
    ],
)

common_complete = common.merge(
    complete_keys,
    on=[
        "Ticker",
        "Date",
    ],
    how="inner",
    validate="many_to_one",
)

dropped_incomplete_dates = int(
    n_target_dates
    -
    len(
        complete_keys
    )
)

print()
print(
    f"Ticker/Date completi 4/4: "
    f"{len(complete_keys):,}"
)

print(
    f"Ticker/Date incompleti esclusi: "
    f"{dropped_incomplete_dates:,}"
)

common = (
    common_complete
    .sort_values(
        [
            "Ticker",
            "Date",
            "SIDE",
            "HORIZON_W",
        ]
    )
    .reset_index(
        drop=True
    )
)

expected_rows = (
    len(
        complete_keys
    )
    *
    4
)

final_duplicates = int(
    common.duplicated(
        [
            "Ticker",
            "Date",
            "SIDE",
            "HORIZON_W",
        ]
    ).sum()
)

print(
    f"Righe finali attese: "
    f"{expected_rows:,}"
)

print(
    f"Righe finali effettive: "
    f"{len(common):,}"
)

print(
    f"Duplicati finali: "
    f"{final_duplicates:,}"
)

if len(common) != expected_rows:
    raise RuntimeError(
        "Numero righe finali inatteso."
    )

if final_duplicates:
    raise RuntimeError(
        "Duplicati dataset finale."
    )


# =============================================================================
# TARGET NUMERIC AUDIT
# =============================================================================

banner(
    "AUDIT NUMERICO TARGET"
)

for metric in TARGET_METRICS:

    values = pd.to_numeric(
        common[
            metric
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    bad = int(
        (
            ~np.isfinite(
                values
            )
        ).sum()
    )

    print(
        f"{metric}: non-finite={bad:,}"
    )

    if bad:
        raise RuntimeError(
            f"Target non finito: {metric}"
        )


# =============================================================================
# SAVE COMMON OOS DATASET
# =============================================================================

banner(
    "SALVATAGGIO DATASET OOS COMUNE"
)

common.to_csv(
    OUTPUT_COMMON,
    index=False,
)

print(
    f"Salvato: {OUTPUT_COMMON}"
)

print(
    "Dimensione: "
    f"{OUTPUT_COMMON.stat().st_size / (1024 ** 2):.2f} MB"
)


# =============================================================================
# PHASE TARGET STUDY
# =============================================================================

banner(
    "STUDIO TARGET PER PHASE"
)

phase_rows = []

for (
    side,
    horizon,
    phase,
), g in common.groupby(
    [
        "SIDE",
        "HORIZON_W",
        "PHASE",
    ],
    observed=True,
):

    row = {
        "SIDE": side,
        "HORIZON_W": int(horizon),
        "PHASE": phase,
        "N": int(len(g)),
        "TICKERS": int(
            g["Ticker"].nunique()
        ),
    }

    for metric in TARGET_METRICS:

        values = pd.to_numeric(
            g[metric],
            errors="coerce",
        )

        row[
            f"{metric}_MEAN"
        ] = float(
            values.mean()
        )

        row[
            f"{metric}_MEDIAN"
        ] = float(
            values.median()
        )

    phase_rows.append(
        row
    )

phase_summary = pd.DataFrame(
    phase_rows
)

phase_summary.to_csv(
    OUTPUT_PHASE_SUMMARY,
    index=False,
)

print(
    f"Salvato: {OUTPUT_PHASE_SUMMARY}"
)


# =============================================================================
# STRENGTH TARGET STUDY
# =============================================================================

banner(
    "STUDIO TARGET PER STRENGTH"
)

strength_rows = []

for (
    side,
    horizon,
    direction,
    strength_class,
), g in common.groupby(
    [
        "SIDE",
        "HORIZON_W",
        "STRENGTH_DIRECTION",
        "STRENGTH_CLASS",
    ],
    observed=True,
):

    row = {
        "SIDE": side,
        "HORIZON_W": int(horizon),
        "STRENGTH_DIRECTION": direction,
        "STRENGTH_CLASS": strength_class,
        "N": int(len(g)),
        "TICKERS": int(
            g["Ticker"].nunique()
        ),
    }

    for metric in TARGET_METRICS:

        values = pd.to_numeric(
            g[metric],
            errors="coerce",
        )

        row[
            f"{metric}_MEAN"
        ] = float(
            values.mean()
        )

        row[
            f"{metric}_MEDIAN"
        ] = float(
            values.median()
        )

    strength_rows.append(
        row
    )

strength_summary = pd.DataFrame(
    strength_rows
)

strength_order = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "VERY_HIGH",
]

strength_summary[
    "STRENGTH_CLASS"
] = pd.Categorical(
    strength_summary[
        "STRENGTH_CLASS"
    ],
    categories=strength_order,
    ordered=True,
)

strength_summary = (
    strength_summary.sort_values(
        [
            "SIDE",
            "HORIZON_W",
            "STRENGTH_DIRECTION",
            "STRENGTH_CLASS",
        ]
    )
)

strength_summary.to_csv(
    OUTPUT_STRENGTH_SUMMARY,
    index=False,
)

print(
    f"Salvato: {OUTPUT_STRENGTH_SUMMARY}"
)


# =============================================================================
# REVERSAL DECILE STUDY
# =============================================================================

banner(
    "STUDIO TARGET PER DECILI REVERSAL 3W"
)

decile_rows = []

for (
    side,
    horizon,
    direction,
), g0 in common.groupby(
    [
        "SIDE",
        "HORIZON_W",
        "STRENGTH_DIRECTION",
    ],
    observed=True,
):

    g = g0.copy()

    ranked = (
        g[
            "REVERSAL_3W"
        ]
        .rank(
            method="first"
        )
    )

    if len(g) < 10:
        continue

    g[
        "REVERSAL_DECILE"
    ] = (
        pd.qcut(
            ranked,
            q=10,
            labels=False,
            duplicates="drop",
        )
        +
        1
    )

    for decile, gd in g.groupby(
        "REVERSAL_DECILE",
        observed=True,
    ):

        row = {
            "SIDE": side,
            "HORIZON_W": int(horizon),
            "STRENGTH_DIRECTION": direction,
            "REVERSAL_DECILE": int(decile),
            "N": int(len(gd)),
            "TICKERS": int(
                gd["Ticker"].nunique()
            ),
            "REVERSAL_1W_MEAN": float(
                gd["REVERSAL_1W"].mean()
            ),
            "REVERSAL_2W_MEAN": float(
                gd["REVERSAL_2W"].mean()
            ),
            "REVERSAL_3W_MEAN": float(
                gd["REVERSAL_3W"].mean()
            ),
        }

        for metric in TARGET_METRICS:

            values = pd.to_numeric(
                gd[metric],
                errors="coerce",
            )

            row[
                f"{metric}_MEAN"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_MEDIAN"
            ] = float(
                values.median()
            )

        decile_rows.append(
            row
        )

reversal_deciles = pd.DataFrame(
    decile_rows
)

reversal_deciles.to_csv(
    OUTPUT_REVERSAL_DECILES,
    index=False,
)

print(
    f"Salvato: {OUTPUT_REVERSAL_DECILES}"
)


# =============================================================================
# CORRELATION STUDY
# =============================================================================

banner(
    "CORRELAZIONI OOS PILASTRI -> TARGET"
)

predictors = [
    "STRENGTH_EXPECTED",
    "REVERSAL_1W",
    "REVERSAL_2W",
    "REVERSAL_3W",
]

correlation_rows = []

for (
    side,
    horizon,
    direction,
), g in common.groupby(
    [
        "SIDE",
        "HORIZON_W",
        "STRENGTH_DIRECTION",
    ],
    observed=True,
):

    for predictor in predictors:

        for metric in TARGET_METRICS:

            rho = spearman_safe(
                g[
                    predictor
                ],
                g[
                    metric
                ],
            )

            n_valid = int(
                g[
                    [
                        predictor,
                        metric,
                    ]
                ]
                .dropna()
                .shape[0]
            )

            correlation_rows.append(
                {
                    "SIDE": side,
                    "HORIZON_W": int(horizon),
                    "PHASE_DIRECTION": direction,
                    "PREDICTOR": predictor,
                    "TARGET_METRIC": metric,
                    "N": n_valid,
                    "SPEARMAN": rho,
                }
            )

correlations = pd.DataFrame(
    correlation_rows
)

correlations.to_csv(
    OUTPUT_CORRELATIONS,
    index=False,
)

print(
    f"Salvato: {OUTPUT_CORRELATIONS}"
)


# =============================================================================
# PRINT STRONGEST CORRELATIONS
# =============================================================================

banner(
    "CORRELAZIONI ASSOLUTE PIU' FORTI"
)

corr_print = (
    correlations
    .dropna(
        subset=[
            "SPEARMAN"
        ]
    )
    .copy()
)

corr_print[
    "ABS_SPEARMAN"
] = (
    corr_print[
        "SPEARMAN"
    ]
    .abs()
)

corr_print = (
    corr_print.sort_values(
        "ABS_SPEARMAN",
        ascending=False,
    )
    .head(40)
)

print(
    corr_print[
        [
            "SIDE",
            "HORIZON_W",
            "PHASE_DIRECTION",
            "PREDICTOR",
            "TARGET_METRIC",
            "N",
            "SPEARMAN",
        ]
    ]
    .to_string(
        index=False
    )
)


# =============================================================================
# KEY TARGET MATRICES
# =============================================================================

banner(
    "TARGET CHIAVE"
)

key_metrics = [
    "MFE_PCT",
    "MAE_PCT",
    "SPEED_PCT_PER_WEEK",
    "FINAL_DIRECTIONAL_RETURN_PCT",
    "PATH_EFFICIENCY",
]

key_corr = correlations.loc[
    correlations[
        "TARGET_METRIC"
    ].isin(
        key_metrics
    )
].copy()

for (
    side,
    horizon,
    direction,
), g in key_corr.groupby(
    [
        "SIDE",
        "HORIZON_W",
        "PHASE_DIRECTION",
    ],
    observed=True,
):

    print()
    print(
        f"{side} {int(horizon)}W | "
        f"PHASE {direction}"
    )

    pivot = g.pivot(
        index="TARGET_METRIC",
        columns="PREDICTOR",
        values="SPEARMAN",
    )

    print(
        pivot.round(
            4
        ).to_string()
    )


# =============================================================================
# SAMPLE STRUCTURE
# =============================================================================

banner(
    "STRUTTURA CAMPIONE FINALE"
)

print(
    "PHASE:"
)

print(
    common[
        [
            "Ticker",
            "Date",
            "PHASE",
        ]
    ]
    .drop_duplicates()
    ["PHASE"]
    .value_counts()
    .to_string()
)

print()
print(
    "DIRECTION:"
)

print(
    common[
        [
            "Ticker",
            "Date",
            "STRENGTH_DIRECTION",
        ]
    ]
    .drop_duplicates()
    ["STRENGTH_DIRECTION"]
    .value_counts()
    .to_string()
)

print()
print(
    "STRENGTH CLASS:"
)

print(
    common[
        [
            "Ticker",
            "Date",
            "STRENGTH_CLASS",
        ]
    ]
    .drop_duplicates()
    ["STRENGTH_CLASS"]
    .value_counts()
    .to_string()
)


# =============================================================================
# METADATA
# =============================================================================

metadata = {
    "version":
        VERSION,

    "purpose":
        (
            "Opportunity target study using "
            "OOS PHASE + calibrated REVERSAL + "
            "OOS STRENGTH + prospective V40.45 targets"
        ),

    "phase_source":
        "V40.39",

    "phase_modified":
        False,

    "reversal_source":
        "V40.40a",

    "reversal_calibration_manifest":
        "V40.40b",

    "reversal_calibration_policy":
        "forward_only",

    "reversal_fold1_policy":
        "RAW only - excluded from calibrated V40.46 sample",

    "reversal_raw_fallback_used":
        False,

    "reversal_modified":
        False,

    "strength_source":
        "V40.43",

    "strength_variant":
        "COMPACT_ROLLING",

    "strength_modified":
        False,

    "target_source":
        "V40.45",

    "reversal_y_true_used_as_feature":
        False,

    "strength_y_num_used_as_feature":
        False,

    "opportunity_model_built":
        False,

    "opportunity_score_1_10_built":
        False,

    "manual_weights_used":
        False,

    "manual_thresholds_used":
        False,

    "reversal_selected_methods":
        method_map,

    "strength_reversal_common_dates_before_target":
        int(
            len(pillars)
        ),

    "complete_target_dates":
        int(
            len(
                complete_keys
            )
        ),

    "final_rows":
        int(
            len(common)
        ),

    "final_tickers":
        int(
            common[
                "Ticker"
            ].nunique()
        ),

    "final_date_min":
        (
            str(
                common[
                    "Date"
                ].min().date()
            )
            if len(common)
            else None
        ),

    "final_date_max":
        (
            str(
                common[
                    "Date"
                ].max().date()
            )
            if len(common)
            else None
        ),

    "phase_strength_conflicts":
        int(
            phase_strength_conflicts
        ),

    "phase_target_conflicts":
        int(
            phase_target_conflicts
        ),

    "final_duplicates":
        int(
            final_duplicates
        ),

    "incomplete_target_dates_excluded":
        int(
            dropped_incomplete_dates
        ),

    "target_metrics_studied":
        TARGET_METRICS,
}

with open(
    OUTPUT_METADATA,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# FINAL FILES
# =============================================================================

banner(
    "FILE GENERATI"
)

for path in [
    OUTPUT_COMMON,
    OUTPUT_PHASE_SUMMARY,
    OUTPUT_STRENGTH_SUMMARY,
    OUTPUT_REVERSAL_DECILES,
    OUTPUT_CORRELATIONS,
    OUTPUT_MANIFEST_USED,
    OUTPUT_METADATA,
]:
    print(
        path
    )


# =============================================================================
# VERDICT
# =============================================================================

banner(
    "VERDETTO V40.46"
)

print(
    "PHASE modificata: NO"
)

print(
    "REVERSAL modificato: NO"
)

print(
    "STRENGTH modificato: NO"
)

print(
    "Reversal Fold 1 RAW-only riconosciuto: SI"
)

print(
    "RAW_PROB usato come fallback: NO"
)

print(
    "Solo Reversal calibrato 1W/2W/3W "
    "della direzione PHASE: SI"
)

print(
    "Y_TRUE Reversal usato come feature: NO"
)

print(
    "y_num Strength usato come feature: NO"
)

print(
    "Modello Opportunity costruito: NO"
)

print(
    "Score Opportunity 1-10 costruito: NO"
)

print(
    "Pesi Opportunity inventati: NO"
)

print(
    "Soglie Opportunity inventate: NO"
)

print()
print(
    f"Ticker/Date Strength+Reversal calibrato: "
    f"{len(pillars):,}"
)

print(
    f"Ticker/Date finali con target 4/4: "
    f"{len(complete_keys):,}"
)

print(
    f"Righe finali: "
    f"{len(common):,}"
)

print(
    f"Ticker finali: "
    f"{common['Ticker'].nunique():,}"
)

if len(common):

    print(
        "Periodo finale: "
        f"{common['Date'].min().date()} "
        "-> "
        f"{common['Date'].max().date()}"
    )

print()
print(
    "V40.46 OPPORTUNITY TARGET STUDY COMPLETATO."
)