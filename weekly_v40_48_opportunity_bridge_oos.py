# ==================================================================================================
# MARKET SENTINEL - V40.48
# OPPORTUNITY BRIDGE OOS BENCHMARK
#
# SCOPO
# -----
# Capire quanto del segnale storico Opportunity V40.37d viene recuperato
# mantenendo il motore tecnico ricco da 413 feature e aggiornando il contesto
# all'architettura corrente:
#
#   TECH413
#   + PHASE V40.39
#   + STRENGTH storico OOS V40.43
#   + REVERSAL storico OOS V40.40 calibrato forward-only
#
# V40.46 fornisce il campione comune già auditato tra:
#   PHASE + STRENGTH OOS + REVERSAL OOS + TARGET V40.45
#
# QUESTO SCRIPT NON:
# - modifica PHASE V40.39
# - modifica REVERSAL V40.40
# - modifica STRENGTH V40.44
# - costruisce uno score 1-10
# - sceglie pesi manuali
# - fa tuning del Random Forest
#
# MODELLO
# -------
# Replica RF storico V40.21 / V40.37d:
#   n_estimators=500
#   max_features="sqrt"
#   min_samples_leaf=4
#   random_state=42
#   n_jobs=1
#
# TARGET
# ------
# Due definizioni SPEED:
#
# 1) CONDITIONAL
#    replica concettualmente V40.37d:
#    se non esiste movimento favorevole -> target NaN -> riga esclusa
#
# 2) COMPLETE
#    metodologia V40.45:
#    se non esiste movimento favorevole -> SPEED = 0
#
# FEATURE SET
# -----------
# TECH_PHASE
# TECH_PHASE_STRENGTH
# TECH_PHASE_REVERSAL
# ALL
#
# IMPORTANTE:
# tutti i feature set vengono confrontati sullo stesso identico campione
# disponibile per il target/fold corrente.
#
# SPLIT
# -----
# Usiamo gli stessi 4 blocchi temporali del V40.47, così il confronto
# V40.47 (pilastri compressi) vs V40.48 (motore tecnico ricco) è leggibile.
#
# Purging:
#   ultima settimana training = posizione test_start - horizon - 1
#
# CHECKPOINT
# ----------
# Ogni job completato viene scritto immediatamente.
# Lo script può essere rilanciato e riprende dai job già completati.
# ==================================================================================================

from pathlib import Path
import gc
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==================================================================================================
# CONFIG
# ==================================================================================================

DATA_DIR = Path("data")

FEATURE_FILE = DATA_DIR / "v40_36b_full200_features.csv"
FEATURE_LIST_FILE = DATA_DIR / "v40_36b_feature_list.csv"

COMMON_FILE = (
    DATA_DIR
    / "v40_46_opportunity_target_study"
    / "v40_46_common_oos_targets.csv"
)

OUT_DIR = (
    DATA_DIR
    / "v40_48_opportunity_bridge_oos"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_RESULTS = OUT_DIR / "v40_48_candidate_results.csv"
OUT_SUMMARY = OUT_DIR / "v40_48_candidate_summary.csv"
OUT_INCREMENTAL = OUT_DIR / "v40_48_incremental_value.csv"
OUT_PREDICTIONS = OUT_DIR / "v40_48_oos_predictions.csv"
OUT_SPLITS = OUT_DIR / "v40_48_split_audit.csv"
OUT_METADATA = OUT_DIR / "v40_48_metadata.json"

CACHE_FEATURES = OUT_DIR / "v40_48_common_413_features.npy"
CACHE_KEYS = OUT_DIR / "v40_48_common_feature_keys.csv"

EXPECTED_FEATURES = 413

SIDES = [
    "BUY",
    "SELL",
]

HORIZONS = [
    5,
    8,
]

TARGET_VARIANTS = [
    "CONDITIONAL",
    "COMPLETE",
]

FEATURE_SETS = [
    "TECH_PHASE",
    "TECH_PHASE_STRENGTH",
    "TECH_PHASE_REVERSAL",
    "ALL",
]

PHASE_CATEGORIES = [
    "BUY",
    "TREND_RIALZISTA",
    "DEBOLEZZA_RIALZISTA",
    "SELL",
    "TREND_RIBASSISTA",
    "DEBOLEZZA_RIBASSISTA",
    "INDECISIONE",
]

STRENGTH_CLASSES = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "VERY_HIGH",
]

RF_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 4,
    "n_jobs": 1,
    "random_state": 42,
}

MIN_TRAIN_ROWS = 1500
MIN_TEST_ROWS = 250

FEATURE_CHUNK_ROWS = 2000

TEST_BLOCKS = [
    {
        "FOLD": 1,
        "TEST_START": "2023-04-07",
        "TEST_END": "2023-12-29",
    },
    {
        "FOLD": 2,
        "TEST_START": "2024-01-05",
        "TEST_END": "2024-12-27",
    },
    {
        "FOLD": 3,
        "TEST_START": "2025-01-03",
        "TEST_END": "2025-12-26",
    },
    {
        "FOLD": 4,
        "TEST_START": "2026-01-02",
        "TEST_END": "2026-07-03",
    },
]


# ==================================================================================================
# UTILITY
# ==================================================================================================

def banner(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)
    print()


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_spearman(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    valid = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    if valid.sum() < 3:
        return np.nan

    if (
        np.nanstd(a[valid]) == 0
        or
        np.nanstd(b[valid]) == 0
    ):
        return np.nan

    return float(
        spearmanr(
            a[valid],
            b[valid],
        ).statistic
    )


def safe_pearson(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    valid = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    if valid.sum() < 3:
        return np.nan

    if (
        np.nanstd(a[valid]) == 0
        or
        np.nanstd(b[valid]) == 0
    ):
        return np.nan

    return float(
        pearsonr(
            a[valid],
            b[valid],
        ).statistic
    )


def normalize_phase(value):
    p = (
        str(value)
        .upper()
        .strip()
        .replace(" ", "_")
    )

    aliases = {
        "TREND_UP": "TREND_RIALZISTA",
        "WEAK_UP": "DEBOLEZZA_RIALZISTA",
        "TREND_DOWN": "TREND_RIBASSISTA",
        "WEAK_DOWN": "DEBOLEZZA_RIBASSISTA",
        "NAN": "INDECISIONE",
        "NONE": "INDECISIONE",
        "UNKNOWN": "INDECISIONE",
    }

    return aliases.get(
        p,
        p,
    )


def load_completed_jobs():
    if not OUT_RESULTS.exists():
        return set()

    try:
        old = pd.read_csv(
            OUT_RESULTS
        )
    except Exception:
        return set()

    needed = {
        "SIDE",
        "HORIZON_W",
        "TARGET_VARIANT",
        "FEATURE_SET",
        "FOLD",
    }

    if not needed.issubset(
        old.columns
    ):
        return set()

    completed = set()

    for row in old.itertuples(
        index=False
    ):
        completed.add(
            (
                str(row.SIDE),
                int(row.HORIZON_W),
                str(row.TARGET_VARIANT),
                str(row.FEATURE_SET),
                int(row.FOLD),
            )
        )

    return completed


def append_row(path, row):
    df = pd.DataFrame(
        [row]
    )

    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def append_predictions(df):
    df.to_csv(
        OUT_PREDICTIONS,
        mode="a",
        header=not OUT_PREDICTIONS.exists(),
        index=False,
    )


# ==================================================================================================
# INPUT AUDIT
# ==================================================================================================

banner(
    "V40.48 - OPPORTUNITY BRIDGE OOS"
)

for path in [
    FEATURE_FILE,
    FEATURE_LIST_FILE,
    COMMON_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"File mancante: {path}"
        )

print(
    f"413 feature: {FEATURE_FILE}"
)

print(
    f"Feature list: {FEATURE_LIST_FILE}"
)

print(
    f"Common OOS:  {COMMON_FILE}"
)


# ==================================================================================================
# FEATURE LIST
# ==================================================================================================

banner(
    "RECUPERO 413 FEATURE TECNICHE"
)

feature_list_df = pd.read_csv(
    FEATURE_LIST_FILE
)

feature_candidates = []

for col in [
    "FEATURE",
    "Feature",
    "feature",
    "COLUMN",
    "Column",
    "column",
]:
    if col in feature_list_df.columns:
        feature_candidates = (
            feature_list_df[col]
            .dropna()
            .astype(str)
            .tolist()
        )
        break

if not feature_candidates:
    if feature_list_df.shape[1] == 1:
        feature_candidates = (
            feature_list_df.iloc[:, 0]
            .dropna()
            .astype(str)
            .tolist()
        )
    else:
        raise RuntimeError(
            "Impossibile identificare la feature list."
        )

base_features = list(
    dict.fromkeys(
        feature_candidates
    )
)

print(
    f"Feature tecniche: {len(base_features)}"
)

if len(base_features) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Attese {EXPECTED_FEATURES} feature, "
        f"trovate {len(base_features)}."
    )

feature_header = pd.read_csv(
    FEATURE_FILE,
    nrows=0,
).columns.tolist()

for required in [
    "Ticker",
    "Date",
]:
    if required not in feature_header:
        raise RuntimeError(
            f"{required} mancante in {FEATURE_FILE}"
        )

missing_features = [
    f
    for f in base_features
    if f not in feature_header
]

if missing_features:
    raise RuntimeError(
        "Feature mancanti: "
        +
        ", ".join(
            missing_features[:30]
        )
    )

print(
    "Audit feature list: OK"
)


# ==================================================================================================
# COMMON OOS V40.46
# ==================================================================================================

banner(
    "CARICAMENTO COMMON OOS V40.46"
)

common = pd.read_csv(
    COMMON_FILE
)

required_common = [
    "Ticker",
    "Date",
    "PHASE",
    "PHASE_DIRECTION",
    "PHASE_AGE",
    "PHASE_CHANGED",
    "SIDE",
    "HORIZON_W",
    "MFE_PCT",
    "SPEED_PCT_PER_WEEK",
    "HAS_FAVORABLE_MOVE",
    "STRENGTH_EXPECTED",
    "STRENGTH_CLASS",
    "REVERSAL_1W",
    "REVERSAL_2W",
    "REVERSAL_3W",
]

missing_common = [
    c
    for c in required_common
    if c not in common.columns
]

if missing_common:
    raise RuntimeError(
        "Colonne mancanti nel V40.46: "
        +
        ", ".join(
            missing_common
        )
    )

common["Ticker"] = (
    common["Ticker"]
    .astype(str)
)

common["Date"] = pd.to_datetime(
    common["Date"],
    errors="coerce",
)

if common["Date"].isna().any():
    raise RuntimeError(
        "Date non valide nel common V40.46."
    )

common["PHASE"] = (
    common["PHASE"]
    .map(
        normalize_phase
    )
)

for c in [
    "PHASE_DIRECTION",
    "PHASE_AGE",
    "PHASE_CHANGED",
    "MFE_PCT",
    "SPEED_PCT_PER_WEEK",
    "HAS_FAVORABLE_MOVE",
    "STRENGTH_EXPECTED",
    "REVERSAL_1W",
    "REVERSAL_2W",
    "REVERSAL_3W",
]:
    common[c] = safe_numeric(
        common[c]
    )

common["STRENGTH_CLASS"] = (
    common["STRENGTH_CLASS"]
    .astype(str)
    .str.upper()
    .str.strip()
)

print(
    f"Righe target:       {len(common):,}"
)

print(
    f"Ticker:             {common['Ticker'].nunique()}"
)

print(
    f"Ticker/Date:        "
    f"{common[['Ticker','Date']].drop_duplicates().shape[0]:,}"
)

print(
    f"Periodo:            "
    f"{common['Date'].min().date()} -> "
    f"{common['Date'].max().date()}"
)

dup_target = int(
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
    f"Duplicati target:   {dup_target}"
)

if dup_target != 0:
    raise RuntimeError(
        "Duplicati Ticker/Date/SIDE/HORIZON nel V40.46."
    )

expected_sides = set(
    common["SIDE"]
    .astype(str)
    .unique()
)

if not set(SIDES).issubset(
    expected_sides
):
    raise RuntimeError(
        f"SIDE inattesi: {expected_sides}"
    )

expected_horizons = set(
    safe_numeric(
        common["HORIZON_W"]
    )
    .dropna()
    .astype(int)
    .unique()
)

if not set(HORIZONS).issubset(
    expected_horizons
):
    raise RuntimeError(
        f"HORIZON inattesi: {expected_horizons}"
    )


# ==================================================================================================
# CONTEXT TABLE - UNA RIGA PER TICKER/DATE
# ==================================================================================================

banner(
    "COSTRUZIONE CONTEXT TABLE"
)

context_cols = [
    "Ticker",
    "Date",
    "PHASE",
    "PHASE_DIRECTION",
    "PHASE_AGE",
    "PHASE_CHANGED",
    "STRENGTH_EXPECTED",
    "STRENGTH_CLASS",
    "REVERSAL_1W",
    "REVERSAL_2W",
    "REVERSAL_3W",
]

context = (
    common[
        context_cols
    ]
    .drop_duplicates()
    .copy()
)

context_dup = int(
    context.duplicated(
        [
            "Ticker",
            "Date",
        ]
    ).sum()
)

print(
    f"Context rows:       {len(context):,}"
)

print(
    f"Duplicati context:  {context_dup}"
)

if context_dup != 0:
    conflict = (
        context.groupby(
            [
                "Ticker",
                "Date",
            ]
        )
        .size()
    )

    conflict = conflict[
        conflict > 1
    ]

    print(
        conflict.head(
            20
        )
    )

    raise RuntimeError(
        "Context V40.46 non univoco per Ticker/Date."
    )

context = (
    context
    .sort_values(
        [
            "Date",
            "Ticker",
        ]
    )
    .reset_index(
        drop=True
    )
)

context["ROW_ID"] = np.arange(
    len(context),
    dtype=int,
)

key_to_row = {
    (
        ticker,
        date.strftime(
            "%Y-%m-%d"
        ),
    ): int(row_id)
    for ticker, date, row_id
    in zip(
        context["Ticker"],
        context["Date"],
        context["ROW_ID"],
    )
}

print(
    f"Chiavi richieste:   {len(key_to_row):,}"
)


# ==================================================================================================
# 413 FEATURES - MEMORY SAFE CACHE
# ==================================================================================================

banner(
    "ALLINEAMENTO 413 FEATURE AL COMMON OOS"
)

if (
    CACHE_FEATURES.exists()
    and
    CACHE_KEYS.exists()
):
    print(
        "Cache 413 già presente."
    )

    cache_keys = pd.read_csv(
        CACHE_KEYS
    )

    cache_keys["Date"] = pd.to_datetime(
        cache_keys["Date"],
        errors="coerce",
    )

    expected_keys = context[
        [
            "Ticker",
            "Date",
            "ROW_ID",
        ]
    ].copy()

    if len(cache_keys) != len(expected_keys):
        raise RuntimeError(
            "Cache key con numero righe inatteso."
        )

    merged_key_check = expected_keys.merge(
        cache_keys,
        on=[
            "Ticker",
            "Date",
            "ROW_ID",
        ],
        how="outer",
        indicator=True,
    )

    if not (
        merged_key_check["_merge"]
        ==
        "both"
    ).all():
        raise RuntimeError(
            "Cache 413 non coerente con common V40.46."
        )

    base_matrix = np.load(
        CACHE_FEATURES,
        mmap_mode="r",
    )

    if base_matrix.shape != (
        len(context),
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            f"Shape cache inattesa: {base_matrix.shape}"
        )

else:

    print(
        "Cache non presente: costruzione streaming."
    )

    base_matrix_write = np.lib.format.open_memmap(
        CACHE_FEATURES,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(context),
            EXPECTED_FEATURES,
        ),
    )

    base_matrix_write[:] = np.nan

    found = np.zeros(
        len(context),
        dtype=bool,
    )

    usecols = (
        [
            "Ticker",
            "Date",
        ]
        +
        base_features
    )

    total_seen = 0
    total_matched = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            FEATURE_FILE,
            usecols=usecols,
            chunksize=FEATURE_CHUNK_ROWS,
        ),
        start=1,
    ):

        total_seen += len(chunk)

        chunk["Ticker"] = (
            chunk["Ticker"]
            .astype(str)
        )

        chunk["Date"] = (
            chunk["Date"]
            .astype(str)
            .str.slice(
                0,
                10,
            )
        )

        row_ids = np.full(
            len(chunk),
            -1,
            dtype=int,
        )

        for i, (
            ticker,
            date_text,
        ) in enumerate(
            zip(
                chunk["Ticker"],
                chunk["Date"],
            )
        ):
            row_ids[i] = key_to_row.get(
                (
                    ticker,
                    date_text,
                ),
                -1,
            )

        mask = (
            row_ids
            >=
            0
        )

        if mask.any():

            selected = chunk.loc[
                mask,
                base_features,
            ]

            values = (
                selected
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .to_numpy(
                    dtype=np.float32,
                )
            )

            selected_row_ids = row_ids[
                mask
            ]

            if found[
                selected_row_ids
            ].any():
                raise RuntimeError(
                    "Duplicazione chiavi durante "
                    "l'allineamento delle 413 feature."
                )

            base_matrix_write[
                selected_row_ids,
                :
            ] = values

            found[
                selected_row_ids
            ] = True

            total_matched += int(
                mask.sum()
            )

        if (
            chunk_no == 1
            or
            chunk_no % 10 == 0
        ):
            print(
                f"Chunk {chunk_no:03d} | "
                f"lette={total_seen:,} | "
                f"match={total_matched:,}"
            )

        del chunk
        gc.collect()

    missing_keys = int(
        (~found).sum()
    )

    print()
    print(
        f"Righe FULL200 lette:       {total_seen:,}"
    )
    print(
        f"Chiavi common trovate:     {found.sum():,}"
    )
    print(
        f"Chiavi common mancanti:    {missing_keys:,}"
    )

    if missing_keys != 0:
        missing_sample = context.loc[
            ~found,
            [
                "Ticker",
                "Date",
            ],
        ].head(
            20
        )

        print(
            missing_sample.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Non tutte le chiavi V40.46 sono presenti "
            "nel V40.36b."
        )

    base_matrix_write.flush()

    context[
        [
            "Ticker",
            "Date",
            "ROW_ID",
        ]
    ].to_csv(
        CACHE_KEYS,
        index=False,
    )

    del base_matrix_write
    gc.collect()

    base_matrix = np.load(
        CACHE_FEATURES,
        mmap_mode="r",
    )

print(
    f"413 matrix shape: {base_matrix.shape}"
)

if base_matrix.shape != (
    len(context),
    EXPECTED_FEATURES,
):
    raise RuntimeError(
        "Shape matrice 413 errata."
    )


# ==================================================================================================
# CONTEXT MATRICES
# ==================================================================================================

banner(
    "COSTRUZIONE PHASE / STRENGTH / REVERSAL CONTEXT"
)

n_rows = len(context)

phase_numeric_names = [
    "PHASE_DIRECTION",
    "PHASE_AGE",
    "PHASE_CHANGED",
]

phase_numeric = (
    context[
        phase_numeric_names
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(
        dtype=np.float32,
    )
)

phase_values = (
    context["PHASE"]
    .astype(str)
    .to_numpy()
)

unknown_phases = sorted(
    set(
        phase_values
    )
    -
    set(
        PHASE_CATEGORIES
    )
)

if unknown_phases:
    raise RuntimeError(
        "PHASE non previste: "
        +
        ", ".join(
            unknown_phases
        )
    )

phase_onehot = np.zeros(
    (
        n_rows,
        len(
            PHASE_CATEGORIES
        ),
    ),
    dtype=np.float32,
)

for j, phase in enumerate(
    PHASE_CATEGORIES
):
    phase_onehot[
        :,
        j
    ] = (
        phase_values
        ==
        phase
    ).astype(
        np.float32
    )

strength_expected = (
    context[
        [
            "STRENGTH_EXPECTED",
        ]
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(
        dtype=np.float32,
    )
)

strength_values = (
    context["STRENGTH_CLASS"]
    .astype(str)
    .to_numpy()
)

unknown_strength = sorted(
    set(
        strength_values
    )
    -
    set(
        STRENGTH_CLASSES
    )
)

if unknown_strength:
    raise RuntimeError(
        "STRENGTH_CLASS non previste: "
        +
        ", ".join(
            unknown_strength
        )
    )

strength_onehot = np.zeros(
    (
        n_rows,
        len(
            STRENGTH_CLASSES
        ),
    ),
    dtype=np.float32,
)

for j, strength_class in enumerate(
    STRENGTH_CLASSES
):
    strength_onehot[
        :,
        j
    ] = (
        strength_values
        ==
        strength_class
    ).astype(
        np.float32
    )

reversal_matrix = (
    context[
        [
            "REVERSAL_1W",
            "REVERSAL_2W",
            "REVERSAL_3W",
        ]
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .to_numpy(
        dtype=np.float32,
    )
)

if not np.isfinite(
    reversal_matrix
).all():
    raise RuntimeError(
        "REVERSAL non finite nel common OOS."
    )

if (
    (
        reversal_matrix
        <
        0
    ).any()
    or
    (
        reversal_matrix
        >
        1
    ).any()
):
    raise RuntimeError(
        "REVERSAL fuori [0,1]."
    )

print(
    f"PHASE numeric:      {phase_numeric.shape[1]}"
)

print(
    f"PHASE one-hot:      {phase_onehot.shape[1]}"
)

print(
    f"STRENGTH numeric:   {strength_expected.shape[1]}"
)

print(
    f"STRENGTH one-hot:   {strength_onehot.shape[1]}"
)

print(
    f"REVERSAL:           {reversal_matrix.shape[1]}"
)


# ==================================================================================================
# TARGET TABLE
# ==================================================================================================

banner(
    "PREPARAZIONE TARGET SPEED"
)

targets = common[
    [
        "Ticker",
        "Date",
        "SIDE",
        "HORIZON_W",
        "SPEED_PCT_PER_WEEK",
        "HAS_FAVORABLE_MOVE",
    ]
].copy()

targets = targets.merge(
    context[
        [
            "Ticker",
            "Date",
            "ROW_ID",
        ]
    ],
    on=[
        "Ticker",
        "Date",
    ],
    how="left",
    validate="many_to_one",
)

if targets["ROW_ID"].isna().any():
    raise RuntimeError(
        "ROW_ID mancante nei target."
    )

targets["ROW_ID"] = (
    targets["ROW_ID"]
    .astype(int)
)

targets["HORIZON_W"] = (
    safe_numeric(
        targets["HORIZON_W"]
    )
    .astype(int)
)

targets["SPEED_COMPLETE"] = (
    safe_numeric(
        targets[
            "SPEED_PCT_PER_WEEK"
        ]
    )
)

targets["HAS_FAVORABLE_MOVE"] = (
    safe_numeric(
        targets[
            "HAS_FAVORABLE_MOVE"
        ]
    )
)

targets["SPEED_CONDITIONAL"] = (
    targets[
        "SPEED_COMPLETE"
    ]
)

targets.loc[
    targets[
        "HAS_FAVORABLE_MOVE"
    ]
    !=
    1,
    "SPEED_CONDITIONAL",
] = np.nan

complete_nonfinite = int(
    (
        ~np.isfinite(
            targets[
                "SPEED_COMPLETE"
            ]
        )
    ).sum()
)

print(
    f"Target rows:                  {len(targets):,}"
)

print(
    f"Complete SPEED nonfinite:     {complete_nonfinite:,}"
)

print(
    f"Conditional SPEED finite:     "
    f"{np.isfinite(targets['SPEED_CONDITIONAL']).sum():,}"
)

if complete_nonfinite != 0:
    raise RuntimeError(
        "SPEED_COMPLETE contiene valori non finite."
    )


# ==================================================================================================
# UNIQUE WEEK CALENDAR
# ==================================================================================================

unique_dates = np.sort(
    context["Date"]
    .drop_duplicates()
    .to_numpy(
        dtype="datetime64[ns]"
    )
)

date_to_position = {
    pd.Timestamp(d): i
    for i, d
    in enumerate(
        pd.to_datetime(
            unique_dates
        )
    )
}

print()
print(
    f"Settimane common: {len(unique_dates)}"
)

print(
    f"Da: {pd.Timestamp(unique_dates[0]).date()}"
)

print(
    f"A:  {pd.Timestamp(unique_dates[-1]).date()}"
)


# ==================================================================================================
# FEATURE MATRIX HELPERS
# ==================================================================================================

def calculate_medians(
    train_rows,
    feature_set,
):

    train_rows = np.asarray(
        train_rows,
        dtype=int,
    )

    x_base = np.asarray(
        base_matrix[
            train_rows,
            :
        ],
        dtype=np.float32,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        base_medians = np.nanmedian(
            x_base,
            axis=0,
        ).astype(
            np.float32
        )

    base_medians[
        ~np.isfinite(
            base_medians
        )
    ] = 0.0

    extra_parts = [
        phase_numeric[
            train_rows,
            :
        ],
    ]

    if feature_set in {
        "TECH_PHASE_STRENGTH",
        "ALL",
    }:
        extra_parts.append(
            strength_expected[
                train_rows,
                :
            ]
        )

    if feature_set in {
        "TECH_PHASE_REVERSAL",
        "ALL",
    }:
        extra_parts.append(
            reversal_matrix[
                train_rows,
                :
            ]
        )

    extra_numeric = np.concatenate(
        extra_parts,
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        extra_medians = np.nanmedian(
            extra_numeric,
            axis=0,
        ).astype(
            np.float32
        )

    extra_medians[
        ~np.isfinite(
            extra_medians
        )
    ] = 0.0

    return (
        base_medians,
        extra_medians,
    )


def build_matrix(
    row_ids,
    feature_set,
    base_medians,
    extra_medians,
):

    row_ids = np.asarray(
        row_ids,
        dtype=int,
    )

    x_base = np.asarray(
        base_matrix[
            row_ids,
            :
        ],
        dtype=np.float32,
    ).copy()

    bad = ~np.isfinite(
        x_base
    )

    if bad.any():
        rr, cc = np.where(
            bad
        )

        x_base[
            rr,
            cc
        ] = base_medians[
            cc
        ]

    numeric_parts = [
        phase_numeric[
            row_ids,
            :
        ],
    ]

    if feature_set in {
        "TECH_PHASE_STRENGTH",
        "ALL",
    }:
        numeric_parts.append(
            strength_expected[
                row_ids,
                :
            ]
        )

    if feature_set in {
        "TECH_PHASE_REVERSAL",
        "ALL",
    }:
        numeric_parts.append(
            reversal_matrix[
                row_ids,
                :
            ]
        )

    x_numeric = np.concatenate(
        numeric_parts,
        axis=1,
    ).astype(
        np.float32,
        copy=True,
    )

    bad_numeric = ~np.isfinite(
        x_numeric
    )

    if bad_numeric.any():
        rr, cc = np.where(
            bad_numeric
        )

        x_numeric[
            rr,
            cc
        ] = extra_medians[
            cc
        ]

    parts = [
        x_base,
        x_numeric,
        phase_onehot[
            row_ids,
            :
        ].astype(
            np.float32,
            copy=False,
        ),
    ]

    if feature_set in {
        "TECH_PHASE_STRENGTH",
        "ALL",
    }:
        parts.append(
            strength_onehot[
                row_ids,
                :
            ].astype(
                np.float32,
                copy=False,
            )
        )

    return np.concatenate(
        parts,
        axis=1,
    )


# ==================================================================================================
# SPLIT AUDIT
# ==================================================================================================

banner(
    "COSTRUZIONE SPLIT PURGED"
)

split_rows = []

for horizon in HORIZONS:

    for block in TEST_BLOCKS:

        fold = int(
            block[
                "FOLD"
            ]
        )

        requested_start = pd.Timestamp(
            block[
                "TEST_START"
            ]
        )

        requested_end = pd.Timestamp(
            block[
                "TEST_END"
            ]
        )

        possible = pd.to_datetime(
            unique_dates
        )

        test_dates = possible[
            (
                possible
                >=
                requested_start
            )
            &
            (
                possible
                <=
                requested_end
            )
        ]

        if len(test_dates) == 0:
            raise RuntimeError(
                f"Nessuna data test fold {fold}."
            )

        test_start = pd.Timestamp(
            test_dates.min()
        )

        test_end = pd.Timestamp(
            test_dates.max()
        )

        test_start_pos = date_to_position[
            test_start
        ]

        train_end_pos = (
            test_start_pos
            -
            horizon
            -
            1
        )

        if train_end_pos < 0:
            raise RuntimeError(
                f"Training insufficiente fold {fold} H={horizon}"
            )

        train_end = pd.Timestamp(
            unique_dates[
                train_end_pos
            ]
        )

        purge_weeks = (
            test_start_pos
            -
            train_end_pos
            -
            1
        )

        purge_ok = (
            purge_weeks
            ==
            horizon
            and
            train_end
            <
            test_start
        )

        split_rows.append(
            {
                "HORIZON_W": horizon,
                "FOLD": fold,
                "TRAIN_END": train_end,
                "TEST_START": test_start,
                "TEST_END": test_end,
                "PURGED_UNIQUE_WEEKS": purge_weeks,
                "EXPECTED_PURGE_WEEKS": horizon,
                "PURGE_OK": purge_ok,
            }
        )

split_audit = pd.DataFrame(
    split_rows
)

split_audit.to_csv(
    OUT_SPLITS,
    index=False,
)

print(
    split_audit.to_string(
        index=False
    )
)

if not split_audit[
    "PURGE_OK"
].all():
    raise RuntimeError(
        "AUDIT PURGING FALLITO."
    )


# ==================================================================================================
# RESUME
# ==================================================================================================

completed_jobs = load_completed_jobs()

total_jobs = (
    len(SIDES)
    *
    len(HORIZONS)
    *
    len(TARGET_VARIANTS)
    *
    len(FEATURE_SETS)
    *
    len(TEST_BLOCKS)
)

print()
print(
    f"Job totali:      {total_jobs}"
)

print(
    f"Job già fatti:   {len(completed_jobs)}"
)

print(
    f"Job rimanenti:   {total_jobs - len(completed_jobs)}"
)


# ==================================================================================================
# TRAIN / TEST
# ==================================================================================================

banner(
    "V40.48 RANDOM FOREST BRIDGE BENCHMARK"
)

job_counter = 0

for side in SIDES:

    for horizon in HORIZONS:

        z_target = targets[
            (
                targets[
                    "SIDE"
                ]
                ==
                side
            )
            &
            (
                targets[
                    "HORIZON_W"
                ]
                ==
                horizon
            )
        ].copy()

        for target_variant in TARGET_VARIANTS:

            target_col = (
                "SPEED_CONDITIONAL"
                if target_variant
                ==
                "CONDITIONAL"
                else
                "SPEED_COMPLETE"
            )

            for feature_set in FEATURE_SETS:

                for block in TEST_BLOCKS:

                    job_counter += 1

                    fold = int(
                        block[
                            "FOLD"
                        ]
                    )

                    job_key = (
                        side,
                        horizon,
                        target_variant,
                        feature_set,
                        fold,
                    )

                    if job_key in completed_jobs:
                        print(
                            f"[{job_counter:03d}/{total_jobs}] "
                            f"SKIP già completato | "
                            f"{side} {horizon}W "
                            f"{target_variant} "
                            f"{feature_set} F{fold}"
                        )
                        continue

                    split = split_audit[
                        (
                            split_audit[
                                "HORIZON_W"
                            ]
                            ==
                            horizon
                        )
                        &
                        (
                            split_audit[
                                "FOLD"
                            ]
                            ==
                            fold
                        )
                    ].iloc[
                        0
                    ]

                    train_end = pd.Timestamp(
                        split[
                            "TRAIN_END"
                        ]
                    )

                    test_start = pd.Timestamp(
                        split[
                            "TEST_START"
                        ]
                    )

                    test_end = pd.Timestamp(
                        split[
                            "TEST_END"
                        ]
                    )

                    z = z_target[
                        [
                            "Ticker",
                            "Date",
                            "ROW_ID",
                            target_col,
                        ]
                    ].copy()

                    z = z.rename(
                        columns={
                            target_col:
                            "Y_TRUE"
                        }
                    )

                    finite_target = np.isfinite(
                        z[
                            "Y_TRUE"
                        ].to_numpy(
                            dtype=float
                        )
                    )

                    train_mask = (
                        (
                            z[
                                "Date"
                            ]
                            <=
                            train_end
                        )
                        &
                        finite_target
                    )

                    test_mask = (
                        (
                            z[
                                "Date"
                            ]
                            >=
                            test_start
                        )
                        &
                        (
                            z[
                                "Date"
                            ]
                            <=
                            test_end
                        )
                        &
                        finite_target
                    )

                    train = z.loc[
                        train_mask
                    ].copy()

                    test = z.loc[
                        test_mask
                    ].copy()

                    if len(train) < MIN_TRAIN_ROWS:
                        raise RuntimeError(
                            f"Training troppo piccolo "
                            f"{side} {horizon}W "
                            f"{target_variant} "
                            f"{feature_set} F{fold}: "
                            f"{len(train)}"
                        )

                    if len(test) < MIN_TEST_ROWS:
                        raise RuntimeError(
                            f"Test troppo piccolo "
                            f"{side} {horizon}W "
                            f"{target_variant} "
                            f"{feature_set} F{fold}: "
                            f"{len(test)}"
                        )

                    train_row_ids = (
                        train[
                            "ROW_ID"
                        ]
                        .to_numpy(
                            dtype=int
                        )
                    )

                    test_row_ids = (
                        test[
                            "ROW_ID"
                        ]
                        .to_numpy(
                            dtype=int
                        )
                    )

                    y_train = (
                        train[
                            "Y_TRUE"
                        ]
                        .to_numpy(
                            dtype=np.float32
                        )
                    )

                    y_test = (
                        test[
                            "Y_TRUE"
                        ]
                        .to_numpy(
                            dtype=np.float32
                        )
                    )

                    (
                        base_medians,
                        extra_medians,
                    ) = calculate_medians(
                        train_row_ids,
                        feature_set,
                    )

                    X_train = build_matrix(
                        train_row_ids,
                        feature_set,
                        base_medians,
                        extra_medians,
                    )

                    X_test = build_matrix(
                        test_row_ids,
                        feature_set,
                        base_medians,
                        extra_medians,
                    )

                    print()
                    print(
                        f"[{job_counter:03d}/{total_jobs}] "
                        f"{side} {horizon}W | "
                        f"{target_variant:11s} | "
                        f"{feature_set:24s} | "
                        f"F{fold} | "
                        f"Train={len(train):5d} | "
                        f"Test={len(test):5d} | "
                        f"P={X_train.shape[1]}"
                    )

                    model = RandomForestRegressor(
                        **RF_PARAMS
                    )

                    model.fit(
                        X_train,
                        y_train,
                    )

                    y_pred = (
                        model.predict(
                            X_test
                        )
                        .astype(
                            np.float32
                        )
                    )

                    spearman = safe_spearman(
                        y_test,
                        y_pred,
                    )

                    pearson = safe_pearson(
                        y_test,
                        y_pred,
                    )

                    mae = float(
                        mean_absolute_error(
                            y_test,
                            y_pred,
                        )
                    )

                    rmse = float(
                        np.sqrt(
                            mean_squared_error(
                                y_test,
                                y_pred,
                            )
                        )
                    )

                    r2 = float(
                        r2_score(
                            y_test,
                            y_pred,
                        )
                    )

                    baseline_value = float(
                        np.nanmedian(
                            y_train
                        )
                    )

                    baseline = np.full(
                        len(y_test),
                        baseline_value,
                        dtype=float,
                    )

                    baseline_mae = float(
                        mean_absolute_error(
                            y_test,
                            baseline,
                        )
                    )

                    mae_improvement_pct = (
                        (
                            baseline_mae
                            -
                            mae
                        )
                        /
                        baseline_mae
                        *
                        100.0
                        if baseline_mae > 0
                        else np.nan
                    )

                    print(
                        f"    Spearman={spearman: .4f} | "
                        f"Pearson={pearson: .4f} | "
                        f"R2={r2: .4f} | "
                        f"MAE={mae: .4f} | "
                        f"ΔMAE={mae_improvement_pct: .2f}%"
                    )

                    result_row = {
                        "SIDE": side,
                        "HORIZON_W": horizon,
                        "TARGET_METRIC": "SPEED",
                        "TARGET_VARIANT": target_variant,
                        "FEATURE_SET": feature_set,
                        "MODEL": "RF_V40_21",
                        "FOLD": fold,
                        "TRAIN_N": len(train),
                        "TEST_N": len(test),
                        "N_FEATURES": int(
                            X_train.shape[
                                1
                            ]
                        ),
                        "TRAIN_END": train_end,
                        "TEST_START": test_start,
                        "TEST_END": test_end,
                        "SPEARMAN": spearman,
                        "PEARSON": pearson,
                        "R2": r2,
                        "MAE": mae,
                        "RMSE": rmse,
                        "BASELINE_MAE": baseline_mae,
                        "MAE_IMPROVEMENT_PCT": mae_improvement_pct,
                    }

                    append_row(
                        OUT_RESULTS,
                        result_row,
                    )

                    pred_df = pd.DataFrame(
                        {
                            "Ticker":
                                test[
                                    "Ticker"
                                ].to_numpy(),

                            "Date":
                                test[
                                    "Date"
                                ].to_numpy(),

                            "SIDE":
                                side,

                            "HORIZON_W":
                                horizon,

                            "TARGET_METRIC":
                                "SPEED",

                            "TARGET_VARIANT":
                                target_variant,

                            "FEATURE_SET":
                                feature_set,

                            "MODEL":
                                "RF_V40_21",

                            "FOLD":
                                fold,

                            "Y_TRUE":
                                y_test,

                            "Y_PRED":
                                y_pred,
                        }
                    )

                    append_predictions(
                        pred_df
                    )

                    completed_jobs.add(
                        job_key
                    )

                    del model
                    del X_train
                    del X_test
                    del y_train
                    del y_test
                    del y_pred
                    del baseline
                    del base_medians
                    del extra_medians
                    del train
                    del test
                    del z

                    gc.collect()


# ==================================================================================================
# FINAL RESULTS
# ==================================================================================================

banner(
    "AGGREGAZIONE RISULTATI V40.48"
)

results = pd.read_csv(
    OUT_RESULTS
)

expected_jobs = total_jobs

actual_jobs = len(
    results[
        [
            "SIDE",
            "HORIZON_W",
            "TARGET_VARIANT",
            "FEATURE_SET",
            "FOLD",
        ]
    ]
    .drop_duplicates()
)

print(
    f"Job attesi:     {expected_jobs}"
)

print(
    f"Job completati: {actual_jobs}"
)

if actual_jobs != expected_jobs:
    raise RuntimeError(
        "Numero job finale non completo."
    )


# ==================================================================================================
# SUMMARY
# ==================================================================================================

summary_rows = []

group_cols = [
    "SIDE",
    "HORIZON_W",
    "TARGET_VARIANT",
    "FEATURE_SET",
]

for keys, g in results.groupby(
    group_cols,
    sort=True,
):

    (
        side,
        horizon,
        target_variant,
        feature_set,
    ) = keys

    summary_rows.append(
        {
            "SIDE": side,
            "HORIZON_W": horizon,
            "TARGET_METRIC": "SPEED",
            "TARGET_VARIANT": target_variant,
            "FEATURE_SET": feature_set,
            "FOLDS": len(g),
            "N_TEST_TOTAL": int(
                g[
                    "TEST_N"
                ].sum()
            ),
            "MEAN_FOLD_SPEARMAN": float(
                g[
                    "SPEARMAN"
                ].mean()
            ),
            "MIN_FOLD_SPEARMAN": float(
                g[
                    "SPEARMAN"
                ].min()
            ),
            "MAX_FOLD_SPEARMAN": float(
                g[
                    "SPEARMAN"
                ].max()
            ),
            "POSITIVE_FOLDS": int(
                (
                    g[
                        "SPEARMAN"
                    ]
                    >
                    0
                ).sum()
            ),
            "MEAN_MAE": float(
                g[
                    "MAE"
                ].mean()
            ),
            "MEAN_BASELINE_MAE": float(
                g[
                    "BASELINE_MAE"
                ].mean()
            ),
            "MEAN_MAE_IMPROVEMENT_PCT": float(
                g[
                    "MAE_IMPROVEMENT_PCT"
                ].mean()
            ),
        }
    )

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

print()
print(
    summary.to_string(
        index=False
    )
)


# ==================================================================================================
# INCREMENTAL VALUE VS TECH_PHASE
# ==================================================================================================

banner(
    "VALORE INCREMENTALE VS TECH_PHASE"
)

incremental_rows = []

for (
    side,
    horizon,
    target_variant,
), g in summary.groupby(
    [
        "SIDE",
        "HORIZON_W",
        "TARGET_VARIANT",
    ]
):

    baseline = g[
        g[
            "FEATURE_SET"
        ]
        ==
        "TECH_PHASE"
    ]

    if len(baseline) != 1:
        raise RuntimeError(
            "Baseline TECH_PHASE non univoca."
        )

    baseline = baseline.iloc[
        0
    ]

    for row in g.itertuples(
        index=False
    ):

        incremental_rows.append(
            {
                "SIDE":
                    side,

                "HORIZON_W":
                    horizon,

                "TARGET_VARIANT":
                    target_variant,

                "FEATURE_SET":
                    row.FEATURE_SET,

                "MEAN_FOLD_SPEARMAN":
                    row.MEAN_FOLD_SPEARMAN,

                "DELTA_SPEARMAN_VS_TECH_PHASE":
                    (
                        row.MEAN_FOLD_SPEARMAN
                        -
                        baseline[
                            "MEAN_FOLD_SPEARMAN"
                        ]
                    ),

                "MEAN_MAE_IMPROVEMENT_PCT":
                    row.MEAN_MAE_IMPROVEMENT_PCT,

                "DELTA_MAE_IMPROVEMENT_VS_TECH_PHASE":
                    (
                        row.MEAN_MAE_IMPROVEMENT_PCT
                        -
                        baseline[
                            "MEAN_MAE_IMPROVEMENT_PCT"
                        ]
                    ),

                "POSITIVE_FOLDS":
                    row.POSITIVE_FOLDS,
            }
        )

incremental = pd.DataFrame(
    incremental_rows
)

incremental.to_csv(
    OUT_INCREMENTAL,
    index=False,
)

print(
    incremental.to_string(
        index=False
    )
)


# ==================================================================================================
# STRUCTURAL VERDICT
# ==================================================================================================

banner(
    "VERDETTO STRUTTURALE V40.48"
)

duplicate_jobs = int(
    results.duplicated(
        [
            "SIDE",
            "HORIZON_W",
            "TARGET_VARIANT",
            "FEATURE_SET",
            "FOLD",
        ]
    ).sum()
)

purge_ok = bool(
    split_audit[
        "PURGE_OK"
    ].all()
)

print(
    f"413 feature tecniche:             {len(base_features)}"
)

print(
    f"Common Ticker/Date:               {len(context):,}"
)

print(
    f"Ticker common:                    {context['Ticker'].nunique()}"
)

print(
    f"Job completati:                   {actual_jobs}/{expected_jobs}"
)

print(
    f"Duplicati job:                    {duplicate_jobs}"
)

print(
    f"Purging corretto:                 {purge_ok}"
)

print(
    "PHASE V40.39 modificata:          NO"
)

print(
    "REVERSAL V40.40 modificato:       NO"
)

print(
    "STRENGTH V40.44 modificata:       NO"
)

print(
    "Opportunity score 1-10 creato:    NO"
)

print(
    "Pesi manuali introdotti:          NO"
)

print(
    "RF modificato rispetto V40.37d:   NO"
)

if duplicate_jobs != 0:
    raise RuntimeError(
        "Duplicati nei job finali."
    )

if not purge_ok:
    raise RuntimeError(
        "Purging finale non valido."
    )


# ==================================================================================================
# METADATA
# ==================================================================================================

metadata = {
    "version":
        "V40.48",

    "purpose":
        "Opportunity Bridge OOS benchmark",

    "feature_source":
        str(
            FEATURE_FILE
        ),

    "feature_list_source":
        str(
            FEATURE_LIST_FILE
        ),

    "common_oos_source":
        str(
            COMMON_FILE
        ),

    "technical_features":
        EXPECTED_FEATURES,

    "feature_sets":
        FEATURE_SETS,

    "target_metric":
        "SPEED_PCT_PER_WEEK",

    "target_variants":
        TARGET_VARIANTS,

    "sides":
        SIDES,

    "horizons_weeks":
        HORIZONS,

    "rf_params":
        RF_PARAMS,

    "test_blocks":
        TEST_BLOCKS,

    "purge_rule":
        "train_end_position = test_start_position - horizon - 1",

    "train_only_imputation":
        True,

    "checkpoint_resume":
        True,

    "phase_v40_39_modified":
        False,

    "reversal_v40_40_modified":
        False,

    "strength_v40_44_modified":
        False,

    "opportunity_score_1_10":
        False,

    "manual_weights":
        False,

    "common_ticker_dates":
        int(
            len(context)
        ),

    "common_tickers":
        int(
            context[
                "Ticker"
            ].nunique()
        ),

    "jobs_expected":
        int(
            expected_jobs
        ),

    "jobs_completed":
        int(
            actual_jobs
        ),

    "structural_pass":
        bool(
            actual_jobs
            ==
            expected_jobs
            and
            duplicate_jobs
            ==
            0
            and
            purge_ok
        ),
}

with open(
    OUT_METADATA,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


banner(
    "V40.48 COMPLETATO"
)

print(
    f"Results:      {OUT_RESULTS}"
)

print(
    f"Summary:      {OUT_SUMMARY}"
)

print(
    f"Incremental:  {OUT_INCREMENTAL}"
)

print(
    f"Predictions:  {OUT_PREDICTIONS}"
)

print(
    f"Split audit:  {OUT_SPLITS}"
)

print(
    f"Metadata:     {OUT_METADATA}"
)

print()
print(
    "V40.48 è un benchmark diagnostico."
)

print(
    "NON congelare ancora Opportunity e NON costruire ancora lo score 1-10."
)