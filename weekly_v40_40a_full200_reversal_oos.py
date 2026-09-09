# ==================================================================================================
# MARKET SENTINEL - V40.40a
# FULL200 WEEKLY REVERSAL MODEL
# MEMORY-SAFE
#
# CORRECTED PURGED WALK-FORWARD
# + FORWARD-ONLY CALIBRATION
# ==================================================================================================
#
# SCOPO
# -----
#
# Primo vero benchmark/retraining WEEKLY sui 200 titoli.
#
# METODOLOGIA:
#
# - FULL200 V40.39
# - 413 feature ESATTE
# - TRUE REVERSAL 1W / 2W / 3W
# - date settimanali realmente uniche
# - purging di settimane INTERE
# - 3 fold temporali
# - nessun random split
# - imputazione train-only
# - RandomForest V40.24b invariato
# - calibrazione forward-only
#
# MEMORY SAFE:
#
# - metadata caricati separatamente
# - feature lette dal CSV a chunk
# - matrice feature salvata come memmap float32 su disco
# - un solo esperimento ML in RAM alla volta
# - predictions salvate progressivamente
# - feature importance salvate progressivamente
# - n_jobs=1 per evitare picchi RAM
#
# IMPORTANTE:
#
# NON cambia il modello matematico.
# NON cambia nessuna feature.
# NON cambia PHASE.
# NON cambia il purging.
#
# ==================================================================================================

from __future__ import annotations

import gc
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


warnings.filterwarnings("ignore")


# ==================================================================================================
# VERSION
# ==================================================================================================

VERSION = "V40.40a-memorysafe"

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================================================================
# INPUT
# ==================================================================================================

INPUT_FILE = (
    DATA_DIR
    /
    "v40_39_full200_features.csv"
)

FEATURE_LIST_FILE = (
    DATA_DIR
    /
    "v40_39_feature_list.csv"
)


# ==================================================================================================
# MEMORY-MAPPED FEATURE MATRIX
# ==================================================================================================

MEMMAP_FILE = (
    DATA_DIR
    /
    "v40_40a_features_float32.dat"
)


# ==================================================================================================
# OUTPUT
# ==================================================================================================

OUT_TARGETS = (
    DATA_DIR
    /
    "v40_40a_reversal_targets.csv"
)

OUT_SPLITS = (
    DATA_DIR
    /
    "v40_40a_temporal_splits.csv"
)

OUT_RESULTS = (
    DATA_DIR
    /
    "v40_40a_oos_results.csv"
)

OUT_PREDICTIONS = (
    DATA_DIR
    /
    "v40_40a_oos_predictions.csv"
)

OUT_CALIBRATION = (
    DATA_DIR
    /
    "v40_40a_calibration_results.csv"
)

OUT_SUMMARY = (
    DATA_DIR
    /
    "v40_40a_summary.csv"
)

OUT_IMPORTANCE = (
    DATA_DIR
    /
    "v40_40a_feature_importance.csv"
)

OUT_TOP_FEATURES = (
    DATA_DIR
    /
    "v40_40a_top_features.csv"
)

OUT_METADATA = (
    DATA_DIR
    /
    "v40_40a_metadata.json"
)


# ==================================================================================================
# CONSTANTS
# ==================================================================================================

EXPECTED_TICKERS = 200
EXPECTED_FEATURES = 413
EXPECTED_ROWS = 141886

HORIZONS = [
    1,
    2,
    3,
]

TEST_START_FRACTIONS = [
    0.55,
    0.70,
    0.85,
]

TEST_END_FRACTIONS = [
    0.70,
    0.85,
    1.00,
]

MIN_TRAIN_ROWS = 400
MIN_TEST_ROWS = 30

RANDOM_STATE = 42

CSV_CHUNK_SIZE = 4000

# --------------------------------------------------------------------------------------------------
# IMPORTANT:
# stesso Random Forest.
# Cambia SOLO il parallelismo runtime.
# --------------------------------------------------------------------------------------------------

N_JOBS = 1


# ==================================================================================================
# RANDOM FOREST
# ==================================================================================================

def build_random_forest():

    return RandomForestClassifier(
        n_estimators=350,
        max_features="sqrt",
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


# ==================================================================================================
# UTILITY
# ==================================================================================================

def section(title):

    print()
    print("=" * 145)
    print(title)
    print("=" * 145)
    print()


def collect():

    gc.collect()


def remove_if_exists(path):

    if path.exists():

        path.unlink()


# ==================================================================================================
# PHASE
# ==================================================================================================

def phase_category(value):

    p = (
        str(value)
        .upper()
        .strip()
        .replace(
            " ",
            "_",
        )
    )

    if p == "BUY":

        return "BUY"

    if p in {
        "TREND_RIALZISTA",
        "TREND_UP",
    }:

        return "TREND_RIALZISTA"

    if p in {
        "DEBOLEZZA_RIALZISTA",
        "WEAK_UP",
    }:

        return "DEBOLEZZA_RIALZISTA"

    if p == "SELL":

        return "SELL"

    if p in {
        "TREND_RIBASSISTA",
        "TREND_DOWN",
    }:

        return "TREND_RIBASSISTA"

    if p in {
        "DEBOLEZZA_RIBASSISTA",
        "WEAK_DOWN",
    }:

        return "DEBOLEZZA_RIBASSISTA"

    return "INDECISIONE"


def phase_family(value):

    p = phase_category(
        value
    )

    if p in {
        "BUY",
        "TREND_RIALZISTA",
        "DEBOLEZZA_RIALZISTA",
    }:

        return "RIALZISTA"

    if p in {
        "SELL",
        "TREND_RIBASSISTA",
        "DEBOLEZZA_RIBASSISTA",
    }:

        return "RIBASSISTA"

    return "INDECISIONE"


# ==================================================================================================
# METRICS
# ==================================================================================================

def safe_auc(
    y_true,
    y_prob,
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    valid = (
        np.isfinite(y_true)
        &
        np.isfinite(y_prob)
    )

    y_true = y_true[
        valid
    ]

    y_prob = y_prob[
        valid
    ]

    if (
        len(y_true) == 0
        or
        len(np.unique(y_true)) < 2
    ):

        return np.nan

    try:

        return roc_auc_score(
            y_true,
            y_prob,
        )

    except Exception:

        return np.nan


def safe_brier(
    y_true,
    y_prob,
):

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    valid = (
        np.isfinite(y_true)
        &
        np.isfinite(y_prob)
    )

    if valid.sum() == 0:

        return np.nan

    return brier_score_loss(
        y_true[
            valid
        ],
        y_prob[
            valid
        ],
    )


def expected_calibration_error(
    y_true,
    y_prob,
    bins=10,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=float,
    )

    valid = (
        np.isfinite(y_true)
        &
        np.isfinite(y_prob)
    )

    y_true = y_true[
        valid
    ]

    y_prob = y_prob[
        valid
    ]

    if len(y_true) == 0:

        return np.nan

    y_prob = np.clip(
        y_prob,
        0.0,
        1.0,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for i in range(bins):

        if i == bins - 1:

            mask = (
                (y_prob >= edges[i])
                &
                (y_prob <= edges[i + 1])
            )

        else:

            mask = (
                (y_prob >= edges[i])
                &
                (y_prob < edges[i + 1])
            )

        n = int(
            mask.sum()
        )

        if n == 0:

            continue

        predicted = float(
            y_prob[
                mask
            ].mean()
        )

        actual = float(
            y_true[
                mask
            ].mean()
        )

        ece += (
            n
            /
            len(y_true)
        ) * abs(
            predicted
            -
            actual
        )

    return float(
        ece
    )


def classification_metrics(
    y_true,
    y_prob,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=float,
    )

    pred = (
        y_prob >= 0.5
    ).astype(
        int
    )

    return {
        "ROC_AUC":
            safe_auc(
                y_true,
                y_prob,
            ),

        "BRIER":
            safe_brier(
                y_true,
                y_prob,
            ),

        "ECE":
            expected_calibration_error(
                y_true,
                y_prob,
            ),

        "BAL_ACC":
            balanced_accuracy_score(
                y_true,
                pred,
            ),

        "PRECISION":
            precision_score(
                y_true,
                pred,
                zero_division=0,
            ),

        "RECALL":
            recall_score(
                y_true,
                pred,
                zero_division=0,
            ),

        "BASE_RATE":
            float(
                y_true.mean()
            ),
    }


# ==================================================================================================
# CALIBRATORS
# ==================================================================================================

class PlattCalibrator:

    def __init__(self):

        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )

        self.constant = None

    def fit(
        self,
        raw_prob,
        y,
    ):

        raw_prob = np.asarray(
            raw_prob,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=int,
        )

        if len(
            np.unique(y)
        ) < 2:

            self.constant = float(
                y.mean()
            )

            return self

        self.model.fit(
            raw_prob.reshape(
                -1,
                1,
            ),
            y,
        )

        return self

    def predict(
        self,
        raw_prob,
    ):

        raw_prob = np.asarray(
            raw_prob,
            dtype=float,
        )

        if self.constant is not None:

            return np.full(
                len(raw_prob),
                self.constant,
            )

        return (
            self.model
            .predict_proba(
                raw_prob.reshape(
                    -1,
                    1,
                )
            )[
                :,
                1
            ]
        )


class IsotonicCalibrator:

    def __init__(self):

        self.model = IsotonicRegression(
            out_of_bounds="clip"
        )

        self.constant = None

    def fit(
        self,
        raw_prob,
        y,
    ):

        raw_prob = np.asarray(
            raw_prob,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=int,
        )

        if len(
            np.unique(y)
        ) < 2:

            self.constant = float(
                y.mean()
            )

            return self

        self.model.fit(
            raw_prob,
            y,
        )

        return self

    def predict(
        self,
        raw_prob,
    ):

        raw_prob = np.asarray(
            raw_prob,
            dtype=float,
        )

        if self.constant is not None:

            return np.full(
                len(raw_prob),
                self.constant,
            )

        return self.model.predict(
            raw_prob
        )


# ==================================================================================================
# LOAD FEATURE LIST
# ==================================================================================================

def load_feature_list():

    section(
        "RECUPERO 413 FEATURE"
    )

    feature_df = pd.read_csv(
        FEATURE_LIST_FILE
    )

    if "Feature" not in feature_df.columns:

        raise RuntimeError(
            "Colonna Feature mancante."
        )

    feature_cols = (
        feature_df[
            "Feature"
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"Feature ML: "
        f"{len(feature_cols)}"
    )

    if len(
        feature_cols
    ) != EXPECTED_FEATURES:

        raise RuntimeError(
            f"Attese {EXPECTED_FEATURES} feature, "
            f"trovate {len(feature_cols)}."
        )

    if len(
        set(feature_cols)
    ) != EXPECTED_FEATURES:

        raise RuntimeError(
            "Feature duplicate nella feature list."
        )

    return feature_cols


# ==================================================================================================
# LOAD METADATA ONLY
# ==================================================================================================

def load_metadata():

    section(
        "CARICAMENTO METADATA FULL200"
    )

    meta = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "Ticker",
            "Date",
            "PHASE",
        ],
        low_memory=False,
    )

    meta[
        "Ticker"
    ] = (
        meta[
            "Ticker"
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    meta[
        "Date"
    ] = (
        pd.to_datetime(
            meta[
                "Date"
            ],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(None)
    )

    if meta[
        [
            "Ticker",
            "Date",
            "PHASE",
        ]
    ].isna().any().any():

        raise RuntimeError(
            "Ticker/Date/PHASE con valori mancanti."
        )

    print(
        f"Righe:      {len(meta)}"
    )

    print(
        f"Ticker:     "
        f"{meta['Ticker'].nunique()}"
    )

    print(
        f"Periodo:    "
        f"{meta['Date'].min().date()} "
        f"-> "
        f"{meta['Date'].max().date()}"
    )

    print(
        f"Duplicati:  "
        f"{meta.duplicated(['Ticker', 'Date']).sum()}"
    )

    if len(meta) != EXPECTED_ROWS:

        raise RuntimeError(
            f"Attese {EXPECTED_ROWS} righe, "
            f"trovate {len(meta)}."
        )

    if (
        meta[
            "Ticker"
        ].nunique()
        !=
        EXPECTED_TICKERS
    ):

        raise RuntimeError(
            "Ticker != 200."
        )

    if (
        meta
        .duplicated(
            [
                "Ticker",
                "Date",
            ]
        )
        .sum()
        !=
        0
    ):

        raise RuntimeError(
            "Duplicati Ticker+Date."
        )

    return meta


# ==================================================================================================
# TARGETS
# ==================================================================================================

def build_targets(meta):

    section(
        "COSTRUZIONE TRUE REVERSAL TARGET"
    )

    # ----------------------------------------------------------------------------------------------
    # IMPORTANTE:
    # Il CSV V40.39 è ordinato Ticker/Date.
    # Comunque costruiamo l'ordine ticker-date esplicitamente
    # SENZA modificare l'ordine fisico delle righe.
    #
    # Il groupby shift lavora correttamente sulle righe di ogni ticker.
    # ----------------------------------------------------------------------------------------------

    meta[
        "PHASE_CATEGORY"
    ] = meta[
        "PHASE"
    ].map(
        phase_category
    )

    meta[
        "PHASE_FAMILY"
    ] = meta[
        "PHASE"
    ].map(
        phase_family
    )

    # Controllo ordinamento interno per ticker

    order_check = (
        meta
        .groupby(
            "Ticker"
        )[
            "Date"
        ]
        .apply(
            lambda x:
                x.is_monotonic_increasing
        )
    )

    if not bool(
        order_check.all()
    ):

        raise RuntimeError(
            "Le date non sono ordinate internamente per ticker."
        )

    target_names = []

    for h in HORIZONS:

        future_family = (
            meta
            .groupby(
                "Ticker",
                sort=False,
            )[
                "PHASE_FAMILY"
            ]
            .shift(
                -h
            )
        )

        bear_col = (
            f"TARGET_BEAR_TO_BULL_{h}W"
        )

        bull_col = (
            f"TARGET_BULL_TO_BEAR_{h}W"
        )

        bear_target = np.full(
            len(meta),
            np.nan,
            dtype=np.float32,
        )

        bull_target = np.full(
            len(meta),
            np.nan,
            dtype=np.float32,
        )

        bear_mask = (
            (
                meta[
                    "PHASE_FAMILY"
                ]
                ==
                "RIBASSISTA"
            )
            &
            future_family.notna()
        ).to_numpy()

        bull_mask = (
            (
                meta[
                    "PHASE_FAMILY"
                ]
                ==
                "RIALZISTA"
            )
            &
            future_family.notna()
        ).to_numpy()

        future_arr = (
            future_family
            .astype(
                object
            )
            .to_numpy()
        )

        bear_target[
            bear_mask
        ] = (
            future_arr[
                bear_mask
            ]
            ==
            "RIALZISTA"
        ).astype(
            np.float32
        )

        bull_target[
            bull_mask
        ] = (
            future_arr[
                bull_mask
            ]
            ==
            "RIBASSISTA"
        ).astype(
            np.float32
        )

        meta[
            bear_col
        ] = bear_target

        meta[
            bull_col
        ] = bull_target

        target_names.extend(
            [
                bear_col,
                bull_col,
            ]
        )

    target_output_cols = [
        "Ticker",
        "Date",
        "PHASE",
        "PHASE_CATEGORY",
        "PHASE_FAMILY",
    ] + target_names

    meta[
        target_output_cols
    ].to_csv(
        OUT_TARGETS,
        index=False,
    )

    for col in target_names:

        valid = meta[
            col
        ].dropna()

        print(
            f"{col:<30} "
            f"N={len(valid):6d} "
            f"Positive={int(valid.sum()):6d} "
            f"Rate={valid.mean():.4f}"
        )

    return meta


# ==================================================================================================
# BUILD FEATURE MEMMAP
# ==================================================================================================

def build_feature_memmap(
    feature_cols,
    n_rows,
):

    section(
        "COSTRUZIONE MATRICE FEATURE MEMORY-MAPPED"
    )

    print(
        "Le 413 feature NON vengono caricate tutte "
        "contemporaneamente in RAM."
    )

    print(
        f"Chunk CSV: "
        f"{CSV_CHUNK_SIZE} righe"
    )

    print(
        f"Matrice: "
        f"{n_rows} x {len(feature_cols)} float32"
    )

    estimated_mb = (
        n_rows
        *
        len(feature_cols)
        *
        4
        /
        1024
        /
        1024
    )

    print(
        f"Dimensione su disco prevista: "
        f"{estimated_mb:.1f} MB"
    )

    remove_if_exists(
        MEMMAP_FILE
    )

    mm = np.memmap(
        MEMMAP_FILE,
        dtype="float32",
        mode="w+",
        shape=(
            n_rows,
            len(feature_cols),
        ),
    )

    dtype_map = {
        col:
            "float32"
        for col in feature_cols
    }

    row_start = 0

    chunk_number = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=feature_cols,
        dtype=dtype_map,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=True,
    ):

        chunk_number += 1

        n = len(
            chunk
        )

        row_end = (
            row_start
            +
            n
        )

        if row_end > n_rows:

            raise RuntimeError(
                "Il CSV contiene più righe del previsto."
            )

        arr = chunk.to_numpy(
            dtype=np.float32,
            copy=False,
        )

        mm[
            row_start:
            row_end,
            :
        ] = arr

        mm.flush()

        print(
            f"Chunk {chunk_number:02d}: "
            f"righe {row_start + 1:6d} "
            f"-> {row_end:6d} "
            f"/ {n_rows}"
        )

        row_start = row_end

        del arr
        del chunk

        collect()

    if row_start != n_rows:

        raise RuntimeError(
            f"Righe memmap: {row_start}, "
            f"attese {n_rows}."
        )

    mm.flush()

    del mm

    collect()

    print()
    print(
        "OK - matrice feature scritta su disco."
    )


# ==================================================================================================
# OPEN MEMMAP
# ==================================================================================================

def open_feature_memmap(
    n_rows,
    n_features,
):

    return np.memmap(
        MEMMAP_FILE,
        dtype="float32",
        mode="r",
        shape=(
            n_rows,
            n_features,
        ),
    )


# ==================================================================================================
# IMPUTATION
# ==================================================================================================

def fit_and_apply_imputer(
    X_train,
    X_test,
):

    # ----------------------------------------------------------------------------------------------
    # Mediana esclusivamente dal TRAIN.
    # ----------------------------------------------------------------------------------------------

    medians = np.nanmedian(
        X_train,
        axis=0,
    ).astype(
        np.float32
    )

    medians[
        ~np.isfinite(
            medians
        )
    ] = 0.0

    # TRAIN

    invalid_train = (
        ~np.isfinite(
            X_train
        )
    )

    if invalid_train.any():

        rows, cols = np.where(
            invalid_train
        )

        X_train[
            rows,
            cols
        ] = medians[
            cols
        ]

        del rows
        del cols

    del invalid_train

    # TEST

    invalid_test = (
        ~np.isfinite(
            X_test
        )
    )

    if invalid_test.any():

        rows, cols = np.where(
            invalid_test
        )

        X_test[
            rows,
            cols
        ] = medians[
            cols
        ]

        del rows
        del cols

    del invalid_test

    return (
        X_train,
        X_test,
        medians,
    )


# ==================================================================================================
# SPLITS
# ==================================================================================================

def build_temporal_splits(meta):

    section(
        "TEMPORAL SPLITS - UNIQUE WEEKS"
    )

    unique_dates = np.array(
        sorted(
            meta[
                "Date"
            ]
            .dropna()
            .unique()
        ),
        dtype="datetime64[ns]",
    )

    n_dates = len(
        unique_dates
    )

    print(
        f"Settimane uniche: "
        f"{n_dates}"
    )

    if n_dates != 767:

        print(
            "ATTENZIONE: V40.37a aveva trovato 767 settimane."
        )

    split_rows = []

    for fold_idx in range(3):

        start_pos = int(
            n_dates
            *
            TEST_START_FRACTIONS[
                fold_idx
            ]
        )

        if fold_idx < 2:

            end_pos = (
                int(
                    n_dates
                    *
                    TEST_END_FRACTIONS[
                        fold_idx
                    ]
                )
                -
                1
            )

        else:

            end_pos = (
                n_dates
                -
                1
            )

        split_rows.append(
            {
                "FOLD":
                    fold_idx + 1,

                "TEST_START_POSITION":
                    start_pos,

                "TEST_END_POSITION":
                    end_pos,

                "TEST_START":
                    pd.Timestamp(
                        unique_dates[
                            start_pos
                        ]
                    ),

                "TEST_END":
                    pd.Timestamp(
                        unique_dates[
                            end_pos
                        ]
                    ),

                "TEST_WEEKS":
                    int(
                        end_pos
                        -
                        start_pos
                        +
                        1
                    ),
            }
        )

    splits = pd.DataFrame(
        split_rows
    )

    splits.to_csv(
        OUT_SPLITS,
        index=False,
    )

    print(
        splits.to_string(
            index=False
        )
    )

    return (
        unique_dates,
        split_rows,
    )


# ==================================================================================================
# APPEND CSV
# ==================================================================================================

def append_csv(
    df,
    path,
):

    exists = path.exists()

    df.to_csv(
        path,
        mode=(
            "a"
            if exists
            else
            "w"
        ),
        header=(
            not exists
        ),
        index=False,
    )


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    section(
        "MARKET SENTINEL - V40.40a\n"
        "FULL200 WEEKLY REVERSAL OOS RETRAINING\n"
        "MEMORY-SAFE"
    )

    # ==============================================================================================
    # INPUT
    # ==============================================================================================

    for path in [
        INPUT_FILE,
        FEATURE_LIST_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File mancante: "
                f"{path}"
            )

    # ==============================================================================================
    # CLEAN OLD OUTPUT
    # ==============================================================================================

    for path in [
        OUT_TARGETS,
        OUT_SPLITS,
        OUT_RESULTS,
        OUT_PREDICTIONS,
        OUT_CALIBRATION,
        OUT_SUMMARY,
        OUT_IMPORTANCE,
        OUT_TOP_FEATURES,
        OUT_METADATA,
    ]:

        remove_if_exists(
            path
        )

    # ==============================================================================================
    # FEATURES
    # ==============================================================================================

    feature_cols = (
        load_feature_list()
    )

    # ==============================================================================================
    # META
    # ==============================================================================================

    meta = load_metadata()

    n_rows = len(
        meta
    )

    # ==============================================================================================
    # TARGET
    # ==============================================================================================

    meta = build_targets(
        meta
    )

    # ==============================================================================================
    # SPLITS
    # ==============================================================================================

    unique_dates, split_rows = (
        build_temporal_splits(
            meta
        )
    )

    # ==============================================================================================
    # MEMMAP
    # ==============================================================================================

    build_feature_memmap(
        feature_cols,
        n_rows,
    )

    feature_matrix = (
        open_feature_memmap(
            n_rows,
            len(
                feature_cols
            ),
        )
    )

    # ==============================================================================================
    # TARGET SPECS
    # ==============================================================================================

    target_specs = []

    for h in HORIZONS:

        target_specs.append(
            {
                "TARGET":
                    f"BEAR_TO_BULL_{h}W",

                "TARGET_COL":
                    f"TARGET_BEAR_TO_BULL_{h}W",

                "CURRENT_FAMILY":
                    "RIBASSISTA",

                "HORIZON_W":
                    h,
            }
        )

    for h in HORIZONS:

        target_specs.append(
            {
                "TARGET":
                    f"BULL_TO_BEAR_{h}W",

                "TARGET_COL":
                    f"TARGET_BULL_TO_BEAR_{h}W",

                "CURRENT_FAMILY":
                    "RIALZISTA",

                "HORIZON_W":
                    h,
            }
        )

    # ==============================================================================================
    # OOS TRAINING
    # ==============================================================================================

    section(
        "VERO MACHINE LEARNING FULL200"
    )

    print(
        "Random Forest: 350 alberi"
    )

    print(
        "413 feature"
    )

    print(
        "6 target x 3 fold = 18 training"
    )

    print(
        "n_jobs=1 per proteggere la RAM"
    )

    result_rows = []

    experiment_counter = 0

    for spec in target_specs:

        target_name = (
            spec[
                "TARGET"
            ]
        )

        target_col = (
            spec[
                "TARGET_COL"
            ]
        )

        family = (
            spec[
                "CURRENT_FAMILY"
            ]
        )

        horizon = int(
            spec[
                "HORIZON_W"
            ]
        )

        section(
            target_name
        )

        for fold_info in split_rows:

            fold = int(
                fold_info[
                    "FOLD"
                ]
            )

            test_start = pd.Timestamp(
                fold_info[
                    "TEST_START"
                ]
            )

            test_end = pd.Timestamp(
                fold_info[
                    "TEST_END"
                ]
            )

            test_start64 = np.datetime64(
                test_start,
                "ns",
            )

            dates_before_test = (
                unique_dates[
                    unique_dates
                    <
                    test_start64
                ]
            )

            if len(
                dates_before_test
            ) <= horizon:

                raise RuntimeError(
                    "Date insufficienti per purging."
                )

            # --------------------------------------------------------------------------------------
            # WHOLE-WEEK PURGING
            # --------------------------------------------------------------------------------------

            purged_train_dates = (
                dates_before_test[
                    :-horizon
                ]
            )

            train_end = pd.Timestamp(
                purged_train_dates[
                    -1
                ]
            )

            train_mask = (
                (
                    meta[
                        "Date"
                    ]
                    <=
                    train_end
                )
                &
                (
                    meta[
                        "PHASE_FAMILY"
                    ]
                    ==
                    family
                )
                &
                meta[
                    target_col
                ].notna()
            ).to_numpy()

            test_mask = (
                (
                    meta[
                        "Date"
                    ]
                    >=
                    test_start
                )
                &
                (
                    meta[
                        "Date"
                    ]
                    <=
                    test_end
                )
                &
                (
                    meta[
                        "PHASE_FAMILY"
                    ]
                    ==
                    family
                )
                &
                meta[
                    target_col
                ].notna()
            ).to_numpy()

            train_idx = np.flatnonzero(
                train_mask
            )

            test_idx = np.flatnonzero(
                test_mask
            )

            del train_mask
            del test_mask

            train_n = len(
                train_idx
            )

            test_n = len(
                test_idx
            )

            if train_n < MIN_TRAIN_ROWS:

                raise RuntimeError(
                    f"{target_name} Fold {fold}: "
                    f"train={train_n}"
                )

            if test_n < MIN_TEST_ROWS:

                raise RuntimeError(
                    f"{target_name} Fold {fold}: "
                    f"test={test_n}"
                )

            # --------------------------------------------------------------------------------------
            # LABELS
            # --------------------------------------------------------------------------------------

            target_values = (
                meta[
                    target_col
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

            y_train = (
                target_values[
                    train_idx
                ]
                .astype(
                    np.int8
                )
            )

            y_test = (
                target_values[
                    test_idx
                ]
                .astype(
                    np.int8
                )
            )

            del target_values

            if len(
                np.unique(
                    y_train
                )
            ) < 2:

                raise RuntimeError(
                    f"{target_name} Fold {fold}: "
                    "train con una sola classe."
                )

            if len(
                np.unique(
                    y_test
                )
            ) < 2:

                raise RuntimeError(
                    f"{target_name} Fold {fold}: "
                    "test con una sola classe."
                )

            # --------------------------------------------------------------------------------------
            # COPY ONLY CURRENT TRAIN/TEST INTO RAM
            # --------------------------------------------------------------------------------------

            print(
                f"Preparazione Fold {fold}: "
                f"Train={train_n} "
                f"Test={test_n}"
            )

            X_train = np.asarray(
                feature_matrix[
                    train_idx,
                    :
                ],
                dtype=np.float32,
            )

            X_test = np.asarray(
                feature_matrix[
                    test_idx,
                    :
                ],
                dtype=np.float32,
            )

            X_train, X_test, medians = (
                fit_and_apply_imputer(
                    X_train,
                    X_test,
                )
            )

            # --------------------------------------------------------------------------------------
            # RANDOM FOREST
            # --------------------------------------------------------------------------------------

            model = build_random_forest()

            model.fit(
                X_train,
                y_train,
            )

            raw_prob = (
                model
                .predict_proba(
                    X_test
                )[
                    :,
                    1
                ]
            )

            metrics = (
                classification_metrics(
                    y_test,
                    raw_prob,
                )
            )

            train_base_rate = float(
                y_train.mean()
            )

            baseline_prob = np.full(
                test_n,
                train_base_rate,
                dtype=np.float32,
            )

            baseline_brier = (
                safe_brier(
                    y_test,
                    baseline_prob,
                )
            )

            experiment_counter += 1

            print(
                f"[{experiment_counter:02d}/18] "
                f"{target_name:<22} "
                f"Fold {fold} "
                f"Train={train_n:6d} "
                f"Test={test_n:6d} "
                f"AUC={metrics['ROC_AUC']:.4f} "
                f"Brier={metrics['BRIER']:.4f}"
            )

            # --------------------------------------------------------------------------------------
            # RESULT
            # --------------------------------------------------------------------------------------

            result_rows.append(
                {
                    "TARGET":
                        target_name,

                    "HORIZON_W":
                        horizon,

                    "FOLD":
                        fold,

                    "TRAIN_END":
                        train_end,

                    "TEST_START":
                        test_start,

                    "TEST_END":
                        test_end,

                    "TRAIN_N":
                        train_n,

                    "TEST_N":
                        test_n,

                    "TRAIN_BASE_RATE":
                        train_base_rate,

                    "TEST_BASE_RATE":
                        float(
                            y_test.mean()
                        ),

                    "ROC_AUC":
                        metrics[
                            "ROC_AUC"
                        ],

                    "BRIER":
                        metrics[
                            "BRIER"
                        ],

                    "ECE":
                        metrics[
                            "ECE"
                        ],

                    "BAL_ACC":
                        metrics[
                            "BAL_ACC"
                        ],

                    "PRECISION":
                        metrics[
                            "PRECISION"
                        ],

                    "RECALL":
                        metrics[
                            "RECALL"
                        ],

                    "BASELINE_BRIER":
                        baseline_brier,

                    "BRIER_IMPROVEMENT":
                        (
                            baseline_brier
                            -
                            metrics[
                                "BRIER"
                            ]
                        ),

                    "PURGED_WEEKS":
                        horizon,
                }
            )

            # --------------------------------------------------------------------------------------
            # SAVE PREDICTIONS IMMEDIATELY
            # --------------------------------------------------------------------------------------

            pred_df = pd.DataFrame(
                {
                    "Ticker":
                        meta.iloc[
                            test_idx
                        ][
                            "Ticker"
                        ].to_numpy(),

                    "Date":
                        meta.iloc[
                            test_idx
                        ][
                            "Date"
                        ].to_numpy(),

                    "PHASE":
                        meta.iloc[
                            test_idx
                        ][
                            "PHASE"
                        ].to_numpy(),

                    "PHASE_FAMILY":
                        meta.iloc[
                            test_idx
                        ][
                            "PHASE_FAMILY"
                        ].to_numpy(),

                    "TARGET":
                        target_name,

                    "HORIZON_W":
                        horizon,

                    "FOLD":
                        fold,

                    "Y_TRUE":
                        y_test.astype(
                            int
                        ),

                    "RAW_PROB":
                        raw_prob,
                }
            )

            append_csv(
                pred_df,
                OUT_PREDICTIONS,
            )

            # --------------------------------------------------------------------------------------
            # SAVE IMPORTANCE IMMEDIATELY
            # --------------------------------------------------------------------------------------

            imp_df = pd.DataFrame(
                {
                    "TARGET":
                        target_name,

                    "HORIZON_W":
                        horizon,

                    "FOLD":
                        fold,

                    "Feature":
                        feature_cols,

                    "Importance":
                        model.feature_importances_,
                }
            )

            append_csv(
                imp_df,
                OUT_IMPORTANCE,
            )

            # --------------------------------------------------------------------------------------
            # CLEAN RAM AGGRESSIVELY
            # --------------------------------------------------------------------------------------

            del X_train
            del X_test
            del medians
            del model
            del raw_prob
            del baseline_prob
            del pred_df
            del imp_df
            del train_idx
            del test_idx
            del y_train
            del y_test

            collect()

    # ==============================================================================================
    # RAW RESULT FILE
    # ==============================================================================================

    results = pd.DataFrame(
        result_rows
    )

    results.to_csv(
        OUT_RESULTS,
        index=False,
    )

    del result_rows

    collect()

    # ==============================================================================================
    # LOAD PREDICTIONS
    #
    # Molto più piccolo della matrice delle 413 feature.
    # ==============================================================================================

    section(
        "CARICAMENTO PREDICTION OOS"
    )

    predictions = pd.read_csv(
        OUT_PREDICTIONS,
        low_memory=False,
    )

    predictions[
        "Date"
    ] = pd.to_datetime(
        predictions[
            "Date"
        ],
        errors="coerce",
    )

    print(
        f"Prediction OOS: "
        f"{len(predictions)}"
    )

    # ==============================================================================================
    # FORWARD-ONLY CALIBRATION
    # ==============================================================================================

    section(
        "FORWARD-ONLY CALIBRATION"
    )

    predictions[
        "PLATT_PROB"
    ] = np.nan

    predictions[
        "ISOTONIC_PROB"
    ] = np.nan

    calibration_rows = []

    for target_name in [
        f"BEAR_TO_BULL_{h}W"
        for h in HORIZONS
    ] + [
        f"BULL_TO_BEAR_{h}W"
        for h in HORIZONS
    ]:

        target_pred = predictions[
            predictions[
                "TARGET"
            ]
            ==
            target_name
        ]

        print()
        print(
            target_name
        )

        for fold in [
            1,
            2,
            3,
        ]:

            current_idx = target_pred[
                target_pred[
                    "FOLD"
                ]
                ==
                fold
            ].index

            if fold == 1:

                print(
                    "  Fold 1: RAW only "
                    "(nessun OOS precedente)"
                )

                continue

            history = target_pred[
                target_pred[
                    "FOLD"
                ]
                <
                fold
            ]

            current = target_pred[
                target_pred[
                    "FOLD"
                ]
                ==
                fold
            ]

            y_history = (
                history[
                    "Y_TRUE"
                ]
                .astype(
                    int
                )
                .to_numpy()
            )

            p_history = (
                history[
                    "RAW_PROB"
                ]
                .astype(
                    float
                )
                .to_numpy()
            )

            y_current = (
                current[
                    "Y_TRUE"
                ]
                .astype(
                    int
                )
                .to_numpy()
            )

            p_current = (
                current[
                    "RAW_PROB"
                ]
                .astype(
                    float
                )
                .to_numpy()
            )

            raw_metrics = (
                classification_metrics(
                    y_current,
                    p_current,
                )
            )

            # --------------------------------------------------------------------------------------
            # PLATT
            # --------------------------------------------------------------------------------------

            platt = (
                PlattCalibrator()
                .fit(
                    p_history,
                    y_history,
                )
            )

            platt_prob = (
                platt.predict(
                    p_current
                )
            )

            predictions.loc[
                current_idx,
                "PLATT_PROB",
            ] = platt_prob

            platt_metrics = (
                classification_metrics(
                    y_current,
                    platt_prob,
                )
            )

            # --------------------------------------------------------------------------------------
            # ISOTONIC
            # --------------------------------------------------------------------------------------

            iso = (
                IsotonicCalibrator()
                .fit(
                    p_history,
                    y_history,
                )
            )

            iso_prob = (
                iso.predict(
                    p_current
                )
            )

            predictions.loc[
                current_idx,
                "ISOTONIC_PROB",
            ] = iso_prob

            iso_metrics = (
                classification_metrics(
                    y_current,
                    iso_prob,
                )
            )

            for method, metrics in [
                (
                    "RAW",
                    raw_metrics,
                ),
                (
                    "PLATT",
                    platt_metrics,
                ),
                (
                    "ISOTONIC",
                    iso_metrics,
                ),
            ]:

                calibration_rows.append(
                    {
                        "TARGET":
                            target_name,

                        "FOLD":
                            fold,

                        "METHOD":
                            method,

                        "CAL_HISTORY_N":
                            len(
                                history
                            ),

                        "TEST_N":
                            len(
                                current
                            ),

                        "ROC_AUC":
                            metrics[
                                "ROC_AUC"
                            ],

                        "BRIER":
                            metrics[
                                "BRIER"
                            ],

                        "ECE":
                            metrics[
                                "ECE"
                            ],

                        "BAL_ACC":
                            metrics[
                                "BAL_ACC"
                            ],

                        "PRECISION":
                            metrics[
                                "PRECISION"
                            ],

                        "RECALL":
                            metrics[
                                "RECALL"
                            ],
                    }
                )

            print(
                f"  Fold {fold}: "
                f"CalHist={len(history):6d} "
                f"Test={len(current):6d} "
                f"RAW={raw_metrics['BRIER']:.4f} "
                f"PLATT={platt_metrics['BRIER']:.4f} "
                f"ISO={iso_metrics['BRIER']:.4f}"
            )

    calibration = pd.DataFrame(
        calibration_rows
    )

    calibration.to_csv(
        OUT_CALIBRATION,
        index=False,
    )

    predictions.to_csv(
        OUT_PREDICTIONS,
        index=False,
    )

    # ==============================================================================================
    # SUMMARY
    # ==============================================================================================

    section(
        "RISULTATI OOS FULL200"
    )

    summary_rows = []

    target_order = [
        f"BEAR_TO_BULL_{h}W"
        for h in HORIZONS
    ] + [
        f"BULL_TO_BEAR_{h}W"
        for h in HORIZONS
    ]

    for target_name in target_order:

        p = predictions[
            predictions[
                "TARGET"
            ]
            ==
            target_name
        ]

        raw_metrics = (
            classification_metrics(
                p[
                    "Y_TRUE"
                ].astype(
                    int
                ),
                p[
                    "RAW_PROB"
                ].astype(
                    float
                ),
            )
        )

        p_cal = p[
            p[
                "FOLD"
            ]
            .isin(
                [
                    2,
                    3,
                ]
            )
        ]

        platt_valid = p_cal[
            "PLATT_PROB"
        ].notna()

        iso_valid = p_cal[
            "ISOTONIC_PROB"
        ].notna()

        p_platt = p_cal.loc[
            platt_valid
        ]

        p_iso = p_cal.loc[
            iso_valid
        ]

        platt_metrics = (
            classification_metrics(
                p_platt[
                    "Y_TRUE"
                ].astype(
                    int
                ),
                p_platt[
                    "PLATT_PROB"
                ].astype(
                    float
                ),
            )
        )

        iso_metrics = (
            classification_metrics(
                p_iso[
                    "Y_TRUE"
                ].astype(
                    int
                ),
                p_iso[
                    "ISOTONIC_PROB"
                ].astype(
                    float
                ),
            )
        )

        # ------------------------------------------------------------------------------------------
        # SELECT CALIBRATION
        #
        # 1. Brier più basso
        # 2. a parità ECE più basso
        # ------------------------------------------------------------------------------------------

        if (
            iso_metrics[
                "BRIER"
            ]
            <
            platt_metrics[
                "BRIER"
            ]
        ):

            selected_method = (
                "ISOTONIC"
            )

            selected_metrics = (
                iso_metrics
            )

        elif (
            platt_metrics[
                "BRIER"
            ]
            <
            iso_metrics[
                "BRIER"
            ]
        ):

            selected_method = (
                "PLATT"
            )

            selected_metrics = (
                platt_metrics
            )

        else:

            if (
                iso_metrics[
                    "ECE"
                ]
                <
                platt_metrics[
                    "ECE"
                ]
            ):

                selected_method = (
                    "ISOTONIC"
                )

                selected_metrics = (
                    iso_metrics
                )

            else:

                selected_method = (
                    "PLATT"
                )

                selected_metrics = (
                    platt_metrics
                )

        horizon = int(
            p[
                "HORIZON_W"
            ].iloc[
                0
            ]
        )

        summary_rows.append(
            {
                "TARGET":
                    target_name,

                "HORIZON_W":
                    horizon,

                "RAW_N":
                    len(
                        p
                    ),

                "RAW_ROC_AUC":
                    raw_metrics[
                        "ROC_AUC"
                    ],

                "RAW_BRIER":
                    raw_metrics[
                        "BRIER"
                    ],

                "RAW_ECE":
                    raw_metrics[
                        "ECE"
                    ],

                "PLATT_N":
                    len(
                        p_platt
                    ),

                "PLATT_ROC_AUC":
                    platt_metrics[
                        "ROC_AUC"
                    ],

                "PLATT_BRIER":
                    platt_metrics[
                        "BRIER"
                    ],

                "PLATT_ECE":
                    platt_metrics[
                        "ECE"
                    ],

                "ISOTONIC_N":
                    len(
                        p_iso
                    ),

                "ISOTONIC_ROC_AUC":
                    iso_metrics[
                        "ROC_AUC"
                    ],

                "ISOTONIC_BRIER":
                    iso_metrics[
                        "BRIER"
                    ],

                "ISOTONIC_ECE":
                    iso_metrics[
                        "ECE"
                    ],

                "SELECTED_CALIBRATION":
                    selected_method,

                "CAL_ROC_AUC":
                    selected_metrics[
                        "ROC_AUC"
                    ],

                "CAL_BRIER":
                    selected_metrics[
                        "BRIER"
                    ],

                "CAL_ECE":
                    selected_metrics[
                        "ECE"
                    ],
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ==============================================================================================
    # FEATURE IMPORTANCE
    # ==============================================================================================

    section(
        "FEATURE IMPORTANCE FULL200"
    )

    importance = pd.read_csv(
        OUT_IMPORTANCE,
        low_memory=False,
    )

    mean_importance = (
        importance
        .groupby(
            "Feature",
            as_index=False,
        )[
            "Importance"
        ]
        .mean()
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    mean_importance[
        "Rank"
    ] = np.arange(
        1,
        len(
            mean_importance
        )
        +
        1
    )

    top_features = (
        mean_importance
        .head(
            50
        )
        .copy()
    )

    top_features.to_csv(
        OUT_TOP_FEATURES,
        index=False,
    )

    print(
        top_features
        .head(
            30
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    # ==============================================================================================
    # GLOBAL DIAGNOSTIC
    # ==============================================================================================

    section(
        "DIAGNOSTICA GLOBALE"
    )

    raw_auc_mean = float(
        summary[
            "RAW_ROC_AUC"
        ].mean()
    )

    raw_auc_min = float(
        summary[
            "RAW_ROC_AUC"
        ].min()
    )

    raw_auc_max = float(
        summary[
            "RAW_ROC_AUC"
        ].max()
    )

    targets_070 = int(
        (
            summary[
                "RAW_ROC_AUC"
            ]
            >=
            0.70
        ).sum()
    )

    targets_065 = int(
        (
            summary[
                "RAW_ROC_AUC"
            ]
            >=
            0.65
        ).sum()
    )

    mean_cal_brier = float(
        summary[
            "CAL_BRIER"
        ].mean()
    )

    mean_cal_ece = float(
        summary[
            "CAL_ECE"
        ].mean()
    )

    print(
        f"Target analizzati:         "
        f"{len(summary)}"
    )

    print(
        f"Mean RAW AUC:              "
        f"{raw_auc_mean:.4f}"
    )

    print(
        f"Min RAW AUC:               "
        f"{raw_auc_min:.4f}"
    )

    print(
        f"Max RAW AUC:               "
        f"{raw_auc_max:.4f}"
    )

    print(
        f"Target RAW AUC >= 0.70:    "
        f"{targets_070}/6"
    )

    print(
        f"Target RAW AUC >= 0.65:    "
        f"{targets_065}/6"
    )

    print(
        f"Mean CAL Brier:            "
        f"{mean_cal_brier:.5f}"
    )

    print(
        f"Mean CAL ECE:              "
        f"{mean_cal_ece:.5f}"
    )

    # ==============================================================================================
    # PASS
    # ==============================================================================================

    pass_model = (
        targets_065 == 6
        and
        targets_070 >= 5
        and
        raw_auc_mean >= 0.75
    )

    # ==============================================================================================
    # METADATA
    # ==============================================================================================

    metadata = {
        "version":
            VERSION,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "input":
            str(
                INPUT_FILE
            ),

        "rows":
            int(
                len(
                    meta
                )
            ),

        "tickers":
            int(
                meta[
                    "Ticker"
                ].nunique()
            ),

        "features":
            int(
                len(
                    feature_cols
                )
            ),

        "horizons":
            HORIZONS,

        "model":
            {
                "type":
                    "RandomForestClassifier",

                "n_estimators":
                    350,

                "max_features":
                    "sqrt",

                "min_samples_leaf":
                    8,

                "class_weight":
                    "balanced_subsample",

                "random_state":
                    RANDOM_STATE,

                "n_jobs":
                    N_JOBS,
            },

        "methodology":
            {
                "random_split":
                    False,

                "true_unique_dates":
                    True,

                "whole_week_purging":
                    True,

                "train_only_imputation":
                    True,

                "forward_only_calibration":
                    True,

                "fold1_calibration":
                    "RAW only",

                "fold2_calibration":
                    "Fold1 OOS",

                "fold3_calibration":
                    "Fold1 + Fold2 OOS",
            },

        "memory_strategy":
            {
                "feature_memmap":
                    True,

                "dtype":
                    "float32",

                "csv_chunk_size":
                    CSV_CHUNK_SIZE,

                "one_model_at_a_time":
                    True,

                "n_jobs_runtime":
                    N_JOBS,
            },

        "raw_auc_mean":
            raw_auc_mean,

        "raw_auc_min":
            raw_auc_min,

        "raw_auc_max":
            raw_auc_max,

        "targets_auc_ge_070":
            targets_070,

        "targets_auc_ge_065":
            targets_065,

        "mean_cal_brier":
            mean_cal_brier,

        "mean_cal_ece":
            mean_cal_ece,

        "pass":
            bool(
                pass_model
            ),

        "production_model_fitted":
            False,
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

    # ==============================================================================================
    # FINAL
    # ==============================================================================================

    section(
        "VERDETTO V40.40a"
    )

    if pass_model:

        print(
            "OK - FULL200 REVERSAL MODEL "
            "SUPERA IL NUOVO TEST OOS."
        )

        print()

        print(
            f"Mean RAW AUC:           "
            f"{raw_auc_mean:.4f}"
        )

        print(
            f"Min RAW AUC:            "
            f"{raw_auc_min:.4f}"
        )

        print(
            f"Target >= 0.70:         "
            f"{targets_070}/6"
        )

        print(
            f"Target >= 0.65:         "
            f"{targets_065}/6"
        )

        print()

        print(
            "PURGING CORRETTO:       SI"
        )

        print(
            "413 FEATURE:            SI"
        )

        print(
            "FULL200:                SI"
        )

        print(
            "CALIBRAZIONE FORWARD:   SI"
        )

        print(
            "MEMORY SAFE:            SI"
        )

        print(
            "PRODUCTION MODEL:       NON ANCORA"
        )

        print()

        print(
            "PROSSIMO STEP:"
        )

        print(
            "V40.37c - FIT FINALE PRODUCTION "
            "FULL200."
        )

    else:

        print(
            "ATTENZIONE - FULL200 NON SUPERA "
            "TUTTE LE SOGLIE OOS."
        )

        print()

        print(
            "NON PROCEDERE A V40.37c."
        )

        print()

        print(
            "Analizzeremo i risultati prima "
            "di qualsiasi modifica."
        )

    # ==============================================================================================
    # CLOSE MEMMAP
    # ==============================================================================================

    del feature_matrix

    collect()


# ==================================================================================================
# RUN
# ==================================================================================================

if __name__ == "__main__":

    main()