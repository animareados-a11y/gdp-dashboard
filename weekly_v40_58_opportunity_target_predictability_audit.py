"""
MarketSentinel
V40.58 - OPPORTUNITY TARGET PREDICTABILITY AUDIT
================================================

SCOPO
-----
Stabilire empiricamente quali caratteristiche economiche di un trend
PHASE completo siano realmente prevedibili dalla configurazione tecnica
disponibile all'inizio del trend.

NON sceglie ancora il target finale Opportunity.
NON crea score Opportunity.
NON crea classi LOW/MEDIUM/HIGH/VERY_HIGH.
NON modifica PHASE, REVERSAL o STRENGTH.

EPISODI
-------
BULL:
    configurazione tecnica al BUY1 e nelle settimane precedenti.

BEAR:
    configurazione tecnica al SELL1 e nelle settimane precedenti.

TARGET STUDIATI
---------------
1. directional_realized_pct
2. mfe_pct
3. mae_pct
4. duration_weeks
5. speed_to_mfe_pct_per_week
6. path_efficiency

PROFONDITA TEMPORALI
--------------------
T
T + T-1
T + T-1 + T-2
T + T-1 + T-2 + T-3
T + T-1 + T-2 + T-3 + T-4

VALIDAZIONE
-----------
Walk-forward temporale OOS.

Regola anti-leakage:
un episodio può essere utilizzato nel TRAIN soltanto se il suo
end_date è strettamente precedente all'inizio del TEST.

PREPROCESSING
-------------
Ogni decisione viene presa esclusivamente sul TRAIN del fold:
- colonne utilizzabili
- mediane di imputazione
- rimozione colonne costanti
- feature screening

FEATURE SCREENING
-----------------
Spearman assoluta feature-target calcolata SOLO sul TRAIN.
Massimo 160 feature.

MODELLI
-------
1. ExtraTreesRegressor
2. RandomForestRegressor

STRUTTURA
---------
6 target
x 2 direzioni
x 5 rappresentazioni temporali
x 5 fold OOS
x 2 modelli
= 600 job.

V40.58 è esclusivamente un audit di predicibilità.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# CONFIG
# ============================================================

VERSION = "V40.58"

INPUT_FILE = Path(
    "data/v40_54_opportunity_temporal_feature_study/"
    "v40_54_opportunity_temporal_dataset.csv"
)

OUT_DIR = Path(
    "data/v40_58_opportunity_target_predictability_audit"
)

OUT_RESULTS = (
    OUT_DIR /
    "v40_58_oos_results.csv"
)

OUT_PREDICTIONS = (
    OUT_DIR /
    "v40_58_oos_predictions.csv"
)

OUT_FEATURES = (
    OUT_DIR /
    "v40_58_selected_features.csv"
)

OUT_SPLIT_AUDIT = (
    OUT_DIR /
    "v40_58_split_audit.csv"
)

OUT_DECILES = (
    OUT_DIR /
    "v40_58_prediction_deciles.csv"
)

OUT_SUMMARY = (
    OUT_DIR /
    "v40_58_summary.csv"
)

OUT_METADATA = (
    OUT_DIR /
    "v40_58_metadata.json"
)

CHECKPOINT_FILE = (
    OUT_DIR /
    "v40_58_checkpoint.csv"
)


TARGETS = [
    "directional_realized_pct",
    "mfe_pct",
    "mae_pct",
    "duration_weeks",
    "speed_to_mfe_pct_per_week",
    "path_efficiency",
]

FEATURE_COUNT_PER_SNAPSHOT = 413

MAX_SELECTED_FEATURES = 160

MIN_TRAIN_ROWS = 1000

RANDOM_STATE = 42

N_JOBS = 1


# ============================================================
# TEMPORAL REPRESENTATIONS
# ============================================================

REPRESENTATIONS = {
    "T": [
        "T",
    ],
    "T_T1": [
        "T",
        "T_MINUS_1",
    ],
    "T_T1_T2": [
        "T",
        "T_MINUS_1",
        "T_MINUS_2",
    ],
    "T_T1_T2_T3": [
        "T",
        "T_MINUS_1",
        "T_MINUS_2",
        "T_MINUS_3",
    ],
    "T_T1_T2_T3_T4": [
        "T",
        "T_MINUS_1",
        "T_MINUS_2",
        "T_MINUS_3",
        "T_MINUS_4",
    ],
}


# ============================================================
# OOS WINDOWS
# ============================================================

OOS_WINDOWS = [
    (
        "F1",
        "2022-01-01",
        "2022-12-31",
    ),
    (
        "F2",
        "2023-01-01",
        "2023-12-31",
    ),
    (
        "F3",
        "2024-01-01",
        "2024-12-31",
    ),
    (
        "F4",
        "2025-01-01",
        "2025-12-31",
    ),
    (
        "F5",
        "2026-01-01",
        "2026-08-28",
    ),
]


# ============================================================
# MODELS
# ============================================================

MODEL_NAMES = [
    "EXTRA_TREES",
    "RANDOM_FOREST",
]


def build_model(
    model_name: str,
):

    if model_name == "EXTRA_TREES":

        return ExtraTreesRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )

    if model_name == "RANDOM_FOREST":

        return RandomForestRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )

    raise ValueError(
        f"Modello sconosciuto: {model_name}"
    )


# ============================================================
# UTILS
# ============================================================

def banner(
    text: str,
) -> None:

    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def safe_spearman(
    y_true,
    y_pred,
) -> float:

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    if mask.sum() < 3:
        return np.nan

    yt = y_true[mask]
    yp = y_pred[mask]

    if (
        np.nanstd(yt) == 0
        or np.nanstd(yp) == 0
    ):
        return np.nan

    result = spearmanr(
        yt,
        yp,
    )

    return float(
        result.statistic
    )


def rmse(
    y_true,
    y_pred,
) -> float:

    return float(
        math.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


# ============================================================
# LOAD
# ============================================================

def load_dataset() -> pd.DataFrame:

    banner(
        "LOAD V40.54 TEMPORAL DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input non trovato: "
            f"{INPUT_FILE}"
        )

    d = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required = {
        "episode_id",
        "Ticker",
        "direction",
        "start_date",
        "end_date",
        *TARGETS,
    }

    missing = sorted(
        required
        - set(d.columns)
    )

    if missing:

        raise KeyError(
            f"Colonne mancanti: "
            f"{missing}"
        )

    d["start_date"] = (
        pd.to_datetime(
            d["start_date"],
            errors="coerce",
        )
    )

    d["end_date"] = (
        pd.to_datetime(
            d["end_date"],
            errors="coerce",
        )
    )

    for target in TARGETS:

        d[target] = (
            pd.to_numeric(
                d[target],
                errors="coerce",
            )
        )

    d = d.dropna(
        subset=[
            "episode_id",
            "Ticker",
            "direction",
            "start_date",
            "end_date",
        ]
    ).copy()

    d = d[
        d["direction"].isin(
            [
                "BULL",
                "BEAR",
            ]
        )
    ].copy()

    d = (
        d.sort_values(
            [
                "start_date",
                "Ticker",
                "episode_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Episodi: "
        f"{len(d):,}"
    )

    print(
        "BULL: "
        f"{(d['direction'] == 'BULL').sum():,}"
    )

    print(
        "BEAR: "
        f"{(d['direction'] == 'BEAR').sum():,}"
    )

    print(
        "Periodo start: "
        f"{d['start_date'].min().date()} "
        "-> "
        f"{d['start_date'].max().date()}"
    )

    print()
    print("Target:")

    for target in TARGETS:

        finite_n = int(
            np.isfinite(
                d[target]
                .to_numpy(
                    dtype=float
                )
            ).sum()
        )

        print(
            f"  {target}: "
            f"{finite_n:,}"
        )

    return d


# ============================================================
# DISCOVER TEMPORAL FEATURES
# ============================================================

def discover_feature_columns(
    d: pd.DataFrame,
):

    banner(
        "DISCOVER TEMPORAL FEATURES"
    )

    snapshots = [
        "T",
        "T_MINUS_1",
        "T_MINUS_2",
        "T_MINUS_3",
        "T_MINUS_4",
    ]

    snapshot_map = {}

    for snapshot in snapshots:

        suffix = (
            "__"
            + snapshot
        )

        cols = [
            c
            for c in d.columns
            if (
                c.endswith(
                    suffix
                )
                and not c.startswith(
                    "SNAPSHOT_DATE__"
                )
            )
        ]

        if len(cols) != (
            FEATURE_COUNT_PER_SNAPSHOT
        ):

            raise RuntimeError(
                f"{snapshot}: "
                f"attese "
                f"{FEATURE_COUNT_PER_SNAPSHOT} "
                f"feature, trovate "
                f"{len(cols)}."
            )

        snapshot_map[
            snapshot
        ] = cols

        print(
            f"{snapshot}: "
            f"{len(cols)}"
        )

    return snapshot_map


def columns_for_representation(
    snapshot_map,
    representation,
):

    snapshots = (
        REPRESENTATIONS[
            representation
        ]
    )

    cols = []

    for snapshot in snapshots:

        cols.extend(
            snapshot_map[
                snapshot
            ]
        )

    return cols


# ============================================================
# TRAIN-ONLY PREPROCESSING
# ============================================================

def train_only_prepare(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target: str,
):

    X_train = (
        train[
            feature_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    X_test = (
        test[
            feature_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    y_train = (
        train[target]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # 1. Keep columns with usable TRAIN data.
    # --------------------------------------------------------

    finite_count = (
        np.isfinite(
            X_train.to_numpy(
                dtype=np.float32
            )
        )
        .sum(
            axis=0
        )
    )

    usable_mask = (
        finite_count >= 20
    )

    usable_cols = [
        c
        for c, ok
        in zip(
            feature_cols,
            usable_mask,
        )
        if ok
    ]

    X_train = (
        X_train[
            usable_cols
        ]
    )

    X_test = (
        X_test[
            usable_cols
        ]
    )

    # --------------------------------------------------------
    # 2. TRAIN medians only.
    # --------------------------------------------------------

    medians = (
        X_train
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .median(
            axis=0
        )
    )

    valid_median = (
        medians.notna()
    )

    usable_cols = (
        medians.index[
            valid_median
        ]
        .tolist()
    )

    medians = (
        medians[
            usable_cols
        ]
    )

    X_train = (
        X_train[
            usable_cols
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            medians
        )
    )

    X_test = (
        X_test[
            usable_cols
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            medians
        )
    )

    # --------------------------------------------------------
    # 3. Remove TRAIN-constant columns.
    # --------------------------------------------------------

    nunique = (
        X_train.nunique(
            dropna=False
        )
    )

    nonconstant_cols = (
        nunique[
            nunique > 1
        ]
        .index
        .tolist()
    )

    X_train = (
        X_train[
            nonconstant_cols
        ]
    )

    X_test = (
        X_test[
            nonconstant_cols
        ]
    )

    # --------------------------------------------------------
    # 4. TRAIN-only Spearman screening.
    # --------------------------------------------------------

    ranking = []

    for col in (
        nonconstant_cols
    ):

        x = (
            X_train[col]
            .to_numpy(
                dtype=float
            )
        )

        rho = safe_spearman(
            x,
            y_train,
        )

        if np.isfinite(
            rho
        ):

            ranking.append(
                (
                    col,
                    float(rho),
                    abs(
                        float(rho)
                    ),
                )
            )

    ranking.sort(
        key=lambda z: (
            -z[2],
            z[0],
        )
    )

    selected = (
        ranking[
            :MAX_SELECTED_FEATURES
        ]
    )

    selected_cols = [
        x[0]
        for x in selected
    ]

    if not selected_cols:

        raise RuntimeError(
            "Nessuna feature "
            "selezionabile."
        )

    X_train_np = (
        X_train[
            selected_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    X_test_np = (
        X_test[
            selected_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    return (
        X_train_np,
        X_test_np,
        selected,
        len(feature_cols),
        len(usable_cols),
        len(nonconstant_cols),
    )


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint():

    if not (
        CHECKPOINT_FILE.exists()
    ):
        return set()

    d = pd.read_csv(
        CHECKPOINT_FILE
    )

    if "job_id" not in (
        d.columns
    ):
        return set()

    return set(
        d["job_id"]
        .astype(str)
        .tolist()
    )


def append_checkpoint(
    job_id: str,
):

    row = pd.DataFrame(
        [
            {
                "job_id":
                    job_id,
                "completed_at":
                    pd.Timestamp.now()
                    .isoformat(),
            }
        ]
    )

    header = (
        not CHECKPOINT_FILE.exists()
    )

    row.to_csv(
        CHECKPOINT_FILE,
        mode="a",
        header=header,
        index=False,
    )


# ============================================================
# OOS BENCHMARK
# ============================================================

def run_benchmark(
    d: pd.DataFrame,
    snapshot_map,
):

    banner(
        "RUN MULTI-TARGET OOS BENCHMARK"
    )

    completed_jobs = (
        load_checkpoint()
    )

    result_rows = []
    prediction_rows = []
    feature_rows = []
    split_rows = []

    # --------------------------------------------------------
    # Resume existing accumulated outputs.
    # --------------------------------------------------------

    if OUT_RESULTS.exists():

        result_rows = (
            pd.read_csv(
                OUT_RESULTS
            )
            .to_dict(
                "records"
            )
        )

    if OUT_PREDICTIONS.exists():

        prediction_rows = (
            pd.read_csv(
                OUT_PREDICTIONS
            )
            .to_dict(
                "records"
            )
        )

    # --------------------------------------------------------
    # RECOVERY:
    # A job is resumable only if its OOS predictions exist.
    # This repairs interrupted runs where the checkpoint may
    # contain a job whose prediction rows were not preserved.
    # --------------------------------------------------------

    prediction_job_ids = {
        str(x.get("job_id"))
        for x in prediction_rows
        if x.get("job_id") is not None
    }

    checkpoint_jobs = set(
        completed_jobs
    )

    completed_jobs = (
        checkpoint_jobs
        & prediction_job_ids
    )

    print(
        "RECOVERY RESUME | "
        f"checkpoint={len(checkpoint_jobs)} | "
        f"predictions={len(prediction_job_ids)} | "
        f"really_complete={len(completed_jobs)} | "
        f"to_rebuild={len(checkpoint_jobs - completed_jobs)}"
    )

    if OUT_FEATURES.exists():

        feature_rows = (
            pd.read_csv(
                OUT_FEATURES
            )
            .to_dict(
                "records"
            )
        )

    if OUT_SPLIT_AUDIT.exists():

        split_rows = (
            pd.read_csv(
                OUT_SPLIT_AUDIT
            )
            .to_dict(
                "records"
            )
        )

    # --------------------------------------------------------
    # RECOVERY CLEANUP:
    # Jobs without preserved OOS predictions must be rebuilt.
    # Remove their old result/feature rows first so rerunning
    # replaces them instead of creating duplicates.
    # Existing prediction-complete jobs remain untouched.
    # --------------------------------------------------------

    rebuild_jobs = (
        checkpoint_jobs
        - completed_jobs
    )

    result_rows = [
        x
        for x in result_rows
        if str(x.get("job_id"))
        not in rebuild_jobs
    ]

    feature_rows = [
        x
        for x in feature_rows
        if str(x.get("job_id"))
        not in rebuild_jobs
    ]

    print(
        "RECOVERY CLEANUP | "
        f"rebuild_jobs={len(rebuild_jobs)} | "
        f"results_preserved={len(result_rows)} | "
        f"features_preserved={len(feature_rows)}"
    )

    total_jobs = (
        len(TARGETS)
        * 2
        * len(REPRESENTATIONS)
        * len(OOS_WINDOWS)
        * len(MODEL_NAMES)
    )

    job_counter = 0

    for target in TARGETS:

        banner(
            f"TARGET: {target}"
        )

        for direction in [
            "BULL",
            "BEAR",
        ]:

            dd = d[
                d["direction"]
                == direction
            ].copy()

            # Only rows where this target exists.
            dd = dd[
                np.isfinite(
                    dd[target]
                    .to_numpy(
                        dtype=float
                    )
                )
            ].copy()

            for representation in (
                REPRESENTATIONS
            ):

                feature_cols = (
                    columns_for_representation(
                        snapshot_map,
                        representation,
                    )
                )

                for (
                    fold,
                    test_start_s,
                    test_end_s,
                ) in OOS_WINDOWS:

                    test_start = (
                        pd.Timestamp(
                            test_start_s
                        )
                    )

                    test_end = (
                        pd.Timestamp(
                            test_end_s
                        )
                    )

                    # --------------------------------------------
                    # TEST:
                    # episode begins inside OOS window.
                    # --------------------------------------------

                    test_mask = (
                        (
                            dd["start_date"]
                            >= test_start
                        )
                        & (
                            dd["start_date"]
                            <= test_end
                        )
                    )

                    test = (
                        dd[
                            test_mask
                        ]
                        .copy()
                    )

                    # --------------------------------------------
                    # TRAIN:
                    # episode target completely known before
                    # beginning of OOS period.
                    # --------------------------------------------

                    train_mask = (
                        dd["end_date"]
                        < test_start
                    )

                    train = (
                        dd[
                            train_mask
                        ]
                        .copy()
                    )

                    overlap_failures = int(
                        (
                            train[
                                "end_date"
                            ]
                            >= test_start
                        ).sum()
                    )

                    future_train_failures = int(
                        (
                            train[
                                "start_date"
                            ]
                            >= test_start
                        ).sum()
                    )

                    split_base = {
                        "target":
                            target,
                        "direction":
                            direction,
                        "representation":
                            representation,
                        "fold":
                            fold,
                        "test_start":
                            test_start.date(),
                        "test_end":
                            test_end.date(),
                        "train_n":
                            int(
                                len(train)
                            ),
                        "test_n":
                            int(
                                len(test)
                            ),
                        "train_start_min":
                            (
                                train[
                                    "start_date"
                                ].min()
                                if len(train)
                                else pd.NaT
                            ),
                        "train_end_max":
                            (
                                train[
                                    "end_date"
                                ].max()
                                if len(train)
                                else pd.NaT
                            ),
                        "test_start_min":
                            (
                                test[
                                    "start_date"
                                ].min()
                                if len(test)
                                else pd.NaT
                            ),
                        "test_start_max":
                            (
                                test[
                                    "start_date"
                                ].max()
                                if len(test)
                                else pd.NaT
                            ),
                        "overlap_failures":
                            overlap_failures,
                        "future_train_failures":
                            future_train_failures,
                    }

                    existing_split = any(
                        (
                            str(
                                x.get(
                                    "target"
                                )
                            )
                            == target
                            and str(
                                x.get(
                                    "direction"
                                )
                            )
                            == direction
                            and str(
                                x.get(
                                    "representation"
                                )
                            )
                            == representation
                            and str(
                                x.get(
                                    "fold"
                                )
                            )
                            == fold
                        )
                        for x in split_rows
                    )

                    if not existing_split:

                        split_rows.append(
                            split_base
                        )

                    split_key = (
                        target,
                        direction,
                        representation,
                        fold,
                    )

                    if len(train) < (
                        MIN_TRAIN_ROWS
                    ):

                        print(
                            f"SKIP {split_key}: "
                            f"train={len(train):,}"
                        )
                        continue

                    if len(test) < 20:

                        print(
                            f"SKIP {split_key}: "
                            f"test={len(test):,}"
                        )
                        continue

                    if overlap_failures:

                        raise RuntimeError(
                            "Leakage overlap in "
                            f"{split_key}"
                        )

                    if future_train_failures:

                        raise RuntimeError(
                            "Future train rows in "
                            f"{split_key}"
                        )

                    (
                        X_train,
                        X_test,
                        selected,
                        requested_n,
                        usable_n,
                        nonconstant_n,
                    ) = train_only_prepare(
                        train,
                        test,
                        feature_cols,
                        target,
                    )

                    y_train = (
                        train[target]
                        .astype(float)
                        .to_numpy()
                    )

                    y_test = (
                        test[target]
                        .astype(float)
                        .to_numpy()
                    )

                    for model_name in (
                        MODEL_NAMES
                    ):

                        job_counter += 1

                        job_id = (
                            f"{target}|"
                            f"{direction}|"
                            f"{representation}|"
                            f"{fold}|"
                            f"{model_name}"
                        )

                        print()
                        print(
                            f"[{job_counter}/"
                            f"{total_jobs}] "
                            f"{job_id}"
                        )

                        if job_id in (
                            completed_jobs
                        ):

                            print(
                                "RESUME: "
                                "gia completato."
                            )
                            continue

                        t0 = time.time()

                        model = (
                            build_model(
                                model_name
                            )
                        )

                        model.fit(
                            X_train,
                            y_train,
                        )

                        pred = (
                            model.predict(
                                X_test
                            )
                        )

                        rho = (
                            safe_spearman(
                                y_test,
                                pred,
                            )
                        )

                        mae = float(
                            mean_absolute_error(
                                y_test,
                                pred,
                            )
                        )

                        this_rmse = (
                            rmse(
                                y_test,
                                pred,
                            )
                        )

                        elapsed = (
                            time.time()
                            - t0
                        )

                        # ----------------------------------------
                        # IDEMPOTENT JOB WRITE:
                        # If an interrupted previous run left partial
                        # rows for this job, remove them before writing
                        # the freshly recomputed complete job.
                        # This changes persistence only, never ML logic.
                        # ----------------------------------------

                        result_rows = [
                            x
                            for x in result_rows
                            if str(x.get("job_id")) != str(job_id)
                        ]

                        feature_rows = [
                            x
                            for x in feature_rows
                            if str(x.get("job_id")) != str(job_id)
                        ]

                        prediction_rows = [
                            x
                            for x in prediction_rows
                            if str(x.get("job_id")) != str(job_id)
                        ]

                        result_rows.append(
                            {
                                "job_id":
                                    job_id,
                                "target":
                                    target,
                                "direction":
                                    direction,
                                "representation":
                                    representation,
                                "fold":
                                    fold,
                                "model":
                                    model_name,
                                "train_n":
                                    int(
                                        len(train)
                                    ),
                                "test_n":
                                    int(
                                        len(test)
                                    ),
                                "requested_features":
                                    int(
                                        requested_n
                                    ),
                                "usable_features":
                                    int(
                                        usable_n
                                    ),
                                "nonconstant_features":
                                    int(
                                        nonconstant_n
                                    ),
                                "selected_features":
                                    int(
                                        len(
                                            selected
                                        )
                                    ),
                                "spearman":
                                    rho,
                                "mae":
                                    mae,
                                "rmse":
                                    this_rmse,
                                "seconds":
                                    float(
                                        elapsed
                                    ),
                            }
                        )

                        for (
                            feature,
                            feature_rho,
                            feature_abs_rho,
                        ) in selected:

                            feature_rows.append(
                                {
                                    "job_id":
                                        job_id,
                                    "target":
                                        target,
                                    "direction":
                                        direction,
                                    "representation":
                                        representation,
                                    "fold":
                                        fold,
                                    "model":
                                        model_name,
                                    "feature":
                                        feature,
                                    "train_spearman":
                                        feature_rho,
                                    "train_abs_spearman":
                                        feature_abs_rho,
                                }
                            )

                        for (
                            episode_id,
                            ticker,
                            start_date,
                            end_date,
                            actual,
                            predicted,
                        ) in zip(
                            test[
                                "episode_id"
                            ],
                            test[
                                "Ticker"
                            ],
                            test[
                                "start_date"
                            ],
                            test[
                                "end_date"
                            ],
                            y_test,
                            pred,
                        ):

                            prediction_rows.append(
                                {
                                    "job_id":
                                        job_id,
                                    "target":
                                        target,
                                    "direction":
                                        direction,
                                    "representation":
                                        representation,
                                    "fold":
                                        fold,
                                    "model":
                                        model_name,
                                    "episode_id":
                                        episode_id,
                                    "Ticker":
                                        ticker,
                                    "start_date":
                                        start_date,
                                    "end_date":
                                        end_date,
                                    "actual_target":
                                        float(
                                            actual
                                        ),
                                    "predicted_target":
                                        float(
                                            predicted
                                        ),
                                }
                            )

                        # ----------------------------------------
                        # Persist after every completed job.
                        # ----------------------------------------

                        pd.DataFrame(
                            result_rows
                        ).to_csv(
                            OUT_RESULTS,
                            index=False,
                        )

                        pd.DataFrame(
                            prediction_rows
                        ).to_csv(
                            OUT_PREDICTIONS,
                            index=False,
                        )

                        pd.DataFrame(
                            feature_rows
                        ).to_csv(
                            OUT_FEATURES,
                            index=False,
                        )

                        pd.DataFrame(
                            split_rows
                        ).to_csv(
                            OUT_SPLIT_AUDIT,
                            index=False,
                        )

                        append_checkpoint(
                            job_id
                        )

                        completed_jobs.add(
                            job_id
                        )

                        print(
                            f"Spearman="
                            f"{rho:.6f} | "
                            f"MAE="
                            f"{mae:.4f} | "
                            f"RMSE="
                            f"{this_rmse:.4f} | "
                            f"{elapsed:.1f}s"
                        )

    return (
        pd.DataFrame(
            result_rows
        ),
        pd.DataFrame(
            prediction_rows
        ),
        pd.DataFrame(
            feature_rows
        ),
        pd.DataFrame(
            split_rows
        ),
    )


# ============================================================
# DECILE VALIDATION
# ============================================================

def build_decile_validation(
    predictions: pd.DataFrame,
):

    banner(
        "OOS PREDICTION DECILE VALIDATION"
    )

    rows = []

    if predictions.empty:

        return pd.DataFrame()

    group_cols = [
        "target",
        "direction",
        "representation",
        "model",
    ]

    for keys, g in (
        predictions.groupby(
            group_cols,
            sort=True,
        )
    ):

        (
            target,
            direction,
            representation,
            model,
        ) = keys

        g = g.copy()

        pct = (
            g[
                "predicted_target"
            ]
            .rank(
                method="first",
                pct=True,
            )
        )

        g[
            "prediction_decile"
        ] = (
            np.ceil(
                pct * 10
            )
            .clip(
                1,
                10,
            )
            .astype(int)
        )

        for decile, x in (
            g.groupby(
                "prediction_decile",
                sort=True,
            )
        ):

            rows.append(
                {
                    "target":
                        target,
                    "direction":
                        direction,
                    "representation":
                        representation,
                    "model":
                        model,
                    "prediction_decile":
                        int(decile),
                    "N":
                        int(len(x)),
                    "predicted_target_mean":
                        float(
                            x[
                                "predicted_target"
                            ].mean()
                        ),
                    "predicted_target_median":
                        float(
                            x[
                                "predicted_target"
                            ].median()
                        ),
                    "actual_target_mean":
                        float(
                            x[
                                "actual_target"
                            ].mean()
                        ),
                    "actual_target_median":
                        float(
                            x[
                                "actual_target"
                            ].median()
                        ),
                }
            )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUT_DECILES,
        index=False,
    )

    return out


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
):

    banner(
        "BUILD OOS SUMMARY"
    )

    rows = []

    if results.empty:

        return pd.DataFrame()

    group_cols = [
        "target",
        "direction",
        "representation",
        "model",
    ]

    for keys, g in (
        results.groupby(
            group_cols,
            sort=True,
        )
    ):

        (
            target,
            direction,
            representation,
            model,
        ) = keys

        p = predictions[
            (
                predictions[
                    "target"
                ]
                == target
            )
            & (
                predictions[
                    "direction"
                ]
                == direction
            )
            & (
                predictions[
                    "representation"
                ]
                == representation
            )
            & (
                predictions[
                    "model"
                ]
                == model
            )
        ]

        fold_rhos = (
            g[
                "spearman"
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        rows.append(
            {
                "target":
                    target,
                "direction":
                    direction,
                "representation":
                    representation,
                "model":
                    model,
                "folds":
                    int(
                        len(g)
                    ),
                "oos_N":
                    int(
                        len(p)
                    ),
                "pooled_spearman":
                    safe_spearman(
                        p[
                            "actual_target"
                        ],
                        p[
                            "predicted_target"
                        ],
                    ),
                "mean_fold_spearman":
                    (
                        float(
                            np.mean(
                                fold_rhos
                            )
                        )
                        if len(
                            fold_rhos
                        )
                        else np.nan
                    ),
                "median_fold_spearman":
                    (
                        float(
                            np.median(
                                fold_rhos
                            )
                        )
                        if len(
                            fold_rhos
                        )
                        else np.nan
                    ),
                "min_fold_spearman":
                    (
                        float(
                            np.min(
                                fold_rhos
                            )
                        )
                        if len(
                            fold_rhos
                        )
                        else np.nan
                    ),
                "positive_spearman_folds":
                    int(
                        (
                            fold_rhos
                            > 0
                        ).sum()
                    ),
                "mean_mae":
                    float(
                        g[
                            "mae"
                        ].mean()
                    ),
                "mean_rmse":
                    float(
                        g[
                            "rmse"
                        ].mean()
                    ),
            }
        )

    out = pd.DataFrame(
        rows
    )

    out = (
        out.sort_values(
            [
                "target",
                "direction",
                "pooled_spearman",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    out.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    print()

    for target in TARGETS:

        print()
        print("-" * 78)
        print(
            f"TARGET: {target}"
        )
        print("-" * 78)

        x = out[
            out["target"]
            == target
        ]

        print(
            x.round(
                6
            )
            .to_string(
                index=False
            )
        )

    return out


# ============================================================
# FINAL AUDIT
# ============================================================

def final_audit(
    split_audit: pd.DataFrame,
    results: pd.DataFrame,
):

    banner(
        "FINAL OOS AUDIT"
    )

    overlap_failures = int(
        split_audit[
            "overlap_failures"
        ].sum()
    )

    future_failures = int(
        split_audit[
            "future_train_failures"
        ].sum()
    )

    duplicate_jobs = int(
        results[
            "job_id"
        ]
        .duplicated()
        .sum()
    )

    expected_jobs = (
        len(TARGETS)
        * 2
        * len(REPRESENTATIONS)
        * len(OOS_WINDOWS)
        * len(MODEL_NAMES)
    )

    actual_jobs = int(
        results[
            "job_id"
        ].nunique()
    )

    expected_splits = (
        len(TARGETS)
        * 2
        * len(REPRESENTATIONS)
        * len(OOS_WINDOWS)
    )

    actual_splits = int(
        split_audit[
            [
                "target",
                "direction",
                "representation",
                "fold",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Overlap leakage failures: "
        f"{overlap_failures}"
    )

    print(
        "Future train failures: "
        f"{future_failures}"
    )

    print(
        "Duplicate completed jobs: "
        f"{duplicate_jobs}"
    )

    print(
        "Completed jobs: "
        f"{actual_jobs}/"
        f"{expected_jobs}"
    )

    print(
        "Unique target/split combinations: "
        f"{actual_splits}/"
        f"{expected_splits}"
    )

    passed = (
        overlap_failures == 0
        and future_failures == 0
        and duplicate_jobs == 0
        and actual_jobs == expected_jobs
        and actual_splits == expected_splits
    )

    print()

    print(
        "OOS STRUCTURAL AUDIT: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "MARKETSENTINEL V40.58 "
        "OPPORTUNITY TARGET "
        "PREDICTABILITY AUDIT"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    d = load_dataset()

    snapshot_map = (
        discover_feature_columns(
            d
        )
    )

    (
        results,
        predictions,
        selected_features,
        split_audit,
    ) = run_benchmark(
        d,
        snapshot_map,
    )

    deciles = (
        build_decile_validation(
            predictions
        )
    )

    summary = (
        build_summary(
            results,
            predictions,
        )
    )

    audit_pass = (
        final_audit(
            split_audit,
            results,
        )
    )

    metadata = {
        "version":
            VERSION,

        "input":
            str(
                INPUT_FILE
            ),

        "purpose":
            (
                "Causal OOS comparison of "
                "candidate economic targets "
                "for Opportunity."
            ),

        "targets":
            TARGETS,

        "target_semantics": {
            "directional_realized_pct":
                (
                    "Directional return from "
                    "episode start to episode end. "
                    "Positive is favorable for both "
                    "BULL and BEAR."
                ),

            "mfe_pct":
                (
                    "Maximum favorable excursion "
                    "during the complete episode."
                ),

            "mae_pct":
                (
                    "Maximum adverse excursion "
                    "during the complete episode."
                ),

            "duration_weeks":
                (
                    "Duration of the complete "
                    "directional episode in weeks."
                ),

            "speed_to_mfe_pct_per_week":
                (
                    "Maximum favorable excursion "
                    "divided by weeks required "
                    "to reach MFE."
                ),

            "path_efficiency":
                (
                    "Episode path-efficiency metric "
                    "from V40.54."
                ),
        },

        "directions": [
            "BULL",
            "BEAR",
        ],

        "representations":
            REPRESENTATIONS,

        "models":
            MODEL_NAMES,

        "oos_windows": [
            {
                "fold":
                    fold,
                "test_start":
                    start,
                "test_end":
                    end,
            }
            for (
                fold,
                start,
                end,
            )
            in OOS_WINDOWS
        ],

        "purge_rule":
            (
                "Training episode end_date "
                "must be strictly before "
                "test_start."
            ),

        "feature_screening":
            (
                "Train-only absolute Spearman "
                f"against each target, top "
                f"{MAX_SELECTED_FEATURES}."
            ),

        "imputation":
            "Train-only median.",

        "constant_filter":
            "Train-only.",

        "random_state":
            RANDOM_STATE,

        "n_jobs":
            N_JOBS,

        "expected_jobs":
            (
                len(TARGETS)
                * 2
                * len(REPRESENTATIONS)
                * len(OOS_WINDOWS)
                * len(MODEL_NAMES)
            ),

        "structural_audit_pass":
            bool(
                audit_pass
            ),

        "phase_modified":
            False,

        "reversal_modified":
            False,

        "strength_modified":
            False,

        "score_1_10_created":
            False,

        "opportunity_classes_created":
            False,

        "production_model_created":
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
        )

    banner(
        "OUTPUT"
    )

    for path in [
        OUT_RESULTS,
        OUT_PREDICTIONS,
        OUT_FEATURES,
        OUT_SPLIT_AUDIT,
        OUT_DECILES,
        OUT_SUMMARY,
        OUT_METADATA,
    ]:

        print(
            path
        )

    banner(
        "V40.58 COMPLETE"
    )

    print(
        "Questo e' un audit OOS "
        "multi-target, NON un "
        "modello production."
    )

    print(
        "Nessuno score Opportunity "
        "1-10 e' stato creato."
    )

    print(
        "Nessuna classe "
        "debole/medio/forte/"
        "molto forte e' stata "
        "imposta."
    )

    print(
        "PHASE, REVERSAL e STRENGTH "
        "non sono stati modificati."
    )


if __name__ == "__main__":
    main()