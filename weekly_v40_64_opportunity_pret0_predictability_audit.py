"""
MarketSentinel
V40.64 — OPPORTUNITY PRE-T0 PREDICTABILITY AUDIT

PURPOSE
-------
Test whether the frozen V40.39 technical configuration contains
causal OOS information before the start T0 of a TRUE directional trend.

This is a DIAGNOSTIC / PREDICTABILITY study.

It DOES NOT:
- create the final Opportunity target Y
- create Opportunity score 1-10
- choose an Opportunity time window
- use fixed forward-return horizons
- modify PHASE
- modify REVERSAL
- modify STRENGTH
- modify V40.62 or V40.63

Two complementary analyses are produced.

A) TRUE-TREND PRE-T0 EVENT STUDY
   Descriptive coverage of the technical observations preceding
   completed TRUE trends.

B) N8/K3 TRUE-vs-FAIL OOS CLASSIFICATION
   For movements preceded by >=8 weeks of opposite SAR regime:
       TRUE_TREND_N8_K3       -> y = 1
       RETRACEMENT_N8_K3_FAIL -> y = 0

   The classifier is evaluated at exact relative positions around
   movement T0.

IMPORTANT:
K3 is used only to establish the historical outcome label.
K3 status, confirmation date/week, future economic endpoint and all
other future-derived fields are NEVER model features.

FEATURES
--------
Exactly the 413 frozen V40.39 features listed in:
    data/v40_39_feature_list.csv

Preprocessing is TRAIN-ONLY, following V40.58 discipline:
- >=20 finite TRAIN observations
- TRAIN medians
- remove TRAIN-constant features
- TRAIN-only abs(Spearman) screening
- max 160 selected features

OOS
---
F1 2022
F2 2023
F3 2024
F4 2025
F5 2026-01-01 -> 2026-08-28

For an event to enter TRAIN, its TRUE/FAIL outcome must already be
known before the beginning of the OOS TEST window.

No final methodological threshold is selected by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
)


# =============================================================================
# CONFIG
# =============================================================================

VERSION = "V40.64"

FEATURE_FILE = Path(
    "data/v40_39_full200_features.csv"
)

FEATURE_LIST_FILE = Path(
    "data/v40_39_feature_list.csv"
)

SAR_FILE = Path(
    "data/v40_62_opportunity_true_trend_target_audit/"
    "v40_62_sar_movements.csv"
)

OUTDIR = Path(
    "data/v40_64_opportunity_pret0_predictability_audit"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

EVENT_FILE = (
    OUTDIR
    / "v40_64_n8_event_dataset.csv"
)

RESULT_FILE = (
    OUTDIR
    / "v40_64_oos_results.csv"
)

PREDICTION_FILE = (
    OUTDIR
    / "v40_64_oos_predictions.csv"
)

SELECTED_FILE = (
    OUTDIR
    / "v40_64_selected_features.csv"
)

DISTANCE_FILE = (
    OUTDIR
    / "v40_64_distance_coverage.csv"
)

AUDIT_FILE = (
    OUTDIR
    / "v40_64_structural_audit.csv"
)

METADATA_FILE = (
    OUTDIR
    / "v40_64_metadata.json"
)


MAX_SELECTED_FEATURES = 160

MIN_FINITE_TRAIN = 20

MIN_TRAIN_ROWS = 1000

RANDOM_STATE = 42


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


# Exact relative positions to movement T0.
#
# Negative = BEFORE T0
# Zero     = T0 itself
# Positive = AFTER T0
#
# This is an audit grid, NOT an Opportunity horizon.
RELATIVE_WEEKS = list(
    range(
        -40,
        6,
    )
)


TRUE_CLASS = (
    "TRUE_TREND_N8_K3"
)

FAIL_CLASS = (
    "RETRACEMENT_N8_K3_FAIL"
)


# =============================================================================
# HELPERS
# =============================================================================

def safe_spearman(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if mask.sum() < 3:
        return np.nan

    xx = x[mask]
    yy = y[mask]

    if (
        np.unique(xx).size < 2
        or np.unique(yy).size < 2
    ):
        return np.nan

    try:
        r = spearmanr(
            xx,
            yy,
        ).statistic
    except Exception:
        return np.nan

    return (
        float(r)
        if np.isfinite(r)
        else np.nan
    )


def add_check(
    rows,
    name,
    value,
    expected,
    passed,
):
    rows.append(
        {
            "check": name,
            "value": value,
            "expected": expected,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
        }
    )


def safe_auc(
    y,
    p,
):
    y = np.asarray(y)
    p = np.asarray(p)

    if np.unique(y).size < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def safe_ap(
    y,
    p,
):
    y = np.asarray(y)
    p = np.asarray(p)

    if np.unique(y).size < 2:
        return np.nan

    return float(
        average_precision_score(
            y,
            p,
        )
    )


# =============================================================================
# LOAD
# =============================================================================

def load_sources():

    for p in [
        FEATURE_FILE,
        FEATURE_LIST_FILE,
        SAR_FILE,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing source: {p}"
            )

    feature_list = pd.read_csv(
        FEATURE_LIST_FILE
    )

    if (
        list(
            feature_list.columns
        )
        != [
            "Feature",
            "FeatureIndex",
        ]
    ):
        raise RuntimeError(
            "Unexpected V40.39 "
            "feature-list schema."
        )

    feature_list = (
        feature_list
        .sort_values(
            "FeatureIndex"
        )
        .reset_index(
            drop=True
        )
    )

    feature_cols = (
        feature_list[
            "Feature"
        ]
        .astype(str)
        .tolist()
    )

    if len(feature_cols) != 413:
        raise RuntimeError(
            "Frozen V40.39 feature "
            f"count != 413: "
            f"{len(feature_cols)}"
        )

    header = pd.read_csv(
        FEATURE_FILE,
        nrows=0,
    ).columns.tolist()

    required = [
        "Ticker",
        "Date",
    ]

    missing_keys = [
        c
        for c in required
        if c not in header
    ]

    if missing_keys:
        raise RuntimeError(
            f"Missing keys in feature file: "
            f"{missing_keys}"
        )

    missing_features = [
        c
        for c in feature_cols
        if c not in header
    ]

    if missing_features:
        raise RuntimeError(
            "Frozen feature list not fully "
            "present in V40.39 feature file. "
            f"Missing: "
            f"{missing_features[:20]}"
        )

    usecols = (
        [
            "Ticker",
            "Date",
        ]
        + feature_cols
    )

    print(
        "\nLoading frozen V40.39 "
        "feature matrix..."
    )

    features = pd.read_csv(
        FEATURE_FILE,
        usecols=usecols,
    )

    features["Date"] = pd.to_datetime(
        features["Date"]
    )

    features = (
        features
        .sort_values(
            [
                "Ticker",
                "Date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    sar = pd.read_csv(
        SAR_FILE
    )

    for c in [
        "sar_start_date",
        "sar_run_end_date",
        "k3_confirmation_date",
    ]:
        if c in sar.columns:
            sar[c] = pd.to_datetime(
                sar[c],
                errors="coerce",
            )

    sar = (
        sar
        .sort_values(
            [
                "Ticker",
                "sar_start_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        features,
        sar,
        feature_cols,
    )


# =============================================================================
# EVENT OUTCOME KNOWLEDGE DATE
# =============================================================================

def add_outcome_known_date(
    sar,
):
    """
    Historical label knowledge date.

    TRUE N8/K3:
        once K3 has occurred, TRUE-vs-FAIL status is known.

    FAIL N8/K3:
        failure is known only when the SAR run ends without K3.

    This field is used ONLY for causal TRAIN eligibility.
    It is never a model feature.
    """

    x = sar[
        sar[
            "trend_classification"
        ].isin(
            [
                TRUE_CLASS,
                FAIL_CLASS,
            ]
        )
    ].copy()

    x["label"] = np.where(
        x[
            "trend_classification"
        ]
        == TRUE_CLASS,
        1,
        0,
    ).astype(int)

    x["outcome_known_date"] = (
        pd.NaT
    )

    true_mask = (
        x[
            "trend_classification"
        ]
        == TRUE_CLASS
    )

    fail_mask = (
        x[
            "trend_classification"
        ]
        == FAIL_CLASS
    )

    x.loc[
        true_mask,
        "outcome_known_date",
    ] = x.loc[
        true_mask,
        "k3_confirmation_date",
    ]

    x.loc[
        fail_mask,
        "outcome_known_date",
    ] = x.loc[
        fail_mask,
        "sar_run_end_date",
    ]

    x[
        "outcome_known_date"
    ] = pd.to_datetime(
        x[
            "outcome_known_date"
        ],
        errors="coerce",
    )

    return x


# =============================================================================
# BUILD EXACT RELATIVE-WEEK EVENT DATASET
# =============================================================================

def build_distance_dataset(
    features,
    events,
    feature_cols,
    relative_week,
):
    """
    Build ONLY one exact relative-week slice.

    Semantics are identical to the PRE_STREAMING implementation:
    - same N8 events
    - same SAR movement T0
    - same positional weekly lookup
    - same 413 frozen V40.39 features

    The only change is memory architecture: one relative_week at a time.
    """

    parts = []

    rel = int(relative_week)

    event_groups = {
        ticker: g.copy()
        for ticker, g in events.groupby("Ticker", sort=False)
    }

    for ticker, g0 in features.groupby("Ticker", sort=False):

        if ticker not in event_groups:
            continue

        g = g0.sort_values("Date").reset_index(drop=True)
        ev = (
            event_groups[ticker]
            .sort_values("sar_start_date")
            .reset_index(drop=True)
        )

        dates = g["Date"].to_numpy(dtype="datetime64[ns]")

        date_to_idx = {
            d: i
            for i, d in enumerate(dates)
        }

        feature_matrix = (
            g[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )

        meta_rows = []
        feature_idx = []

        for r in ev.itertuples(index=False):

            t0 = np.datetime64(r.sar_start_date)

            if t0 not in date_to_idx:
                raise RuntimeError(
                    f"T0 not found: {ticker} {r.sar_start_date}"
                )

            t0_idx = int(date_to_idx[t0])
            j = t0_idx + rel

            if j < 0 or j >= len(g):
                continue

            meta_rows.append(
                {
                    "Ticker": ticker,
                    "direction": r.direction,
                    "trend_classification": r.trend_classification,
                    "label": int(r.label),
                    "sar_start_date": r.sar_start_date,
                    "outcome_known_date": r.outcome_known_date,
                    "previous_sar_age": r.previous_sar_age,
                    "relative_week": rel,
                    "observation_date": g.at[j, "Date"],
                }
            )

            feature_idx.append(j)

        if meta_rows:

            meta = pd.DataFrame(meta_rows)

            feat = pd.DataFrame(
                feature_matrix[
                    np.asarray(feature_idx, dtype=np.int32)
                ],
                columns=feature_cols,
            )

            parts.append(
                pd.concat(
                    [
                        meta.reset_index(drop=True),
                        feat.reset_index(drop=True),
                    ],
                    axis=1,
                )
            )

    if not parts:
        raise RuntimeError(
            f"No rows constructed for relative_week={rel}"
        )

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    for c in [
        "sar_start_date",
        "observation_date",
        "outcome_known_date",
    ]:
        out[c] = pd.to_datetime(out[c])

    return out


# =============================================================================
# DISTANCE COVERAGE
# =============================================================================

def make_distance_coverage(
    event_data,
):
    rows = []

    for (
        direction,
        rel,
        label,
    ), g in event_data.groupby(
        [
            "direction",
            "relative_week",
            "label",
        ],
        sort=True,
    ):

        rows.append(
            {
                "direction": direction,
                "relative_week": int(
                    rel
                ),
                "label": int(
                    label
                ),
                "class": (
                    "TRUE"
                    if int(label) == 1
                    else "FAIL"
                ),
                "N": int(
                    len(g)
                ),
                "unique_events": int(
                    g[
                        [
                            "Ticker",
                            "sar_start_date",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# TRAIN-ONLY PREPROCESSING
# =============================================================================

def train_only_prepare(
    train,
    test,
    feature_cols,
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
        train[
            "label"
        ]
        .astype(int)
        .to_numpy()
    )

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
        finite_count
        >= MIN_FINITE_TRAIN
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

    if not usable_cols:
        raise RuntimeError(
            "No usable TRAIN features."
        )

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

    if not nonconstant_cols:
        raise RuntimeError(
            "No nonconstant TRAIN features."
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

    ranking = []

    for col in nonconstant_cols:

        rho = safe_spearman(
            X_train[
                col
            ].to_numpy(
                dtype=float
            ),
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

    selected = ranking[
        :MAX_SELECTED_FEATURES
    ]

    selected_cols = [
        z[0]
        for z in selected
    ]

    if not selected_cols:
        raise RuntimeError(
            "No selectable features."
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


# =============================================================================
# MODELS
# =============================================================================

def model_factories():
    """
    Small OOS model comparison.

    The winning model is NOT assumed a priori.
    """

    return {
        "RF": lambda: RandomForestClassifier(
            n_estimators=320,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),

        "EXTRA_TREES": lambda: ExtraTreesClassifier(
            n_estimators=320,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),

        "HIST_GB": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


# =============================================================================
# OOS AUDIT
# =============================================================================

def run_oos(
    event_data,
    feature_cols,
    relative_week,
):

    result_rows = []
    prediction_rows = []
    selected_rows = []

    factories = (
        model_factories()
    )

    jobs = []

    rel = int(relative_week)

    for direction in [
        "BULL",
        "BEAR",
    ]:

        for (
            fold,
            test_start_s,
            test_end_s,
        ) in OOS_WINDOWS:

            for model_name in (
                factories.keys()
            ):

                jobs.append(
                    (
                        direction,
                        rel,
                        fold,
                        test_start_s,
                        test_end_s,
                        model_name,
                    )
                )

    total_jobs = len(
        jobs
    )

    print(
        f"\nOOS jobs: "
        f"{total_jobs:,}"
    )

    done = 0

    for (
        direction,
        rel,
        fold,
        test_start_s,
        test_end_s,
        model_name,
    ) in jobs:

        done += 1

        dd = event_data[
            (
                event_data[
                    "direction"
                ]
                == direction
            )
            & (
                event_data[
                    "relative_week"
                ]
                == rel
            )
        ].copy()

        test_start = pd.Timestamp(
            test_start_s
        )

        test_end = pd.Timestamp(
            test_end_s
        )

        # ---------------------------------------------------------
        # TEST:
        # event T0 begins inside the OOS calendar window.
        #
        # The feature observation can be before T0 because rel may
        # be negative. Assignment to TEST is by EVENT start date,
        # preventing the same event from crossing TRAIN/TEST.
        # ---------------------------------------------------------

        test = dd[
            (
                dd[
                    "sar_start_date"
                ]
                >= test_start
            )
            & (
                dd[
                    "sar_start_date"
                ]
                <= test_end
            )
        ].copy()

        # ---------------------------------------------------------
        # TRAIN:
        # TRUE/FAIL label must have become fully known before the
        # beginning of the OOS period.
        # ---------------------------------------------------------

        train = dd[
            dd[
                "outcome_known_date"
            ]
            < test_start
        ].copy()

        status = "OK"

        if len(train) < MIN_TRAIN_ROWS:
            status = (
                "SKIP_MIN_TRAIN"
            )

        elif len(test) == 0:
            status = (
                "SKIP_NO_TEST"
            )

        elif (
            train[
                "label"
            ].nunique()
            < 2
        ):
            status = (
                "SKIP_TRAIN_ONE_CLASS"
            )

        elif (
            test[
                "label"
            ].nunique()
            < 2
        ):
            status = (
                "SKIP_TEST_ONE_CLASS"
            )

        if status != "OK":

            result_rows.append(
                {
                    "direction": direction,
                    "relative_week": rel,
                    "fold": fold,
                    "model": model_name,
                    "status": status,
                    "train_N": len(
                        train
                    ),
                    "test_N": len(
                        test
                    ),
                    "train_positive_rate": (
                        train[
                            "label"
                        ].mean()
                        if len(train)
                        else np.nan
                    ),
                    "test_positive_rate": (
                        test[
                            "label"
                        ].mean()
                        if len(test)
                        else np.nan
                    ),
                }
            )

            continue

        (
            X_train,
            X_test,
            selected,
            n_input,
            n_usable,
            n_nonconstant,
        ) = train_only_prepare(
            train,
            test,
            feature_cols,
        )

        y_train = (
            train[
                "label"
            ]
            .astype(int)
            .to_numpy()
        )

        y_test = (
            test[
                "label"
            ]
            .astype(int)
            .to_numpy()
        )

        model = factories[
            model_name
        ]()

        model.fit(
            X_train,
            y_train,
        )

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise RuntimeError(
                f"{model_name} has no "
                "predict_proba."
            )

        p = model.predict_proba(
            X_test
        )[:, 1]

        pred_class = (
            p >= 0.5
        ).astype(int)

        auc = safe_auc(
            y_test,
            p,
        )

        ap = safe_ap(
            y_test,
            p,
        )

        bal_acc = float(
            balanced_accuracy_score(
                y_test,
                pred_class,
            )
        )

        brier = float(
            brier_score_loss(
                y_test,
                p,
            )
        )

        result_rows.append(
            {
                "direction": direction,
                "relative_week": rel,
                "fold": fold,
                "model": model_name,
                "status": "OK",
                "train_N": int(
                    len(train)
                ),
                "test_N": int(
                    len(test)
                ),
                "train_positive_rate": float(
                    train[
                        "label"
                    ].mean()
                ),
                "test_positive_rate": float(
                    test[
                        "label"
                    ].mean()
                ),
                "roc_auc": auc,
                "average_precision": ap,
                "balanced_accuracy": (
                    bal_acc
                ),
                "brier": brier,
                "input_features": (
                    n_input
                ),
                "usable_features": (
                    n_usable
                ),
                "nonconstant_features": (
                    n_nonconstant
                ),
                "selected_features": (
                    len(selected)
                ),
            }
        )

        for rank, (
            feature,
            rho,
            abs_rho,
        ) in enumerate(
            selected,
            start=1,
        ):

            selected_rows.append(
                {
                    "direction": direction,
                    "relative_week": rel,
                    "fold": fold,
                    "model": model_name,
                    "rank": rank,
                    "feature": feature,
                    "train_spearman": rho,
                    "train_abs_spearman": (
                        abs_rho
                    ),
                }
            )

        test_reset = (
            test.reset_index(
                drop=True
            )
        )

        for i in range(
            len(test_reset)
        ):

            prediction_rows.append(
                {
                    "Ticker": (
                        test_reset.at[
                            i,
                            "Ticker",
                        ]
                    ),
                    "direction": direction,
                    "relative_week": rel,
                    "fold": fold,
                    "model": model_name,
                    "sar_start_date": (
                        test_reset.at[
                            i,
                            "sar_start_date",
                        ]
                    ),
                    "observation_date": (
                        test_reset.at[
                            i,
                            "observation_date",
                        ]
                    ),
                    "trend_classification": (
                        test_reset.at[
                            i,
                            "trend_classification",
                        ]
                    ),
                    "label": int(
                        y_test[i]
                    ),
                    "prob_true": float(
                        p[i]
                    ),
                }
            )

        if (
            done % 100 == 0
            or done == total_jobs
        ):
            print(
                f"  {done:,}/"
                f"{total_jobs:,} jobs"
            )

    return (
        pd.DataFrame(
            result_rows
        ),
        pd.DataFrame(
            prediction_rows
        ),
        pd.DataFrame(
            selected_rows
        ),
    )


# =============================================================================
# STRUCTURAL AUDIT
# =============================================================================

def structural_audit(
    features,
    events,
    event_data,
    feature_cols,
):

    rows = []

    add_check(
        rows,
        "frozen_feature_count",
        len(feature_cols),
        413,
        len(feature_cols) == 413,
    )

    add_check(
        rows,
        "feature_source_tickers",
        features[
            "Ticker"
        ].nunique(),
        200,
        features[
            "Ticker"
        ].nunique()
        == 200,
    )

    event_dup = int(
        events.duplicated(
            [
                "Ticker",
                "sar_start_date",
            ]
        ).sum()
    )

    add_check(
        rows,
        "n8_event_key_duplicates",
        event_dup,
        0,
        event_dup == 0,
    )

    counts = (
        events.groupby(
            [
                "direction",
                "trend_classification",
            ]
        )
        .size()
    )

    expected = {
        (
            "BULL",
            TRUE_CLASS,
        ): 2638,
        (
            "BULL",
            FAIL_CLASS,
        ): 857,
        (
            "BEAR",
            TRUE_CLASS,
        ): 2678,
        (
            "BEAR",
            FAIL_CLASS,
        ): 1898,
    }

    for key, exp in (
        expected.items()
    ):

        actual = int(
            counts.get(
                key,
                0,
            )
        )

        name = (
            f"{key[0].lower()}_"
            f"{'true' if key[1] == TRUE_CLASS else 'fail'}"
            "_events"
        )

        add_check(
            rows,
            name,
            actual,
            exp,
            actual == exp,
        )

    bad_prev_age = int(
        (
            pd.to_numeric(
                events[
                    "previous_sar_age"
                ],
                errors="coerce",
            )
            < 8
        ).sum()
    )

    add_check(
        rows,
        "n8_event_previous_age_lt8",
        bad_prev_age,
        0,
        bad_prev_age == 0,
    )

    missing_known = int(
        events[
            "outcome_known_date"
        ]
        .isna()
        .sum()
    )

    add_check(
        rows,
        "missing_outcome_known_date",
        missing_known,
        0,
        missing_known == 0,
    )

    true_events = events[
        events[
            "label"
        ]
        == 1
    ]

    bad_true_known = int(
        (
            true_events[
                "outcome_known_date"
            ]
            != true_events[
                "k3_confirmation_date"
            ]
        ).sum()
    )

    add_check(
        rows,
        "true_label_known_not_k3_date",
        bad_true_known,
        0,
        bad_true_known == 0,
    )

    fail_events = events[
        events[
            "label"
        ]
        == 0
    ]

    bad_fail_known = int(
        (
            fail_events[
                "outcome_known_date"
            ]
            != fail_events[
                "sar_run_end_date"
            ]
        ).sum()
    )

    add_check(
        rows,
        "fail_label_known_not_run_end",
        bad_fail_known,
        0,
        bad_fail_known == 0,
    )

    key_dup = int(
        event_data.duplicated(
            [
                "Ticker",
                "sar_start_date",
                "relative_week",
            ]
        ).sum()
    )

    add_check(
        rows,
        "event_relative_key_duplicates",
        key_dup,
        0,
        key_dup == 0,
    )

    rel_values = (
        event_data["relative_week"]
        .dropna()
        .astype(int)
        .unique()
    )

    add_check(
        rows,
        "single_relative_week_slice",
        len(rel_values),
        1,
        len(rel_values) == 1,
    )

    if len(rel_values) == 1:

        rel = int(rel_values[0])

        delta_days = (
            (
                event_data["observation_date"]
                - event_data["sar_start_date"]
            )
            .dt.days
        )

        expected_days = rel * 7

        bad = int(
            (
                delta_days
                != expected_days
            ).sum()
        )

        add_check(
            rows,
            f"relative_week_{rel}_"
            "not_exact_7day_spacing",
            bad,
            0,
            bad == 0,
        )

        if rel == 0:

            bad_rel0_count = int(
                (
                    event_data["observation_date"]
                    != event_data["sar_start_date"]
                ).sum()
            )

            add_check(
                rows,
                "relative_week_0_not_T0",
                bad_rel0_count,
                0,
                bad_rel0_count == 0,
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# RESULT SUMMARY
# =============================================================================

def print_oos_summary(
    results,
):

    ok = results[
        results[
            "status"
        ]
        == "OK"
    ].copy()

    if len(ok) == 0:
        print(
            "\nNO COMPLETED OOS JOBS."
        )
        return

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OOS MODEL SUMMARY"
    )

    print(
        "=" * 80
    )

    s = (
        ok.groupby(
            [
                "direction",
                "relative_week",
                "model",
            ]
        )
        .agg(
            folds=(
                "fold",
                "nunique",
            ),
            auc_mean=(
                "roc_auc",
                "mean",
            ),
            auc_min=(
                "roc_auc",
                "min",
            ),
            ap_mean=(
                "average_precision",
                "mean",
            ),
            bal_acc_mean=(
                "balanced_accuracy",
                "mean",
            ),
            brier_mean=(
                "brier",
                "mean",
            ),
        )
        .reset_index()
    )

    # Compact report around T0 plus selected earlier positions.
    report_weeks = {
        -40,
        -30,
        -20,
        -15,
        -12,
        -10,
        -8,
        -6,
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
    }

    report = s[
        s[
            "relative_week"
        ].isin(
            report_weeks
        )
    ]

    print(
        report.to_string(
            index=False
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("MARKETSENTINEL V40.64")
    print("OPPORTUNITY PRE-T0 PREDICTABILITY AUDIT")
    print("STREAMING + CHECKPOINT/RESUME")
    print("=" * 80)

    checkpoint_file = OUTDIR / "v40_64_checkpoint.json"

    (
        features,
        sar,
        feature_cols,
    ) = load_sources()

    print(f"FEATURE ROWS: {len(features):,}")
    print(f"TICKERS: {features['Ticker'].nunique():,}")
    print(f"FROZEN FEATURES: {len(feature_cols):,}")

    events = add_outcome_known_date(sar)

    print(f"\nN8 EVENTS: {len(events):,}")
    print(
        events.groupby(
            ["direction", "trend_classification"]
        ).size().to_string()
    )

    # -------------------------------------------------------------
    # Small event manifest only.
    # We deliberately DO NOT save the 413-feature matrix for all
    # event-distance combinations.
    # -------------------------------------------------------------

    event_manifest_cols = [
        "Ticker",
        "direction",
        "trend_classification",
        "label",
        "sar_start_date",
        "outcome_known_date",
        "previous_sar_age",
        "k3_confirmation_date",
        "sar_run_end_date",
    ]

    events[
        event_manifest_cols
    ].to_csv(
        EVENT_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # RESUME STATE
    # -------------------------------------------------------------

    completed = []

    if checkpoint_file.exists():

        with open(
            checkpoint_file,
            "r",
            encoding="utf-8",
        ) as f:
            cp = json.load(f)

        completed = [
            int(x)
            for x in cp.get(
                "completed_relative_weeks",
                [],
            )
        ]

        completed = sorted(
            set(completed)
        )

        print(
            "\nRESUME: completed relative weeks = "
            f"{completed}"
        )

    else:
        print("\nRESUME: no previous checkpoint.")

    # -------------------------------------------------------------
    # Existing accumulated outputs, if any.
    # -------------------------------------------------------------

    def read_existing(path):
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
        return pd.DataFrame()

    all_results = read_existing(RESULT_FILE)
    all_predictions = read_existing(PREDICTION_FILE)
    all_selected = read_existing(SELECTED_FILE)
    all_coverage = read_existing(DISTANCE_FILE)
    all_audit = read_existing(AUDIT_FILE)

    total_event_distance_rows = 0

    if not all_coverage.empty and "N" in all_coverage.columns:
        # Coverage has one row per direction / distance / label.
        # Sum N to recover exact already-materialized row count.
        total_event_distance_rows = int(
            pd.to_numeric(
                all_coverage["N"],
                errors="coerce",
            ).fillna(0).sum()
        )

    # -------------------------------------------------------------
    # STREAM ONE RELATIVE WEEK AT A TIME
    # -------------------------------------------------------------

    for pos, rel in enumerate(
        RELATIVE_WEEKS,
        start=1,
    ):

        rel = int(rel)

        if rel in completed:
            print(
                f"\n[{pos}/{len(RELATIVE_WEEKS)}] "
                f"relative_week={rel:+d} "
                "already complete -> SKIP"
            )
            continue

        print("\n" + "=" * 80)
        print(
            f"[{pos}/{len(RELATIVE_WEEKS)}] "
            f"RELATIVE WEEK {rel:+d}"
        )
        print("=" * 80)

        event_data = build_distance_dataset(
            features,
            events,
            feature_cols,
            rel,
        )

        print(
            f"SLICE ROWS: {len(event_data):,}"
        )

        # ---------------------------------------------------------
        # STRUCTURAL AUDIT FOR THIS EXACT DISTANCE
        # ---------------------------------------------------------

        audit = structural_audit(
            features,
            events,
            event_data,
            feature_cols,
        )

        audit.insert(
            0,
            "relative_week",
            rel,
        )

        all_pass = bool(
            (audit["status"] == "PASS").all()
        )

        print("\nSTRUCTURAL AUDIT")
        print(audit.to_string(index=False))

        if not all_pass:
            print(
                "\nFINAL RESULT: FAIL "
                f"AT RELATIVE WEEK {rel:+d}"
            )
            print("DO NOT CONTINUE OOS MODELS.")

            # Remove an eventual previous partial copy of this rel.
            if (
                not all_audit.empty
                and "relative_week" in all_audit.columns
            ):
                all_audit = all_audit[
                    pd.to_numeric(
                        all_audit["relative_week"],
                        errors="coerce",
                    ) != rel
                ]

            all_audit = pd.concat(
                [all_audit, audit],
                ignore_index=True,
            )

            all_audit.to_csv(
                AUDIT_FILE,
                index=False,
            )

            return

        coverage = make_distance_coverage(
            event_data
        )

        # ---------------------------------------------------------
        # OOS: 2 directions x 5 folds x 3 models = 30 jobs.
        # ---------------------------------------------------------

        (
            results,
            predictions,
            selected,
        ) = run_oos(
            event_data,
            feature_cols,
            rel,
        )

        # ---------------------------------------------------------
        # IDEMPOTENT REPLACEMENT:
        # if a previous attempt died before checkpointing, remove
        # any partial rows for this distance before appending.
        # ---------------------------------------------------------

        def remove_rel(df):
            if (
                df.empty
                or "relative_week" not in df.columns
            ):
                return df

            return df[
                pd.to_numeric(
                    df["relative_week"],
                    errors="coerce",
                ) != rel
            ].copy()

        all_results = remove_rel(all_results)
        all_predictions = remove_rel(all_predictions)
        all_selected = remove_rel(all_selected)
        all_coverage = remove_rel(all_coverage)
        all_audit = remove_rel(all_audit)

        all_results = pd.concat(
            [all_results, results],
            ignore_index=True,
        )

        all_predictions = pd.concat(
            [all_predictions, predictions],
            ignore_index=True,
        )

        all_selected = pd.concat(
            [all_selected, selected],
            ignore_index=True,
        )

        all_coverage = pd.concat(
            [all_coverage, coverage],
            ignore_index=True,
        )

        all_audit = pd.concat(
            [all_audit, audit],
            ignore_index=True,
        )

        # ---------------------------------------------------------
        # SAVE ALL ACCUMULATED OUTPUTS BEFORE CHECKPOINT.
        # ---------------------------------------------------------

        all_results.to_csv(
            RESULT_FILE,
            index=False,
        )

        all_predictions.to_csv(
            PREDICTION_FILE,
            index=False,
        )

        all_selected.to_csv(
            SELECTED_FILE,
            index=False,
        )

        all_coverage.to_csv(
            DISTANCE_FILE,
            index=False,
        )

        all_audit.to_csv(
            AUDIT_FILE,
            index=False,
        )

        # ---------------------------------------------------------
        # CHECKPOINT ONLY AFTER ALL OUTPUTS ARE SAFELY WRITTEN.
        # ---------------------------------------------------------

        completed.append(rel)
        completed = sorted(set(completed))

        with open(
            checkpoint_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "version": VERSION,
                    "completed_relative_weeks": completed,
                    "last_completed_relative_week": rel,
                },
                f,
                indent=2,
            )

        total_event_distance_rows = int(
            pd.to_numeric(
                all_coverage["N"],
                errors="coerce",
            ).fillna(0).sum()
        )

        print(
            f"\nCHECKPOINT SAVED: "
            f"relative_week={rel:+d}"
        )

        print(
            f"COMPLETED DISTANCES: "
            f"{len(completed)}/"
            f"{len(RELATIVE_WEEKS)}"
        )

        # Explicitly release the large slice before next distance.
        del event_data
        del coverage
        del audit
        del results
        del predictions
        del selected

    # -------------------------------------------------------------
    # FINAL VALIDATION
    # -------------------------------------------------------------

    expected_rel = sorted(
        int(x)
        for x in RELATIVE_WEEKS
    )

    completed = sorted(
        set(
            int(x)
            for x in completed
        )
    )

    complete_grid = (
        completed == expected_rel
    )

    if not complete_grid:
        print(
            "\nRUN INCOMPLETE: "
            f"{len(completed)}/"
            f"{len(expected_rel)} distances."
        )
        print(
            "Re-run the same command to resume."
        )
        return

    # Reload final persisted outputs so final summary reflects disk.
    results = pd.read_csv(
        RESULT_FILE
    )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    selected = pd.read_csv(
        SELECTED_FILE
    )

    coverage = pd.read_csv(
        DISTANCE_FILE
    )

    audit = pd.read_csv(
        AUDIT_FILE
    )

    final_audit_pass = bool(
        (audit["status"] == "PASS").all()
    )

    ok = results[
        results["status"] == "OK"
    ].copy()

    print_oos_summary(
        results
    )

    total_event_distance_rows = int(
        pd.to_numeric(
            coverage["N"],
            errors="coerce",
        ).fillna(0).sum()
    )

    metadata = {
        "version": VERSION,
        "execution_architecture": (
            "STREAMING_ONE_RELATIVE_WEEK_AT_A_TIME"
        ),
        "checkpoint_resume": True,
        "feature_source": str(FEATURE_FILE),
        "feature_list_source": str(
            FEATURE_LIST_FILE
        ),
        "sar_source": str(SAR_FILE),
        "frozen_feature_count": int(
            len(feature_cols)
        ),
        "max_selected_features": (
            MAX_SELECTED_FEATURES
        ),
        "relative_week_min": int(
            min(RELATIVE_WEEKS)
        ),
        "relative_week_max": int(
            max(RELATIVE_WEEKS)
        ),
        "relative_weeks_completed": int(
            len(completed)
        ),
        "n8_events": int(
            len(events)
        ),
        "event_distance_rows": int(
            total_event_distance_rows
        ),
        "oos_jobs_total": int(
            len(results)
        ),
        "oos_jobs_ok": int(
            len(ok)
        ),
        "models": list(
            model_factories().keys()
        ),
        "final_opportunity_target_created": False,
        "opportunity_score_created": False,
        "opportunity_window_selected": False,
        "fixed_return_horizon_used": False,
        "k3_used_as_model_feature": False,
        "future_endpoint_used_as_model_feature": False,
        "train_only_preprocessing": True,
        "causal_label_availability_train_filter": True,
        "phase_modified": False,
        "reversal_modified": False,
        "strength_modified": False,
        "v40_62_modified": False,
        "v40_63_modified": False,
        "structural_audit_pass": (
            final_audit_pass
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        "V40.64 STRUCTURAL AUDIT: "
        + (
            "PASS"
            if final_audit_pass
            else "FAIL"
        )
    )

    print(
        f"RELATIVE WEEKS COMPLETED: "
        f"{len(completed)}/"
        f"{len(RELATIVE_WEEKS)}"
    )

    print(
        f"OOS JOBS RECORDED: "
        f"{len(results):,}"
    )

    print(
        f"OOS JOBS OK: "
        f"{len(ok):,}"
    )

    print(
        "\nNo final Opportunity target, "
        "window or score has been selected."
    )

    print("\nOutputs:")
    print(f"  {EVENT_FILE}")
    print(f"  {RESULT_FILE}")
    print(f"  {PREDICTION_FILE}")
    print(f"  {SELECTED_FILE}")
    print(f"  {DISTANCE_FILE}")
    print(f"  {AUDIT_FILE}")
    print(f"  {METADATA_FILE}")
    print(f"  {checkpoint_file}")


if __name__ == "__main__":
    main()