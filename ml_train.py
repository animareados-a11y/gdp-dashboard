import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# MARKET SENTINEL
# MACHINE LEARNING TRAINER V1
# ============================================================
#
# PRIMO VERO MODELLO MACHINE LEARNING
#
# Obiettivo:
#
# BUY MODEL
#   Prevedere se nelle prossime 15 sedute
#   il titolo salirà almeno del 2%.
#
# SELL MODEL
#   Prevedere se nelle prossime 15 sedute
#   il titolo scenderà almeno del 2%.
#
# IMPORTANTISSIMO:
#
# Il dataset viene separato CRONOLOGICAMENTE.
#
# Il modello impara dal passato.
# Il futuro rimane completamente fuori
# dal training.
#
# ============================================================


warnings.filterwarnings(
    "ignore"
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

DATASET_FILE = os.path.join(
    "data",
    "ml_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    "data",
    "ml_models"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# ORIZZONTE PRINCIPALE
# ------------------------------------------------------------

HORIZON = 15


BUY_TARGET = (
    f"target_up_{HORIZON}d"
)

SELL_TARGET = (
    f"target_down_{HORIZON}d"
)


# ------------------------------------------------------------
# TRAIN / TEST
#
# Circa 75% del periodo = apprendimento
# Ultimo 25% = esame fuori campione
# ------------------------------------------------------------

TEST_FRACTION = 0.25


# ------------------------------------------------------------
# PURGE GAP
#
# Creiamo una zona vuota fra TRAIN e TEST.
#
# Serve perché il target a 15 sedute
# utilizza dati successivi alla data
# della fotografia.
#
# In questo modo evitiamo che il TRAIN
# "sconfini" nel periodo TEST.
# ------------------------------------------------------------

PURGE_CALENDAR_DAYS = 35


RANDOM_STATE = 42


# ------------------------------------------------------------
# FILTRI FEATURES
# ------------------------------------------------------------

# Scartiamo caratteristiche con
# più del 55% di valori mancanti.

MAX_MISSING_RATIO = 0.55


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(
        DATASET_FILE
    ):

        raise FileNotFoundError(
            f"Non trovo il dataset: "
            f"{DATASET_FILE}"
        )

    df = pd.read_csv(
        DATASET_FILE
    )

    if df.empty:

        raise RuntimeError(
            "Il dataset ML è vuoto."
        )

    required = [
        "ticker",
        "market",
        "date",
        BUY_TARGET,
        SELL_TARGET,
    ]

    missing = [
        column
        for column
        in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Nel dataset mancano colonne: "
            +
            ", ".join(
                missing
            )
        )

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date"
        ]
    )

    df = (
        df
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(
    df
):
    """
    Selezioniamo SOLO informazioni
    disponibili al momento della previsione.

    Qualsiasi colonna target_* viene esclusa.
    """

    excluded = {
        "ticker",
        "market",
        "date",
    }

    candidates = []

    for column in df.columns:

        if column in excluded:
            continue

        if column.startswith(
            "target_"
        ):
            continue

        candidates.append(
            column
        )

    # --------------------------------------------------------
    # CONVERSIONE NUMERICA
    # --------------------------------------------------------

    for column in candidates:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce"
        )

    selected = []

    dropped_missing = []

    dropped_constant = []

    for column in candidates:

        missing_ratio = (
            df[
                column
            ]
            .isna()
            .mean()
        )

        if (
            missing_ratio
            >
            MAX_MISSING_RATIO
        ):

            dropped_missing.append(
                column
            )

            continue

        unique_values = (
            df[
                column
            ]
            .dropna()
            .nunique()
        )

        if unique_values <= 1:

            dropped_constant.append(
                column
            )

            continue

        selected.append(
            column
        )

    return (
        selected,
        dropped_missing,
        dropped_constant,
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df
):
    """
    NON facciamo uno split casuale.

    Il modello impara sul periodo vecchio
    e viene esaminato sul periodo successivo.
    """

    unique_dates = (
        df[
            "date"
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    if len(
        unique_dates
    ) < 20:

        raise RuntimeError(
            "Numero di date insufficiente."
        )

    split_position = int(
        len(
            unique_dates
        )
        *
        (
            1
            -
            TEST_FRACTION
        )
    )

    split_position = min(
        max(
            split_position,
            1
        ),
        len(
            unique_dates
        )
        -
        1
    )

    test_start = (
        unique_dates.iloc[
            split_position
        ]
    )

    train_end = (
        test_start
        -
        pd.Timedelta(
            days=
                PURGE_CALENDAR_DAYS
        )
    )

    train = df[
        df[
            "date"
        ]
        <
        train_end
    ].copy()

    test = df[
        df[
            "date"
        ]
        >=
        test_start
    ].copy()

    if train.empty:

        raise RuntimeError(
            "TRAIN vuoto."
        )

    if test.empty:

        raise RuntimeError(
            "TEST vuoto."
        )

    return (
        train,
        test,
        train_end,
        test_start,
    )


# ============================================================
# MODELS
# ============================================================

def build_logistic_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    C=0.50,
                    random_state=
                        RANDOM_STATE
                )
            ),
        ]
    )


def build_random_forest():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,

                    max_depth=9,

                    min_samples_leaf=5,

                    min_samples_split=10,

                    max_features="sqrt",

                    class_weight=
                        "balanced_subsample",

                    random_state=
                        RANDOM_STATE,

                    n_jobs=-1
                )
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def safe_auc(
    y_true,
    probabilities
):

    try:

        if (
            pd.Series(
                y_true
            )
            .nunique()
            <
            2
        ):

            return np.nan

        return roc_auc_score(
            y_true,
            probabilities
        )

    except Exception:

        return np.nan


def probability_lift(
    y_true,
    probabilities,
    top_fraction=0.20
):
    """
    Misura molto intuitiva.

    Prendiamo il 20% delle previsioni
    con probabilità più alta.

    Vediamo quante volte l'evento
    si è realmente verificato.

    Se il modello funziona,
    questa percentuale dovrebbe essere
    significativamente superiore
    alla frequenza media.
    """

    temp = pd.DataFrame(
        {
            "real":
                np.asarray(
                    y_true
                ),

            "probability":
                np.asarray(
                    probabilities
                ),
        }
    )

    temp = temp.sort_values(
        "probability",
        ascending=False
    )

    n_top = max(
        int(
            len(
                temp
            )
            *
            top_fraction
        ),
        1
    )

    top = temp.head(
        n_top
    )

    base_rate = (
        temp[
            "real"
        ]
        .mean()
    )

    top_rate = (
        top[
            "real"
        ]
        .mean()
    )

    if base_rate > 0:

        lift = (
            top_rate
            /
            base_rate
        )

    else:

        lift = np.nan

    return (
        base_rate,
        top_rate,
        lift,
    )


def evaluate_model(
    model,
    X_test,
    y_test
):

    probabilities = (
        model.predict_proba(
            X_test
        )[
            :,
            1
        ]
    )

    predictions = (
        probabilities
        >=
        0.50
    ).astype(
        int
    )

    auc = safe_auc(
        y_test,
        probabilities
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        brier = brier_score_loss(
            y_test,
            probabilities
        )

    except Exception:

        brier = np.nan

    (
        base_rate,
        top_rate,
        lift
    ) = probability_lift(
        y_test,
        probabilities
    )

    return {

        "auc":
            auc,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "brier":
            brier,

        "base_rate":
            base_rate,

        "top20_rate":
            top_rate,

        "top20_lift":
            lift,

        "predictions":
            predictions,

        "probabilities":
            probabilities,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def feature_importance(
    model,
    feature_columns,
    model_name
):

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    if model_name == "Random Forest":

        importance = (
            classifier
            .feature_importances_
        )

    else:

        coefficients = (
            classifier
            .coef_[
                0
            ]
        )

        importance = np.abs(
            coefficients
        )

    result = pd.DataFrame(
        {
            "Feature":
                feature_columns,

            "Importance":
                importance,
        }
    )

    total = (
        result[
            "Importance"
        ]
        .sum()
    )

    if total > 0:

        result[
            "Importance %"
        ] = (
            result[
                "Importance"
            ]
            /
            total
            *
            100
        )

    else:

        result[
            "Importance %"
        ] = 0.0

    result = (
        result
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# FEATURE GROUPS
# ============================================================

def feature_group(
    feature
):

    name = (
        str(
            feature
        )
        .lower()
    )

    if "sar" in name:

        return "SAR"

    if "macd" in name:

        return "MACD"

    if (
        "bollinger"
        in name
        or
        "bb_"
        in name
        or
        "bbmid"
        in name
    ):

        return "Bollinger"

    if (
        "rsi"
        in name
        or
        "stoch"
        in name
    ):

        return "RSI / Stoch"

    if (
        "chaikin"
        in name
        or
        "cmf"
        in name
        or
        "flow"
        in name
    ):

        return "Flussi"

    if "aroon" in name:

        return "Aroon"

    if (
        "adx"
        in name
        or
        "di_plus"
        in name
        or
        "di_minus"
        in name
        or
        "di_diff"
        in name
    ):

        return "ADX / DI"

    if "volume" in name:

        return "Volume"

    if "momentum" in name:

        return "Momentum"

    if "alignment" in name:

        return "Multi-Timeframe"

    if (
        "early_bullish"
        in name
        or
        "early_bearish"
        in name
        or
        "aligned_bullish"
        in name
        or
        "aligned_bearish"
        in name
    ):

        return "Multi-Timeframe"

    if name.startswith(
        "engine_"
    ):

        return "Engine attuale"

    if (
        "price_change"
        in name
        or
        name.endswith(
            "_close"
        )
        or
        "_ha_"
        in name
    ):

        return "Prezzo / Heikin Ashi"

    if (
        "reversal"
        in name
        or
        "trend_age"
        in name
        or
        "overextension"
        in name
    ):

        return "Trend / Reversal"

    return "Altro"


def summarize_feature_groups(
    importance_df
):

    current = (
        importance_df
        .copy()
    )

    current[
        "Group"
    ] = current[
        "Feature"
    ].apply(
        feature_group
    )

    summary = (
        current
        .groupby(
            "Group",
            as_index=False
        )[
            "Importance"
        ]
        .sum()
    )

    total = (
        summary[
            "Importance"
        ]
        .sum()
    )

    if total > 0:

        summary[
            "Importance %"
        ] = (
            summary[
                "Importance"
            ]
            /
            total
            *
            100
        )

    else:

        summary[
            "Importance %"
        ] = 0

    summary = (
        summary
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# ============================================================
# MODEL COMPARISON
# ============================================================

def train_target_model(
    train,
    test,
    features,
    target,
    target_name
):

    print()
    print(
        "============================================"
    )
    print(
        f" {target_name} MODEL"
    )
    print(
        "============================================"
    )

    train_current = train.dropna(
        subset=[
            target
        ]
    ).copy()

    test_current = test.dropna(
        subset=[
            target
        ]
    ).copy()

    train_current[
        target
    ] = (
        pd.to_numeric(
            train_current[
                target
            ],
            errors="coerce"
        )
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    test_current[
        target
    ] = (
        pd.to_numeric(
            test_current[
                target
            ],
            errors="coerce"
        )
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    X_train = (
        train_current[
            features
        ]
        .copy()
    )

    X_test = (
        test_current[
            features
        ]
        .copy()
    )

    y_train = (
        train_current[
            target
        ]
    )

    y_test = (
        test_current[
            target
        ]
    )

    print()
    print(
        f"TRAIN osservazioni: "
        f"{len(X_train)}"
    )

    print(
        f"TEST osservazioni:  "
        f"{len(X_test)}"
    )

    print(
        f"Evento nel TRAIN:   "
        f"{y_train.mean() * 100:.1f}%"
    )

    print(
        f"Evento nel TEST:    "
        f"{y_test.mean() * 100:.1f}%"
    )

    candidates = {
        "Logistic Regression":
            build_logistic_model(),

        "Random Forest":
            build_random_forest(),
    }

    results = []

    trained_models = {}

    evaluations = {}

    for (
        model_name,
        model
    ) in candidates.items():

        print()
        print(
            f"Addestramento "
            f"{model_name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        evaluation = (
            evaluate_model(
                model,
                X_test,
                y_test
            )
        )

        trained_models[
            model_name
        ] = model

        evaluations[
            model_name
        ] = evaluation

        results.append(
            {
                "Target":
                    target_name,

                "Model":
                    model_name,

                "ROC AUC":
                    evaluation[
                        "auc"
                    ],

                "Accuracy":
                    evaluation[
                        "accuracy"
                    ],

                "Balanced Accuracy":
                    evaluation[
                        "balanced_accuracy"
                    ],

                "Precision":
                    evaluation[
                        "precision"
                    ],

                "Recall":
                    evaluation[
                        "recall"
                    ],

                "F1":
                    evaluation[
                        "f1"
                    ],

                "Brier":
                    evaluation[
                        "brier"
                    ],

                "Base Rate":
                    evaluation[
                        "base_rate"
                    ],

                "Top 20% Event Rate":
                    evaluation[
                        "top20_rate"
                    ],

                "Top 20% Lift":
                    evaluation[
                        "top20_lift"
                    ],
            }
        )

        print(
            f"ROC AUC:            "
            f"{evaluation['auc']:.3f}"
        )

        print(
            f"Balanced Accuracy:  "
            f"{evaluation['balanced_accuracy']:.3f}"
        )

        print(
            f"Precision:           "
            f"{evaluation['precision']:.3f}"
        )

        print(
            f"Recall:              "
            f"{evaluation['recall']:.3f}"
        )

        print(
            f"F1:                  "
            f"{evaluation['f1']:.3f}"
        )

        print(
            f"Top 20% lift:        "
            f"{evaluation['top20_lift']:.2f}x"
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # SCEGLIAMO IL MIGLIORE
    #
    # Prima ROC AUC.
    # In caso di valori non disponibili:
    # Balanced Accuracy.
    # --------------------------------------------------------

    sortable = (
        results_df
        .copy()
    )

    sortable[
        "AUC Sort"
    ] = sortable[
        "ROC AUC"
    ].fillna(
        -999
    )

    sortable = (
        sortable
        .sort_values(
            [
                "AUC Sort",
                "Balanced Accuracy",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    best_name = (
        sortable.iloc[
            0
        ][
            "Model"
        ]
    )

    best_model = (
        trained_models[
            best_name
        ]
    )

    best_evaluation = (
        evaluations[
            best_name
        ]
    )

    print()
    print(
        "--------------------------------------------"
    )

    print(
        f"MIGLIOR MODELLO {target_name}: "
        f"{best_name}"
    )

    print(
        "--------------------------------------------"
    )

    importance = feature_importance(
        best_model,
        features,
        best_name
    )

    group_summary = (
        summarize_feature_groups(
            importance
        )
    )

    # --------------------------------------------------------
    # PREVISIONI TEST
    # --------------------------------------------------------

    predictions = pd.DataFrame(
        {
            "date":
                test_current[
                    "date"
                ].values,

            "ticker":
                test_current[
                    "ticker"
                ].values,

            "market":
                test_current[
                    "market"
                ].values,

            "real":
                y_test.values,

            "probability":
                best_evaluation[
                    "probabilities"
                ],

            "prediction":
                best_evaluation[
                    "predictions"
                ],
        }
    )

    return {
        "results":
            results_df,

        "best_name":
            best_name,

        "best_model":
            best_model,

        "evaluation":
            best_evaluation,

        "importance":
            importance,

        "groups":
            group_summary,

        "predictions":
            predictions,

        "feature_columns":
            features,
    }


# ============================================================
# PRINT TOP FEATURES
# ============================================================

def print_top_features(
    title,
    dataframe,
    limit=25
):

    print()
    print(
        "============================================"
    )

    print(
        f" TOP FEATURES — {title}"
    )

    print(
        "============================================"
    )

    print()

    display = (
        dataframe
        .head(
            limit
        )
        .copy()
    )

    display[
        "Importance %"
    ] = (
        display[
            "Importance %"
        ]
        .round(
            2
        )
    )

    for position, row in (
        display.iterrows()
    ):

        print(
            f"{position + 1:2d}. "
            f"{row['Feature']:<48} "
            f"{row['Importance %']:6.2f}%"
        )


def print_group_summary(
    title,
    dataframe
):

    print()
    print(
        "============================================"
    )

    print(
        f" FAMIGLIE INDICATORI — {title}"
    )

    print(
        "============================================"
    )

    print()

    for _, row in (
        dataframe.iterrows()
    ):

        print(
            f"{row['Group']:<24} "
            f"{row['Importance %']:6.2f}%"
        )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model_package(
    filename,
    result,
    target,
    train_end,
    test_start
):

    package = {
        "model":
            result[
                "best_model"
            ],

        "model_name":
            result[
                "best_name"
            ],

        "target":
            target,

        "horizon":
            HORIZON,

        "feature_columns":
            result[
                "feature_columns"
            ],

        "train_end":
            train_end,

        "test_start":
            test_start,

        "metrics":
            {
                key:
                    value

                for key, value
                in result[
                    "evaluation"
                ].items()

                if key
                not in [
                    "predictions",
                    "probabilities",
                ]
            },
    }

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    joblib.dump(
        package,
        path
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "============================================"
    )
    print(
        " MARKET SENTINEL"
    )
    print(
        " MACHINE LEARNING TRAINER V1"
    )
    print(
        "============================================"
    )
    print()

    print(
        "Caricamento dataset..."
    )

    df = load_dataset()

    print(
        f"Righe dataset: "
        f"{len(df)}"
    )

    print(
        f"Titoli:        "
        f"{df['ticker'].nunique()}"
    )

    print(
        f"Dal:           "
        f"{df['date'].min()}"
    )

    print(
        f"Al:            "
        f"{df['date'].max()}"
    )

    # ========================================================
    # FEATURES
    # ========================================================

    (
        feature_columns,
        dropped_missing,
        dropped_constant,
    ) = get_feature_columns(
        df
    )

    print()
    print(
        f"Features utilizzate: "
        f"{len(feature_columns)}"
    )

    print(
        f"Scartate per troppi NaN: "
        f"{len(dropped_missing)}"
    )

    print(
        f"Scartate perché costanti: "
        f"{len(dropped_constant)}"
    )

    # ========================================================
    # TIME SPLIT
    # ========================================================

    (
        train,
        test,
        train_end,
        test_start,
    ) = chronological_split(
        df
    )

    print()
    print(
        "============================================"
    )

    print(
        " SEPARAZIONE TEMPORALE"
    )

    print(
        "============================================"
    )

    print()

    print(
        "TRAIN:"
    )

    print(
        f"  dal "
        f"{train['date'].min()}"
    )

    print(
        f"  al  "
        f"{train['date'].max()}"
    )

    print()

    print(
        "PURGE GAP:"
    )

    print(
        f"  fino a "
        f"{test_start}"
    )

    print()

    print(
        "TEST MAI VISTO:"
    )

    print(
        f"  dal "
        f"{test['date'].min()}"
    )

    print(
        f"  al  "
        f"{test['date'].max()}"
    )

    # ========================================================
    # BUY
    # ========================================================

    buy_result = train_target_model(
        train,
        test,
        feature_columns,
        BUY_TARGET,
        "BUY"
    )

    # ========================================================
    # SELL
    # ========================================================

    sell_result = train_target_model(
        train,
        test,
        feature_columns,
        SELL_TARGET,
        "SELL"
    )

    # ========================================================
    # PRINT IMPORTANCE
    # ========================================================

    print_top_features(
        "BUY",
        buy_result[
            "importance"
        ]
    )

    print_group_summary(
        "BUY",
        buy_result[
            "groups"
        ]
    )

    print_top_features(
        "SELL",
        sell_result[
            "importance"
        ]
    )

    print_group_summary(
        "SELL",
        sell_result[
            "groups"
        ]
    )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    buy_model_path = save_model_package(
        "market_sentinel_buy_15d.joblib",
        buy_result,
        BUY_TARGET,
        train_end,
        test_start
    )

    sell_model_path = save_model_package(
        "market_sentinel_sell_15d.joblib",
        sell_result,
        SELL_TARGET,
        train_end,
        test_start
    )

    # ========================================================
    # SAVE REPORTS
    # ========================================================

    metrics = pd.concat(
        [
            buy_result[
                "results"
            ],
            sell_result[
                "results"
            ],
        ],
        ignore_index=True
    )

    metrics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_metrics.csv"
        ),
        index=False
    )

    buy_result[
        "importance"
    ].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "buy_feature_importance.csv"
        ),
        index=False
    )

    sell_result[
        "importance"
    ].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "sell_feature_importance.csv"
        ),
        index=False
    )

    buy_result[
        "groups"
    ].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "buy_indicator_groups.csv"
        ),
        index=False
    )

    sell_result[
        "groups"
    ].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "sell_indicator_groups.csv"
        ),
        index=False
    )

    buy_predictions = (
        buy_result[
            "predictions"
        ]
        .rename(
            columns={
                "real":
                    "buy_real",

                "probability":
                    "buy_probability",

                "prediction":
                    "buy_prediction",
            }
        )
    )

    sell_predictions = (
        sell_result[
            "predictions"
        ]
        .rename(
            columns={
                "real":
                    "sell_real",

                "probability":
                    "sell_probability",

                "prediction":
                    "sell_prediction",
            }
        )
    )

    test_predictions = pd.merge(
        buy_predictions,
        sell_predictions,
        on=[
            "date",
            "ticker",
            "market",
        ],
        how="outer"
    )

    test_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_predictions.csv"
        ),
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "============================================"
    )
    print(
        " MACHINE LEARNING COMPLETATO"
    )
    print(
        "============================================"
    )

    print()

    print(
        f"BUY modello:  "
        f"{buy_result['best_name']}"
    )

    print(
        f"BUY ROC AUC:  "
        f"{buy_result['evaluation']['auc']:.3f}"
    )

    print(
        f"BUY Lift 20%: "
        f"{buy_result['evaluation']['top20_lift']:.2f}x"
    )

    print()

    print(
        f"SELL modello:  "
        f"{sell_result['best_name']}"
    )

    print(
        f"SELL ROC AUC:  "
        f"{sell_result['evaluation']['auc']:.3f}"
    )

    print(
        f"SELL Lift 20%: "
        f"{sell_result['evaluation']['top20_lift']:.2f}x"
    )

    print()

    print(
        "Modelli salvati:"
    )

    print(
        buy_model_path
    )

    print(
        sell_model_path
    )

    print()

    print(
        "IMPORTANTE:"
    )

    print(
        "ROC AUC = 0.50 significa nessuna "
        "capacità predittiva."
    )

    print(
        "Valori superiori a 0.50 indicano "
        "capacità predittiva crescente."
    )

    print(
        "Non giudicheremo il modello "
        "da un solo numero: guarderemo "
        "anche Lift, stabilità e feature importance."
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()