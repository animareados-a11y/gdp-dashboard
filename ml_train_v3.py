import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

warnings.filterwarnings("ignore")


# ============================================================
# MARKET SENTINEL
# MACHINE LEARNING TRAINER V3
# "IL TRENO"
# ============================================================
#
# OBIETTIVO
#
# Addestrare modelli BUY e SELL usando:
#
#   data/ml_dataset_v3.csv
#
# Il modello NON deve vedere nessuna informazione futura.
#
# Confronteremo:
#
# 1. Logistic Regression
#    -> benchmark lineare
#
# 2. Random Forest
#    -> modello non lineare
#
# 3. Extra Trees
#    -> modello non lineare, molto utile per molte feature
#
# 4. HistGradientBoosting
#    -> modello non lineare basato su boosting
#
# La separazione TRAIN / TEST è TEMPORALE:
# il modello impara dal passato e viene valutato
# su un periodo successivo mai visto.
#
# ============================================================


INPUT_FILE = os.path.join(
    "data",
    "ml_dataset_v3.csv",
)

MODEL_DIR = os.path.join(
    "data",
    "ml_models",
)

BUY_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "market_sentinel_buy_v3.joblib",
)

SELL_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "market_sentinel_sell_v3.joblib",
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

TEST_FRACTION = 0.30

# Gap fra train e test.
# Serve a ridurre contaminazioni dovute
# agli orizzonti futuri dei target.
PURGE_DAYS = 25

RANDOM_STATE = 42

TOP_FEATURES_TO_PRINT = 40

TOP_PERCENTILE = 0.20


# ============================================================
# COLONNE CHE IL MODELLO NON DEVE VEDERE
# ============================================================

META_COLUMNS = {
    "ticker",
    "market",
    "date",
}

# Qualsiasi colonna che inizi con questi prefissi
# contiene direttamente o indirettamente informazioni future.
FORBIDDEN_PREFIXES = (
    "future_",
    "target_",
)

FORBIDDEN_EXACT = {
    "train_state_v3",
    "buy_strength_v3",
    "sell_strength_v3",
    "direction_strength_v3",
    "target_buy_event",
    "target_sell_event",
}


# ============================================================
# HELPERS
# ============================================================

def safe_auc(
    y_true,
    probabilities,
):
    try:
        return roc_auc_score(
            y_true,
            probabilities,
        )
    except Exception:
        return np.nan


def top_percent_lift(
    y_true,
    probabilities,
    fraction=TOP_PERCENTILE,
):
    """
    Esempio:

    baseline BUY = 40%

    nel 20% delle osservazioni che il modello
    considera migliori, BUY = 60%

    lift = 60 / 40 = 1.50x
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(y_true) == 0:
        return np.nan

    baseline = np.mean(
        y_true
    )

    if (
        not np.isfinite(baseline)
        or baseline <= 0
    ):
        return np.nan

    number = max(
        1,
        int(
            len(y_true)
            *
            fraction
        ),
    )

    order = np.argsort(
        probabilities
    )[::-1]

    selected = y_true[
        order[:number]
    ]

    selected_rate = np.mean(
        selected
    )

    return (
        selected_rate
        /
        baseline
    )


def probability_bucket_table(
    y_true,
    probabilities,
):
    """
    Divide le previsioni in 5 gruppi.

    Ci interessa vedere se passando
    dai punteggi bassi a quelli alti
    aumenta realmente la frequenza
    degli eventi BUY/SELL.
    """

    frame = pd.DataFrame(
        {
            "target": np.asarray(
                y_true,
                dtype=float,
            ),
            "probability": np.asarray(
                probabilities,
                dtype=float,
            ),
        }
    )

    try:

        frame["bucket"] = pd.qcut(
            frame["probability"],
            q=5,
            duplicates="drop",
        )

        summary = (
            frame
            .groupby(
                "bucket",
                observed=True,
            )
            .agg(
                observations=(
                    "target",
                    "size",
                ),
                event_rate=(
                    "target",
                    "mean",
                ),
                avg_probability=(
                    "probability",
                    "mean",
                ),
            )
        )

        summary[
            "event_rate"
        ] *= 100

        summary[
            "avg_probability"
        ] *= 100

        return summary

    except Exception:
        return pd.DataFrame()


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(
    dataframe
):

    features = []

    excluded_future = []

    for column in dataframe.columns:

        if column in META_COLUMNS:
            continue

        if column in FORBIDDEN_EXACT:

            excluded_future.append(
                column
            )

            continue

        if any(
            column.startswith(prefix)
            for prefix
            in FORBIDDEN_PREFIXES
        ):

            excluded_future.append(
                column
            )

            continue

        features.append(
            column
        )

    return (
        features,
        excluded_future,
    )


def clean_feature_matrix(
    dataframe,
    feature_columns,
):
    """
    Converte tutto in numerico e rimuove:

    - feature completamente vuote
    - feature quasi completamente vuote
    - feature costanti

    Non riempiamo qui i NaN:
    lo farà la pipeline usando SOLO il TRAIN.
    """

    X = dataframe[
        feature_columns
    ].copy()

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    removed_nan = []

    removed_constant = []

    # ========================================================
    # TROPPI NaN
    # ========================================================

    keep = []

    for column in X.columns:

        missing_ratio = (
            X[column]
            .isna()
            .mean()
        )

        if missing_ratio > 0.60:

            removed_nan.append(
                column
            )

        else:

            keep.append(
                column
            )

    X = X[
        keep
    ]

    # ========================================================
    # COSTANTI
    # ========================================================

    keep = []

    for column in X.columns:

        unique_values = (
            X[column]
            .dropna()
            .nunique()
        )

        if unique_values <= 1:

            removed_constant.append(
                column
            )

        else:

            keep.append(
                column
            )

    X = X[
        keep
    ]

    return (
        X,
        removed_nan,
        removed_constant,
    )


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(
    dataframe,
):
    """
    Split temporale.

    Prima parte:
        TRAIN

    Ultima parte:
        TEST mai visto

    Fra i due:
        PURGE GAP
    """

    dates = pd.to_datetime(
        dataframe["date"],
        utc=True,
        errors="coerce",
    )

    valid_dates = (
        dates
        .dropna()
        .sort_values()
        .unique()
    )

    if len(valid_dates) < 20:

        raise RuntimeError(
            "Serie temporale insufficiente."
        )

    split_position = int(
        len(valid_dates)
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
            1,
        ),
        len(valid_dates) - 1,
    )

    test_start = pd.Timestamp(
        valid_dates[
            split_position
        ]
    )

    purge_start = (
        test_start
        -
        pd.Timedelta(
            days=PURGE_DAYS
        )
    )

    train_mask = (
        dates
        <
        purge_start
    )

    test_mask = (
        dates
        >=
        test_start
    )

    return (
        train_mask,
        test_mask,
        purge_start,
        test_start,
    )


# ============================================================
# MODELS
# ============================================================

def build_models():
    """
    Costruiamo modelli volutamente diversi.

    Logistic:
        benchmark lineare.

    Random Forest:
        relazioni non lineari e interazioni.

    Extra Trees:
        molte interazioni, utile con molte feature.

    HistGradientBoosting:
        boosting non lineare.
    """

    models = {}

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    models[
        "Logistic Regression"
    ] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    models[
        "Random Forest"
    ] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=450,
                    max_depth=12,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # ========================================================
    # EXTRA TREES
    # ========================================================

    models[
        "Extra Trees"
    ] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=14,
                    min_samples_leaf=6,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # ========================================================
    # HIST GRADIENT BOOSTING
    # ========================================================

    models[
        "HistGradientBoosting"
    ] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=250,
                    max_leaf_nodes=31,
                    min_samples_leaf=25,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return models


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities
        >=
        0.50
    ).astype(
        int
    )

    metrics = {

        "roc_auc":
            safe_auc(
                y_test,
                probabilities,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_test,
                predictions,
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),

        "lift_20":
            top_percent_lift(
                y_test,
                probabilities,
            ),
    }

    return (
        metrics,
        probabilities,
    )


# ============================================================
# MODEL SELECTION
# ============================================================

def model_selection_score(
    metrics
):
    """
    Non scegliamo il vincitore soltanto
    sulla ROC AUC.

    Diamo particolare importanza al Lift:
    Market Sentinel deve soprattutto
    riconoscere le situazioni MIGLIORI.
    """

    auc = metrics.get(
        "roc_auc",
        np.nan,
    )

    lift = metrics.get(
        "lift_20",
        np.nan,
    )

    bal = metrics.get(
        "balanced_accuracy",
        np.nan,
    )

    if not np.isfinite(
        auc
    ):
        auc = 0.5

    if not np.isfinite(
        lift
    ):
        lift = 1.0

    if not np.isfinite(
        bal
    ):
        bal = 0.5

    # normalizzazione approssimativa

    auc_component = max(
        auc - 0.5,
        0,
    )

    lift_component = max(
        lift - 1.0,
        0,
    )

    bal_component = max(
        bal - 0.5,
        0,
    )

    return (
        0.45
        *
        auc_component

        +

        0.40
        *
        lift_component

        +

        0.15
        *
        bal_component
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(
    fitted_pipeline,
    feature_names,
):
    """
    Restituisce importanza normalizzata.

    Per modelli ad alberi:
        feature_importances_

    Per Logistic:
        valore assoluto coefficienti
    """

    try:

        model = fitted_pipeline.named_steps[
            "model"
        ]

        if hasattr(
            model,
            "feature_importances_",
        ):

            importance = np.asarray(
                model.feature_importances_,
                dtype=float,
            )

        elif hasattr(
            model,
            "coef_",
        ):

            importance = np.abs(
                np.asarray(
                    model.coef_[0],
                    dtype=float,
                )
            )

        else:

            return pd.DataFrame()

        if (
            len(importance)
            !=
            len(feature_names)
        ):
            return pd.DataFrame()

        total = np.sum(
            importance
        )

        if (
            not np.isfinite(total)
            or total <= 0
        ):
            return pd.DataFrame()

        importance = (
            importance
            /
            total
            *
            100
        )

        result = pd.DataFrame(
            {
                "feature":
                    feature_names,

                "importance":
                    importance,
            }
        )

        result = (
            result
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        return result

    except Exception:
        return pd.DataFrame()


# ============================================================
# FAMIGLIE DI INDICATORI
# ============================================================

def feature_family(
    feature
):

    name = str(
        feature
    ).lower()

    if "sar" in name:
        return "SAR"

    if "macd" in name:
        return "MACD"

    if (
        "bollinger" in name
        or "_bb_" in name
    ):
        return "Bollinger"

    if (
        "rsi" in name
        or "stoch" in name
    ):
        return "RSI / Stoch"

    if (
        "chaikin" in name
        or "_ad_" in name
        or "cmf" in name
    ):
        return "Flussi"

    if (
        "_ha_" in name
        or "heikin" in name
    ):
        return "Heikin Ashi"

    if (
        "adx" in name
        or "diplus" in name
        or "diminus" in name
        or "di_plus" in name
        or "di_minus" in name
    ):
        return "ADX / DI"

    if "aroon" in name:
        return "Aroon"

    if "momentum" in name:
        return "Momentum"

    if "volume" in name:
        return "Volume"

    if (
        "setup_" in name
        or "mtf_" in name
    ):
        return "Setup / Multi-Timeframe"

    if "engine_" in name:
        return "Engine attuale"

    if (
        "trend" in name
        or "reversal" in name
    ):
        return "Trend / Reversal"

    if (
        "price" in name
        or "close" in name
    ):
        return "Prezzo"

    return "Altro"


def family_importance_table(
    importance_frame
):

    if (
        importance_frame is None
        or importance_frame.empty
    ):
        return pd.DataFrame()

    frame = importance_frame.copy()

    frame["family"] = frame[
        "feature"
    ].apply(
        feature_family
    )

    result = (
        frame
        .groupby(
            "family",
            as_index=False,
        )[
            "importance"
        ]
        .sum()
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# TRAIN ONE SIDE
# ============================================================

def train_direction(
    direction_name,
    target_column,
    dataframe,
    X,
    train_mask,
    test_mask,
):

    print()
    print(
        "============================================"
    )

    print(
        f" {direction_name} MODEL"
    )

    print(
        "============================================"
    )

    print()

    target = pd.to_numeric(
        dataframe[
            target_column
        ],
        errors="coerce",
    )

    valid_target = (
        target.notna()
    )

    current_train_mask = (
        train_mask
        &
        valid_target
    )

    current_test_mask = (
        test_mask
        &
        valid_target
    )

    X_train = X.loc[
        current_train_mask
    ]

    y_train = target.loc[
        current_train_mask
    ].astype(
        int
    )

    X_test = X.loc[
        current_test_mask
    ]

    y_test = target.loc[
        current_test_mask
    ].astype(
        int
    )

    if (
        len(X_train) == 0
        or len(X_test) == 0
    ):

        raise RuntimeError(
            f"Dati insufficienti per {direction_name}"
        )

    print(
        f"TRAIN osservazioni: "
        f"{len(X_train)}"
    )

    print(
        f"TEST osservazioni:  "
        f"{len(X_test)}"
    )

    print()

    print(
        f"Evento nel TRAIN:   "
        f"{y_train.mean() * 100:.1f}%"
    )

    print(
        f"Evento nel TEST:    "
        f"{y_test.mean() * 100:.1f}%"
    )

    models = build_models()

    results = []

    fitted_models = {}

    probabilities_by_model = {}

    # ========================================================
    # TRAIN
    # ========================================================

    for (
        model_name,
        model,
    ) in models.items():

        print()
        print(
            f"Addestramento "
            f"{model_name}..."
        )

        try:

            model.fit(
                X_train,
                y_train,
            )

            (
                metrics,
                probabilities,
            ) = evaluate_model(
                model,
                X_test,
                y_test,
            )

            selection_score = (
                model_selection_score(
                    metrics
                )
            )

            fitted_models[
                model_name
            ] = model

            probabilities_by_model[
                model_name
            ] = probabilities

            result = {
                "model":
                    model_name,

                **metrics,

                "selection_score":
                    selection_score,
            }

            results.append(
                result
            )

            print(
                f"ROC AUC:            "
                f"{metrics['roc_auc']:.3f}"
            )

            print(
                f"Balanced Accuracy:  "
                f"{metrics['balanced_accuracy']:.3f}"
            )

            print(
                f"Precision:          "
                f"{metrics['precision']:.3f}"
            )

            print(
                f"Recall:             "
                f"{metrics['recall']:.3f}"
            )

            print(
                f"F1:                 "
                f"{metrics['f1']:.3f}"
            )

            print(
                f"Top 20% lift:       "
                f"{metrics['lift_20']:.2f}x"
            )

        except Exception as exc:

            print(
                f"ERRORE: "
                f"{exc}"
            )

    if not results:

        raise RuntimeError(
            f"Nessun modello addestrato "
            f"per {direction_name}"
        )

    results_frame = pd.DataFrame(
        results
    )

    results_frame = (
        results_frame
        .sort_values(
            [
                "selection_score",
                "roc_auc",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    winner_name = results_frame.loc[
        0,
        "model"
    ]

    winner = fitted_models[
        winner_name
    ]

    winner_probabilities = (
        probabilities_by_model[
            winner_name
        ]
    )

    print()
    print(
        "--------------------------------------------"
    )

    print(
        f"MIGLIOR MODELLO "
        f"{direction_name}: "
        f"{winner_name}"
    )

    print(
        "--------------------------------------------"
    )

    # ========================================================
    # PROBABILITY BUCKETS
    # ========================================================

    bucket_table = (
        probability_bucket_table(
            y_test,
            winner_probabilities,
        )
    )

    print()
    print(
        "PROBABILITY BUCKETS"
    )

    print()

    if not bucket_table.empty:

        print(
            bucket_table
            .round(
                2
            )
            .to_string()
        )

    # ========================================================
    # IMPORTANCE
    # ========================================================

    importance = (
        extract_feature_importance(
            winner,
            list(
                X.columns
            ),
        )
    )

    family_table = (
        family_importance_table(
            importance
        )
    )

    return {
        "winner_name":
            winner_name,

        "winner":
            winner,

        "results":
            results_frame,

        "importance":
            importance,

        "families":
            family_table,

        "bucket_table":
            bucket_table,

        "test_target":
            y_test,

        "test_probabilities":
            winner_probabilities,
    }


# ============================================================
# PRINT IMPORTANCE
# ============================================================

def print_importance(
    title,
    result,
):

    importance = result[
        "importance"
    ]

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

    if importance.empty:

        print(
            "Feature importance non disponibile."
        )

    else:

        top = importance.head(
            TOP_FEATURES_TO_PRINT
        )

        for index, row in top.iterrows():

            print(
                f"{index + 1:>2}. "
                f"{row['feature']:<60} "
                f"{row['importance']:>6.2f}%"
            )

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

    families = result[
        "families"
    ]

    if families.empty:

        print(
            "Non disponibile."
        )

    else:

        for _, row in families.iterrows():

            print(
                f"{row['family']:<30} "
                f"{row['importance']:>6.2f}%"
            )


# ============================================================
# SAVE MODEL PACKAGE
# ============================================================

def save_model_package(
    path,
    result,
    feature_columns,
    direction,
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    package = {

        "version":
            "V3",

        "direction":
            direction,

        "model_name":
            result[
                "winner_name"
            ],

        "model":
            result[
                "winner"
            ],

        "features":
            list(
                feature_columns
            ),

        "results":
            result[
                "results"
            ],

        "importance":
            result[
                "importance"
            ],

        "family_importance":
            result[
                "families"
            ],

        "target_definition":
            "Market Sentinel Train V3",

        "purge_days":
            PURGE_DAYS,

        "test_fraction":
            TEST_FRACTION,
    }

    joblib.dump(
        package,
        path,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "============================================"
    )

    print(
        " MARKET SENTINEL"
    )

    print(
        " MACHINE LEARNING TRAINER V3"
    )

    print(
        ' "IL TRENO"'
    )

    print(
        "============================================"
    )

    print()

    # ========================================================
    # LOAD
    # ========================================================

    if not os.path.exists(
        INPUT_FILE
    ):

        raise FileNotFoundError(
            f"File non trovato: "
            f"{INPUT_FILE}"
        )

    print(
        "Caricamento dataset..."
    )

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        utc=True,
        errors="coerce",
    )

    dataframe = (
        dataframe
        .dropna(
            subset=[
                "date"
            ]
        )
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

    print(
        f"Righe dataset: "
        f"{len(dataframe)}"
    )

    print(
        f"Titoli:        "
        f"{dataframe['ticker'].nunique()}"
    )

    print(
        f"Dal:           "
        f"{dataframe['date'].min()}"
    )

    print(
        f"Al:            "
        f"{dataframe['date'].max()}"
    )

    # ========================================================
    # FEATURES
    # ========================================================

    (
        candidate_features,
        excluded_future,
    ) = get_feature_columns(
        dataframe
    )

    (
        X,
        removed_nan,
        removed_constant,
    ) = clean_feature_matrix(
        dataframe,
        candidate_features,
    )

    print()
    print(
        f"Features utilizzate: "
        f"{X.shape[1]}"
    )

    print(
        f"Colonne future escluse: "
        f"{len(excluded_future)}"
    )

    print(
        f"Scartate per troppi NaN: "
        f"{len(removed_nan)}"
    )

    print(
        f"Scartate perché costanti: "
        f"{len(removed_constant)}"
    )

    # ========================================================
    # CONTROLLO ANTI-LEAKAGE
    # ========================================================

    suspicious = [

        column

        for column in X.columns

        if (
            column.startswith(
                "future_"
            )
            or
            column.startswith(
                "target_"
            )
            or
            column
            in FORBIDDEN_EXACT
        )
    ]

    if suspicious:

        raise RuntimeError(
            "ATTENZIONE: leakage rilevato. "
            f"Colonne future presenti: "
            f"{suspicious}"
        )

    print()
    print(
        "Controllo leakage: OK"
    )

    # ========================================================
    # TEMPORAL SPLIT
    # ========================================================

    (
        train_mask,
        test_mask,
        purge_start,
        test_start,
    ) = temporal_split(
        dataframe
    )

    train_dates = dataframe.loc[
        train_mask,
        "date"
    ]

    test_dates = dataframe.loc[
        test_mask,
        "date"
    ]

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
        f"{train_dates.min()}"
    )

    print(
        f"  al  "
        f"{train_dates.max()}"
    )

    print()

    print(
        "PURGE GAP:"
    )

    print(
        f"  dal "
        f"{purge_start}"
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
        f"{test_dates.min()}"
    )

    print(
        f"  al  "
        f"{test_dates.max()}"
    )

    # ========================================================
    # BUY
    # ========================================================

    buy_result = train_direction(
        direction_name="BUY",
        target_column="target_buy_v3",
        dataframe=dataframe,
        X=X,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    # ========================================================
    # SELL
    # ========================================================

    sell_result = train_direction(
        direction_name="SELL",
        target_column="target_sell_v3",
        dataframe=dataframe,
        X=X,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    # ========================================================
    # IMPORTANCE
    # ========================================================

    print_importance(
        "BUY",
        buy_result,
    )

    print_importance(
        "SELL",
        sell_result,
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_model_package(
        BUY_MODEL_FILE,
        buy_result,
        X.columns,
        "BUY",
    )

    save_model_package(
        SELL_MODEL_FILE,
        sell_result,
        X.columns,
        "SELL",
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    buy_best = (
        buy_result[
            "results"
        ]
        .iloc[0]
    )

    sell_best = (
        sell_result[
            "results"
        ]
        .iloc[0]
    )

    print()
    print(
        "============================================"
    )

    print(
        " MACHINE LEARNING V3 COMPLETATO"
    )

    print(
        "============================================"
    )

    print()

    print(
        f"BUY modello:  "
        f"{buy_result['winner_name']}"
    )

    print(
        f"BUY ROC AUC:  "
        f"{buy_best['roc_auc']:.3f}"
    )

    print(
        f"BUY Lift 20%: "
        f"{buy_best['lift_20']:.2f}x"
    )

    print()

    print(
        f"SELL modello:  "
        f"{sell_result['winner_name']}"
    )

    print(
        f"SELL ROC AUC:  "
        f"{sell_best['roc_auc']:.3f}"
    )

    print(
        f"SELL Lift 20%: "
        f"{sell_best['lift_20']:.2f}x"
    )

    print()

    print(
        "Modelli salvati:"
    )

    print(
        BUY_MODEL_FILE
    )

    print(
        SELL_MODEL_FILE
    )

    print()

    print(
        "NOTA:"
    )

    print(
        "ROC AUC = 0.50 significa nessuna "
        "capacità di ordinare correttamente "
        "i casi positivi rispetto ai negativi."
    )

    print()

    print(
        "Il Lift 20% ci dice quanto il gruppo "
        "selezionato come migliore dal modello "
        "è più ricco di veri BUY/SELL rispetto "
        "alla media del mercato."
    )

    print()

    print(
        "NON modificheremo ancora i pesi "
        "dell'engine.py."
    )

    print(
        "Prima confronteremo risultati, "
        "feature importance e famiglie "
        "di indicatori."
    )

    print()