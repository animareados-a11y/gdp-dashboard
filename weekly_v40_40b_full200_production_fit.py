# ============================================================
# MARKET SENTINEL - V40.40b
# FULL200 WEEKLY REVERSAL - FINAL PRODUCTION FIT
#
# VERSIONE DEFINITIVA
#
# PRINCIPI:
# - FULL200
# - 413 feature congelate
# - RandomForest congelata
# - nessun nuovo tuning
# - nessun peso manuale
# - calibrazione production PREDECLARED da V40.25
# - V40.40a usata come validazione OOS e calibration sample
# - la scelta esplorativa PLATT/ISOTONIC osservata in V40.40a
#   NON viene usata per scegliere il calibratore production
#
# IMPORTANTE:
# V40.40a ha validato il modello con temporal split corretto,
# unique weeks e purging H settimane.
#
# V40.40b NON è un nuovo test.
# È il fit finale production sull'intera storia disponibile.
# ============================================================

from pathlib import Path
import gc
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURAZIONE
# ============================================================

DATA_DIR = Path("data")

FEATURE_DATA_FILE = DATA_DIR / "v40_39_full200_features.csv"
FEATURE_LIST_FILE = DATA_DIR / "v40_39_feature_list.csv"

OOS_PREDICTIONS_FILE = DATA_DIR / "v40_40a_oos_predictions.csv"
OOS_SUMMARY_FILE = DATA_DIR / "v40_40a_summary.csv"

MEMMAP_FILE = DATA_DIR / "v40_40a_features_float32.dat"

OUTPUT_DIR = DATA_DIR / "v40_40b_weekly_production"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MANIFEST_FILE = DATA_DIR / "v40_40b_model_manifest.csv"
CALIBRATION_MANIFEST_FILE = DATA_DIR / "v40_40b_calibration_manifest.csv"
TRAINING_SUMMARY_FILE = DATA_DIR / "v40_40b_training_summary.csv"
METADATA_FILE = DATA_DIR / "v40_40b_metadata.json"


# ============================================================
# RANDOM FOREST CONGELATA
# ============================================================

RF_PARAMS = {
    "n_estimators": 350,
    "max_features": "sqrt",
    "min_samples_leaf": 8,
    "class_weight": "balanced_subsample",
    "random_state": 42,

    # Memory-safe.
    # Non modifica la metodologia statistica.
    "n_jobs": 1,
}


# ============================================================
# TARGET PRODUCTION
# ============================================================

TARGET_CONFIG = {
    "BEAR_TO_BULL_1W": {
        "family": "RIBASSISTA",
        "future_family": "RIALZISTA",
        "horizon": 1,
    },
    "BEAR_TO_BULL_2W": {
        "family": "RIBASSISTA",
        "future_family": "RIALZISTA",
        "horizon": 2,
    },
    "BEAR_TO_BULL_3W": {
        "family": "RIBASSISTA",
        "future_family": "RIALZISTA",
        "horizon": 3,
    },
    "BULL_TO_BEAR_1W": {
        "family": "RIALZISTA",
        "future_family": "RIBASSISTA",
        "horizon": 1,
    },
    "BULL_TO_BEAR_2W": {
        "family": "RIALZISTA",
        "future_family": "RIBASSISTA",
        "horizon": 2,
    },
    "BULL_TO_BEAR_3W": {
        "family": "RIALZISTA",
        "future_family": "RIBASSISTA",
        "horizon": 3,
    },
}


# ============================================================
# CALIBRAZIONE PRODUCTION
#
# PREDECLARED POLICY
#
# Questa mappatura NON viene scelta usando i risultati V40.40a.
#
# È la policy storica V40.25 già congelata PRIMA della nuova
# validazione FULL200.
#
# Questo evita model-selection leakage nella scelta tra
# PLATT e ISOTONIC.
# ============================================================

PRODUCTION_CALIBRATION = {
    "BEAR_TO_BULL_1W": "PLATT",
    "BEAR_TO_BULL_2W": "ISOTONIC",
    "BEAR_TO_BULL_3W": "PLATT",
    "BULL_TO_BEAR_1W": "ISOTONIC",
    "BULL_TO_BEAR_2W": "PLATT",
    "BULL_TO_BEAR_3W": "PLATT",
}


# ============================================================
# METODI ESPLORATIVI OSSERVATI IN V40.40a
#
# SOLO DOCUMENTAZIONE.
#
# NON vengono utilizzati per decidere la calibrazione
# production.
# ============================================================

V40_40A_EXPLORATORY_SELECTION = {
    "BEAR_TO_BULL_1W": "ISOTONIC",
    "BEAR_TO_BULL_2W": "PLATT",
    "BEAR_TO_BULL_3W": "PLATT",
    "BULL_TO_BEAR_1W": "ISOTONIC",
    "BULL_TO_BEAR_2W": "ISOTONIC",
    "BULL_TO_BEAR_3W": "ISOTONIC",
}


# ============================================================
# UTILITIES
# ============================================================

def section(title):
    print()
    print("=" * 145)
    print(title)
    print("=" * 145)
    print()


def normalize_phase(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .upper()
        .replace("À", "A")
        .replace("È", "E")
        .replace("É", "E")
        .replace("Ì", "I")
        .replace("Ò", "O")
        .replace("Ù", "U")
    )


def phase_family(value):
    phase = normalize_phase(value)

    bullish = {
        "BUY",
        "TREND_RIALZISTA",
        "DEBOLEZZA_RIALZISTA",
    }

    bearish = {
        "SELL",
        "TREND_RIBASSISTA",
        "DEBOLEZZA_RIBASSISTA",
    }

    if phase in bullish:
        return "RIALZISTA"

    if phase in bearish:
        return "RIBASSISTA"

    return "INDECISIONE"


def find_column(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        key = candidate.lower()
        if key in lookup:
            return lookup[key]

    return None


# ============================================================
# CALIBRATORI
# ============================================================

class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

    def fit(self, raw_prob, y):
        x = np.asarray(raw_prob, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=int)

        self.model.fit(x, y)

        return self

    def predict(self, raw_prob):
        x = np.asarray(raw_prob, dtype=float).reshape(-1, 1)

        return self.model.predict_proba(x)[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

    def fit(self, raw_prob, y):
        x = np.asarray(raw_prob, dtype=float)
        y = np.asarray(y, dtype=int)

        self.model.fit(x, y)

        return self

    def predict(self, raw_prob):
        x = np.asarray(raw_prob, dtype=float)

        return np.asarray(self.model.predict(x), dtype=float)


def build_calibrator(method):
    method = str(method).upper()

    if method == "PLATT":
        return PlattCalibrator()

    if method == "ISOTONIC":
        return IsotonicCalibrator()

    raise ValueError(f"Metodo calibrazione non riconosciuto: {method}")


# ============================================================
# CONTROLLI INPUT
# ============================================================

section(
    "MARKET SENTINEL - V40.40b\n"
    "FULL200 WEEKLY REVERSAL - FINAL PRODUCTION FIT\n"
    "CALIBRATION POLICY PREDECLARED V40.25"
)

required_files = [
    FEATURE_DATA_FILE,
    FEATURE_LIST_FILE,
    OOS_PREDICTIONS_FILE,
    OOS_SUMMARY_FILE,
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(f"File richiesto non trovato: {file}")

print("Input trovati:")
for file in required_files:
    print(f"  OK - {file}")


# ============================================================
# RECUPERO FEATURE LIST
# ============================================================

section("RECUPERO 413 FEATURE CONGELATE")

feature_list_df = pd.read_csv(FEATURE_LIST_FILE)

if len(feature_list_df.columns) == 0:
    raise RuntimeError("Feature list vuota.")

feature_col_name = None

for candidate in ["FEATURE", "Feature", "feature", "FEATURE_NAME", "FeatureName"]:
    if candidate in feature_list_df.columns:
        feature_col_name = candidate
        break

if feature_col_name is None:
    feature_col_name = feature_list_df.columns[0]

feature_cols = (
    feature_list_df[feature_col_name]
    .dropna()
    .astype(str)
    .tolist()
)

feature_cols = list(dict.fromkeys(feature_cols))

print(f"Feature recuperate: {len(feature_cols)}")

if len(feature_cols) != 413:
    raise RuntimeError(
        f"Attese 413 feature, trovate {len(feature_cols)}."
    )


# ============================================================
# CARICAMENTO METADATA FULL200
# ============================================================

section("CARICAMENTO METADATA FULL200")

header = pd.read_csv(FEATURE_DATA_FILE, nrows=0)
all_columns = list(header.columns)

ticker_col = find_column(
    all_columns,
    ["Ticker", "TICKER", "ticker"]
)

date_col = find_column(
    all_columns,
    ["Date", "DATE", "date"]
)

phase_col = find_column(
    all_columns,
    ["PHASE", "Phase", "phase"]
)

if ticker_col is None:
    raise RuntimeError("Colonna Ticker non trovata.")

if date_col is None:
    raise RuntimeError("Colonna Date non trovata.")

if phase_col is None:
    raise RuntimeError("Colonna PHASE non trovata.")

missing_features = [
    c for c in feature_cols
    if c not in all_columns
]

if missing_features:
    raise RuntimeError(
        f"Feature mancanti nel dataset FULL200: {missing_features[:20]}"
    )

metadata = pd.read_csv(
    FEATURE_DATA_FILE,
    usecols=[ticker_col, date_col, phase_col],
)

metadata[date_col] = pd.to_datetime(
    metadata[date_col],
    utc=True,
    errors="coerce",
).dt.tz_convert(None)

metadata = metadata.reset_index(drop=True)

metadata["PHASE_FAMILY"] = metadata[phase_col].apply(
    phase_family
)

print(f"Righe:      {len(metadata)}")
print(f"Ticker:     {metadata[ticker_col].nunique()}")
print(
    f"Periodo:    "
    f"{metadata[date_col].min().date()} -> "
    f"{metadata[date_col].max().date()}"
)
print(
    f"Duplicati:  "
    f"{metadata.duplicated(subset=[ticker_col, date_col]).sum()}"
)

if len(metadata) != 141886:
    raise RuntimeError(
        f"Attese 141886 righe FULL200, trovate {len(metadata)}."
    )

if metadata[ticker_col].nunique() != 200:
    raise RuntimeError(
        f"Attesi 200 ticker, trovati {metadata[ticker_col].nunique()}."
    )

if metadata.duplicated(
    subset=[ticker_col, date_col]
).sum() != 0:
    raise RuntimeError("Duplicati Ticker+Date nel FULL200.")


# ============================================================
# COSTRUZIONE TRUE REVERSAL TARGET
# ============================================================

section("COSTRUZIONE TRUE REVERSAL TARGET PRODUCTION")

targets = {}

for target_name, cfg in TARGET_CONFIG.items():

    horizon = cfg["horizon"]
    current_family = cfg["family"]
    future_family_target = cfg["future_family"]

    future_family = (
        metadata
        .groupby(ticker_col, sort=False)["PHASE_FAMILY"]
        .shift(-horizon)
    )

    valid_future = future_family.notna()

    target = pd.Series(
        np.nan,
        index=metadata.index,
        dtype=float,
    )

    eligible = (
        (metadata["PHASE_FAMILY"] == current_family)
        & valid_future
    )

    target.loc[eligible] = (
        future_family.loc[eligible] == future_family_target
    ).astype(float)

    targets[target_name] = target

    valid = target.notna()

    n = int(valid.sum())
    positives = int(target.loc[valid].sum())
    rate = positives / n if n else np.nan

    print(
        f"{target_name:<25} "
        f"N={n:>7} "
        f"Positive={positives:>6} "
        f"Rate={rate:.4f}"
    )


# ============================================================
# VALIDAZIONE OOS V40.40a
#
# Non scegliamo il calibratore qui.
# Controlliamo soltanto che la validazione OOS sia quella
# attesa e documentiamo la scelta esplorativa V40.40a.
# ============================================================

section("AUDIT V40.40a - VALIDAZIONE OOS")

oos_summary = pd.read_csv(OOS_SUMMARY_FILE)

target_summary_col = find_column(
    oos_summary.columns,
    ["TARGET", "Target", "target"]
)

if target_summary_col is None:
    raise RuntimeError(
        "Colonna TARGET non trovata in v40_40a_summary.csv"
    )

selected_summary_col = find_column(
    oos_summary.columns,
    [
        "SELECTED_CALIBRATION",
        "CALIBRATION_METHOD",
        "SELECTED_METHOD",
    ],
)

raw_auc_col = find_column(
    oos_summary.columns,
    [
        "RAW_ROC_AUC",
        "RAW_AUC",
        "ROC_AUC_RAW",
    ],
)

summary_targets = set(
    oos_summary[target_summary_col]
    .dropna()
    .astype(str)
)

expected_targets = set(TARGET_CONFIG.keys())

missing_summary_targets = expected_targets - summary_targets

if missing_summary_targets:
    raise RuntimeError(
        "Target mancanti nel summary V40.40a: "
        f"{sorted(missing_summary_targets)}"
    )

print("Target V40.40a presenti: OK")

if raw_auc_col is not None:

    auc_check = (
        oos_summary[
            oos_summary[target_summary_col].isin(expected_targets)
        ][
            [target_summary_col, raw_auc_col]
        ]
        .drop_duplicates(subset=[target_summary_col])
        .copy()
    )

    auc_check[raw_auc_col] = pd.to_numeric(
        auc_check[raw_auc_col],
        errors="coerce",
    )

    print()
    print("RAW OOS AUC V40.40a:")

    for _, row in auc_check.iterrows():
        print(
            f"  {row[target_summary_col]:<25} "
            f"{row[raw_auc_col]:.6f}"
        )

    if auc_check[raw_auc_col].isna().any():
        raise RuntimeError(
            "AUC RAW V40.40a non leggibile."
        )

    if (auc_check[raw_auc_col] < 0.65).any():
        raise RuntimeError(
            "Almeno un target V40.40a ha RAW AUC < 0.65."
        )

    mean_auc = float(
        auc_check[raw_auc_col].mean()
    )

    count_070 = int(
        (auc_check[raw_auc_col] >= 0.70).sum()
    )

    print()
    print(f"Mean RAW AUC:      {mean_auc:.6f}")
    print(f"Target >= 0.70:    {count_070}/6")

    if mean_auc < 0.75:
        raise RuntimeError(
            "Mean RAW AUC V40.40a < 0.75."
        )

    if count_070 < 5:
        raise RuntimeError(
            "Meno di 5 target su 6 con RAW AUC >= 0.70."
        )

else:
    print(
        "Colonna RAW AUC non identificata nel summary. "
        "Audit quantitativo demandato all'output V40.40a già congelato."
    )


# ============================================================
# AUDIT DELLA SCELTA ESPLORATIVA V40.40a
#
# NON deve coincidere con la policy production.
# Viene solo registrata.
# ============================================================

observed_exploratory = {}

if selected_summary_col is not None:

    print()
    print("Scelta calibrazione osservata in V40.40a:")
    print("(DIAGNOSTICA ESPLORATIVA - NON POLICY PRODUCTION)")
    print()

    for target_name in TARGET_CONFIG:

        rows = oos_summary[
            oos_summary[target_summary_col].astype(str)
            == target_name
        ]

        if len(rows) == 0:
            continue

        method = str(
            rows.iloc[0][selected_summary_col]
        ).upper()

        observed_exploratory[target_name] = method

        print(
            f"  {target_name:<25} "
            f"V40.40a={method:<10} "
            f"PRODUCTION={PRODUCTION_CALIBRATION[target_name]}"
        )

else:
    print()
    print(
        "Colonna SELECTED_CALIBRATION non trovata. "
        "Nessun problema: non è necessaria per la policy production."
    )


# ============================================================
# POLICY PRODUCTION PREDECLARED
# ============================================================

section("CALIBRAZIONE PRODUCTION PREDECLARED V40.25")

for target_name in TARGET_CONFIG:

    print(
        f"{target_name:<25} "
        f"-> {PRODUCTION_CALIBRATION[target_name]}"
    )

print()
print(
    "NOTA METODOLOGICA:"
)
print(
    "La scelta PLATT/ISOTONIC NON deriva dalle performance "
    "osservate in V40.40a."
)
print(
    "È la policy storica V40.25, stabilita prima della "
    "validazione FULL200."
)


# ============================================================
# CARICAMENTO OOS PREDICTIONS V40.40a
#
# Le prediction OOS vengono utilizzate per FITTARE i parametri
# del calibratore production.
#
# NON vengono utilizzate per scegliere il TIPO di calibratore.
# ============================================================

section("CARICAMENTO OOS PREDICTIONS V40.40a")

oos = pd.read_csv(OOS_PREDICTIONS_FILE)

oos_target_col = find_column(
    oos.columns,
    ["TARGET", "Target", "target"]
)

oos_y_col = find_column(
    oos.columns,
    [
        "Y_TRUE",
        "y_true",
        "TARGET_VALUE",
        "ACTUAL",
    ],
)

oos_prob_col = find_column(
    oos.columns,
    [
        "Y_PROB",
        "RAW_PROB",
        "RAW_PROBABILITY",
        "PROBABILITY",
        "PRED_PROB",
    ],
)

if oos_target_col is None:
    raise RuntimeError(
        "TARGET non trovato nelle OOS predictions."
    )

if oos_y_col is None:
    raise RuntimeError(
        "Y_TRUE non trovato nelle OOS predictions."
    )

if oos_prob_col is None:
    raise RuntimeError(
        "Probabilità RAW non trovata nelle OOS predictions."
    )

oos[oos_y_col] = pd.to_numeric(
    oos[oos_y_col],
    errors="coerce",
)

oos[oos_prob_col] = pd.to_numeric(
    oos[oos_prob_col],
    errors="coerce",
)

print(f"Prediction OOS totali: {len(oos)}")
print(f"Target OOS: {oos[oos_target_col].nunique()}")


# ============================================================
# MEMMAP FEATURE MATRIX
# ============================================================

section("PREPARAZIONE MATRICE FEATURE FULL200")

expected_shape = (
    len(metadata),
    len(feature_cols),
)

expected_bytes = (
    expected_shape[0]
    * expected_shape[1]
    * np.dtype("float32").itemsize
)

use_existing_memmap = False

if MEMMAP_FILE.exists():

    actual_bytes = MEMMAP_FILE.stat().st_size

    if actual_bytes == expected_bytes:
        use_existing_memmap = True

        print("Memmap V40.40a compatibile trovato.")
        print(f"File: {MEMMAP_FILE}")
        print(
            f"Shape: "
            f"{expected_shape[0]} x {expected_shape[1]}"
        )

    else:
        print(
            "Memmap esistente non compatibile. "
            "Verrà ricostruito."
        )


if not use_existing_memmap:

    print("Ricostruzione memmap float32...")

    X_mem = np.memmap(
        MEMMAP_FILE,
        dtype="float32",
        mode="w+",
        shape=expected_shape,
    )

    row_start = 0
    chunk_size = 4000

    for chunk_id, chunk in enumerate(
        pd.read_csv(
            FEATURE_DATA_FILE,
            usecols=feature_cols,
            chunksize=chunk_size,
        ),
        start=1,
    ):

        chunk = chunk.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        for col in feature_cols:
            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="coerce",
            )

        arr = chunk.to_numpy(
            dtype=np.float32,
            copy=True,
        )

        row_end = row_start + len(arr)

        X_mem[row_start:row_end, :] = arr

        print(
            f"Chunk {chunk_id:02d}: "
            f"{row_start + 1:>7} -> "
            f"{row_end:>7} / {expected_shape[0]}"
        )

        row_start = row_end

        del chunk
        del arr
        gc.collect()

    X_mem.flush()

    if row_start != expected_shape[0]:
        raise RuntimeError(
            "Numero righe memmap diverso dal dataset."
        )

    del X_mem
    gc.collect()


X_mem = np.memmap(
    MEMMAP_FILE,
    dtype="float32",
    mode="r",
    shape=expected_shape,
)

print()
print("Matrice feature pronta.")
print(
    f"Shape: {X_mem.shape[0]} x {X_mem.shape[1]}"
)


# ============================================================
# FIT PRODUCTION
# ============================================================

section("FIT FINAL PRODUCTION FULL200")

model_manifest = []
calibration_manifest = []
training_summary = []

for target_index, (target_name, cfg) in enumerate(
    TARGET_CONFIG.items(),
    start=1,
):

    section(
        f"[{target_index}/6] {target_name}"
    )

    target_series = targets[target_name]

    valid_mask = target_series.notna().to_numpy()

    idx = np.flatnonzero(valid_mask)

    y = (
        target_series.iloc[idx]
        .astype(int)
        .to_numpy()
    )

    train_n = len(idx)
    positive_n = int(y.sum())
    base_rate = float(y.mean())

    train_dates = metadata.iloc[idx][date_col]

    train_start = train_dates.min()
    train_end = train_dates.max()

    train_tickers = metadata.iloc[idx][ticker_col].nunique()

    print(f"Family:      {cfg['family']}")
    print(f"Horizon:     {cfg['horizon']}W")
    print(f"Train N:     {train_n}")
    print(f"Positive:    {positive_n}")
    print(f"Base rate:   {base_rate:.6f}")
    print(f"Ticker:      {train_tickers}")
    print(
        f"Periodo:     "
        f"{train_start.date()} -> {train_end.date()}"
    )

    if train_tickers != 200:
        raise RuntimeError(
            f"{target_name}: attesi 200 ticker, "
            f"trovati {train_tickers}."
        )

    if len(np.unique(y)) != 2:
        raise RuntimeError(
            f"{target_name}: target non binario."
        )

    # --------------------------------------------------------
    # Copia SOLO del campione necessario
    # --------------------------------------------------------

    print()
    print("Copia matrice training in RAM...")

    X_train = np.asarray(
        X_mem[idx, :],
        dtype=np.float32,
    ).copy()

    # --------------------------------------------------------
    # Imputation train-only
    #
    # In production il training set è tutta la storia
    # disponibile. Le mediane vengono quindi calcolate
    # esclusivamente sul training production.
    # --------------------------------------------------------

    print("Calcolo mediane training...")

    with np.errstate(
        all="ignore"
    ):
        medians = np.nanmedian(
            X_train,
            axis=0,
        )

    medians = np.asarray(
        medians,
        dtype=np.float32,
    )

    medians[~np.isfinite(medians)] = 0.0

    nan_rows, nan_cols = np.where(
        ~np.isfinite(X_train)
    )

    if len(nan_rows) > 0:
        X_train[nan_rows, nan_cols] = medians[nan_cols]

    if not np.isfinite(X_train).all():
        raise RuntimeError(
            f"{target_name}: NaN/inf rimasti dopo imputation."
        )

    # --------------------------------------------------------
    # RANDOM FOREST PRODUCTION
    # --------------------------------------------------------

    print("Training RandomForest production...")

    rf = RandomForestClassifier(
        **RF_PARAMS
    )

    rf.fit(
        X_train,
        y,
    )

    # --------------------------------------------------------
    # FIT CALIBRATORE
    #
    # TIPO = PREDECLARED V40.25
    # PARAMETRI = FIT su tutte le prediction OOS V40.40a
    # --------------------------------------------------------

    method = PRODUCTION_CALIBRATION[
        target_name
    ]

    oos_target = oos[
        oos[oos_target_col].astype(str)
        == target_name
    ].copy()

    oos_target = oos_target[
        np.isfinite(oos_target[oos_y_col])
        & np.isfinite(oos_target[oos_prob_col])
    ].copy()

    if len(oos_target) == 0:
        raise RuntimeError(
            f"{target_name}: nessuna OOS prediction "
            "disponibile per calibrazione."
        )

    y_cal = (
        oos_target[oos_y_col]
        .astype(int)
        .to_numpy()
    )

    p_cal = (
        oos_target[oos_prob_col]
        .astype(float)
        .to_numpy()
    )

    if len(np.unique(y_cal)) != 2:
        raise RuntimeError(
            f"{target_name}: OOS calibration target "
            "non contiene entrambe le classi."
        )

    print()
    print(
        f"Calibratore production: {method}"
    )
    print(
        f"OOS calibration N:      {len(y_cal)}"
    )
    print(
        f"OOS event rate:         {y_cal.mean():.6f}"
    )

    calibrator = build_calibrator(
        method
    )

    calibrator.fit(
        p_cal,
        y_cal,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    model_file = (
        OUTPUT_DIR
        / f"{target_name}_random_forest.pkl"
    )

    median_file = (
        OUTPUT_DIR
        / f"{target_name}_medians.npy"
    )

    calibrator_file = (
        OUTPUT_DIR
        / f"{target_name}_{method.lower()}_calibrator.pkl"
    )

    feature_file = (
        OUTPUT_DIR
        / f"{target_name}_features.json"
    )

    with open(
        model_file,
        "wb",
    ) as f:
        pickle.dump(
            rf,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    np.save(
        median_file,
        medians,
    )

    with open(
        calibrator_file,
        "wb",
    ) as f:
        pickle.dump(
            calibrator,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with open(
        feature_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_cols,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"MODEL:       {model_file}")
    print(f"MEDIANS:     {median_file}")
    print(f"CALIBRATOR:  {calibrator_file}")
    print(f"FEATURES:    {feature_file}")

    model_manifest.append(
        {
            "TARGET": target_name,
            "HORIZON_W": cfg["horizon"],
            "PHASE_FAMILY": cfg["family"],
            "TRAIN_N": train_n,
            "POSITIVE_N": positive_n,
            "BASE_RATE": base_rate,
            "TICKERS": train_tickers,
            "TRAIN_START": train_start.date(),
            "TRAIN_END": train_end.date(),
            "FEATURES": len(feature_cols),
            "MODEL_FILE": str(model_file),
            "MEDIAN_FILE": str(median_file),
            "FEATURE_FILE": str(feature_file),
        }
    )

    calibration_manifest.append(
        {
            "TARGET": target_name,
            "METHOD": method,
            "POLICY": "PREDECLARED_V40_25",
            "OOS_N": len(y_cal),
            "OOS_EVENT_RATE": float(
                y_cal.mean()
            ),
            "CALIBRATION_SAMPLE": (
                "V40.40a corrected-purging OOS predictions"
            ),
            "METHOD_SELECTION_SOURCE": (
                "Historical frozen V40.25 policy"
            ),
            "V40_37B_EXPLORATORY_METHOD": (
                observed_exploratory.get(
                    target_name,
                    V40_40A_EXPLORATORY_SELECTION[
                        target_name
                    ],
                )
            ),
            "EXPLORATORY_METHOD_USED_FOR_PRODUCTION": False,
            "CALIBRATOR_FILE": str(
                calibrator_file
            ),
        }
    )

    training_summary.append(
        {
            "TARGET": target_name,
            "TRAIN_N": train_n,
            "POSITIVE_N": positive_n,
            "NEGATIVE_N": train_n - positive_n,
            "BASE_RATE": base_rate,
            "FEATURES": len(feature_cols),
            "RF_TREES": RF_PARAMS[
                "n_estimators"
            ],
            "RF_MAX_FEATURES": RF_PARAMS[
                "max_features"
            ],
            "RF_MIN_SAMPLES_LEAF": RF_PARAMS[
                "min_samples_leaf"
            ],
            "RF_CLASS_WEIGHT": RF_PARAMS[
                "class_weight"
            ],
            "RF_RANDOM_STATE": RF_PARAMS[
                "random_state"
            ],
            "CALIBRATION_METHOD": method,
            "CALIBRATION_POLICY": (
                "PREDECLARED_V40_25"
            ),
        }
    )

    del X_train
    del y
    del rf
    del medians
    del calibrator
    del oos_target
    del y_cal
    del p_cal
    del idx

    gc.collect()

    print()
    print(
        f"OK - {target_name} PRODUCTION COMPLETATO."
    )


# ============================================================
# SALVATAGGIO MANIFEST
# ============================================================

section("SALVATAGGIO MANIFEST PRODUCTION")

model_manifest_df = pd.DataFrame(
    model_manifest
)

calibration_manifest_df = pd.DataFrame(
    calibration_manifest
)

training_summary_df = pd.DataFrame(
    training_summary
)

model_manifest_df.to_csv(
    MODEL_MANIFEST_FILE,
    index=False,
)

calibration_manifest_df.to_csv(
    CALIBRATION_MANIFEST_FILE,
    index=False,
)

training_summary_df.to_csv(
    TRAINING_SUMMARY_FILE,
    index=False,
)

print(model_manifest_df[
    [
        "TARGET",
        "HORIZON_W",
        "PHASE_FAMILY",
        "TRAIN_N",
        "POSITIVE_N",
        "BASE_RATE",
        "TICKERS",
        "TRAIN_START",
        "TRAIN_END",
        "FEATURES",
    ]
].to_string(index=False))

print()

print(calibration_manifest_df[
    [
        "TARGET",
        "METHOD",
        "POLICY",
        "OOS_N",
        "OOS_EVENT_RATE",
        "V40_37B_EXPLORATORY_METHOD",
        "EXPLORATORY_METHOD_USED_FOR_PRODUCTION",
    ]
].to_string(index=False))


# ============================================================
# METADATA FINALE
# ============================================================

metadata_json = {
    "version": "V40.40b",
    "component": "WEEKLY_REVERSAL_PRODUCTION",
    "status": "FINAL_PRODUCTION_FIT",
    "universe": "FULL200",
    "ticker_count": 200,
    "feature_count": 413,

    "training_dataset": str(
        FEATURE_DATA_FILE
    ),

    "training_rows_total": int(
        len(metadata)
    ),

    "training_period_start": str(
        metadata[date_col].min().date()
    ),

    "training_period_end": str(
        metadata[date_col].max().date()
    ),

    "targets": list(
        TARGET_CONFIG.keys()
    ),

    "horizons_weeks": [
        1,
        2,
        3,
    ],

    "random_forest": RF_PARAMS,

    "production_calibration_policy": {
        "name": "PREDECLARED_V40_25",
        "description": (
            "Calibration method mapping was frozen historically "
            "in V40.25 before the V40.37 FULL200 OOS validation."
        ),
        "mapping": PRODUCTION_CALIBRATION,
    },

    "v40_40a_role": {
        "raw_model_validation": (
            "Corrected temporal OOS validation with unique weeks "
            "and whole-week purging."
        ),
        "calibration_sample": (
            "OOS raw predictions used to fit the parameters "
            "of the predeclared production calibrator."
        ),
        "calibration_method_selection": (
            "Exploratory only. NOT used for production method selection."
        ),
        "exploratory_mapping": (
            observed_exploratory
            if observed_exploratory
            else V40_40A_EXPLORATORY_SELECTION
        ),
    },

    "leakage_policy": {
        "temporal_split": True,
        "unique_week_dates": True,
        "whole_week_purging": True,
        "train_only_imputation": True,
        "manual_weights": False,
        "random_split": False,
        "calibration_method_selected_on_v40_40a_oos": False,
    },

    "production_notes": [
        (
            "V40.40a raw OOS validation remains the independent "
            "performance benchmark for the FULL200 model."
        ),
        (
            "V40.40a PLATT vs ISOTONIC comparison is retained "
            "only as exploratory diagnostic information."
        ),
        (
            "Production calibration method is predeclared from "
            "historical V40.25 policy."
        ),
        (
            "Calibration parameters are fitted using V40.40a "
            "OOS raw probabilities."
        ),
        (
            "Production RandomForest is fitted on all eligible "
            "FULL200 historical observations."
        ),
        (
            "No new tuning was performed in V40.40b."
        ),
    ],
}

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata_json,
        f,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


# ============================================================
# CONTROLLI FINALI
# ============================================================

section("CONTROLLI FINALI V40.40b")

expected_model_files = []
expected_median_files = []
expected_calibrator_files = []
expected_feature_files = []

for target_name in TARGET_CONFIG:

    method = PRODUCTION_CALIBRATION[
        target_name
    ]

    expected_model_files.append(
        OUTPUT_DIR
        / f"{target_name}_random_forest.pkl"
    )

    expected_median_files.append(
        OUTPUT_DIR
        / f"{target_name}_medians.npy"
    )

    expected_calibrator_files.append(
        OUTPUT_DIR
        / f"{target_name}_{method.lower()}_calibrator.pkl"
    )

    expected_feature_files.append(
        OUTPUT_DIR
        / f"{target_name}_features.json"
    )


models_ok = all(
    p.exists()
    for p in expected_model_files
)

medians_ok = all(
    p.exists()
    for p in expected_median_files
)

calibrators_ok = all(
    p.exists()
    for p in expected_calibrator_files
)

features_ok = all(
    p.exists()
    for p in expected_feature_files
)

targets_ok = (
    set(model_manifest_df["TARGET"])
    == set(TARGET_CONFIG.keys())
)

policy_ok = (
    set(
        calibration_manifest_df[
            "POLICY"
        ]
    )
    == {"PREDECLARED_V40_25"}
)

exploratory_not_used = (
    calibration_manifest_df[
        "EXPLORATORY_METHOD_USED_FOR_PRODUCTION"
    ]
    .eq(False)
    .all()
)

print(
    f"Modelli attesi:             "
    f"{len(TARGET_CONFIG)}"
)

print(
    f"Modelli prodotti:           "
    f"{len(model_manifest_df)}"
)

print(
    f"Target corretti:            "
    f"{targets_ok}"
)

print(
    f"File modello presenti:      "
    f"{models_ok}"
)

print(
    f"File mediane presenti:      "
    f"{medians_ok}"
)

print(
    f"File calibratori presenti:  "
    f"{calibrators_ok}"
)

print(
    f"File feature presenti:      "
    f"{features_ok}"
)

print(
    f"Policy PREDECLARED:         "
    f"{policy_ok}"
)

print(
    f"Selection OOS non usata:    "
    f"{exploratory_not_used}"
)


all_ok = all(
    [
        len(model_manifest_df) == 6,
        targets_ok,
        models_ok,
        medians_ok,
        calibrators_ok,
        features_ok,
        policy_ok,
        exploratory_not_used,
    ]
)

if not all_ok:
    raise RuntimeError(
        "V40.40b: almeno un controllo finale NON è passato."
    )


# ============================================================
# VERDETTO
# ============================================================

section("VERDETTO V40.40b")

print(
    "OK - WEEKLY REVERSAL PRODUCTION MODEL COMPLETATO."
)

print()
print("Ticker:                  200")
print("Feature:                 413")
print("Modelli RF:              6")
print("Orizzonti:               1W / 2W / 3W")
print("Configurazione RF:       CONGELATA")
print("Calibrazione method:     PREDECLARED V40.25")
print("Calibration sample:      V40.40a OOS")
print("Selection su OOS:        NO")
print("Nuovo tuning:            NO")
print("Pesi manuali:            NO")
print("Production model:        SI")

print()
print(
    "V40.40b WEEKLY REVERSAL PRODUCTION "
    "DEFINITIVO CONGELATO."
)

print()
print("File principali:")
print(f"  {MODEL_MANIFEST_FILE}")
print(f"  {CALIBRATION_MANIFEST_FILE}")
print(f"  {TRAINING_SUMMARY_FILE}")
print(f"  {METADATA_FILE}")
print(f"  {OUTPUT_DIR}")

print()
print(
    "NEXT STEP: WEEKLY FULL200 OPERATIONAL REPORT + PDF."
)