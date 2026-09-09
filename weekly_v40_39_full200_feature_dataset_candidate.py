# ==================================================================================================
# MARKET SENTINEL - V40.39
# FULL200 WEEKLY PHASE + FEATURE DATASET
# MEMORY-SAFE VERSION
# ==================================================================================================
#
# SCOPO
# -----
#
# Questa versione replica la logica scientifica di V40.36 ma riduce
# significativamente il consumo di RAM.
#
# CAMBIA SOLO:
#
#   - motore PHASE: V40.10 -> V40.39 CANONICAL (V5 validata)
#
# NON CAMBIA:
#
#   - V40.12
#   - V40.15
#   - indicatori
#   - breadth FULL200
#   - 413 feature ML
#   - criteri anti-leakage
#
# OTTIMIZZAZIONI:
#
#   1. recupero anticipato delle 413 feature necessarie
#   2. eliminazione delle colonne inutili appena possibile
#   3. eliminazione esplicita dei DataFrame intermedi
#   4. garbage collection
#   5. conversione delle feature ML a float32
#   6. salvataggio CSV a chunk
#   7. nessuna copia completa inutile del dataset finale
#
# INPUT
# -----
#
# data/v40_35_full200_weekly.csv
# data/v40_18_ml_features.csv
#
# OUTPUT
# ------
#
# data/v40_39_full200_phase12.csv
# data/v40_39_full200_features.csv
# data/v40_39_feature_list.csv
# data/v40_39_feature_coverage.csv
# data/v40_39_phase_distribution.csv
# data/v40_39_source_distribution.csv
# data/v40_39_metadata.json
#
# ==================================================================================================

from __future__ import annotations

import gc
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import weekly_v40_39_phase_canonical as v4039
import weekly_v40_12_phase_strength_dataset as v4012
import weekly_v40_15_score_feature_dataset as v4015


warnings.filterwarnings("ignore")


# ==================================================================================================
# VERSION
# ==================================================================================================

VERSION = "V40.39"

DATA_DIR = Path("data")

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================================================================
# INPUT
# ==================================================================================================

FULL200_FILE = (
    DATA_DIR
    /
    "v40_35_full200_weekly.csv"
)

REFERENCE_FEATURE_FILE = (
    DATA_DIR
    /
    "v40_18_ml_features.csv"
)


# ==================================================================================================
# OUTPUT
# ==================================================================================================

OUT_PHASE12 = (
    DATA_DIR
    /
    "v40_39_full200_phase12.csv"
)

OUT_FEATURES = (
    DATA_DIR
    /
    "v40_39_full200_features.csv"
)

OUT_FEATURE_LIST = (
    DATA_DIR
    /
    "v40_39_feature_list.csv"
)

OUT_FEATURE_COVERAGE = (
    DATA_DIR
    /
    "v40_39_feature_coverage.csv"
)

OUT_PHASE_DISTRIBUTION = (
    DATA_DIR
    /
    "v40_39_phase_distribution.csv"
)

OUT_SOURCE_DISTRIBUTION = (
    DATA_DIR
    /
    "v40_39_source_distribution.csv"
)

OUT_METADATA = (
    DATA_DIR
    /
    "v40_39_metadata.json"
)


# ==================================================================================================
# EXPECTED
# ==================================================================================================

EXPECTED_TICKERS = 200
EXPECTED_FEATURES = 413

FROZEN_END_DATE = pd.Timestamp(
    "2026-08-28"
)

CSV_CHUNK_SIZE = 10000


# ==================================================================================================
# UTILITY
# ==================================================================================================

def section(title):

    print()
    print("=" * 150)
    print(title)
    print("=" * 150)
    print()


def memory_mb(df):

    if df is None:

        return 0.0

    try:

        return (
            df.memory_usage(
                deep=True
            ).sum()
            /
            1024
            /
            1024
        )

    except Exception:

        return np.nan


def print_memory(
    name,
    df,
):

    mb = memory_mb(
        df
    )

    print(
        f"{name:<35} "
        f"{mb:10.1f} MB"
    )


def collect_memory():

    gc.collect()


# ==================================================================================================
# NORMALIZE
# ==================================================================================================

def normalize_dataset(df):

    if "Ticker" in df.columns:

        df["Ticker"] = (
            df["Ticker"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    if "Date" in df.columns:

        df["Date"] = (
            pd.to_datetime(
                df["Date"],
                utc=True,
                errors="coerce",
            )
            .dt.tz_convert(None)
        )

    return df


# ==================================================================================================
# PHASE HELPERS
# ==================================================================================================

def normalize_phase(value):

    return (
        str(value)
        .upper()
        .strip()
        .replace(
            " ",
            "_",
        )
    )


def phase_family(value):

    p = normalize_phase(
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
# RECOVER EXACT 413 FEATURES
#
# Replica il criterio già utilizzato in V40.36 / V40.24b.
# ==================================================================================================

def recover_reference_features(
    reference,
):

    excluded_exact = {
        "Ticker",
        "Date",
        "PHASE",
        "PHASE_CATEGORY",
        "PHASE_FAMILY",
    }

    candidate_features = []

    for col in reference.columns:

        if col in excluded_exact:

            continue

        upper = str(
            col
        ).upper()

        if upper.startswith(
            "TARGET_"
        ):

            continue

        if upper.startswith(
            "FUTURE_"
        ):

            continue

        if (
            "PHASE_"
            in upper
            or
            upper
            ==
            "PHASE"
        ):

            continue

        if (
            upper.endswith(
                "_ID"
            )
            or
            "RUN_ID"
            in upper
            or
            upper
            ==
            "INDEX"
        ):

            continue

        if pd.api.types.is_numeric_dtype(
            reference[
                col
            ]
        ):

            candidate_features.append(
                col
            )

    clean_features = []

    for col in candidate_features:

        s = pd.to_numeric(
            reference[
                col
            ],
            errors="coerce",
        )

        s = s.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        if s.notna().sum() < 100:

            continue

        if s.nunique(
            dropna=True
        ) <= 1:

            continue

        clean_features.append(
            col
        )

    return clean_features


# ==================================================================================================
# PHASE V40.39 + V40.12
# ==================================================================================================

def build_phase12(
    raw,
):

    results = []

    tickers = sorted(
        raw[
            "Ticker"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    total = len(
        tickers
    )

    for i, ticker in enumerate(
        tickers,
        start=1,
    ):

        mask = (
            raw[
                "Ticker"
            ]
            ==
            ticker
        )

        g = (
            raw.loc[
                mask
            ]
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        source = None

        if (
            "SOURCE_UNIVERSE"
            in g.columns
        ):

            source_values = (
                g[
                    "SOURCE_UNIVERSE"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if source_values:

                source = (
                    source_values[
                        0
                    ]
                )

        raw_n = len(
            g
        )

        # ------------------------------------------------------------------------------------------
        # V40.39 PHASE CANONICAL
        # ------------------------------------------------------------------------------------------

        phase10 = (
            v4039.process_ticker(
                g
            )
        )

        del g

        # ------------------------------------------------------------------------------------------
        # V40.12
        # ------------------------------------------------------------------------------------------

        phase10 = (
            v4012.normalize_base_columns(
                phase10
            )
        )

        phase12 = (
            v4012.add_phase_strength_features(
                phase10
            )
        )

        del phase10

        phase12 = normalize_dataset(
            phase12
        )

        if source is not None:

            phase12[
                "SOURCE_UNIVERSE"
            ] = source

        results.append(
            phase12
        )

        print(
            f"[{i:03d}/{total:03d}] "
            f"{ticker:<10} "
            f"raw={raw_n:4d} "
            f"-> phase12={len(phase12):4d}"
        )

        if (
            i
            %
            20
            ==
            0
        ):

            collect_memory()

    if not results:

        raise RuntimeError(
            "Nessun ticker elaborato."
        )

    out = pd.concat(
        results,
        ignore_index=True,
        copy=False,
    )

    del results

    collect_memory()

    out = normalize_dataset(
        out
    )

    out.sort_values(
        [
            "Ticker",
            "Date",
        ],
        inplace=True,
    )

    out.reset_index(
        drop=True,
        inplace=True,
    )

    return out


# ==================================================================================================
# REDUCE PHASE DATASET
#
# Conserviamo:
#
# - colonne operative
# - PHASE
# - contesto PHASE
# - tutte le feature ML reference già presenti
#
# ==================================================================================================

def reduce_phase_dataset(
    phase12,
    reference_features,
):

    core = [
        "Ticker",
        "Date",
        "SOURCE_UNIVERSE",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "PHASE",
        "V4012_PHASE_DIRECTION",
        "V4012_PHASE_AGE",
        "V4012_PHASE_CHANGED",
    ]

    wanted = []

    for col in core:

        if (
            col
            in phase12.columns
            and
            col
            not in wanted
        ):

            wanted.append(
                col
            )

    for col in reference_features:

        if (
            col
            in phase12.columns
            and
            col
            not in wanted
        ):

            wanted.append(
                col
            )

    reduced = phase12.loc[
        :,
        wanted,
    ]

    return reduced


# ==================================================================================================
# NUMERIC OPTIMIZATION
# ==================================================================================================

def optimize_numeric_features(
    df,
    feature_cols,
):

    for i, col in enumerate(
        feature_cols,
        start=1,
    ):

        if col not in df.columns:

            continue

        s = pd.to_numeric(
            df[
                col
            ],
            errors="coerce",
        )

        s = s.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        df[
            col
        ] = s.astype(
            "float32"
        )

        if (
            i
            %
            50
            ==
            0
        ):

            collect_memory()

    return df


# ==================================================================================================
# FEATURE COVERAGE
# ==================================================================================================

def build_feature_coverage(
    dataset,
    feature_cols,
):

    rows = []

    total_rows = len(
        dataset
    )

    for col in feature_cols:

        s = dataset[
            col
        ]

        non_null = int(
            s.notna().sum()
        )

        missing = int(
            total_rows
            -
            non_null
        )

        unique = int(
            s.nunique(
                dropna=True
            )
        )

        rows.append(
            {
                "Feature":
                    col,

                "Rows":
                    total_rows,

                "NonNull":
                    non_null,

                "Missing":
                    missing,

                "MissingPct":
                    (
                        100.0
                        *
                        missing
                        /
                        total_rows
                    )
                    if total_rows
                    else np.nan,

                "Unique":
                    unique,
            }
        )

    coverage = pd.DataFrame(
        rows
    )

    coverage.sort_values(
        [
            "MissingPct",
            "Feature",
        ],
        ascending=[
            False,
            True,
        ],
        inplace=True,
    )

    coverage.reset_index(
        drop=True,
        inplace=True,
    )

    return coverage


# ==================================================================================================
# CHUNKED CSV SAVE
# ==================================================================================================

def save_csv_chunked(
    df,
    path,
):

    print(
        f"Salvataggio: {path}"
    )

    df.to_csv(
        path,
        index=False,
        chunksize=CSV_CHUNK_SIZE,
    )


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():

    section(
        "MARKET SENTINEL - V40.39\n"
        "FULL200 WEEKLY PHASE + FEATURE DATASET\n"
        "MEMORY-SAFE VERSION"
    )

    # ==============================================================================================
    # INPUT CHECK
    # ==============================================================================================

    for path in [
        FULL200_FILE,
        REFERENCE_FEATURE_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File mancante: "
                f"{path}"
            )

    # ==============================================================================================
    # LOAD REFERENCE FIRST
    # ==============================================================================================

    section(
        "RECUPERO FEATURE REFERENCE V40.24b"
    )

    reference = pd.read_csv(
        REFERENCE_FEATURE_FILE,
        low_memory=False,
    )

    reference = normalize_dataset(
        reference
    )

    print_memory(
        "Reference V40.18:",
        reference,
    )

    reference_features = (
        recover_reference_features(
            reference
        )
    )

    print(
        f"Feature reference recuperate: "
        f"{len(reference_features)}"
    )

    if (
        len(
            reference_features
        )
        !=
        EXPECTED_FEATURES
    ):

        raise RuntimeError(
            f"Attese {EXPECTED_FEATURES} feature, "
            f"trovate {len(reference_features)}."
        )

    feature_list = pd.DataFrame(
        {
            "Feature":
                reference_features,

            "FeatureIndex":
                np.arange(
                    1,
                    len(
                        reference_features
                    )
                    +
                    1,
                ),
        }
    )

    feature_list.to_csv(
        OUT_FEATURE_LIST,
        index=False,
    )

    # ----------------------------------------------------------------------------------------------
    # Reference non serve più in RAM
    # ----------------------------------------------------------------------------------------------

    del reference

    collect_memory()

    print(
        "Reference V40.18 liberata dalla RAM."
    )

    # ==============================================================================================
    # LOAD FULL200
    # ==============================================================================================

    section(
        "CARICAMENTO FULL200"
    )

    full_raw = pd.read_csv(
        FULL200_FILE,
        low_memory=False,
    )

    full_raw = normalize_dataset(
        full_raw
    )

    print_memory(
        "FULL200 RAW:",
        full_raw,
    )

    print(
        f"Righe:      "
        f"{len(full_raw)}"
    )

    print(
        f"Ticker:     "
        f"{full_raw['Ticker'].nunique()}"
    )

    print(
        f"Periodo:    "
        f"{full_raw['Date'].min()} "
        f"-> "
        f"{full_raw['Date'].max()}"
    )

    duplicate_raw = int(
        full_raw
        .duplicated(
            subset=[
                "Ticker",
                "Date",
            ]
        )
        .sum()
    )

    print(
        f"Duplicati:  "
        f"{duplicate_raw}"
    )

    if (
        full_raw[
            "Ticker"
        ].nunique()
        !=
        EXPECTED_TICKERS
    ):

        raise RuntimeError(
            "FULL200 non contiene 200 ticker."
        )

    if duplicate_raw != 0:

        raise RuntimeError(
            "Duplicati Ticker+Date nel FULL200."
        )

    if (
        full_raw[
            "Date"
        ].max()
        >
        FROZEN_END_DATE
    ):

        raise RuntimeError(
            "FULL200 contiene date oltre "
            "il frozen end date."
        )

    # ==============================================================================================
    # SOURCE MAP
    # ==============================================================================================

    source_map = (
        full_raw[
            [
                "Ticker",
                "SOURCE_UNIVERSE",
            ]
        ]
        .drop_duplicates(
            subset=[
                "Ticker"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ==============================================================================================
    # PHASE
    # ==============================================================================================

    section(
        "V40.39 + V40.12"
    )

    phase12 = build_phase12(
        full_raw
    )

    print()

    print_memory(
        "PHASE12 completo:",
        phase12,
    )

    duplicate_phase = int(
        phase12
        .duplicated(
            subset=[
                "Ticker",
                "Date",
            ]
        )
        .sum()
    )

    print(
        f"Righe PHASE12: "
        f"{len(phase12)}"
    )

    print(
        f"Ticker PHASE12: "
        f"{phase12['Ticker'].nunique()}"
    )

    print(
        f"Duplicati PHASE12: "
        f"{duplicate_phase}"
    )

    if duplicate_phase != 0:

        raise RuntimeError(
            "Duplicati PHASE12."
        )

    if (
        phase12[
            "Ticker"
        ].nunique()
        !=
        EXPECTED_TICKERS
    ):

        raise RuntimeError(
            "PHASE12 non contiene 200 ticker."
        )

    # ==============================================================================================
    # PHASE DISTRIBUTION
    # ==============================================================================================

    if (
        "PHASE"
        not in phase12.columns
    ):

        raise RuntimeError(
            "Colonna PHASE mancante."
        )

    phase_distribution = (
        phase12[
            "PHASE"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "PHASE"
        )
        .reset_index(
            name="N"
        )
    )

    phase_distribution[
        "Pct"
    ] = (
        phase_distribution[
            "N"
        ]
        /
        len(
            phase12
        )
        *
        100.0
    )

    phase_distribution.to_csv(
        OUT_PHASE_DISTRIBUTION,
        index=False,
    )

    section(
        "DISTRIBUZIONE PHASE"
    )

    print(
        phase_distribution.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # ==============================================================================================
    # SAVE REDUCED PHASE12
    # ==============================================================================================

    reduced_phase = (
        reduce_phase_dataset(
            phase12,
            reference_features,
        )
    )

    print()

    print_memory(
        "PHASE12 ridotto:",
        reduced_phase,
    )

    save_csv_chunked(
        reduced_phase,
        OUT_PHASE12,
    )

    # ==============================================================================================
    # IMPORTANT:
    #
    # Teniamo reduced_phase.
    # Eliminiamo phase12 completo.
    # ==============================================================================================

    del phase12

    collect_memory()

    print(
        "PHASE12 completo liberato dalla RAM."
    )

    # ==============================================================================================
    # V40.15 INDICATORS
    # ==============================================================================================

    section(
        "V40.15 - INDICATORI FULL200"
    )

    raw_v4015 = (
        v4015.prepare_raw(
            full_raw
        )
    )

    raw_v4015 = normalize_dataset(
        raw_v4015
    )

    print_memory(
        "RAW preparato V40.15:",
        raw_v4015,
    )

    indicator_dataset = (
        v4015.calculate_all_raw_features(
            raw_v4015
        )
    )

    indicator_dataset = normalize_dataset(
        indicator_dataset
    )

    del raw_v4015

    collect_memory()

    print()

    print(
        f"Righe indicatori: "
        f"{len(indicator_dataset)}"
    )

    print(
        f"Ticker indicatori: "
        f"{indicator_dataset['Ticker'].nunique()}"
    )

    print_memory(
        "Indicator dataset:",
        indicator_dataset,
    )

    # ==============================================================================================
    # RAW NON SERVE PIU'
    # ==============================================================================================

    del full_raw

    collect_memory()

    print(
        "FULL200 RAW liberato dalla RAM."
    )

    # ==============================================================================================
    # BREADTH FULL200
    # ==============================================================================================

    section(
        "V40.15 - BREADTH FULL200"
    )

    print(
        "Calcolo breadth sul nuovo universo "
        "FULL200..."
    )

    indicator_dataset = (
        v4015.add_cross_sectional_breadth(
            indicator_dataset
        )
    )

    indicator_dataset = normalize_dataset(
        indicator_dataset
    )

    breadth_cols = [
        col
        for col in indicator_dataset.columns
        if (
            "BREADTH"
            in str(col).upper()
            or
            "ADVANCE_DECLINE"
            in str(col).upper()
        )
    ]

    print()

    print(
        f"Feature breadth trovate: "
        f"{len(breadth_cols)}"
    )

    for col in breadth_cols:

        print(
            f"  + {col}"
        )

    v4015_cols = [
        col
        for col in indicator_dataset.columns
        if str(
            col
        ).startswith(
            "V4015_"
        )
    ]

    print()

    print(
        f"Feature V4015 disponibili: "
        f"{len(v4015_cols)}"
    )

    print_memory(
        "Indicatori + breadth:",
        indicator_dataset,
    )

    # ==============================================================================================
    # KEEP ONLY NEEDED INDICATOR FEATURES
    # ==============================================================================================

    already_in_phase = set(
        reduced_phase.columns
    )

    needed_from_indicator = [
        col
        for col in reference_features
        if (
            col
            in indicator_dataset.columns
            and
            col
            not in already_in_phase
        )
    ]

    indicator_keep = [
        "Ticker",
        "Date",
    ]

    indicator_keep.extend(
        needed_from_indicator
    )

    indicator_keep = list(
        dict.fromkeys(
            indicator_keep
        )
    )

    indicator_reduced = indicator_dataset.loc[
        :,
        indicator_keep,
    ]

    print()

    print(
        f"Feature ML recuperate da V40.15: "
        f"{len(needed_from_indicator)}"
    )

    print_memory(
        "Indicator dataset ridotto:",
        indicator_reduced,
    )

    del indicator_dataset

    collect_memory()

    print(
        "Indicator dataset completo liberato dalla RAM."
    )

    # ==============================================================================================
    # MERGE
    # ==============================================================================================

    section(
        "MERGE MEMORY-SAFE"
    )

    features = reduced_phase.merge(
        indicator_reduced,
        on=[
            "Ticker",
            "Date",
        ],
        how="left",
        validate="one_to_one",
        copy=False,
    )

    del reduced_phase
    del indicator_reduced

    collect_memory()

    print_memory(
        "Dataset dopo merge:",
        features,
    )

    print(
        f"Righe: "
        f"{len(features)}"
    )

    print(
        f"Ticker: "
        f"{features['Ticker'].nunique()}"
    )

    duplicate_final = int(
        features
        .duplicated(
            subset=[
                "Ticker",
                "Date",
            ]
        )
        .sum()
    )

    print(
        f"Duplicati: "
        f"{duplicate_final}"
    )

    if duplicate_final != 0:

        raise RuntimeError(
            "Duplicati dopo merge."
        )

    # ==============================================================================================
    # CHECK 413
    # ==============================================================================================

    section(
        "CONTROLLO 413 FEATURE"
    )

    missing_features = [
        col
        for col in reference_features
        if col not in features.columns
    ]

    print(
        f"Feature reference: "
        f"{len(reference_features)}"
    )

    print(
        f"Feature mancanti: "
        f"{len(missing_features)}"
    )

    if missing_features:

        for col in missing_features:

            print(
                f" - {col}"
            )

        raise RuntimeError(
            "Mancano feature ML."
        )

    suspicious = [
        col
        for col in reference_features
        if (
            str(col)
            .upper()
            .startswith(
                "TARGET_"
            )
            or
            str(col)
            .upper()
            .startswith(
                "FUTURE_"
            )
        )
    ]

    print(
        f"Leakage sospetta: "
        f"{len(suspicious)}"
    )

    if suspicious:

        raise RuntimeError(
            f"Feature leakage: "
            f"{suspicious}"
        )

    # ==============================================================================================
    # FLOAT32
    # ==============================================================================================

    section(
        "OTTIMIZZAZIONE MEMORIA FEATURE ML"
    )

    print_memory(
        "Prima float32:",
        features,
    )

    features = (
        optimize_numeric_features(
            features,
            reference_features,
        )
    )

    collect_memory()

    print_memory(
        "Dopo float32:",
        features,
    )

    # ==============================================================================================
    # ALL NULL
    # ==============================================================================================

    all_null = [
        col
        for col in reference_features
        if (
            features[
                col
            ]
            .notna()
            .sum()
            ==
            0
        )
    ]

    print()

    print(
        f"Feature completamente NaN: "
        f"{len(all_null)}"
    )

    if all_null:

        for col in all_null:

            print(
                f" - {col}"
            )

        raise RuntimeError(
            "Feature all-NaN presenti."
        )

    # ==============================================================================================
    # FEATURE COVERAGE
    # ==============================================================================================

    section(
        "FEATURE COVERAGE"
    )

    feature_coverage = (
        build_feature_coverage(
            features,
            reference_features,
        )
    )

    feature_coverage.to_csv(
        OUT_FEATURE_COVERAGE,
        index=False,
    )

    print(
        "Peggiori 30 feature per missing:"
    )

    print()

    print(
        feature_coverage
        .head(30)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    max_missing_pct = float(
        feature_coverage[
            "MissingPct"
        ].max()
    )

    # ==============================================================================================
    # SOURCE DISTRIBUTION
    # ==============================================================================================

    if (
        "SOURCE_UNIVERSE"
        not in features.columns
    ):

        features = features.merge(
            source_map,
            on="Ticker",
            how="left",
            validate="many_to_one",
            copy=False,
        )

    source_distribution = (
        features
        .groupby(
            "SOURCE_UNIVERSE",
            as_index=False,
        )
        .agg(
            Tickers=(
                "Ticker",
                "nunique",
            ),

            Rows=(
                "Ticker",
                "size",
            ),

            Start=(
                "Date",
                "min",
            ),

            End=(
                "Date",
                "max",
            ),
        )
    )

    source_distribution.to_csv(
        OUT_SOURCE_DISTRIBUTION,
        index=False,
    )

    section(
        "DISTRIBUZIONE PER UNIVERSO"
    )

    print(
        source_distribution.to_string(
            index=False
        )
    )

    # ==============================================================================================
    # PHASE FAMILY
    # ==============================================================================================

    features[
        "PHASE_FAMILY"
    ] = (
        features[
            "PHASE"
        ]
        .map(
            phase_family
        )
    )

    # ==============================================================================================
    # FINAL OUTPUT COLUMNS
    #
    # Non facciamo una copia completa del dataframe.
    # Costruiamo solo la lista ordinata delle colonne.
    # ==============================================================================================

    core_columns = []

    for col in [
        "Ticker",
        "Date",
        "SOURCE_UNIVERSE",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "PHASE",
        "V4012_PHASE_DIRECTION",
        "V4012_PHASE_AGE",
        "V4012_PHASE_CHANGED",
        "PHASE_FAMILY",
    ]:

        if (
            col
            in features.columns
            and
            col
            not in core_columns
        ):

            core_columns.append(
                col
            )

    final_columns = (
        core_columns
        +
        [
            col
            for col in reference_features
            if col not in core_columns
        ]
    )

    final_columns = list(
        dict.fromkeys(
            final_columns
        )
    )

    # ==============================================================================================
    # FINAL CHECK BEFORE SAVE
    # ==============================================================================================

    section(
        "CONTROLLO FINALE PRIMA DEL SALVATAGGIO"
    )

    final_tickers = int(
        features[
            "Ticker"
        ].nunique()
    )

    final_rows = int(
        len(
            features
        )
    )

    min_date = (
        features[
            "Date"
        ].min()
    )

    max_date = (
        features[
            "Date"
        ].max()
    )

    print(
        f"Ticker:                "
        f"{final_tickers}"
    )

    print(
        f"Righe:                 "
        f"{final_rows}"
    )

    print(
        f"Feature ML:            "
        f"{len(reference_features)}"
    )

    print(
        f"Feature mancanti:      "
        f"{len(missing_features)}"
    )

    print(
        f"Feature all-NaN:       "
        f"{len(all_null)}"
    )

    print(
        f"Leakage:               "
        f"{len(suspicious)}"
    )

    print(
        f"Duplicati:              "
        f"{duplicate_final}"
    )

    print(
        f"Max missing feature:    "
        f"{max_missing_pct:.4f}%"
    )

    print(
        f"Periodo:                "
        f"{min_date.date()} "
        f"-> "
        f"{max_date.date()}"
    )

    print_memory(
        "Dataset pronto al salvataggio:",
        features,
    )

    # ==============================================================================================
    # SAVE FINAL DATASET IN CHUNKS
    # ==============================================================================================

    section(
        "SALVATAGGIO DATASET FINALE A CHUNK"
    )

    if OUT_FEATURES.exists():

        OUT_FEATURES.unlink()

    first_chunk = True

    total_saved = 0

    for start in range(
        0,
        final_rows,
        CSV_CHUNK_SIZE,
    ):

        end = min(
            start
            +
            CSV_CHUNK_SIZE,
            final_rows,
        )

        chunk = features.iloc[
            start:end
        ]

        chunk.loc[
            :,
            final_columns
        ].to_csv(
            OUT_FEATURES,
            mode=(
                "w"
                if first_chunk
                else
                "a"
            ),
            header=first_chunk,
            index=False,
        )

        saved_now = (
            end
            -
            start
        )

        total_saved += (
            saved_now
        )

        print(
            f"Salvate righe "
            f"{start + 1:>6} "
            f"-> "
            f"{end:>6} "
            f"/ "
            f"{final_rows}"
        )

        first_chunk = False

        del chunk

        collect_memory()

    print()

    print(
        f"Righe salvate: "
        f"{total_saved}"
    )

    if (
        total_saved
        !=
        final_rows
    ):

        raise RuntimeError(
            "Numero righe salvate non coerente."
        )

    # ==============================================================================================
    # READ-BACK CONTROL
    #
    # Leggiamo solo colonne minime per non riempire nuovamente la RAM.
    # ==============================================================================================

    section(
        "CONTROLLO FILE SCRITTO"
    )

    check_rows = 0
    check_tickers = set()
    check_min_date = None
    check_max_date = None

    for check_chunk in pd.read_csv(
        OUT_FEATURES,
        usecols=[
            "Ticker",
            "Date",
        ],
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    ):

        check_chunk[
            "Date"
        ] = pd.to_datetime(
            check_chunk[
                "Date"
            ],
            errors="coerce",
        )

        check_rows += len(
            check_chunk
        )

        check_tickers.update(
            check_chunk[
                "Ticker"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

        local_min = (
            check_chunk[
                "Date"
            ].min()
        )

        local_max = (
            check_chunk[
                "Date"
            ].max()
        )

        if (
            check_min_date
            is None
            or
            local_min
            <
            check_min_date
        ):

            check_min_date = local_min

        if (
            check_max_date
            is None
            or
            local_max
            >
            check_max_date
        ):

            check_max_date = local_max

        del check_chunk

        collect_memory()

    print(
        f"Righe file scritto: "
        f"{check_rows}"
    )

    print(
        f"Ticker file scritto: "
        f"{len(check_tickers)}"
    )

    print(
        f"Periodo file scritto: "
        f"{check_min_date.date()} "
        f"-> "
        f"{check_max_date.date()}"
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

        "purpose":
            (
                "Memory-safe FULL200 weekly PHASE and "
                "feature dataset for final Market Sentinel retraining."
            ),

        "input":
            str(
                FULL200_FILE
            ),

        "reference_feature_file":
            str(
                REFERENCE_FEATURE_FILE
            ),

        "rows":
            final_rows,

        "tickers":
            final_tickers,

        "min_date":
            str(
                min_date
            ),

        "max_date":
            str(
                max_date
            ),

        "reference_features":
            int(
                len(
                    reference_features
                )
            ),

        "expected_features":
            EXPECTED_FEATURES,

        "missing_features":
            missing_features,

        "all_null_features":
            all_null,

        "leakage_features":
            suspicious,

        "duplicate_ticker_date":
            duplicate_final,

        "max_feature_missing_pct":
            max_missing_pct,

        "breadth_policy":
            (
                "Breadth recomputed on FULL200 for the "
                "new final production retraining universe."
            ),

        "breadth_recomputed_full200":
            True,

        "memory_safe":
            True,

        "float32_ml_features":
            True,

        "csv_chunk_size":
            CSV_CHUNK_SIZE,

        "written_rows_check":
            check_rows,

        "written_tickers_check":
            len(
                check_tickers
            ),

        "model_training":
            False,

        "model_tuning":
            False,

        "model_calibration":
            False,

        "v40_10_modified":
            False,

        "v40_39_phase_canonical":
            True,

        "phase_modified":
            True,

        "v40_12_modified":
            False,

        "v40_15_modified":
            False,

        "v40_24b_modified":
            False,

        "v40_25_modified":
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
    # FINAL VERDICT
    # ==============================================================================================

    section(
        "VERDETTO V40.39"
    )

    passed = (
        final_tickers
        ==
        EXPECTED_TICKERS

        and

        len(
            reference_features
        )
        ==
        EXPECTED_FEATURES

        and

        len(
            missing_features
        )
        ==
        0

        and

        len(
            all_null
        )
        ==
        0

        and

        len(
            suspicious
        )
        ==
        0

        and

        duplicate_final
        ==
        0

        and

        max_date
        <=
        FROZEN_END_DATE

        and

        check_rows
        ==
        final_rows

        and

        len(
            check_tickers
        )
        ==
        EXPECTED_TICKERS
    )

    if passed:

        print(
            "OK - V40.39 COMPLETATA CORRETTAMENTE"
        )

        print()

        print(
            f"Ticker:                "
            f"{final_tickers}"
        )

        print(
            f"Righe:                 "
            f"{final_rows}"
        )

        print(
            f"Feature ML reference:  "
            f"{len(reference_features)}"
        )

        print(
            f"Feature mancanti:      "
            f"{len(missing_features)}"
        )

        print(
            f"Feature all-NaN:       "
            f"{len(all_null)}"
        )

        print(
            f"Leakage:               "
            f"{len(suspicious)}"
        )

        print(
            f"Duplicati:              "
            f"{duplicate_final}"
        )

        print(
            f"Max missing feature:    "
            f"{max_missing_pct:.4f}%"
        )

        print(
            f"Periodo:                "
            f"{min_date.date()} "
            f"-> "
            f"{max_date.date()}"
        )

        print()

        print(
            "Breadth FULL200:        SI"
        )

        print(
            "Feature float32:        SI"
        )

        print(
            "Salvataggio a chunk:    SI"
        )

        print(
            "Training ML eseguito:   NO"
        )

        print()

        print(
            "FULL200 FEATURE DATASET CONGELATO."
        )

        print()

        print(
            "PROSSIMO STEP:"
        )

        print(
            "V40.37 - RETRAINING + CALIBRAZIONE "
            "WEEKLY FULL200."
        )

    else:

        print(
            "ATTENZIONE - V40.39 NON HA SUPERATO "
            "TUTTI I CONTROLLI."
        )

        print()

        print(
            "NON PROCEDERE AL RETRAINING."
        )


# ==================================================================================================
# RUN
# ==================================================================================================

if __name__ == "__main__":

    main()