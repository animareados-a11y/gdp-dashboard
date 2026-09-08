import os

import numpy as np
import pandas as pd


# ============================================================
# MARKET SENTINEL
# TARGET ENGINE V3
# "IL TRENO"
# ============================================================
#
# SCOPO
#
# NON chiediamo più soltanto:
#
#   "ha fatto +4% prima di -2%?"
#
# Vogliamo descrivere una domanda più vicina
# all'obiettivo reale di Market Sentinel:
#
# BUY:
#   "da qui è partito un movimento rialzista significativo?"
#
# SELL:
#   "da qui è partito un movimento ribassista significativo?"
#
# Non sappiamo in anticipo:
#
# - quanto durerà il movimento
# - se sarà +4%, +7%, +12%...
# - se durerà 5, 10, 15 o 20 sedute
#
# Per questo analizziamo contemporaneamente
# diversi orizzonti temporali.
#
# ============================================================


INPUT_FILE = os.path.join(
    "data",
    "ml_dataset_v2.csv",
)

OUTPUT_FILE = os.path.join(
    "data",
    "ml_dataset_v3.csv",
)


# ============================================================
# ORIZZONTI DISPONIBILI NEL DATASET
# ============================================================

HORIZONS = [
    5,
    10,
    15,
    20,
]


# ============================================================
# SOGLIE BASE
# ============================================================
#
# Non sono "target rigidi".
#
# Servono solo a distinguere:
#
# - rumore normale
# - movimento sufficientemente interessante
#
# La vera informazione sarà anche CONTINUA:
#
# buy_strength_v3
# sell_strength_v3
#
# ============================================================

MIN_SIGNIFICANT_MOVE = 3.0

STRONG_MOVE = 6.0

DOMINANCE_RATIO = 1.30

MIN_STRENGTH_FOR_EVENT = 5.5


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value,
    default=np.nan,
):
    try:

        value = float(
            value
        )

        if np.isfinite(
            value
        ):
            return value

    except Exception:
        pass

    return default


def clip10(
    value
):
    try:

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):
            return np.nan

        return float(
            np.clip(
                value,
                0.0,
                10.0,
            )
        )

    except Exception:
        return np.nan


# ============================================================
# NORMALIZZAZIONE FUTURE MAX DOWN
# ============================================================

def prepare_down_columns(
    dataframe
):
    """
    future_max_down può essere stato salvato:

        - come valore negativo, es. -5.2
        oppure
        - come magnitudine positiva, es. 5.2

    Identifichiamo automaticamente il formato
    e creiamo sempre una MAGNITUDINE POSITIVA.
    """

    result = dataframe.copy()

    for horizon in HORIZONS:

        column = (
            f"future_max_down_{horizon}d"
        )

        if column not in result.columns:
            continue

        series = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        valid = series.dropna()

        if valid.empty:

            result[
                f"_down_mag_{horizon}d"
            ] = np.nan

            continue

        median = float(
            valid.median()
        )

        # Se normalmente è negativo:
        # -5.2 -> 5.2

        if median <= 0:

            magnitude = (
                -series
                .clip(
                    upper=0
                )
            )

        # Se è già positivo:
        # 5.2 -> 5.2

        else:

            magnitude = (
                series
                .clip(
                    lower=0
                )
            )

        result[
            f"_down_mag_{horizon}d"
        ] = magnitude

    return result


# ============================================================
# SCORE DI PERSISTENZA
# ============================================================

def persistence_up(
    row
):
    """
    Quanto il movimento rialzista continua
    ad essere visibile a 5/10/15/20 giorni.
    """

    values = []

    weights = {
        5: 0.15,
        10: 0.30,
        15: 0.30,
        20: 0.25,
    }

    for horizon, weight in weights.items():

        value = safe_float(
            row.get(
                f"future_return_{horizon}d"
            )
        )

        if np.isfinite(
            value
        ):

            # Da 0 a 1.
            #
            # +5% o più = piena persistenza positiva.

            normalized = float(
                np.clip(
                    value / 5.0,
                    0.0,
                    1.0,
                )
            )

            values.append(
                (
                    normalized,
                    weight,
                )
            )

    if not values:
        return np.nan

    total_weight = sum(
        weight
        for _, weight
        in values
    )

    return (
        sum(
            value * weight
            for value, weight
            in values
        )
        /
        total_weight
    )


def persistence_down(
    row
):
    """
    Versione speculare per il movimento ribassista.
    """

    values = []

    weights = {
        5: 0.15,
        10: 0.30,
        15: 0.30,
        20: 0.25,
    }

    for horizon, weight in weights.items():

        value = safe_float(
            row.get(
                f"future_return_{horizon}d"
            )
        )

        if np.isfinite(
            value
        ):

            normalized = float(
                np.clip(
                    (-value) / 5.0,
                    0.0,
                    1.0,
                )
            )

            values.append(
                (
                    normalized,
                    weight,
                )
            )

    if not values:
        return np.nan

    total_weight = sum(
        weight
        for _, weight
        in values
    )

    return (
        sum(
            value * weight
            for value, weight
            in values
        )
        /
        total_weight
    )


# ============================================================
# VELOCITA' DI PARTENZA DEL TRENO
# ============================================================

def early_up_score(
    row
):
    """
    Quanto rapidamente compare il movimento rialzista.

    Diamo maggiore valore a un movimento
    già visibile entro 5-10 sedute.
    """

    up5 = max(
        safe_float(
            row.get(
                "future_max_up_5d"
            ),
            0,
        ),
        0,
    )

    up10 = max(
        safe_float(
            row.get(
                "future_max_up_10d"
            ),
            0,
        ),
        0,
    )

    ret5 = safe_float(
        row.get(
            "future_return_5d"
        ),
        0,
    )

    ret10 = safe_float(
        row.get(
            "future_return_10d"
        ),
        0,
    )

    score = (
        0.35
        *
        np.clip(
            up5 / 3.0,
            0,
            1,
        )

        +

        0.30
        *
        np.clip(
            up10 / 4.0,
            0,
            1,
        )

        +

        0.15
        *
        np.clip(
            ret5 / 2.0,
            0,
            1,
        )

        +

        0.20
        *
        np.clip(
            ret10 / 3.0,
            0,
            1,
        )
    )

    return float(
        np.clip(
            score,
            0,
            1,
        )
    )


def early_down_score(
    row
):

    down5 = max(
        safe_float(
            row.get(
                "_down_mag_5d"
            ),
            0,
        ),
        0,
    )

    down10 = max(
        safe_float(
            row.get(
                "_down_mag_10d"
            ),
            0,
        ),
        0,
    )

    ret5 = safe_float(
        row.get(
            "future_return_5d"
        ),
        0,
    )

    ret10 = safe_float(
        row.get(
            "future_return_10d"
        ),
        0,
    )

    score = (
        0.35
        *
        np.clip(
            down5 / 3.0,
            0,
            1,
        )

        +

        0.30
        *
        np.clip(
            down10 / 4.0,
            0,
            1,
        )

        +

        0.15
        *
        np.clip(
            (-ret5) / 2.0,
            0,
            1,
        )

        +

        0.20
        *
        np.clip(
            (-ret10) / 3.0,
            0,
            1,
        )
    )

    return float(
        np.clip(
            score,
            0,
            1,
        )
    )


# ============================================================
# CALCOLO FUTURE EXCURSION
# ============================================================

def future_excursion_summary(
    row
):

    ups = []
    downs = []

    for horizon in HORIZONS:

        up = safe_float(
            row.get(
                f"future_max_up_{horizon}d"
            )
        )

        down = safe_float(
            row.get(
                f"_down_mag_{horizon}d"
            )
        )

        if np.isfinite(
            up
        ):

            ups.append(
                max(
                    up,
                    0,
                )
            )

        if np.isfinite(
            down
        ):

            downs.append(
                max(
                    down,
                    0,
                )
            )

    best_up = (
        max(
            ups
        )
        if ups
        else np.nan
    )

    best_down = (
        max(
            downs
        )
        if downs
        else np.nan
    )

    return (
        best_up,
        best_down,
    )


# ============================================================
# BUY / SELL STRENGTH
# ============================================================

def calculate_train_scores(
    row
):
    """
    Crea due score CONTINUI 0-10.

    BUY STRENGTH:
        quanto il futuro assomiglia
        alla partenza di un trend rialzista.

    SELL STRENGTH:
        quanto il futuro assomiglia
        alla partenza di un trend ribassista.
    """

    (
        best_up,
        best_down,
    ) = future_excursion_summary(
        row
    )

    if (
        not np.isfinite(
            best_up
        )
        or
        not np.isfinite(
            best_down
        )
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # ========================================================
    # MAGNITUDINE
    # ========================================================

    buy_magnitude = np.clip(
        best_up
        /
        STRONG_MOVE,
        0,
        1,
    )

    sell_magnitude = np.clip(
        best_down
        /
        STRONG_MOVE,
        0,
        1,
    )

    # ========================================================
    # DOMINANZA DIREZIONALE
    #
    # Se sale +8 e scende -1:
    # fortissima dominanza BUY.
    #
    # Se sale +5 e scende -5:
    # nessuna vera dominanza.
    # ========================================================

    total_move = max(
        best_up
        +
        best_down,
        1e-9,
    )

    buy_dominance = (
        best_up
        /
        total_move
    )

    sell_dominance = (
        best_down
        /
        total_move
    )

    # ========================================================
    # PARTENZA PRECOCE
    # ========================================================

    early_buy = early_up_score(
        row
    )

    early_sell = early_down_score(
        row
    )

    # ========================================================
    # PERSISTENZA
    # ========================================================

    persist_buy = persistence_up(
        row
    )

    persist_sell = persistence_down(
        row
    )

    if not np.isfinite(
        persist_buy
    ):
        persist_buy = 0.0

    if not np.isfinite(
        persist_sell
    ):
        persist_sell = 0.0

    # ========================================================
    # SCORE FINALE
    #
    # Magnitudine: 35%
    # Dominanza:   30%
    # Partenza:    20%
    # Persistenza: 15%
    #
    # ========================================================

    buy_score = 10 * (
        0.35
        *
        buy_magnitude

        +

        0.30
        *
        buy_dominance

        +

        0.20
        *
        early_buy

        +

        0.15
        *
        persist_buy
    )

    sell_score = 10 * (
        0.35
        *
        sell_magnitude

        +

        0.30
        *
        sell_dominance

        +

        0.20
        *
        early_sell

        +

        0.15
        *
        persist_sell
    )

    return (
        clip10(
            buy_score
        ),
        clip10(
            sell_score
        ),
        best_up,
        best_down,
    )


# ============================================================
# TARGET BINARI
# ============================================================

def classify_train_event(
    buy_strength,
    sell_strength,
    best_up,
    best_down,
):
    """
    Costruiamo anche target 0/1
    per gli algoritmi di classificazione.

    IMPORTANTE:

    non basta superare una soglia.

    BUY deve essere:
        - significativo
        - dominante rispetto al ribasso

    SELL speculare.
    """

    if not all(
        np.isfinite(
            value
        )
        for value
        in [
            buy_strength,
            sell_strength,
            best_up,
            best_down,
        ]
    ):

        return (
            np.nan,
            np.nan,
            "NO_DATA",
        )

    # ========================================================
    # BUY
    # ========================================================

    buy_significant = (
        best_up
        >=
        MIN_SIGNIFICANT_MOVE
    )

    buy_dominant = (
        best_up
        >=
        (
            best_down
            *
            DOMINANCE_RATIO
        )
    )

    buy_event = (
        buy_significant
        and
        buy_dominant
        and
        buy_strength
        >=
        MIN_STRENGTH_FOR_EVENT
    )

    # ========================================================
    # SELL
    # ========================================================

    sell_significant = (
        best_down
        >=
        MIN_SIGNIFICANT_MOVE
    )

    sell_dominant = (
        best_down
        >=
        (
            best_up
            *
            DOMINANCE_RATIO
        )
    )

    sell_event = (
        sell_significant
        and
        sell_dominant
        and
        sell_strength
        >=
        MIN_STRENGTH_FOR_EVENT
    )

    # ========================================================
    # EVITIAMO CONTRADDIZIONE
    # ========================================================

    if (
        buy_event
        and
        sell_event
    ):

        if (
            buy_strength
            >
            sell_strength
        ):

            sell_event = False

        elif (
            sell_strength
            >
            buy_strength
        ):

            buy_event = False

        else:

            buy_event = False
            sell_event = False

    # ========================================================
    # STATO
    # ========================================================

    if buy_event:

        state = (
            "BUY_TRAIN_START"
        )

    elif sell_event:

        state = (
            "SELL_TRAIN_START"
        )

    else:

        state = (
            "NEUTRAL"
        )

    return (
        int(
            buy_event
        ),
        int(
            sell_event
        ),
        state,
    )


# ============================================================
# ELABORAZIONE RIGA
# ============================================================

def process_row(
    row
):

    (
        buy_strength,
        sell_strength,
        best_up,
        best_down,
    ) = calculate_train_scores(
        row
    )

    (
        target_buy,
        target_sell,
        train_state,
    ) = classify_train_event(
        buy_strength,
        sell_strength,
        best_up,
        best_down,
    )

    return pd.Series(
        {
            "future_best_up_v3":
                best_up,

            "future_best_down_v3":
                best_down,

            "buy_strength_v3":
                buy_strength,

            "sell_strength_v3":
                sell_strength,

            "target_buy_v3":
                target_buy,

            "target_sell_v3":
                target_sell,

            "train_state_v3":
                train_state,

            # Differenziale utile anche
            # per eventuale regressione futura.

            "direction_strength_v3":
                (
                    buy_strength
                    -
                    sell_strength

                    if (
                        np.isfinite(
                            buy_strength
                        )
                        and
                        np.isfinite(
                            sell_strength
                        )
                    )

                    else
                    np.nan
                ),
        }
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    dataframe
):

    print()
    print(
        "============================================"
    )

    print(
        " TARGET V3 COMPLETATI"
    )

    print(
        "============================================"
    )

    print()

    print(
        f"Righe: "
        f"{len(dataframe)}"
    )

    print(
        f"Titoli: "
        f"{dataframe['ticker'].nunique()}"
    )

    print()

    buy = pd.to_numeric(
        dataframe[
            "target_buy_v3"
        ],
        errors="coerce",
    )

    sell = pd.to_numeric(
        dataframe[
            "target_sell_v3"
        ],
        errors="coerce",
    )

    valid_buy = (
        buy.dropna()
    )

    valid_sell = (
        sell.dropna()
    )

    print(
        f"BUY TRAIN START: "
        f"{valid_buy.mean() * 100:.1f}%"
    )

    print(
        f"SELL TRAIN START: "
        f"{valid_sell.mean() * 100:.1f}%"
    )

    print()

    print(
        "Distribuzione stati:"
    )

    print()

    print(
        dataframe[
            "train_state_v3"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()

    print(
        "BUY strength medio:"
    )

    print(
        round(
            dataframe[
                "buy_strength_v3"
            ].mean(),
            3,
        )
    )

    print()

    print(
        "SELL strength medio:"
    )

    print(
        round(
            dataframe[
                "sell_strength_v3"
            ].mean(),
            3,
        )
    )

    print()

    print(
        "BUY strength percentile:"
    )

    print(
        dataframe[
            "buy_strength_v3"
        ]
        .quantile(
            [
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
        .round(
            3
        )
        .to_string()
    )

    print()

    print(
        "SELL strength percentile:"
    )

    print(
        dataframe[
            "sell_strength_v3"
        ]
        .quantile(
            [
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
        .round(
            3
        )
        .to_string()
    )

    print()


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
        " TARGET ENGINE V3"
    )

    print(
        ' "IL TRENO"'
    )

    print(
        "============================================"
    )

    print()

    print(
        "Caricamento dataset V2.1..."
    )

    if not os.path.exists(
        INPUT_FILE
    ):

        raise FileNotFoundError(
            f"File non trovato: "
            f"{INPUT_FILE}"
        )

    dataset = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Righe caricate: "
        f"{len(dataset)}"
    )

    print()

    print(
        "Preparazione movimenti futuri..."
    )

    dataset = prepare_down_columns(
        dataset
    )

    print(
        "Calcolo nuovi target..."
    )

    new_targets = dataset.apply(
        process_row,
        axis=1,
    )

    # ========================================================
    # RIMUOVIAMO LE COLONNE TEMPORANEE
    # ========================================================

    temporary_columns = [
        column

        for column
        in dataset.columns

        if column.startswith(
            "_down_mag_"
        )
    ]

    dataset = dataset.drop(
        columns=temporary_columns,
        errors="ignore",
    )

    # ========================================================
    # AGGIUNGIAMO TARGET V3
    # ========================================================

    dataset = pd.concat(
        [
            dataset.reset_index(
                drop=True
            ),
            new_targets.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        "data",
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(
        dataset
    )

    print(
        "File salvato:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "IMPORTANTE:"
    )

    print(
        "buy_strength_v3 e sell_strength_v3 "
        "sono score CONTINUI 0-10."
    )

    print(
        "target_buy_v3 e target_sell_v3 "
        "sono le etichette per il Machine Learning."
    )

    print()

    print(
        "Il target non richiede più che il mercato "
        "faccia esattamente +4% prima di -2%."
    )

    print(
        "Cerca invece un movimento significativo, "
        "dominante, precoce e persistente."
    )

    print()