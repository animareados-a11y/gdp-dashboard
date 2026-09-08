"""
MarketSentinel
DIAGNOSI V40.5 - USCITA DALLA LATERALITA / INDECISIONE

OBIETTIVO
---------
INDECISIONE deve essere una categoria RESIDUALE:

    BUY
    TREND_RIALZISTA
    DEBOLEZZA_RIALZISTA
    SELL
    TREND_RIBASSISTA
    DEBOLEZZA_RIBASSISTA

e solo quando nessuna fase direzionale descrive correttamente
il mercato:

    INDECISIONE / LATERALITA

Questo diagnostico analizza le settimane nelle quali:

    PHASE V40.5 == INDECISIONE
    LATERAL_SIGNAL == 0

cioe' i punti nei quali il motore continua a dichiarare
INDECISIONE anche se il segnale che aveva identificato la
lateralita' non e' piu' attivo.

NON modifica nessun motore.
NON introduce nuove soglie.
NON modifica PHASE.
NON modifica file di produzione.

Focus iniziale:
    FBK.MI
    UCG.MI
    BBVA.MC
    TIT.MI
"""

from pathlib import Path

import numpy as np
import pandas as pd

import weekly_v40_5_core_phase_engine as v405


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

TICKERS = [
    "FBK.MI",
    "UCG.MI",
    "BBVA.MC",
    "TIT.MI",
]


def banner(text):
    print("\n" + "=" * 165)
    print(text)
    print("=" * 165)


def num_series(df, col, default=0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)

    return (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(default)
    )


def get_value(row, col, default=np.nan):
    if col not in row.index:
        return default
    return row[col]


def main():

    banner("CARICAMENTO")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    print(f"Input: {INPUT_FILE}")
    print(f"Righe input: {len(df):,}")

    all_rows = []

    # ============================================================
    # PROCESSO TICKER
    # ============================================================

    for ticker in TICKERS:

        raw = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if raw.empty:
            print(f"{ticker}: NON TROVATO")
            continue

        g = v405.process_ticker(raw)

        g = (
            g.sort_values("Date")
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Informazioni future SOLO DIAGNOSTICHE.
        #
        # Non vengono utilizzate per creare PHASE.
        # Servono esclusivamente a capire se una settimana che
        # V40.5 chiama INDECISIONE era realmente laterale oppure
        # stava gia' entrando in un movimento direzionale.
        # --------------------------------------------------------

        close = pd.to_numeric(
            g["Close"],
            errors="coerce",
        )

        g["RET_FWD_1W"] = (
            close.shift(-1) / close - 1.0
        ) * 100.0

        g["RET_FWD_2W"] = (
            close.shift(-2) / close - 1.0
        ) * 100.0

        g["RET_FWD_3W"] = (
            close.shift(-3) / close - 1.0
        ) * 100.0

        # --------------------------------------------------------
        # Identifichiamo gli episodi contigui di INDECISIONE.
        # --------------------------------------------------------

        phase = g["PHASE"].astype(str)

        is_ind = phase.eq("INDECISIONE")

        episode_start = (
            is_ind
            & ~is_ind.shift(1, fill_value=False)
        )

        episode_id = episode_start.cumsum()

        g["_IND_EPISODE_ID"] = np.where(
            is_ind,
            episode_id,
            np.nan,
        )

        # --------------------------------------------------------
        # Righe problematiche:
        #
        # V40.5 dice INDECISIONE
        # ma il LATERAL_SIGNAL corrente e' spento.
        # --------------------------------------------------------

        lateral = num_series(
            g,
            "LATERAL_SIGNAL",
            0,
        )

        mask = (
            is_ind
            & lateral.eq(0)
        )

        candidates = g.loc[mask].copy()

        if candidates.empty:
            continue

        for idx, row in candidates.iterrows():

            ep_id = row["_IND_EPISODE_ID"]

            ep = g[
                g["_IND_EPISODE_ID"] == ep_id
            ]

            ep_start_idx = ep.index.min()

            weeks_from_ind_start = (
                idx - ep_start_idx + 1
            )

            # Ultima settimana precedente in cui il lateral signal
            # era effettivamente attivo all'interno dell'episodio.
            ep_until_now = g.loc[
                ep_start_idx:idx
            ]

            lateral_until_now = num_series(
                ep_until_now,
                "LATERAL_SIGNAL",
                0,
            )

            active_lateral_rows = (
                ep_until_now[
                    lateral_until_now.eq(1)
                ]
            )

            if not active_lateral_rows.empty:

                last_lateral_idx = (
                    active_lateral_rows.index.max()
                )

                weeks_since_lateral = (
                    idx - last_lateral_idx
                )

                last_lateral_date = g.loc[
                    last_lateral_idx,
                    "Date",
                ]

            else:

                last_lateral_idx = np.nan
                weeks_since_lateral = np.nan
                last_lateral_date = pd.NaT

            # ----------------------------------------------------
            # Segnali V40.5 gia' esistenti
            # ----------------------------------------------------

            sar_side = get_value(
                row,
                "SAR_SIDE",
            )

            sar_age = get_value(
                row,
                "SAR_AGE",
            )

            weak_up = int(
                get_value(
                    row,
                    "WEAK_UP_TRIGGER",
                    0,
                ) or 0
            )

            weak_down = int(
                get_value(
                    row,
                    "WEAK_DOWN_TRIGGER",
                    0,
                ) or 0
            )

            bull_raw = int(
                get_value(
                    row,
                    "BULL_RECOVERY_RAW",
                    0,
                ) or 0
            )

            bear_raw = int(
                get_value(
                    row,
                    "BEAR_RECOVERY_RAW",
                    0,
                ) or 0
            )

            bull_conf = int(
                get_value(
                    row,
                    "BULL_RECOVERY_CONFIRMED",
                    0,
                ) or 0
            )

            bear_conf = int(
                get_value(
                    row,
                    "BEAR_RECOVERY_CONFIRMED",
                    0,
                ) or 0
            )

            # ----------------------------------------------------
            # Indicatori tecnici disponibili.
            #
            # Non imponiamo condizioni nuove.
            # Li stampiamo per capire cosa distingue i casi.
            # ----------------------------------------------------

            record = {
                "Ticker": ticker,
                "Date": row["Date"],
                "Close": get_value(row, "Close"),

                "IND_EPISODE":
                    int(ep_id),

                "WEEK_FROM_IND_START":
                    int(weeks_from_ind_start),

                "LAST_LATERAL_DATE":
                    last_lateral_date,

                "WEEKS_SINCE_LATERAL":
                    weeks_since_lateral,

                "SAR_SIDE":
                    sar_side,

                "SAR_AGE":
                    sar_age,

                "WEAK_UP":
                    weak_up,

                "WEAK_DOWN":
                    weak_down,

                "BULL_RAW":
                    bull_raw,

                "BEAR_RAW":
                    bear_raw,

                "BULL_CONF":
                    bull_conf,

                "BEAR_CONF":
                    bear_conf,

                "HA_DIRECTION":
                    get_value(
                        row,
                        "HA_DIRECTION",
                    ),

                "HA_INDECISION":
                    get_value(
                        row,
                        "HA_INDECISION",
                    ),

                "HA_BODY_RATIO":
                    get_value(
                        row,
                        "HA_BODY_RATIO",
                    ),

                "BB_POSITION":
                    get_value(
                        row,
                        "BB_POSITION",
                    ),

                "RSI":
                    get_value(
                        row,
                        "RSI",
                    ),

                "MACD":
                    get_value(
                        row,
                        "MACD",
                    ),

                "MACD_SIGNAL":
                    get_value(
                        row,
                        "MACD_SIGNAL",
                    ),

                "MACD_HIST":
                    get_value(
                        row,
                        "MACD_HIST",
                    ),

                "VOLUME_OSC":
                    get_value(
                        row,
                        "VOLUME_OSC",
                    ),

                "RET_FWD_1W":
                    row["RET_FWD_1W"],

                "RET_FWD_2W":
                    row["RET_FWD_2W"],

                "RET_FWD_3W":
                    row["RET_FWD_3W"],
            }

            all_rows.append(record)

    result = pd.DataFrame(all_rows)

    if result.empty:
        banner("RISULTATO")
        print(
            "Nessuna riga INDECISIONE con "
            "LATERAL_SIGNAL=0 trovata."
        )
        return

    # ============================================================
    # SUMMARY
    # ============================================================

    banner(
        "SUMMARY - INDECISIONE MA LATERAL_SIGNAL = 0"
    )

    print(
        f"Righe candidate: {len(result):,}"
    )

    print(
        f"Episodi distinti: "
        f"{result[['Ticker', 'IND_EPISODE']].drop_duplicates().shape[0]:,}"
    )

    print(
        "\nPer ticker:"
    )

    print(
        result.groupby("Ticker")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    # ============================================================
    # SEGNALE TECNICO CORRENTE
    # ============================================================

    def current_evidence(row):

        if row["SAR_SIDE"] == 1:

            if row["WEAK_UP"] == 1:
                return "BULL_WEAKNESS"

            if (
                row["BULL_RAW"] == 1
                or row["BULL_CONF"] == 1
            ):
                return "BULL_TREND_RECOVERY"

            return "BULL_NO_EXIT_EVIDENCE"

        if row["SAR_SIDE"] == -1:

            if row["WEAK_DOWN"] == 1:
                return "BEAR_WEAKNESS"

            if (
                row["BEAR_RAW"] == 1
                or row["BEAR_CONF"] == 1
            ):
                return "BEAR_TREND_RECOVERY"

            return "BEAR_NO_EXIT_EVIDENCE"

        return "NO_SAR_DIRECTION"

    result["CURRENT_EVIDENCE"] = (
        result.apply(
            current_evidence,
            axis=1,
        )
    )

    banner(
        "EVIDENZA TECNICA CORRENTE"
    )

    print(
        result["CURRENT_EVIDENCE"]
        .value_counts()
        .to_string()
    )

    # ============================================================
    # COSA SUCCEDE DOPO?
    # ============================================================

    banner(
        "RENDIMENTI FUTURI PER EVIDENZA CORRENTE "
        "(SOLO DIAGNOSTICA)"
    )

    summary = (
        result
        .groupby("CURRENT_EVIDENCE")
        .agg(
            N=("Ticker", "size"),

            AVG_1W=(
                "RET_FWD_1W",
                "mean",
            ),

            MEDIAN_1W=(
                "RET_FWD_1W",
                "median",
            ),

            AVG_2W=(
                "RET_FWD_2W",
                "mean",
            ),

            MEDIAN_2W=(
                "RET_FWD_2W",
                "median",
            ),

            AVG_3W=(
                "RET_FWD_3W",
                "mean",
            ),

            MEDIAN_3W=(
                "RET_FWD_3W",
                "median",
            ),
        )
        .round(3)
    )

    print(summary.to_string())

    # ============================================================
    # DOPO QUANTO TEMPO DALL'ULTIMA VERA LATERALITA?
    # ============================================================

    banner(
        "SETTIMANE DALL'ULTIMO LATERAL_SIGNAL=1"
    )

    valid_since = result[
        result["WEEKS_SINCE_LATERAL"]
        .notna()
    ]

    if not valid_since.empty:

        print(
            valid_since[
                "WEEKS_SINCE_LATERAL"
            ]
            .describe(
                percentiles=[
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                ]
            )
            .round(3)
            .to_string()
        )

    # ============================================================
    # CASI PIU' LONTANI DALLA LATERALITA
    # ============================================================

    banner(
        "TOP 30 - V40.5 ANCORA INDECISIONE "
        "MOLTE SETTIMANE DOPO L'ULTIMO LATERAL_SIGNAL"
    )

    top_cols = [
        "Ticker",
        "Date",
        "Close",
        "WEEK_FROM_IND_START",
        "LAST_LATERAL_DATE",
        "WEEKS_SINCE_LATERAL",
        "SAR_SIDE",
        "SAR_AGE",
        "CURRENT_EVIDENCE",
        "WEAK_UP",
        "WEAK_DOWN",
        "BULL_RAW",
        "BEAR_RAW",
        "RET_FWD_1W",
        "RET_FWD_2W",
        "RET_FWD_3W",
    ]

    print(
        result[
            result[
                "WEEKS_SINCE_LATERAL"
            ].notna()
        ]
        .sort_values(
            "WEEKS_SINCE_LATERAL",
            ascending=False,
        )
        .head(30)[top_cols]
        .round(3)
        .to_string(index=False)
    )

    # ============================================================
    # FINECO 2026
    # ============================================================

    banner(
        "FINECO 2026 - USCITA DALLA LATERALITA"
    )

    fineco = result[
        (result["Ticker"] == "FBK.MI")
        & (
            result["Date"]
            >= pd.Timestamp(
                "2026-06-01",
                tz="UTC",
            )
        )
    ]

    print(
        fineco[top_cols]
        .round(3)
        .to_string(index=False)
    )

    # ============================================================
    # CASI CON RECOVERY TREND
    # ============================================================

    banner(
        "CASI CON RECOVERY TREND MA ANCORA INDECISIONE"
    )

    recovery = result[
        result["CURRENT_EVIDENCE"].isin(
            [
                "BULL_TREND_RECOVERY",
                "BEAR_TREND_RECOVERY",
            ]
        )
    ]

    print(
        recovery[top_cols]
        .round(3)
        .to_string(index=False)
    )

    # ============================================================
    # CASI CON DEBOLEZZA MA ANCORA INDECISIONE
    # ============================================================

    banner(
        "CASI CON DEBOLEZZA MA ANCORA INDECISIONE"
    )

    weakness = result[
        result["CURRENT_EVIDENCE"].isin(
            [
                "BULL_WEAKNESS",
                "BEAR_WEAKNESS",
            ]
        )
    ]

    print(
        weakness[top_cols]
        .round(3)
        .to_string(index=False)
    )

    # ============================================================
    # SICUREZZA
    # ============================================================

    banner("FINE DIAGNOSI")

    print(
        "Nessuna nuova soglia PHASE introdotta."
    )

    print(
        "Nessun file di produzione modificato."
    )

    print(
        "I rendimenti futuri sono usati SOLO per diagnosi "
        "e NON entrano nella classificazione."
    )


if __name__ == "__main__":
    main()