"""
MarketSentinel
DIAGNOSI LONG-TREND CONFIRMATION - K3 vs K4
===========================================

OBIETTIVO
---------
Verificare una regola di conferma aggiuntiva dopo un trend opposto lungo.

Esempio BUY:
- trend ribassista precedente lungo almeno N settimane;
- SAR gira bullish;
- NON dichiariamo immediatamente BUY;
- attendiamo conferme Heikin-Ashi rialziste e decisive.

K=3:
    TEST1 -> TEST2 -> BUY1 sulla terza conferma valida.

K=4:
    TEST1 -> TEST2 -> TEST3 -> BUY1 sulla quarta conferma valida.

Simmetrico per SELL.

Confrontiamo due modalità:

1) CUMULATIVE
   Le HA valide nella nuova direzione si accumulano.
   Le settimane non valide non aumentano il conteggio,
   ma NON lo azzerano.

2) CONSECUTIVE
   Servono K HA valide consecutive nella nuova direzione.
   Una settimana non valida azzera il conteggio.

IMPORTANTE
----------
- SOLO diagnostica.
- NON modifica V40.10.
- NON modifica V4.
- NON modifica PHASE di produzione.
- NON usa rendimenti futuri per creare la regola.
- La lunghezza futura del SAR run viene usata SOLO per valutare
  ex-post quanto la regola avrebbe filtrato run brevi.

CHECKPOINT
----------
Salva progressivamente un CSV per ticker.
Se il processo viene interrotto, il rilancio riparte dai ticker
già completati.

OUTPUT
------
data/diagnosi_phase_long_trend_confirmation_k34_events.csv
data/diagnosi_phase_long_trend_confirmation_k34_summary.csv
data/diagnosi_phase_long_trend_confirmation_k34_checkpoint.csv
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v4 as v4

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

OUT_EVENTS = Path(
    "data/diagnosi_phase_long_trend_confirmation_k34_events.csv"
)
OUT_SUMMARY = Path(
    "data/diagnosi_phase_long_trend_confirmation_k34_summary.csv"
)
OUT_CHECKPOINT = Path(
    "data/diagnosi_phase_long_trend_confirmation_k34_checkpoint.csv"
)

N_VALUES = [6, 8, 10, 12]
K_VALUES = [3, 4]
MODES = ["CUMULATIVE", "CONSECUTIVE"]


# ============================================================
# UTILS
# ============================================================

def _num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def _ticker_col(df: pd.DataFrame) -> str:
    c = _first_existing(
        df,
        ["Ticker", "TICKER", "ticker", "Symbol", "SYMBOL", "symbol"],
    )
    if c is None:
        raise KeyError("Colonna ticker non trovata.")
    return c


def _date_col(df: pd.DataFrame) -> str:
    c = _first_existing(
        df,
        ["Date", "DATE", "date", "Datetime", "DATETIME", "datetime"],
    )
    if c is None:
        raise KeyError("Colonna data non trovata.")
    return c


def _phase_col(df: pd.DataFrame) -> str | None:
    return _first_existing(
        df,
        [
            "PHASE_DYNAMICS_V4",
            "PHASE",
            "PHASE_V409",
        ],
    )


def _prev_sar_age_col(df: pd.DataFrame) -> str | None:
    return _first_existing(
        df,
        [
            "V4010_PREV_SAR_AGE",
            "PREV_SAR_AGE",
            "V4010_PREVIOUS_SAR_AGE",
        ],
    )


def _run_qualified_col(df: pd.DataFrame) -> str | None:
    return _first_existing(
        df,
        [
            "V4010_RUN_QUALIFIED",
            "RUN_QUALIFIED",
        ],
    )


def _run_id_col(df: pd.DataFrame) -> str | None:
    return _first_existing(
        df,
        [
            "V4010_SAR_RUN_ID",
            "SAR_RUN_ID",
        ],
    )


# ============================================================
# HA LOGIC
# ============================================================

def _valid_ha_confirmation(row: pd.Series, direction: str) -> bool:
    """
    Conferma HA valida:
    BUY  -> HA_DIRECTION = +1 e HA_INDECISION = 0
    SELL -> HA_DIRECTION = -1 e HA_INDECISION = 0
    """

    ha_dir = pd.to_numeric(
        pd.Series([row.get("HA_DIRECTION", np.nan)]),
        errors="coerce",
    ).iloc[0]

    ha_ind = pd.to_numeric(
        pd.Series([row.get("HA_INDECISION", np.nan)]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(ha_dir) or pd.isna(ha_ind):
        return False

    if int(ha_ind) != 0:
        return False

    if direction == "BUY":
        return int(ha_dir) == 1

    return int(ha_dir) == -1


# ============================================================
# SAR RUN
# ============================================================

def _same_sar_run(
    g: pd.DataFrame,
    start_pos: int,
    side: int,
) -> pd.DataFrame:

    end = start_pos

    for j in range(start_pos, len(g)):
        current_side = pd.to_numeric(
            pd.Series([g.iloc[j].get("SAR_SIDE", np.nan)]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(current_side):
            break

        if int(current_side) != side:
            break

        end = j

    return g.iloc[start_pos:end + 1].copy()


def _confirmation_week(
    run: pd.DataFrame,
    direction: str,
    k: int,
    mode: str,
) -> tuple[int | None, int, int]:
    """
    Restituisce:
    - settimana relativa in cui scatterebbe BUY1/SELL1;
    - numero totale HA valide nel run;
    - massimo numero di HA valide consecutive.

    La settimana del flip SAR è week=1.
    """

    cumulative = 0
    consecutive = 0
    max_consecutive = 0
    confirmation_week = None

    for week, (_, row) in enumerate(run.iterrows(), start=1):

        valid = _valid_ha_confirmation(row, direction)

        if valid:
            cumulative += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            if mode == "CONSECUTIVE":
                consecutive = 0

        count = cumulative if mode == "CUMULATIVE" else consecutive

        if confirmation_week is None and count >= k:
            confirmation_week = week

    return confirmation_week, cumulative, max_consecutive


# ============================================================
# PROCESS ONE TICKER
# ============================================================

def process_ticker(raw: pd.DataFrame, ticker: str) -> list[dict]:

    g = raw.copy()

    # V4 completo in modalità shadow
    g = v4.process_ticker(g).reset_index(drop=True)

    date_col = _date_col(g)
    phase_col = _phase_col(g)
    prev_age_col = _prev_sar_age_col(g)
    qualified_col = _run_qualified_col(g)
    run_id_col = _run_id_col(g)

    if prev_age_col is None:
        raise KeyError(
            f"{ticker}: colonna previous SAR age non trovata."
        )

    if qualified_col is None:
        raise KeyError(
            f"{ticker}: colonna V4010_RUN_QUALIFIED non trovata."
        )

    sar_age = _num(g["SAR_AGE"], default=-1)
    sar_side = _num(g["SAR_SIDE"], default=0)
    run_qualified = _num(g[qualified_col], default=0)

    # Solo flip SAR che V40.10 considera qualificati
    flip_positions = np.where(
        (sar_age == 1) &
        (run_qualified == 1) &
        (sar_side.isin([1, -1]))
    )[0]

    records = []

    for pos in flip_positions:

        row = g.iloc[pos]

        side = int(sar_side.iloc[pos])
        direction = "BUY" if side == 1 else "SELL"

        prev_sar_age = pd.to_numeric(
            pd.Series([row.get(prev_age_col, np.nan)]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(prev_sar_age):
            prev_sar_age = 0

        prev_sar_age = int(prev_sar_age)

        run = _same_sar_run(
            g=g,
            start_pos=pos,
            side=side,
        )

        actual_run_length = len(run)

        phase_at_flip = (
            str(row.get(phase_col, ""))
            if phase_col is not None
            else ""
        )

        previous_structure_confirmed = pd.to_numeric(
            pd.Series([
                row.get(
                    "V4010_PREVIOUS_STRUCTURE_CONFIRMED",
                    row.get(
                        "PREVIOUS_STRUCTURE_CONFIRMED",
                        np.nan,
                    ),
                )
            ]),
            errors="coerce",
        ).iloc[0]

        run_id = (
            row.get(run_id_col, np.nan)
            if run_id_col is not None
            else np.nan
        )

        for n in N_VALUES:

            if prev_sar_age < n:
                continue

            for mode in MODES:

                for k in K_VALUES:

                    (
                        confirmation_week,
                        total_valid_ha,
                        max_consecutive_ha,
                    ) = _confirmation_week(
                        run=run,
                        direction=direction,
                        k=k,
                        mode=mode,
                    )

                    reached = confirmation_week is not None

                    if reached:
                        remaining_weeks_after_confirmation = (
                            actual_run_length - confirmation_week
                        )
                    else:
                        remaining_weeks_after_confirmation = np.nan

                    # Run breve che la nuova regola evita completamente
                    blocked_le2 = (
                        actual_run_length <= 2 and not reached
                    )

                    blocked_le3 = (
                        actual_run_length <= 3 and not reached
                    )

                    blocked_le4 = (
                        actual_run_length <= 4 and not reached
                    )

                    # Conferma arrivata proprio alla fine del run:
                    # formalmente raggiunta, ma senza persistenza successiva.
                    confirmed_last_week = (
                        reached and
                        confirmation_week == actual_run_length
                    )

                    # Almeno una settimana ulteriore nello stesso SAR run
                    survived_1w_after_confirmation = (
                        reached and
                        remaining_weeks_after_confirmation >= 1
                    )

                    # Almeno due settimane ulteriori
                    survived_2w_after_confirmation = (
                        reached and
                        remaining_weeks_after_confirmation >= 2
                    )

                    records.append(
                        {
                            "TICKER": ticker,
                            "DATE": row.get(date_col),
                            "DIRECTION": direction,
                            "N_PREV_TREND_MIN": n,
                            "K_HA_CONFIRMATIONS": k,
                            "MODE": mode,
                            "PREV_SAR_AGE": prev_sar_age,
                            "ACTUAL_NEW_SAR_RUN_LENGTH": actual_run_length,
                            "RUN_ID": run_id,
                            "PHASE_AT_FLIP": phase_at_flip,
                            "PREVIOUS_STRUCTURE_CONFIRMED":
                                previous_structure_confirmed,
                            "TOTAL_VALID_HA_IN_RUN": total_valid_ha,
                            "MAX_CONSECUTIVE_VALID_HA":
                                max_consecutive_ha,
                            "CONFIRMATION_REACHED": int(reached),
                            "CONFIRMATION_WEEK":
                                confirmation_week
                                if reached else np.nan,
                            "REMAINING_WEEKS_AFTER_CONFIRMATION":
                                remaining_weeks_after_confirmation,
                            "CONFIRMED_LAST_WEEK":
                                int(confirmed_last_week),
                            "SURVIVED_1W_AFTER_CONFIRMATION":
                                int(survived_1w_after_confirmation),
                            "SURVIVED_2W_AFTER_CONFIRMATION":
                                int(survived_2w_after_confirmation),
                            "SHORT_RUN_LE2W":
                                int(actual_run_length <= 2),
                            "SHORT_RUN_LE2W_BLOCKED":
                                int(blocked_le2),
                            "SHORT_RUN_LE3W":
                                int(actual_run_length <= 3),
                            "SHORT_RUN_LE3W_BLOCKED":
                                int(blocked_le3),
                            "SHORT_RUN_LE4W":
                                int(actual_run_length <= 4),
                            "SHORT_RUN_LE4W_BLOCKED":
                                int(blocked_le4),
                        }
                    )

    return records


# ============================================================
# SUMMARY
# ============================================================

def _pct(num, den):
    if den == 0:
        return np.nan
    return 100.0 * num / den


def build_summary(events: pd.DataFrame) -> pd.DataFrame:

    rows = []

    group_cols = [
        "DIRECTION",
        "N_PREV_TREND_MIN",
        "K_HA_CONFIRMATIONS",
        "MODE",
    ]

    for keys, x in events.groupby(group_cols, dropna=False):

        direction, n, k, mode = keys

        events_n = len(x)

        reached = int(x["CONFIRMATION_REACHED"].sum())
        not_reached = events_n - reached

        confirmed = x[x["CONFIRMATION_REACHED"] == 1]

        short2 = int(x["SHORT_RUN_LE2W"].sum())
        blocked2 = int(x["SHORT_RUN_LE2W_BLOCKED"].sum())

        short3 = int(x["SHORT_RUN_LE3W"].sum())
        blocked3 = int(x["SHORT_RUN_LE3W_BLOCKED"].sum())

        short4 = int(x["SHORT_RUN_LE4W"].sum())
        blocked4 = int(x["SHORT_RUN_LE4W_BLOCKED"].sum())

        confirmed_last = int(x["CONFIRMED_LAST_WEEK"].sum())

        survived1 = int(
            x["SURVIVED_1W_AFTER_CONFIRMATION"].sum()
        )

        survived2 = int(
            x["SURVIVED_2W_AFTER_CONFIRMATION"].sum()
        )

        rows.append(
            {
                "DIRECTION": direction,
                "N_PREV_TREND_MIN": int(n),
                "K_HA_CONFIRMATIONS": int(k),
                "MODE": mode,

                "EVENTS": events_n,

                "CONFIRMATION_REACHED": reached,
                "CONFIRMATION_NOT_REACHED": not_reached,
                "CONFIRMATION_REACHED_PCT":
                    _pct(reached, events_n),

                "MEDIAN_CONFIRMATION_WEEK":
                    confirmed["CONFIRMATION_WEEK"].median()
                    if len(confirmed) else np.nan,

                "MEAN_CONFIRMATION_WEEK":
                    confirmed["CONFIRMATION_WEEK"].mean()
                    if len(confirmed) else np.nan,

                "CONFIRMED_LAST_WEEK": confirmed_last,
                "CONFIRMED_LAST_WEEK_PCT_OF_REACHED":
                    _pct(confirmed_last, reached),

                "SURVIVED_1W_AFTER_CONFIRMATION":
                    survived1,
                "SURVIVED_1W_AFTER_CONFIRMATION_PCT":
                    _pct(survived1, reached),

                "SURVIVED_2W_AFTER_CONFIRMATION":
                    survived2,
                "SURVIVED_2W_AFTER_CONFIRMATION_PCT":
                    _pct(survived2, reached),

                "SHORT_RUN_LE2W": short2,
                "SHORT_RUN_LE2W_BLOCKED": blocked2,
                "SHORT_RUN_LE2W_BLOCKED_PCT":
                    _pct(blocked2, short2),

                "SHORT_RUN_LE3W": short3,
                "SHORT_RUN_LE3W_BLOCKED": blocked3,
                "SHORT_RUN_LE3W_BLOCKED_PCT":
                    _pct(blocked3, short3),

                "SHORT_RUN_LE4W": short4,
                "SHORT_RUN_LE4W_BLOCKED": blocked4,
                "SHORT_RUN_LE4W_BLOCKED_PCT":
                    _pct(blocked4, short4),
            }
        )

    out = pd.DataFrame(rows)

    return out.sort_values(
        [
            "DIRECTION",
            "N_PREV_TREND_MIN",
            "MODE",
            "K_HA_CONFIRMATIONS",
        ]
    ).reset_index(drop=True)


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint() -> pd.DataFrame:

    if not OUT_CHECKPOINT.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(OUT_CHECKPOINT)
    except Exception:
        return pd.DataFrame()


def save_checkpoint(records: list[dict]) -> None:

    if not records:
        return

    pd.DataFrame(records).to_csv(
        OUT_CHECKPOINT,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MARKETSENTINEL")
    print("DIAGNOSI LONG-TREND CONFIRMATION - K3 vs K4")
    print("=" * 80)
    print()
    print(f"Input: {INPUT_FILE}")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File non trovato: {INPUT_FILE}"
        )

    print("Lettura FULL200...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    ticker_col = _ticker_col(df)
    date_col = _date_col(df)

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    tickers = sorted(
        df[ticker_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(f"Ticker: {len(tickers)}")
    print()

    checkpoint = load_checkpoint()

    if len(checkpoint):
        completed_tickers = set(
            checkpoint["TICKER"]
            .dropna()
            .astype(str)
            .unique()
        )

        all_records = checkpoint.to_dict("records")

        print(
            f"Checkpoint trovato: "
            f"{len(completed_tickers)}/{len(tickers)} ticker già completati."
        )
        print()
    else:
        completed_tickers = set()
        all_records = []

    errors = []

    for i, ticker in enumerate(tickers, start=1):

        if ticker in completed_tickers:
            print(
                f"[{i:03d}/{len(tickers):03d}] "
                f"{ticker} - checkpoint"
            )
            continue

        print(
            f"[{i:03d}/{len(tickers):03d}] {ticker}",
            flush=True,
        )

        raw_ticker = (
            df.loc[
                df[ticker_col].astype(str) == ticker
            ]
            .copy()
            .sort_values(date_col)
            .reset_index(drop=True)
        )

        try:
            recs = process_ticker(
                raw=raw_ticker,
                ticker=ticker,
            )

            all_records.extend(recs)

            # Riga sentinella se il ticker non genera eventi,
            # così al riavvio sappiamo comunque che è stato elaborato.
            if not recs:
                all_records.append(
                    {
                        "TICKER": ticker,
                        "DATE": np.nan,
                        "DIRECTION": "__NO_EVENT__",
                        "N_PREV_TREND_MIN": np.nan,
                        "K_HA_CONFIRMATIONS": np.nan,
                        "MODE": "__CHECKPOINT__",
                    }
                )

            save_checkpoint(all_records)

        except Exception as e:
            errors.append(
                {
                    "TICKER": ticker,
                    "ERROR": repr(e),
                }
            )

            print(
                f"  ERRORE: {repr(e)}",
                flush=True,
            )

    events = pd.DataFrame(all_records)

    # Rimuove le righe sentinella
    real_events = events[
        events["DIRECTION"].isin(["BUY", "SELL"])
    ].copy()

    real_events.to_csv(
        OUT_EVENTS,
        index=False,
    )

    summary = build_summary(real_events)

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    print()
    print("=" * 80)
    print("RISULTATI")
    print("=" * 80)

    print()
    print(f"Ticker input: {len(tickers)}")
    print(f"Errori: {len(errors)}")
    print(f"Eventi diagnostici: {len(real_events)}")

    print()
    print(
        summary[
            [
                "DIRECTION",
                "N_PREV_TREND_MIN",
                "MODE",
                "K_HA_CONFIRMATIONS",
                "EVENTS",
                "CONFIRMATION_REACHED_PCT",
                "MEDIAN_CONFIRMATION_WEEK",
                "SHORT_RUN_LE2W_BLOCKED_PCT",
                "SHORT_RUN_LE3W_BLOCKED_PCT",
                "SHORT_RUN_LE4W_BLOCKED_PCT",
                "CONFIRMED_LAST_WEEK_PCT_OF_REACHED",
                "SURVIVED_1W_AFTER_CONFIRMATION_PCT",
            ]
        ].to_string(index=False)
    )

    if errors:
        print()
        print("ERRORI:")
        for err in errors:
            print(
                f"  {err['TICKER']}: {err['ERROR']}"
            )

    print()
    print("Output:")
    print(f"  {OUT_EVENTS}")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_CHECKPOINT}")

    print()
    print("DIAGNOSI K3/K4 COMPLETATA.")


if __name__ == "__main__":
    main()