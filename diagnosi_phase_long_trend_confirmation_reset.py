"""
MarketSentinel
DIAGNOSI LONG-TREND CONFIRMATION - CUMULATIVE vs RESET
======================================================

OBIETTIVO
---------
Ultimo test diagnostico della regola di conferma dopo un trend opposto lungo.

Configurazione candidata già emersa dai test FULL200:

    N = 8 settimane di precedente SAR regime
    K = 3 conferme HA decisive

Confrontiamo:

1) CUMULATIVE
   - HA valida nella nuova direzione -> +1
   - HA indecisione -> non conta, non azzera
   - HA decisa opposta -> non conta, non azzera

2) RESET
   - HA valida nella nuova direzione -> +1
   - HA indecisione -> non conta, non azzera
   - HA decisa opposta -> AZZERA il conteggio

Esempio BUY RESET:

    verde decisa       -> TEST1
    indecisione        -> resta TEST1
    verde decisa       -> TEST2
    rossa decisa       -> RESET a 0
    verde decisa       -> TEST1
    verde decisa       -> TEST2
    verde decisa       -> BUY1

Simmetrico per SELL.

IMPORTANTE
----------
- SOLO diagnostica SHADOW.
- NON modifica V40.10.
- NON modifica V4.
- NON modifica PHASE di produzione.
- NON usa rendimenti futuri per creare la regola.
- La durata futura del SAR run serve solo alla valutazione ex-post.

CHECKPOINT
----------
Salva ogni ticker completato.
Se Codespaces interrompe il processo, il rilancio riparte dal checkpoint.

OUTPUT
------
data/diagnosi_phase_long_trend_confirmation_reset_events.csv
data/diagnosi_phase_long_trend_confirmation_reset_summary.csv
data/diagnosi_phase_long_trend_confirmation_reset_cases.csv
data/diagnosi_phase_long_trend_confirmation_reset_checkpoint.csv
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
    "data/diagnosi_phase_long_trend_confirmation_reset_events.csv"
)

OUT_SUMMARY = Path(
    "data/diagnosi_phase_long_trend_confirmation_reset_summary.csv"
)

OUT_CASES = Path(
    "data/diagnosi_phase_long_trend_confirmation_reset_cases.csv"
)

OUT_CHECKPOINT = Path(
    "data/diagnosi_phase_long_trend_confirmation_reset_checkpoint.csv"
)

N_PREV_TREND_MIN = 8
K_HA_CONFIRMATIONS = 3

MODES = [
    "CUMULATIVE",
    "RESET",
]


# ============================================================
# BASIC UTILS
# ============================================================

def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def _ticker_col(df: pd.DataFrame) -> str:
    c = _first_existing(
        df,
        [
            "Ticker",
            "TICKER",
            "ticker",
            "Symbol",
            "SYMBOL",
            "symbol",
        ],
    )

    if c is None:
        raise KeyError("Colonna ticker non trovata.")

    return c


def _date_col(df: pd.DataFrame) -> str:
    c = _first_existing(
        df,
        [
            "Date",
            "DATE",
            "date",
            "Datetime",
            "DATETIME",
            "datetime",
        ],
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


def _qualified_col(df: pd.DataFrame) -> str | None:
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


def _to_num(value, default=np.nan):
    x = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(x):
        return default

    return x


def _pct(num, den):
    if den == 0:
        return np.nan

    return 100.0 * num / den


# ============================================================
# HA CLASSIFICATION
# ============================================================

def _ha_state(row: pd.Series, direction: str) -> str:
    """
    Restituisce:

        VALID
        INDECISION
        OPPOSITE
        UNKNOWN

    BUY:
        VALID    = HA_DIRECTION +1, HA_INDECISION 0
        OPPOSITE = HA_DIRECTION -1, HA_INDECISION 0

    SELL:
        VALID    = HA_DIRECTION -1, HA_INDECISION 0
        OPPOSITE = HA_DIRECTION +1, HA_INDECISION 0
    """

    ha_dir = _to_num(
        row.get("HA_DIRECTION", np.nan),
        default=np.nan,
    )

    ha_ind = _to_num(
        row.get("HA_INDECISION", np.nan),
        default=np.nan,
    )

    if pd.isna(ha_dir) or pd.isna(ha_ind):
        return "UNKNOWN"

    if int(ha_ind) == 1:
        return "INDECISION"

    expected = 1 if direction == "BUY" else -1

    if int(ha_dir) == expected:
        return "VALID"

    if int(ha_dir) == -expected:
        return "OPPOSITE"

    return "UNKNOWN"


# ============================================================
# SAR RUN
# ============================================================

def _extract_same_sar_run(
    g: pd.DataFrame,
    start_pos: int,
    side: int,
) -> pd.DataFrame:

    end_pos = start_pos

    for pos in range(start_pos, len(g)):

        current_side = _to_num(
            g.iloc[pos].get("SAR_SIDE", np.nan),
            default=np.nan,
        )

        if pd.isna(current_side):
            break

        if int(current_side) != side:
            break

        end_pos = pos

    return g.iloc[start_pos:end_pos + 1].copy()


# ============================================================
# CONFIRMATION
# ============================================================

def _evaluate_confirmation(
    run: pd.DataFrame,
    direction: str,
    mode: str,
) -> dict:

    count = 0
    confirmation_week = None

    valid_total = 0
    indecision_total = 0
    opposite_total = 0
    reset_total = 0

    max_count_before_confirmation = 0

    states = []

    for week, (_, row) in enumerate(
        run.iterrows(),
        start=1,
    ):

        state = _ha_state(
            row=row,
            direction=direction,
        )

        states.append(state)

        if state == "VALID":
            valid_total += 1
            count += 1

        elif state == "INDECISION":
            indecision_total += 1

        elif state == "OPPOSITE":
            opposite_total += 1

            if mode == "RESET":
                if count > 0:
                    reset_total += 1

                count = 0

        # UNKNOWN non modifica il conteggio

        max_count_before_confirmation = max(
            max_count_before_confirmation,
            count,
        )

        if (
            confirmation_week is None
            and count >= K_HA_CONFIRMATIONS
        ):
            confirmation_week = week

    reached = confirmation_week is not None

    return {
        "CONFIRMATION_REACHED": int(reached),

        "CONFIRMATION_WEEK":
            confirmation_week
            if reached
            else np.nan,

        "TOTAL_VALID_HA": valid_total,

        "TOTAL_INDECISION_HA": indecision_total,

        "TOTAL_OPPOSITE_HA": opposite_total,

        "RESET_COUNT": reset_total,

        "MAX_TEST_COUNT":
            max_count_before_confirmation,

        "HA_SEQUENCE":
            "|".join(states),
    }


# ============================================================
# PROCESS ONE TICKER
# ============================================================

def process_ticker(
    raw: pd.DataFrame,
    ticker: str,
) -> list[dict]:

    g = v4.process_ticker(
        raw.copy()
    ).reset_index(drop=True)

    date_col = _date_col(g)
    phase_col = _phase_col(g)

    prev_age_col = _prev_sar_age_col(g)
    qualified_col = _qualified_col(g)
    run_id_col = _run_id_col(g)

    if prev_age_col is None:
        raise KeyError(
            f"{ticker}: previous SAR age non trovato."
        )

    if qualified_col is None:
        raise KeyError(
            f"{ticker}: RUN_QUALIFIED non trovato."
        )

    sar_age = pd.to_numeric(
        g["SAR_AGE"],
        errors="coerce",
    )

    sar_side = pd.to_numeric(
        g["SAR_SIDE"],
        errors="coerce",
    )

    qualified = pd.to_numeric(
        g[qualified_col],
        errors="coerce",
    ).fillna(0)

    flip_positions = np.where(
        (sar_age == 1)
        & (qualified == 1)
        & (sar_side.isin([1, -1]))
    )[0]

    records = []

    for pos in flip_positions:

        row = g.iloc[pos]

        side = int(sar_side.iloc[pos])

        direction = (
            "BUY"
            if side == 1
            else "SELL"
        )

        prev_sar_age = int(
            _to_num(
                row.get(
                    prev_age_col,
                    0,
                ),
                default=0,
            )
        )

        # Ci interessano solo i flip dopo trend lungo >= 8
        if prev_sar_age < N_PREV_TREND_MIN:
            continue

        run = _extract_same_sar_run(
            g=g,
            start_pos=pos,
            side=side,
        )

        actual_run_length = len(run)

        phase_at_flip = (
            str(
                row.get(
                    phase_col,
                    "",
                )
            )
            if phase_col is not None
            else ""
        )

        run_id = (
            row.get(
                run_id_col,
                np.nan,
            )
            if run_id_col is not None
            else np.nan
        )

        prev_structure_confirmed = _to_num(
            row.get(
                "V4010_PREVIOUS_STRUCTURE_CONFIRMED",
                row.get(
                    "PREVIOUS_STRUCTURE_CONFIRMED",
                    np.nan,
                ),
            ),
            default=np.nan,
        )

        mode_results = {}

        for mode in MODES:

            result = _evaluate_confirmation(
                run=run,
                direction=direction,
                mode=mode,
            )

            mode_results[mode] = result

        cumulative_week = mode_results[
            "CUMULATIVE"
        ]["CONFIRMATION_WEEK"]

        reset_week = mode_results[
            "RESET"
        ]["CONFIRMATION_WEEK"]

        cumulative_reached = mode_results[
            "CUMULATIVE"
        ]["CONFIRMATION_REACHED"]

        reset_reached = mode_results[
            "RESET"
        ]["CONFIRMATION_REACHED"]

        # Effetto specifico del RESET
        reset_changes_result = int(
            (
                cumulative_reached
                != reset_reached
            )
            or (
                cumulative_reached == 1
                and reset_reached == 1
                and cumulative_week != reset_week
            )
        )

        for mode in MODES:

            result = mode_results[mode]

            reached = bool(
                result[
                    "CONFIRMATION_REACHED"
                ]
            )

            confirmation_week = result[
                "CONFIRMATION_WEEK"
            ]

            if reached:
                remaining = (
                    actual_run_length
                    - int(confirmation_week)
                )
            else:
                remaining = np.nan

            confirmed_last_week = int(
                reached
                and int(confirmation_week)
                == actual_run_length
            )

            survived_1w = int(
                reached
                and remaining >= 1
            )

            survived_2w = int(
                reached
                and remaining >= 2
            )

            records.append(
                {
                    "TICKER": ticker,

                    "DATE":
                        row.get(date_col),

                    "DIRECTION":
                        direction,

                    "MODE":
                        mode,

                    "N_PREV_TREND_MIN":
                        N_PREV_TREND_MIN,

                    "K_HA_CONFIRMATIONS":
                        K_HA_CONFIRMATIONS,

                    "PREV_SAR_AGE":
                        prev_sar_age,

                    "ACTUAL_NEW_SAR_RUN_LENGTH":
                        actual_run_length,

                    "RUN_ID":
                        run_id,

                    "PHASE_AT_FLIP":
                        phase_at_flip,

                    "PREVIOUS_STRUCTURE_CONFIRMED":
                        prev_structure_confirmed,

                    "CONFIRMATION_REACHED":
                        result[
                            "CONFIRMATION_REACHED"
                        ],

                    "CONFIRMATION_WEEK":
                        result[
                            "CONFIRMATION_WEEK"
                        ],

                    "TOTAL_VALID_HA":
                        result[
                            "TOTAL_VALID_HA"
                        ],

                    "TOTAL_INDECISION_HA":
                        result[
                            "TOTAL_INDECISION_HA"
                        ],

                    "TOTAL_OPPOSITE_HA":
                        result[
                            "TOTAL_OPPOSITE_HA"
                        ],

                    "RESET_COUNT":
                        result[
                            "RESET_COUNT"
                        ],

                    "MAX_TEST_COUNT":
                        result[
                            "MAX_TEST_COUNT"
                        ],

                    "HA_SEQUENCE":
                        result[
                            "HA_SEQUENCE"
                        ],

                    "REMAINING_WEEKS_AFTER_CONFIRMATION":
                        remaining,

                    "CONFIRMED_LAST_WEEK":
                        confirmed_last_week,

                    "SURVIVED_1W_AFTER_CONFIRMATION":
                        survived_1w,

                    "SURVIVED_2W_AFTER_CONFIRMATION":
                        survived_2w,

                    "SHORT_RUN_LE2W":
                        int(
                            actual_run_length <= 2
                        ),

                    "SHORT_RUN_LE2W_BLOCKED":
                        int(
                            actual_run_length <= 2
                            and not reached
                        ),

                    "SHORT_RUN_LE3W":
                        int(
                            actual_run_length <= 3
                        ),

                    "SHORT_RUN_LE3W_BLOCKED":
                        int(
                            actual_run_length <= 3
                            and not reached
                        ),

                    "SHORT_RUN_LE4W":
                        int(
                            actual_run_length <= 4
                        ),

                    "SHORT_RUN_LE4W_BLOCKED":
                        int(
                            actual_run_length <= 4
                            and not reached
                        ),

                    "RESET_CHANGES_RESULT":
                        reset_changes_result,
                }
            )

    return records


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        direction,
        mode,
    ), x in events.groupby(
        [
            "DIRECTION",
            "MODE",
        ],
        dropna=False,
    ):

        total = len(x)

        reached = int(
            x[
                "CONFIRMATION_REACHED"
            ].sum()
        )

        confirmed = x[
            x[
                "CONFIRMATION_REACHED"
            ] == 1
        ]

        short2 = int(
            x[
                "SHORT_RUN_LE2W"
            ].sum()
        )

        blocked2 = int(
            x[
                "SHORT_RUN_LE2W_BLOCKED"
            ].sum()
        )

        short3 = int(
            x[
                "SHORT_RUN_LE3W"
            ].sum()
        )

        blocked3 = int(
            x[
                "SHORT_RUN_LE3W_BLOCKED"
            ].sum()
        )

        short4 = int(
            x[
                "SHORT_RUN_LE4W"
            ].sum()
        )

        blocked4 = int(
            x[
                "SHORT_RUN_LE4W_BLOCKED"
            ].sum()
        )

        reset_changed = int(
            x[
                "RESET_CHANGES_RESULT"
            ].sum()
        )

        confirmed_last = int(
            x[
                "CONFIRMED_LAST_WEEK"
            ].sum()
        )

        survived1 = int(
            x[
                "SURVIVED_1W_AFTER_CONFIRMATION"
            ].sum()
        )

        survived2 = int(
            x[
                "SURVIVED_2W_AFTER_CONFIRMATION"
            ].sum()
        )

        rows.append(
            {
                "DIRECTION":
                    direction,

                "MODE":
                    mode,

                "EVENTS":
                    total,

                "CONFIRMATION_REACHED":
                    reached,

                "CONFIRMATION_REACHED_PCT":
                    _pct(
                        reached,
                        total,
                    ),

                "MEDIAN_CONFIRMATION_WEEK":
                    (
                        confirmed[
                            "CONFIRMATION_WEEK"
                        ].median()
                        if len(confirmed)
                        else np.nan
                    ),

                "MEAN_CONFIRMATION_WEEK":
                    (
                        confirmed[
                            "CONFIRMATION_WEEK"
                        ].mean()
                        if len(confirmed)
                        else np.nan
                    ),

                "SHORT_RUN_LE2W":
                    short2,

                "SHORT_RUN_LE2W_BLOCKED":
                    blocked2,

                "SHORT_RUN_LE2W_BLOCKED_PCT":
                    _pct(
                        blocked2,
                        short2,
                    ),

                "SHORT_RUN_LE3W":
                    short3,

                "SHORT_RUN_LE3W_BLOCKED":
                    blocked3,

                "SHORT_RUN_LE3W_BLOCKED_PCT":
                    _pct(
                        blocked3,
                        short3,
                    ),

                "SHORT_RUN_LE4W":
                    short4,

                "SHORT_RUN_LE4W_BLOCKED":
                    blocked4,

                "SHORT_RUN_LE4W_BLOCKED_PCT":
                    _pct(
                        blocked4,
                        short4,
                    ),

                "CONFIRMED_LAST_WEEK":
                    confirmed_last,

                "CONFIRMED_LAST_WEEK_PCT":
                    _pct(
                        confirmed_last,
                        reached,
                    ),

                "SURVIVED_1W_AFTER_CONFIRMATION":
                    survived1,

                "SURVIVED_1W_AFTER_CONFIRMATION_PCT":
                    _pct(
                        survived1,
                        reached,
                    ),

                "SURVIVED_2W_AFTER_CONFIRMATION":
                    survived2,

                "SURVIVED_2W_AFTER_CONFIRMATION_PCT":
                    _pct(
                        survived2,
                        reached,
                    ),

                "RESET_CHANGES_RESULT":
                    reset_changed,

                "RESET_CHANGES_RESULT_PCT":
                    _pct(
                        reset_changed,
                        total,
                    ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "DIRECTION",
                "MODE",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# IMPORTANT CASES
# ============================================================

def build_cases(
    events: pd.DataFrame,
) -> pd.DataFrame:

    # Solo una riga per evento/mode
    x = events.copy()

    # Casi recenti già studiati + casi in cui RESET
    # cambia davvero il risultato rispetto a CUMULATIVE.
    important = (
        (
            x["TICKER"].isin(
                [
                    "ISRG",
                    "CRWD",
                ]
            )
            & pd.to_datetime(
                x["DATE"],
                errors="coerce",
            ).dt.year.ge(2025)
        )
        |
        (
            x[
                "RESET_CHANGES_RESULT"
            ] == 1
        )
    )

    cols = [
        "TICKER",
        "DATE",
        "DIRECTION",
        "MODE",
        "PREV_SAR_AGE",
        "ACTUAL_NEW_SAR_RUN_LENGTH",
        "CONFIRMATION_REACHED",
        "CONFIRMATION_WEEK",
        "TOTAL_VALID_HA",
        "TOTAL_INDECISION_HA",
        "TOTAL_OPPOSITE_HA",
        "RESET_COUNT",
        "HA_SEQUENCE",
        "RESET_CHANGES_RESULT",
    ]

    out = x.loc[
        important,
        cols,
    ].copy()

    return out.sort_values(
        [
            "TICKER",
            "DATE",
            "MODE",
        ]
    ).reset_index(drop=True)


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint() -> pd.DataFrame:

    if not OUT_CHECKPOINT.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            OUT_CHECKPOINT,
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()


def save_checkpoint(
    records: list[dict],
) -> None:

    if not records:
        return

    pd.DataFrame(
        records
    ).to_csv(
        OUT_CHECKPOINT,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("MARKETSENTINEL")
    print("LONG-TREND CONFIRMATION: CUMULATIVE vs RESET")
    print("=" * 78)

    print()
    print(f"Input: {INPUT_FILE}")
    print(
        f"Regola: N={N_PREV_TREND_MIN}, "
        f"K={K_HA_CONFIRMATIONS}"
    )
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
        df[
            ticker_col
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Ticker: {len(tickers)}"
    )
    print()

    checkpoint = load_checkpoint()

    if len(checkpoint):

        completed_tickers = set(
            checkpoint[
                "TICKER"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        all_records = checkpoint.to_dict(
            "records"
        )

        print(
            f"Checkpoint trovato: "
            f"{len(completed_tickers)}/"
            f"{len(tickers)} ticker."
        )
        print()

    else:

        completed_tickers = set()
        all_records = []

    errors = []

    for i, ticker in enumerate(
        tickers,
        start=1,
    ):

        if ticker in completed_tickers:

            print(
                f"[{i:03d}/{len(tickers):03d}] "
                f"{ticker} - checkpoint",
                flush=True,
            )

            continue

        print(
            f"[{i:03d}/{len(tickers):03d}] "
            f"{ticker}",
            flush=True,
        )

        raw_ticker = (
            df.loc[
                df[
                    ticker_col
                ].astype(str) == ticker
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

            if recs:

                all_records.extend(
                    recs
                )

            else:

                # Sentinella checkpoint
                all_records.append(
                    {
                        "TICKER":
                            ticker,

                        "DATE":
                            np.nan,

                        "DIRECTION":
                            "__NO_EVENT__",

                        "MODE":
                            "__CHECKPOINT__",
                    }
                )

            save_checkpoint(
                all_records
            )

        except Exception as exc:

            errors.append(
                {
                    "TICKER":
                        ticker,

                    "ERROR":
                        repr(exc),
                }
            )

            print(
                f"  ERRORE: {repr(exc)}",
                flush=True,
            )

    events_all = pd.DataFrame(
        all_records
    )

    events = events_all.loc[
        events_all[
            "DIRECTION"
        ].isin(
            [
                "BUY",
                "SELL",
            ]
        )
    ].copy()

    events.to_csv(
        OUT_EVENTS,
        index=False,
    )

    summary = build_summary(
        events
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    cases = build_cases(
        events
    )

    cases.to_csv(
        OUT_CASES,
        index=False,
    )

    print()
    print("=" * 78)
    print("RISULTATI")
    print("=" * 78)
    print()

    print(
        summary[
            [
                "DIRECTION",
                "MODE",
                "EVENTS",
                "CONFIRMATION_REACHED_PCT",
                "MEDIAN_CONFIRMATION_WEEK",
                "SHORT_RUN_LE2W_BLOCKED_PCT",
                "SHORT_RUN_LE3W_BLOCKED_PCT",
                "SHORT_RUN_LE4W_BLOCKED_PCT",
                "SURVIVED_1W_AFTER_CONFIRMATION_PCT",
                "RESET_CHANGES_RESULT_PCT",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Errori: {len(errors)}"
    )

    if errors:

        for err in errors:
            print(
                f"{err['TICKER']}: "
                f"{err['ERROR']}"
            )

    print()
    print("Output:")
    print(
        f"  {OUT_EVENTS}"
    )
    print(
        f"  {OUT_SUMMARY}"
    )
    print(
        f"  {OUT_CASES}"
    )
    print(
        f"  {OUT_CHECKPOINT}"
    )

    print()
    print(
        "DIAGNOSI RESET COMPLETATA."
    )


if __name__ == "__main__":
    main()