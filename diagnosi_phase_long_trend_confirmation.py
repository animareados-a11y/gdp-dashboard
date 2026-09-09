"""
MarketSentinel
DIAGNOSI PHASE - LONG TREND CONFIRMATION

OBIETTIVO
---------
Studiare una possibile regola prudenziale per i cambi di regime dopo
un trend SAR opposto molto lungo.

Problema osservato:
    dopo molte settimane di trend ribassista/rialzista, un semplice
    flip del SAR può rappresentare soltanto un rimbalzo/correzione
    temporanea e non necessariamente l'inizio di un nuovo trend.

Esempio concettuale:
    lungo trend ribassista
        -> SAR gira bullish
        -> 1/2 settimane di rimbalzo/incertezza
        -> SAR torna bearish

In questi casi non vogliamo necessariamente dichiarare subito:
    BUY1 / BUY2

Lo stesso principio deve valere simmetricamente per SELL.

QUESTO SCRIPT:
--------------
- NON modifica V40.10
- NON modifica V4
- NON modifica PHASE production
- NON modifica engine.py
- NON usa dati futuri per decidere la PHASE
- è esclusivamente diagnostico / SHADOW

Testa:

    N = 6, 8, 10, 12

dove N è la durata minima del precedente regime SAR opposto.

Per ogni flip qualificato V40.10 misura poi cosa succede al nuovo
regime SAR:

    durata 1 settimana
    durata 2 settimane
    durata 3 settimane
    durata >= 4 settimane

Studia inoltre le prime settimane del nuovo regime e verifica
quante presentano una Heikin Ashi:

    - direzionale nella nuova direzione
    - non di indecisione

Vengono simulate due ipotesi:

    K = 2 conferme HA valide
    K = 3 conferme HA valide

IMPORTANTE
----------
La parte che osserva la durata successiva del regime serve SOLO
alla validazione statistica ex-post della regola.

NON è una regola utilizzabile in produzione e NON viene utilizzata
per classificare causalmente la PHASE.

OUTPUT
------
data/diagnosi_phase_long_trend_confirmation_events.csv
data/diagnosi_phase_long_trend_confirmation_summary.csv
data/diagnosi_phase_long_trend_confirmation_nk.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v4 as v4


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

OUTPUT_DIR = Path("data")

OUT_EVENTS = (
    OUTPUT_DIR
    / "diagnosi_phase_long_trend_confirmation_events.csv"
)

OUT_SUMMARY = (
    OUTPUT_DIR
    / "diagnosi_phase_long_trend_confirmation_summary.csv"
)

OUT_NK = (
    OUTPUT_DIR
    / "diagnosi_phase_long_trend_confirmation_nk.csv"
)

N_VALUES = [6, 8, 10, 12]
K_VALUES = [2, 3]


# ============================================================
# HELPERS
# ============================================================

def _num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _valid_ha_confirmation(
    row: pd.Series,
    direction: str,
) -> bool:
    """
    Conferma HA semplice e volutamente trasparente.

    BUY:
        HA_DIRECTION == +1
        HA_INDECISION == 0

    SELL:
        HA_DIRECTION == -1
        HA_INDECISION == 0

    Non aggiungiamo RSI/MACD/Bollinger:
    vogliamo prima isolare il principio proposto sulle HA.
    """

    ha_direction = pd.to_numeric(
        pd.Series([row.get("HA_DIRECTION", 0)]),
        errors="coerce",
    ).fillna(0).iloc[0]

    ha_indecision = pd.to_numeric(
        pd.Series([row.get("HA_INDECISION", 0)]),
        errors="coerce",
    ).fillna(0).iloc[0]

    if int(ha_indecision) != 0:
        return False

    if direction == "BUY":
        return int(ha_direction) == 1

    if direction == "SELL":
        return int(ha_direction) == -1

    return False


def _future_run_length(
    g: pd.DataFrame,
    start_pos: int,
    side: int,
) -> int:
    """
    Durata effettiva ex-post del nuovo regime SAR.

    ATTENZIONE:
    questa funzione guarda le settimane successive.
    Serve ESCLUSIVAMENTE alla diagnosi/validazione storica.
    """

    n = 0

    for pos in range(start_pos, len(g)):
        current_side = int(
            pd.to_numeric(
                pd.Series([g.iloc[pos].get("SAR_SIDE", 0)]),
                errors="coerce",
            ).fillna(0).iloc[0]
        )

        if current_side != side:
            break

        n += 1

    return n


def _confirmation_week(
    g: pd.DataFrame,
    start_pos: int,
    side: int,
    direction: str,
    k: int,
) -> tuple[int | None, int]:
    """
    Cerca causalmente la K-esima HA valida all'interno dello stesso
    regime SAR.

    Restituisce:
        offset settimana rispetto al flip (1 = settimana del flip)
        numero totale di conferme osservate prima della fine del run

    Le HA di indecisione non contano.
    Le HA nella direzione opposta non contano.

    Il conteggio delle conferme non richiede consecutività:
    una settimana di indecisione non azzera le conferme precedenti.
    """

    valid_count = 0
    kth_week = None

    for pos in range(start_pos, len(g)):
        row = g.iloc[pos]

        current_side = int(
            pd.to_numeric(
                pd.Series([row.get("SAR_SIDE", 0)]),
                errors="coerce",
            ).fillna(0).iloc[0]
        )

        if current_side != side:
            break

        if _valid_ha_confirmation(row, direction):
            valid_count += 1

            if valid_count == k and kth_week is None:
                kth_week = (pos - start_pos) + 1

    return kth_week, valid_count


# ============================================================
# PROCESS ONE TICKER
# ============================================================

def process_ticker(
    ticker_df: pd.DataFrame,
) -> list[dict]:

    ticker_df = (
        ticker_df
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    x = v4.process_ticker(ticker_df)

    x = (
        x
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    events: list[dict] = []

    sar_age = _num(x["SAR_AGE"])
    run_qualified = _num(x["V4010_RUN_QUALIFIED"])

    # Un nuovo regime SAR comincia a SAR_AGE == 1.
    # Consideriamo qui SOLO i flip che V40.10 aveva qualificato.
    flip_positions = x.index[
        (sar_age == 1)
        & (run_qualified == 1)
    ].tolist()

    for pos in flip_positions:

        row = x.iloc[pos]

        side = int(
            pd.to_numeric(
                pd.Series([row.get("SAR_SIDE", 0)]),
                errors="coerce",
            ).fillna(0).iloc[0]
        )

        if side not in (-1, 1):
            continue

        direction = "BUY" if side == 1 else "SELL"

        prev_sar_age = float(
            pd.to_numeric(
                pd.Series(
                    [row.get("V4010_PREV_SAR_AGE", np.nan)]
                ),
                errors="coerce",
            ).iloc[0]
        )

        if not np.isfinite(prev_sar_age):
            continue

        actual_run_length = _future_run_length(
            x,
            pos,
            side,
        )

        k2_week, total_valid_ha = _confirmation_week(
            x,
            pos,
            side,
            direction,
            2,
        )

        k3_week, _ = _confirmation_week(
            x,
            pos,
            side,
            direction,
            3,
        )

        event = {
            "Ticker": row.get("Ticker"),
            "Date": row.get("Date"),
            "DIRECTION": direction,
            "SAR_SIDE": side,
            "PREV_SAR_AGE": prev_sar_age,
            "ACTUAL_NEW_RUN_LENGTH": actual_run_length,
            "RUN_1W": int(actual_run_length == 1),
            "RUN_2W": int(actual_run_length == 2),
            "RUN_3W": int(actual_run_length == 3),
            "RUN_GE4W": int(actual_run_length >= 4),
            "TOTAL_VALID_HA_IN_RUN": total_valid_ha,
            "K2_REACHED": int(k2_week is not None),
            "K2_CONFIRMATION_WEEK": k2_week,
            "K3_REACHED": int(k3_week is not None),
            "K3_CONFIRMATION_WEEK": k3_week,
            "PHASE_AT_FLIP_V31": row.get(
                "PHASE_DYNAMICS_V31"
            ),
            "PHASE_AT_FLIP_V4": row.get(
                "PHASE_DYNAMICS_V4"
            ),
            "PREVIOUS_STRUCTURE_CONFIRMED": int(
                pd.to_numeric(
                    pd.Series(
                        [
                            row.get(
                                "V4010_PREVIOUS_STRUCTURE_CONFIRMED",
                                0,
                            )
                        ]
                    ),
                    errors="coerce",
                ).fillna(0).iloc[0]
            ),
        }

        events.append(event)

    return events


# ============================================================
# SUMMARIES
# ============================================================

def build_summary(events: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for direction in ["BUY", "SELL"]:

        z_dir = events[
            events["DIRECTION"] == direction
        ].copy()

        for n in N_VALUES:

            z = z_dir[
                z_dir["PREV_SAR_AGE"] >= n
            ].copy()

            total = len(z)

            if total == 0:
                continue

            rows.append(
                {
                    "DIRECTION": direction,
                    "N_PREV_TREND_MIN": n,
                    "EVENTS": total,
                    "RUN_1W": int(z["RUN_1W"].sum()),
                    "RUN_2W": int(z["RUN_2W"].sum()),
                    "RUN_3W": int(z["RUN_3W"].sum()),
                    "RUN_GE4W": int(z["RUN_GE4W"].sum()),
                    "RUN_LE2W": int(
                        (
                            z["ACTUAL_NEW_RUN_LENGTH"] <= 2
                        ).sum()
                    ),
                    "RUN_LE3W": int(
                        (
                            z["ACTUAL_NEW_RUN_LENGTH"] <= 3
                        ).sum()
                    ),
                    "RUN_LE2W_PCT": (
                        100.0
                        * (
                            z["ACTUAL_NEW_RUN_LENGTH"] <= 2
                        ).mean()
                    ),
                    "RUN_LE3W_PCT": (
                        100.0
                        * (
                            z["ACTUAL_NEW_RUN_LENGTH"] <= 3
                        ).mean()
                    ),
                    "MEDIAN_NEW_RUN_LENGTH": float(
                        z["ACTUAL_NEW_RUN_LENGTH"].median()
                    ),
                    "MEAN_NEW_RUN_LENGTH": float(
                        z["ACTUAL_NEW_RUN_LENGTH"].mean()
                    ),
                    "K2_REACHED_PCT": (
                        100.0 * z["K2_REACHED"].mean()
                    ),
                    "K3_REACHED_PCT": (
                        100.0 * z["K3_REACHED"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_nk_summary(events: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for direction in ["BUY", "SELL"]:

        z_dir = events[
            events["DIRECTION"] == direction
        ].copy()

        for n in N_VALUES:

            z_n = z_dir[
                z_dir["PREV_SAR_AGE"] >= n
            ].copy()

            if len(z_n) == 0:
                continue

            for k in K_VALUES:

                reached_col = f"K{k}_REACHED"
                week_col = f"K{k}_CONFIRMATION_WEEK"

                reached = z_n[
                    z_n[reached_col] == 1
                ].copy()

                not_reached = z_n[
                    z_n[reached_col] == 0
                ].copy()

                short_2 = (
                    z_n["ACTUAL_NEW_RUN_LENGTH"] <= 2
                )

                short_3 = (
                    z_n["ACTUAL_NEW_RUN_LENGTH"] <= 3
                )

                short2_blocked = (
                    short_2
                    & (z_n[reached_col] == 0)
                )

                short3_blocked = (
                    short_3
                    & (z_n[reached_col] == 0)
                )

                rows.append(
                    {
                        "DIRECTION": direction,
                        "N_PREV_TREND_MIN": n,
                        "K_HA_CONFIRMATIONS": k,
                        "EVENTS": len(z_n),
                        "CONFIRMATION_REACHED": len(reached),
                        "CONFIRMATION_NOT_REACHED": len(
                            not_reached
                        ),
                        "CONFIRMATION_REACHED_PCT": (
                            100.0
                            * len(reached)
                            / len(z_n)
                        ),
                        "MEDIAN_CONFIRMATION_WEEK": (
                            float(
                                reached[
                                    week_col
                                ].median()
                            )
                            if len(reached)
                            else np.nan
                        ),
                        "MEAN_CONFIRMATION_WEEK": (
                            float(
                                reached[
                                    week_col
                                ].mean()
                            )
                            if len(reached)
                            else np.nan
                        ),
                        "SHORT_RUN_LE2W": int(
                            short_2.sum()
                        ),
                        "SHORT_RUN_LE2W_BLOCKED": int(
                            short2_blocked.sum()
                        ),
                        "SHORT_RUN_LE2W_BLOCKED_PCT": (
                            100.0
                            * short2_blocked.sum()
                            / short_2.sum()
                            if short_2.sum()
                            else np.nan
                        ),
                        "SHORT_RUN_LE3W": int(
                            short_3.sum()
                        ),
                        "SHORT_RUN_LE3W_BLOCKED": int(
                            short3_blocked.sum()
                        ),
                        "SHORT_RUN_LE3W_BLOCKED_PCT": (
                            100.0
                            * short3_blocked.sum()
                            / short_3.sum()
                            if short_3.sum()
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)
    print("MARKETSENTINEL")
    print("DIAGNOSI LONG TREND CONFIRMATION")
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nInput: {INPUT_FILE}")

    # engine='python' evita il problema di lettura C
    # incontrato durante i controlli interattivi.
    df = pd.read_csv(
        INPUT_FILE,
        engine="python",
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=["Ticker", "Date"])
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    tickers = (
        df["Ticker"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(f"Righe input : {len(df):,}")
    print(f"Ticker      : {len(tickers):,}")

    all_events: list[dict] = []
    errors: list[tuple[str, str]] = []

    for i, ticker in enumerate(tickers, start=1):

        print(
            f"[{i:03d}/{len(tickers):03d}] {ticker}",
            flush=True,
        )

        g = df[
            df["Ticker"].astype(str) == ticker
        ].copy()

        try:
            events = process_ticker(g)
            all_events.extend(events)

        except Exception as exc:
            errors.append(
                (ticker, repr(exc))
            )

            print(
                f"  ERRORE: {repr(exc)}",
                flush=True,
            )

    events_df = pd.DataFrame(all_events)

    if events_df.empty:
        raise RuntimeError(
            "Nessun evento qualificato trovato."
        )

    events_df = (
        events_df
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    summary_df = build_summary(events_df)
    nk_df = build_nk_summary(events_df)

    events_df.to_csv(
        OUT_EVENTS,
        index=False,
    )

    summary_df.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    nk_df.to_csv(
        OUT_NK,
        index=False,
    )

    print("\n" + "=" * 72)
    print("RISULTATI")
    print("=" * 72)

    print(
        f"\nEventi qualificati analizzati: "
        f"{len(events_df):,}"
    )

    print(
        f"Errori ticker: {len(errors):,}"
    )

    if errors:
        print("\nERRORI:")
        for ticker, err in errors:
            print(f"  {ticker}: {err}")

    print("\n--- DURATA NUOVI REGIMI ---")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n--- TEST N x K ---")
    print(
        nk_df.to_string(
            index=False
        )
    )

    print("\nOutput:")
    print(f"  {OUT_EVENTS}")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_NK}")

    print("\nDIAGNOSI COMPLETATA.")


if __name__ == "__main__":
    main()