"""
MarketSentinel
FULL200 - DIAGNOSI USCITE DA INDECISIONE

OBIETTIVO
---------
Analizzare storicamente, su tutti i titoli FULL200, cosa accade
dopo una fase INDECISIONE.

Il test NON modifica alcun motore.

Per ogni episodio di INDECISIONE misura:
- fase precedente;
- durata dell'episodio;
- SAR side/age;
- stato tecnico nell'ultima settimana di INDECISIONE;
- prima fase successiva osservata;
- se la fase successiva è:
    BUY
    TREND_RIALZISTA
    DEBOLEZZA_RIALZISTA
    SELL
    TREND_RIBASSISTA
    DEBOLEZZA_RIBASSISTA
    oppure ancora/non classificabile.

Serve per progettare una state machine generale
INDECISIONE -> qualunque fase coerente.

INPUT
-----
data/v40_35_full200_weekly.csv

OUTPUT
------
data/diagnosi_full200_indecision_exits.csv
data/diagnosi_full200_indecision_exits_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_5_core_phase_engine as v405


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

OUT_DETAIL = Path(
    "data/diagnosi_full200_indecision_exits.csv"
)

OUT_SUMMARY = Path(
    "data/diagnosi_full200_indecision_exits_summary.csv"
)


# ============================================================
# COSTANTI
# ============================================================

INDECISION = "INDECISIONE"

VALID_EXIT_PHASES = {
    "BUY",
    "TREND_RIALZISTA",
    "DEBOLEZZA_RIALZISTA",
    "SELL",
    "TREND_RIBASSISTA",
    "DEBOLEZZA_RIBASSISTA",
}


# ============================================================
# HELPERS
# ============================================================

def safe_get(row, col, default=None):
    if col not in row.index:
        return default

    value = row[col]

    if pd.isna(value):
        return default

    return value


def classify_direction(phase):
    if phase in {
        "BUY",
        "TREND_RIALZISTA",
        "DEBOLEZZA_RIALZISTA",
    }:
        return "BULL"

    if phase in {
        "SELL",
        "TREND_RIBASSISTA",
        "DEBOLEZZA_RIBASSISTA",
    }:
        return "BEAR"

    if phase == INDECISION:
        return "INDECISION"

    return "OTHER"


# ============================================================
# ANALISI EPISODI
# ============================================================

def analyse_ticker(g: pd.DataFrame) -> list[dict]:

    ticker = str(g["Ticker"].iloc[0])

    x = (
        g.copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    out = v405.process_ticker(x)

    episodes = []

    i = 0
    n = len(out)

    while i < n:

        phase = str(out.loc[i, "PHASE"])

        if phase != INDECISION:
            i += 1
            continue

        # ----------------------------------------------------
        # Trova inizio/fine episodio INDECISIONE
        # ----------------------------------------------------

        start_pos = i

        while (
            i + 1 < n
            and
            str(out.loc[i + 1, "PHASE"]) == INDECISION
        ):
            i += 1

        end_pos = i

        prev_pos = start_pos - 1
        next_pos = end_pos + 1

        prev_phase = (
            str(out.loc[prev_pos, "PHASE"])
            if prev_pos >= 0
            else None
        )

        next_phase = (
            str(out.loc[next_pos, "PHASE"])
            if next_pos < n
            else None
        )

        last_row = out.loc[end_pos]

        first_row = out.loc[start_pos]

        next_row = (
            out.loc[next_pos]
            if next_pos < n
            else None
        )

        exit_phase = (
            next_phase
            if next_phase in VALID_EXIT_PHASES
            else next_phase
        )

        episodes.append(
            {
                "Ticker": ticker,

                "indecision_start":
                    first_row["Date"],

                "indecision_end":
                    last_row["Date"],

                "duration_weeks":
                    end_pos - start_pos + 1,

                "prev_phase":
                    prev_phase,

                "prev_direction":
                    classify_direction(prev_phase),

                "exit_phase":
                    exit_phase,

                "exit_direction":
                    classify_direction(exit_phase),

                # --------------------------------------------
                # Ultima settimana di INDECISIONE
                # --------------------------------------------

                "last_close":
                    safe_get(last_row, "Close"),

                "last_sar_side":
                    safe_get(last_row, "SAR_SIDE"),

                "last_sar_age":
                    safe_get(last_row, "SAR_AGE"),

                "last_weak_up":
                    safe_get(
                        last_row,
                        "WEAK_UP_TRIGGER",
                        0,
                    ),

                "last_weak_down":
                    safe_get(
                        last_row,
                        "WEAK_DOWN_TRIGGER",
                        0,
                    ),

                "last_lateral_signal":
                    safe_get(
                        last_row,
                        "LATERAL_SIGNAL",
                        0,
                    ),

                "last_bull_recovery_raw":
                    safe_get(
                        last_row,
                        "BULL_RECOVERY_RAW",
                        0,
                    ),

                "last_bull_recovery_confirmed":
                    safe_get(
                        last_row,
                        "BULL_RECOVERY_CONFIRMED",
                        0,
                    ),

                "last_bear_recovery_raw":
                    safe_get(
                        last_row,
                        "BEAR_RECOVERY_RAW",
                        0,
                    ),

                "last_bear_recovery_confirmed":
                    safe_get(
                        last_row,
                        "BEAR_RECOVERY_CONFIRMED",
                        0,
                    ),

                "last_ha_direction":
                    safe_get(
                        last_row,
                        "HA_DIRECTION",
                    ),

                "last_ha_indecision":
                    safe_get(
                        last_row,
                        "HA_INDECISION",
                    ),

                "last_bb_position":
                    safe_get(
                        last_row,
                        "BB_POSITION",
                    ),

                "last_rsi_d1":
                    safe_get(
                        last_row,
                        "RSI_D1",
                    ),

                "last_rsi_slope3":
                    safe_get(
                        last_row,
                        "RSI_SLOPE3",
                    ),

                "last_macd_hist_d1":
                    safe_get(
                        last_row,
                        "MACD_HIST_D1",
                    ),

                "last_volume_osc_slope3":
                    safe_get(
                        last_row,
                        "VOLUME_OSC_SLOPE3",
                    ),

                # --------------------------------------------
                # Prima settimana DOPO INDECISIONE
                # --------------------------------------------

                "exit_date":
                    (
                        next_row["Date"]
                        if next_row is not None
                        else None
                    ),

                "exit_sar_side":
                    (
                        safe_get(next_row, "SAR_SIDE")
                        if next_row is not None
                        else None
                    ),

                "exit_sar_age":
                    (
                        safe_get(next_row, "SAR_AGE")
                        if next_row is not None
                        else None
                    ),

                "exit_weak_up":
                    (
                        safe_get(
                            next_row,
                            "WEAK_UP_TRIGGER",
                            0,
                        )
                        if next_row is not None
                        else None
                    ),

                "exit_weak_down":
                    (
                        safe_get(
                            next_row,
                            "WEAK_DOWN_TRIGGER",
                            0,
                        )
                        if next_row is not None
                        else None
                    ),

                "exit_lateral_signal":
                    (
                        safe_get(
                            next_row,
                            "LATERAL_SIGNAL",
                            0,
                        )
                        if next_row is not None
                        else None
                    ),

                "exit_bull_recovery_raw":
                    (
                        safe_get(
                            next_row,
                            "BULL_RECOVERY_RAW",
                            0,
                        )
                        if next_row is not None
                        else None
                    ),

                "exit_bull_recovery_confirmed":
                    (
                        safe_get(
                            next_row,
                            "BULL_RECOVERY_CONFIRMED",
                            0,
                        )
                        if next_row is not None
                        else None
                    ),

                "exit_bear_recovery_raw":
                    (
                        safe_get(
                            next_row,
                            "BEAR_RECOVERY_RAW",
                            0,
                        )
                        if next_row is not None
                        else None
                    ),

                "exit_bear_recovery_confirmed":
                    (
                        safe_get(
                            next_row,
                            "BEAR_RECOVERY_CONFIRMED",
                            0,
                        )
                        if next_row is not None
                        else None
                    ),
            }
        )

        i += 1

    return episodes


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("MARKETSENTINEL - FULL200 INDECISION EXIT DIAGNOSTIC")
    print("=" * 100)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Non trovo {INPUT_FILE}"
        )

    raw = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    raw["Date"] = pd.to_datetime(
        raw["Date"],
        utc=True,
        errors="coerce",
    )

    tickers = (
        raw["Ticker"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"\nTicker trovati: {len(tickers)}"
    )

    all_episodes = []

    for n, ticker in enumerate(tickers, start=1):

        g = raw[
            raw["Ticker"] == ticker
        ].copy()

        eps = analyse_ticker(g)

        all_episodes.extend(eps)

        if (
            n == 1
            or
            n % 10 == 0
            or
            n == len(tickers)
        ):
            print(
                f"[{n:3d}/{len(tickers)}] "
                f"{ticker:<12} "
                f"episodi cumulati: {len(all_episodes):,}"
            )

    result = pd.DataFrame(
        all_episodes
    )

    OUT_DETAIL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT_DETAIL,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    if result.empty:

        print(
            "\nNessun episodio INDECISIONE trovato."
        )

        return

    summary = (
        result
        .groupby(
            [
                "prev_direction",
                "exit_phase",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="episodes"
        )
    )

    total_by_prev = (
        summary
        .groupby(
            "prev_direction"
        )["episodes"]
        .transform("sum")
    )

    summary["pct_within_prev_direction"] = (
        summary["episodes"]
        /
        total_by_prev
        *
        100.0
    )

    summary = summary.sort_values(
        [
            "prev_direction",
            "episodes",
        ],
        ascending=[
            True,
            False,
        ],
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # OUTPUT TERMINALE
    # ========================================================

    print("\n" + "=" * 100)
    print("RISULTATO GENERALE")
    print("=" * 100)

    print(
        f"\nEpisodi INDECISIONE: "
        f"{len(result):,}"
    )

    print(
        f"Ticker coinvolti: "
        f"{result['Ticker'].nunique():,}"
    )

    print(
        f"Durata mediana: "
        f"{result['duration_weeks'].median():.1f} settimane"
    )

    print(
        f"Durata media: "
        f"{result['duration_weeks'].mean():.2f} settimane"
    )

    print("\n" + "=" * 100)
    print("USCITA DA INDECISIONE - TUTTI GLI EPISODI")
    print("=" * 100)

    exit_counts = (
        result["exit_phase"]
        .fillna("NO_EXIT_END_DATA")
        .value_counts()
        .rename_axis("exit_phase")
        .reset_index(name="episodes")
    )

    exit_counts["pct"] = (
        exit_counts["episodes"]
        /
        len(result)
        *
        100.0
    )

    print(
        exit_counts.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}",
        )
    )

    print("\n" + "=" * 100)
    print("USCITA PER DIREZIONE PRECEDENTE")
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}",
        )
    )

    # ========================================================
    # CONTINUITA' vs CAMBIO DIREZIONE
    # ========================================================

    valid = result[
        result["exit_direction"].isin(
            ["BULL", "BEAR"]
        )
        &
        result["prev_direction"].isin(
            ["BULL", "BEAR"]
        )
    ].copy()

    if not valid.empty:

        valid["same_direction"] = (
            valid["prev_direction"]
            ==
            valid["exit_direction"]
        )

        same = int(
            valid["same_direction"].sum()
        )

        changed = int(
            (~valid["same_direction"]).sum()
        )

        print("\n" + "=" * 100)
        print("CONTINUITA' O CAMBIO DIREZIONE")
        print("=" * 100)

        print(
            f"\nStessa direzione: "
            f"{same:,} "
            f"({same / len(valid) * 100:.2f}%)"
        )

        print(
            f"Cambio direzione: "
            f"{changed:,} "
            f"({changed / len(valid) * 100:.2f}%)"
        )

    # ========================================================
    # SEGNALE SAR ALL'USCITA
    # ========================================================

    print("\n" + "=" * 100)
    print("EXIT PHASE vs SAR SIDE")
    print("=" * 100)

    sar_summary = (
        result
        .groupby(
            [
                "exit_phase",
                "exit_sar_side",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="episodes"
        )
        .sort_values(
            "episodes",
            ascending=False,
        )
    )

    print(
        sar_summary.head(30).to_string(
            index=False,
        )
    )

    print("\n" + "=" * 100)
    print("FILE SALVATI")
    print("=" * 100)

    print(
        f"\nDettaglio: {OUT_DETAIL}"
    )

    print(
        f"Summary : {OUT_SUMMARY}"
    )

    print(
        "\nNESSUN MOTORE DI PRODUZIONE È STATO MODIFICATO."
    )


if __name__ == "__main__":
    main()