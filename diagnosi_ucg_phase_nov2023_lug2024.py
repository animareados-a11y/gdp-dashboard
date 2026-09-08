"""
MarketSentinel
DIAGNOSI UNICREDIT PHASE - NOVEMBRE 2023 / LUGLIO 2024

Solo diagnostica.
NON modifica alcun file di produzione.

Obiettivo:
capire settimana per settimana perché UniCredit viene classificata
TREND_RIALZISTA / DEBOLEZZA_RIALZISTA / INDECISIONE /
TREND_RIBASSISTA / DEBOLEZZA_RIBASSISTA.

Confrontiamo:
- V40.5 originale
- Hierarchy Shadow
- V40.9
- indicatori/segnali tecnici già esistenti
"""

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_hierarchy_shadow as hierarchy
import weekly_v40_6_transition_engine as v406
import weekly_v40_7_weakness_severity_engine as v407
import weekly_v40_9_phase_confirmation_fix as v409


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")
TICKER = "UCG.MI"


def main():

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    g = (
        df[df["Ticker"] == TICKER]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if g.empty:
        raise RuntimeError(
            f"Ticker {TICKER} non trovato nel file."
        )

    # =========================================================
    # V40.5 + HIERARCHY
    # =========================================================

    g = hierarchy.process_ticker(g)

    g["PHASE_HIERARCHY"] = (
        g["PHASE_HIERARCHY_SHADOW"]
        .copy()
    )

    g["PHASE"] = (
        g["PHASE_HIERARCHY"]
        .copy()
    )

    # =========================================================
    # V40.6
    # =========================================================

    g = v406.add_middle_band_structure(g)
    g = v406.add_ha_transition(g)
    g = v406.add_rsi_transition(g)
    g = v406.add_macd_transition(g)
    g = v406.add_volume_transition(g)
    g = v406.add_transition_pressure(g)
    g = v406.add_transition_confirmation(g)

    # =========================================================
    # V40.7
    # =========================================================

    g = v407.add_weakness_components(g)
    g = v407.add_weakness_scores(g)
    g = v407.add_weakness_level(g)
    g = v407.add_weakness_labels(g)
    g = v407.add_pre_signal_history(g)

    # =========================================================
    # V40.9
    # =========================================================

    g = v409.add_phase_confirmation_features(g)

    g = v409.apply_v409_phase_fix(g)

    g["PHASE_V409_FINAL"] = (
        g["PHASE"]
        .copy()
    )

    # =========================================================
    # SOLO NOVEMBRE 2023 - LUGLIO 2024
    # =========================================================

    z = g[
        (g["Date"] >= pd.Timestamp("2023-11-01", tz="UTC"))
        &
        (g["Date"] <= pd.Timestamp("2024-07-31", tz="UTC"))
    ].copy()

    # =========================================================
    # COLONNE UTILI
    # =========================================================

    wanted = [
        "Date",
        "Close",

        # SAR
        "SAR_SIDE",
        "SAR_AGE",

        # PHASE
        "PHASE_V405_ORIGINAL",
        "PHASE_HIERARCHY",
        "PHASE_V409_FINAL",

        # V40.5
        "LATERAL_SIGNAL",
        "WEAK_UP_TRIGGER",
        "WEAK_DOWN_TRIGGER",
        "BULL_RECOVERY_RAW",
        "BULL_RECOVERY_CONFIRMED",
        "BEAR_RECOVERY_RAW",
        "BEAR_RECOVERY_CONFIRMED",

        # V40.9
        "V409_BULL_CONFIRMATIONS",
        "V409_BULL_DAMAGE_COUNT",
        "V409_BEAR_CONFIRMATIONS",
        "V409_BEAR_DAMAGE_COUNT",
        "V409_PHASE_CORRECTED",

        # Indicatori
        "RSI_14",
        "MACD",
        "MACD_SIGNAL",
        "MACD_HIST",
        "BB_MIDDLE",
        "BB_UPPER",
        "BB_LOWER",
        "HA_Open",
        "HA_Close",
        "HA_High",
        "HA_Low",
        "VOLUME_OSC",
    ]

    existing = [
        c for c in wanted
        if c in z.columns
    ]

    print("\n" + "=" * 180)
    print(
        "UNICREDIT - DIAGNOSI PHASE "
        "NOVEMBRE 2023 / LUGLIO 2024"
    )
    print("=" * 180)

    print(
        z[existing]
        .to_string(index=False)
    )

    print("\n" + "=" * 180)
    print("COLONNE V40.9 DISPONIBILI")
    print("=" * 180)

    print(
        [
            c for c in g.columns
            if c.startswith("V409_")
        ]
    )

    print("\n" + "=" * 180)
    print(
        "COLONNE HA / BB / RSI / MACD / "
        "VOLUME DISPONIBILI"
    )
    print("=" * 180)

    keys = [
        "HA",
        "BB",
        "RSI",
        "MACD",
        "VOL",
    ]

    technical_cols = [
        c for c in g.columns
        if any(
            key in c.upper()
            for key in keys
        )
    ]

    print(technical_cols)


if __name__ == "__main__":
    main()