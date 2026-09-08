from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_hierarchy_shadow as hierarchy
import weekly_v40_6_transition_engine as v406
import weekly_v40_7_weakness_severity_engine as v407
import weekly_v40_9_phase_confirmation_fix as v409


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")
TICKER = "UCG.MI"


def print_period(df, title, start, end, columns):
    z = df[
        (df["Date"] >= pd.Timestamp(start, tz="UTC"))
        & (df["Date"] <= pd.Timestamp(end, tz="UTC"))
    ].copy()

    existing = [c for c in columns if c in z.columns]

    print("\n")
    print("=" * 220)
    print(title)
    print("=" * 220)
    print(z[existing].to_string(index=False))


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
        raise RuntimeError(f"Ticker {TICKER} non trovato.")

    # V40.5 + HIERARCHY
    g = hierarchy.process_ticker(g)

    g["PHASE_HIERARCHY"] = (
        g["PHASE_HIERARCHY_SHADOW"].copy()
    )

    g["PHASE"] = (
        g["PHASE_HIERARCHY"].copy()
    )

    # V40.6
    g = v406.add_middle_band_structure(g)
    g = v406.add_ha_transition(g)
    g = v406.add_rsi_transition(g)
    g = v406.add_macd_transition(g)
    g = v406.add_volume_transition(g)
    g = v406.add_transition_pressure(g)
    g = v406.add_transition_confirmation(g)

    # V40.7
    g = v407.add_weakness_components(g)
    g = v407.add_weakness_scores(g)
    g = v407.add_weakness_level(g)
    g = v407.add_weakness_labels(g)
    g = v407.add_pre_signal_history(g)

    # V40.9
    g = v409.add_phase_confirmation_features(g)
    g = v409.apply_v409_phase_fix(g)

    g["PHASE_V409_FINAL"] = g["PHASE"].copy()

    columns = [
        "Date",
        "Close",
        "SAR_SIDE",
        "SAR_AGE",
        "PHASE_V405_ORIGINAL",
        "PHASE_HIERARCHY",
        "PHASE_V409_FINAL",

        "HA_Open",
        "HA_Close",
        "HA_DIRECTION",
        "HA_SMALL_BODY",
        "HA_BOTH_WICKS",
        "HA_INDECISION",
        "HA_BODY_SHRINK",
        "HA_BODY_EXPANDING",
        "HA_GREEN_COUNT3",
        "HA_RED_COUNT3",
        "HA_BULL_TRANSITION",
        "HA_BEAR_TRANSITION",

        "BB_MIDDLE",
        "BB_UPPER",
        "BB_LOWER",
        "BB_POSITION",
        "BB_ABOVE_MIDDLE",
        "BB_BELOW_MIDDLE",
        "BB_MIDDLE_DISTANCE_PCT",
        "BB_MIDDLE_BREAK_UP",
        "BB_MIDDLE_BREAK_DOWN",
        "BB_PROGRESS_UP",
        "BB_PROGRESS_DOWN",
        "BB_DETACH_UPPER",
        "BB_DETACH_LOWER",

        "RSI_14",
        "RSI_D1",
        "RSI_SLOPE3",
        "RSI_BULL_PRESSURE",
        "RSI_BEAR_PRESSURE",
        "RSI_UP_COUNT3",
        "RSI_DOWN_COUNT3",
        "RSI_BULL_FILM",
        "RSI_BEAR_FILM",

        "MACD",
        "MACD_SIGNAL",
        "MACD_HIST",
        "MACD_D1",
        "MACD_HIST_D1",
        "MACD_HIST_SLOPE3",
        "MACD_ABOVE_SIGNAL",
        "MACD_BELOW_SIGNAL",
        "MACD_BULL_PRESSURE",
        "MACD_BEAR_PRESSURE",

        "VOLUME_OSC",
        "VOLUME_OSC_D1",
        "VOLUME_OSC_SLOPE3",
        "VOL_BULL_PRESSURE",
        "VOL_BEAR_PRESSURE",

        "V405_HA_WEAK_UP_STRUCTURE",
        "V405_RSI_HARD_WEAK_UP",
        "V405_MACD_HARD_WEAK_UP",
        "V405_VOL_PARTICIPATION_FALLING",
        "V405_WEAK_UP_BB_PATH",
        "V405_WEAK_UP_HA_PATH",
        "WEAK_UP_TRIGGER",

        "V405_BULL_REC_HA",
        "V405_BULL_REC_BB",
        "V405_BULL_REC_RSI",
        "V405_BULL_REC_MACD",
        "V405_BULL_REC_VOLUME",
        "BULL_RECOVERY_RAW",
        "BULL_RECOVERY_CONFIRMED",

        "WU_HA_COMPONENT",
        "WU_BB_OUTER_COMPONENT",
        "WU_BB_MIDDLE_COMPONENT",
        "WU_RSI_COMPONENT",
        "WU_MACD_COMPONENT",
        "WU_VOLUME_COMPONENT",

        "V409_BULL_HA",
        "V409_BULL_BB",
        "V409_BULL_RSI",
        "V409_BULL_MACD",
        "V409_BULL_CONFIRMATIONS",

        "V409_BULL_DAMAGE_HA",
        "V409_BULL_DAMAGE_RSI",
        "V409_BULL_DAMAGE_MACD",
        "V409_BULL_DAMAGE_BB",
        "V409_BULL_DAMAGE_COUNT",

        "V409_PHASE_CORRECTED",
        "V409_CORRECTION_REASON",

        "LATERAL_SIGNAL",
        "HA_OVERLAP",
        "LAT_COLOR_CHANGES",
    ]

    print_period(
        g,
        "CASO A - UCG: DICEMBRE 2023 / GENNAIO 2024",
        "2023-12-01",
        "2024-01-31",
        columns,
    )

    print_period(
        g,
        "CASO B - UCG: APRILE / MAGGIO 2024",
        "2024-04-01",
        "2024-05-31",
        columns,
    )

    print("\n")
    print("=" * 220)
    print("FINE DIAGNOSI")
    print("=" * 220)
    print("Nessun file di produzione è stato modificato.")


if __name__ == "__main__":
    main()
