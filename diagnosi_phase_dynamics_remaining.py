from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow as dynamics


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")


FOCUS = {
    "FBK.MI": [
        "2026-07-03",
        "2026-07-10",
        "2026-07-17",
        "2026-07-24",
        "2026-07-31",
    ],
    "UCG.MI": [
        "2023-12-29",
        "2024-01-05",
        "2024-01-12",
        "2024-01-19",
        "2024-01-26",
        "2024-02-02",
    ],
}


COLS = [
    "Date",
    "Close",
    "SAR_SIDE",
    "SAR_AGE",

    "PHASE_V405_ORIGINAL",
    "PHASE_HIERARCHY_SHADOW",
    "PHASE_DYNAMICS_BASE",
    "PHASE_DYNAMICS_SHADOW",
    "PHASE_DYNAMICS_REASON",

    "HA_DIRECTION",
    "HA_INDECISION",
    "HA_SMALL_BODY",
    "HA_BOTH_WICKS",
    "HA_GREEN_COUNT3",
    "HA_RED_COUNT3",
    "HA_OVERLAP",
    "LAT_COLOR_CHANGES",
    "LATERAL_SIGNAL",

    "WEAK_UP_TRIGGER",
    "V405_HA_WEAK_UP_STRUCTURE",
    "V405_RSI_HARD_WEAK_UP",
    "V405_MACD_HARD_WEAK_UP",
    "V405_VOL_PARTICIPATION_FALLING",

    "RSI_14",
    "RSI_D1",
    "RSI_SLOPE3",

    "MACD_HIST",
    "MACD_HIST_D1",
    "MACD_HIST_SLOPE3",

    "BB_POSITION",
    "BB_ABOVE_MIDDLE",
    "BB_PROGRESS_UP",
    "BB_PROGRESS_DOWN",
    "BB_DETACH_UPPER",

    "V409_BULL_CONFIRMATIONS",
    "V409_BULL_DAMAGE_COUNT",

    "BULL_RECOVERY_RAW",
    "BULL_RECOVERY_CONFIRMED",
]


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

    for ticker, dates in FOCUS.items():

        print("\n")
        print("=" * 150)
        print(ticker)
        print("=" * 150)

        g = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        g = dynamics.process_ticker(g)

        wanted_indexes = set()

        for date_text in dates:

            dt = pd.Timestamp(
                date_text,
                tz="UTC",
            )

            matches = g.index[
                g["Date"] == dt
            ].tolist()

            for idx in matches:

                for j in range(
                    max(0, idx - 2),
                    min(len(g), idx + 3),
                ):
                    wanted_indexes.add(j)

        z = (
            g.loc[sorted(wanted_indexes)]
            .copy()
        )

        existing = [
            c for c in COLS
            if c in z.columns
        ]

        print(
            z[existing].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()