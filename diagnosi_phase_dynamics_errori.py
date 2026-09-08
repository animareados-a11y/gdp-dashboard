from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow as dynamics


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")


ERRORS = {
    "FBK.MI": {
        "2026-05-01": "TREND_RIALZISTA",
        "2026-05-08": "TREND_RIALZISTA",
        "2026-05-29": "DEBOLEZZA_RIALZISTA",
        "2026-06-05": "DEBOLEZZA_RIALZISTA",
        "2026-06-12": "TREND_RIALZISTA",
        "2026-08-21": "DEBOLEZZA_RIALZISTA",
    },

    "UCG.MI": {
        "2023-12-15": "DEBOLEZZA_RIALZISTA",
        "2024-03-08": "TREND_RIALZISTA",
        "2024-04-12": "DEBOLEZZA_RIALZISTA",
        "2024-04-19": "DEBOLEZZA_RIALZISTA",
        "2024-04-26": "DEBOLEZZA_RIALZISTA",
        "2024-05-03": "DEBOLEZZA_RIALZISTA",
        "2024-05-10": "DEBOLEZZA_RIALZISTA",
        "2024-05-17": "DEBOLEZZA_RIALZISTA",
        "2024-05-24": "DEBOLEZZA_RIALZISTA",
        "2024-05-31": "DEBOLEZZA_RIALZISTA",
        "2024-06-28": "TREND_RIBASSISTA",
    },
}


def value(row, col):
    if col not in row.index:
        return "-"
    x = row[col]
    if pd.isna(x):
        return "-"
    return x


def main():

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce",
    )

    for ticker, dates in ERRORS.items():

        print("\n")
        print("=" * 100)
        print(ticker)
        print("=" * 100)

        g = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        g = dynamics.process_ticker(g)

        for date_text, expected in dates.items():

            dt = pd.Timestamp(date_text, tz="UTC")
            z = g[g["Date"] == dt]

            if z.empty:
                continue

            r = z.iloc[0]

            print("\n" + "-" * 100)
            print(
                f"{date_text} | EXPECTED={expected} | "
                f"SHADOW={value(r, 'PHASE_DYNAMICS_SHADOW')} | "
                f"REASON={value(r, 'PHASE_DYNAMICS_REASON')}"
            )

            print(
                "PHASE | "
                f"V405={value(r,'PHASE_V405_ORIGINAL')} | "
                f"HIER={value(r,'PHASE_HIERARCHY_SHADOW')} | "
                f"V409_BASE={value(r,'PHASE_DYNAMICS_BASE')}"
            )

            print(
                "SAR   | "
                f"SIDE={value(r,'SAR_SIDE')} | "
                f"AGE={value(r,'SAR_AGE')}"
            )

            print(
                "HA    | "
                f"DIR={value(r,'HA_DIRECTION')} | "
                f"INDEC={value(r,'HA_INDECISION')} | "
                f"SMALL={value(r,'HA_SMALL_BODY')} | "
                f"BOTH_WICKS={value(r,'HA_BOTH_WICKS')} | "
                f"GREEN3={value(r,'HA_GREEN_COUNT3')} | "
                f"RED3={value(r,'HA_RED_COUNT3')}"
            )

            print(
                "LAT   | "
                f"LATERAL={value(r,'LATERAL_SIGNAL')} | "
                f"OVERLAP={value(r,'HA_OVERLAP')} | "
                f"COLOR_CHANGES={value(r,'LAT_COLOR_CHANGES')}"
            )

            print(
                "WEAK  | "
                f"TRIGGER_UP={value(r,'WEAK_UP_TRIGGER')} | "
                f"TRIGGER_DOWN={value(r,'WEAK_DOWN_TRIGGER')} | "
                f"HA_WEAK={value(r,'V405_HA_WEAK_UP_STRUCTURE')} | "
                f"RSI_WEAK={value(r,'V405_RSI_HARD_WEAK_UP')} | "
                f"MACD_WEAK={value(r,'V405_MACD_HARD_WEAK_UP')} | "
                f"VOL_FALL={value(r,'V405_VOL_PARTICIPATION_FALLING')}"
            )

            print(
                "RSI   | "
                f"RSI={value(r,'RSI_14')} | "
                f"D1={value(r,'RSI_D1')} | "
                f"SLOPE3={value(r,'RSI_SLOPE3')}"
            )

            print(
                "MACD  | "
                f"HIST={value(r,'MACD_HIST')} | "
                f"HIST_D1={value(r,'MACD_HIST_D1')} | "
                f"HIST_SLOPE3={value(r,'MACD_HIST_SLOPE3')}"
            )

            print(
                "BB    | "
                f"POSITION={value(r,'BB_POSITION')} | "
                f"ABOVE_MID={value(r,'BB_ABOVE_MIDDLE')} | "
                f"PROGRESS_UP={value(r,'BB_PROGRESS_UP')} | "
                f"PROGRESS_DOWN={value(r,'BB_PROGRESS_DOWN')} | "
                f"DETACH_UPPER={value(r,'BB_DETACH_UPPER')}"
            )

            print(
                "V409 BULL | "
                f"CONF={value(r,'V409_BULL_CONFIRMATIONS')} | "
                f"DAMAGE={value(r,'V409_BULL_DAMAGE_COUNT')} | "
                f"HA={value(r,'V409_BULL_HA')} | "
                f"BB={value(r,'V409_BULL_BB')} | "
                f"RSI={value(r,'V409_BULL_RSI')} | "
                f"MACD={value(r,'V409_BULL_MACD')}"
            )

            print(
                "V409 BEAR | "
                f"CONF={value(r,'V409_BEAR_CONFIRMATIONS')} | "
                f"DAMAGE={value(r,'V409_BEAR_DAMAGE_COUNT')}"
            )

            print(
                "RECOVERY | "
                f"BULL_RAW={value(r,'BULL_RECOVERY_RAW')} | "
                f"BULL_CONF={value(r,'BULL_RECOVERY_CONFIRMED')} | "
                f"BEAR_RAW={value(r,'BEAR_RECOVERY_RAW')} | "
                f"BEAR_CONF={value(r,'BEAR_RECOVERY_CONFIRMED')}"
            )


if __name__ == "__main__":
    main()