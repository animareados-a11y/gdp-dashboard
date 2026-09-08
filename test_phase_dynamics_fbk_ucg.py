from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow as dynamics


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")


BENCHMARK = {
    "FBK.MI": {
        "2026-05-01": "TREND_RIALZISTA",
        "2026-05-08": "TREND_RIALZISTA",
        "2026-05-15": "TREND_RIALZISTA",
        "2026-05-22": "DEBOLEZZA_RIALZISTA",
        "2026-05-29": "DEBOLEZZA_RIALZISTA",
        "2026-06-05": "DEBOLEZZA_RIALZISTA",
        "2026-06-12": "TREND_RIALZISTA",
        "2026-06-19": "TREND_RIALZISTA",
        "2026-06-26": "TREND_RIALZISTA",
        "2026-07-03": "INDECISIONE",
        "2026-07-10": "TREND_RIALZISTA",
        "2026-07-17": "TREND_RIALZISTA",
        "2026-07-24": "TREND_RIALZISTA",
        "2026-07-31": "TREND_RIALZISTA",
        "2026-08-07": "TREND_RIALZISTA",
        "2026-08-14": "TREND_RIALZISTA",
        "2026-08-21": "DEBOLEZZA_RIALZISTA",
        "2026-08-28": "DEBOLEZZA_RIALZISTA",
    },

    "UCG.MI": {
        "2023-11-03": "TREND_RIALZISTA",
        "2023-11-10": "TREND_RIALZISTA",
        "2023-11-17": "TREND_RIALZISTA",

        "2023-12-15": "DEBOLEZZA_RIALZISTA",
        "2023-12-22": "DEBOLEZZA_RIALZISTA",
        "2023-12-29": "INDECISIONE",

        "2024-01-05": "TREND_RIALZISTA",
        "2024-01-12": "TREND_RIALZISTA",
        "2024-01-19": "TREND_RIALZISTA",
        "2024-01-26": "TREND_RIALZISTA",

        "2024-02-02": "TREND_RIALZISTA",
        "2024-02-09": "TREND_RIALZISTA",
        "2024-02-16": "TREND_RIALZISTA",
        "2024-02-23": "TREND_RIALZISTA",

        "2024-03-01": "TREND_RIALZISTA",
        "2024-03-08": "TREND_RIALZISTA",
        "2024-03-15": "TREND_RIALZISTA",
        "2024-03-22": "TREND_RIALZISTA",
        "2024-03-29": "TREND_RIALZISTA",
        "2024-04-05": "TREND_RIALZISTA",

        "2024-04-12": "DEBOLEZZA_RIALZISTA",
        "2024-04-19": "DEBOLEZZA_RIALZISTA",
        "2024-04-26": "DEBOLEZZA_RIALZISTA",

        "2024-05-03": "DEBOLEZZA_RIALZISTA",
        "2024-05-10": "DEBOLEZZA_RIALZISTA",
        "2024-05-17": "DEBOLEZZA_RIALZISTA",
        "2024-05-24": "DEBOLEZZA_RIALZISTA",
        "2024-05-31": "DEBOLEZZA_RIALZISTA",

        "2024-06-07": "SELL",
        "2024-06-14": "SELL",
        "2024-06-21": "SELL",
        "2024-06-28": "TREND_RIBASSISTA",

        "2024-07-05": "BUY",
        "2024-07-12": "BUY",
        "2024-07-19": "BUY",
        "2024-07-26": "TREND_RIALZISTA",
    },
}


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

    total_ok = 0
    total_test = 0

    for ticker, expected_map in BENCHMARK.items():

        print("\n")
        print("=" * 120)
        print(ticker)
        print("=" * 120)

        g = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if g.empty:
            print("TICKER NON TROVATO")
            continue

        g = dynamics.process_ticker(g)

        rows = []

        for date_text, expected in expected_map.items():

            target_date = pd.Timestamp(
                date_text,
                tz="UTC",
            )

            z = g[g["Date"] == target_date]

            if z.empty:
                rows.append({
                    "Date": date_text,
                    "EXPECTED": expected,
                    "SHADOW": "DATA_NON_TROVATA",
                    "OK": "NO",
                    "REASON": "",
                })
                continue

            row = z.iloc[0]

            shadow = row["PHASE_DYNAMICS_SHADOW"]
            reason = row["PHASE_DYNAMICS_REASON"]

            ok = shadow == expected

            rows.append({
                "Date": date_text,
                "EXPECTED": expected,
                "SHADOW": shadow,
                "OK": "OK" if ok else "NO",
                "REASON": reason,
            })

            total_test += 1

            if ok:
                total_ok += 1

        result = pd.DataFrame(rows)

        print(
            result.to_string(
                index=False
            )
        )

        tested = result[
            result["SHADOW"] != "DATA_NON_TROVATA"
        ]

        matches = (
            tested["EXPECTED"]
            == tested["SHADOW"]
        ).sum()

        print()
        print(
            f"{ticker}: "
            f"{matches}/{len(tested)} corrette "
            f"({matches / len(tested) * 100:.1f}%)"
        )

    print("\n")
    print("=" * 120)

    if total_test > 0:
        print(
            f"TOTALE: {total_ok}/{total_test} "
            f"({total_ok / total_test * 100:.1f}%)"
        )

    print("=" * 120)


if __name__ == "__main__":
    main()