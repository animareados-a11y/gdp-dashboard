"""
MarketSentinel
DIAGNOSI V2 - PERSISTENZA DEBOLEZZA SENZA TRIGGER

Ticker:
FBK.MI
UCG.MI
BBVA.MC
TIT.MI

Misura quante settimane consecutive la shadow rimane in
DEBOLEZZA quando il trigger specifico è già tornato a zero.

Nessun file di produzione modificato.
"""

from pathlib import Path
import pandas as pd

import weekly_v40_5_indecision_exit_causal_shadow_v2 as shadow


INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

TICKERS = [
    "FBK.MI",
    "UCG.MI",
    "BBVA.MC",
    "TIT.MI",
]


def banner(text):
    print("\n" + "=" * 130)
    print(text)
    print("=" * 130)


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

    all_rows = []

    for ticker in TICKERS:

        g = (
            df[df["Ticker"] == ticker]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if g.empty:
            continue

        out = shadow.process_ticker(g)
        all_rows.append(out)

    full = pd.concat(
        all_rows,
        ignore_index=True,
    )

    records = []

    for ticker, g in full.groupby("Ticker"):

        g = (
            g.sort_values("Date")
            .reset_index(drop=True)
        )

        run_start = None
        run_direction = None
        no_trigger_weeks = 0
        max_no_trigger_weeks = 0

        for i, row in g.iterrows():

            phase = str(
                row["PHASE_CAUSAL_SHADOW"]
            )

            if phase == "DEBOLEZZA_RIALZISTA":
                direction = "BULL"
                trigger = int(
                    row.get(
                        "WEAK_UP_TRIGGER",
                        0,
                    )
                )

            elif phase == "DEBOLEZZA_RIBASSISTA":
                direction = "BEAR"
                trigger = int(
                    row.get(
                        "WEAK_DOWN_TRIGGER",
                        0,
                    )
                )

            else:
                direction = None
                trigger = None

            if direction is None:

                if run_start is not None:

                    prev = g.iloc[i - 1]

                    records.append(
                        {
                            "Ticker": ticker,
                            "Direction": run_direction,
                            "Start": run_start,
                            "End": prev["Date"],
                            "TotalWeakWeeks": (
                                i - run_start_pos
                            ),
                            "MaxConsecutiveWeeksTriggerZero":
                                max_no_trigger_weeks,
                            "ExitPhase":
                                phase,
                            "ExitDate":
                                row["Date"],
                        }
                    )

                run_start = None
                run_direction = None
                no_trigger_weeks = 0
                max_no_trigger_weeks = 0

                continue

            if (
                run_start is None
                or
                direction != run_direction
            ):

                run_start = row["Date"]
                run_start_pos = i
                run_direction = direction
                no_trigger_weeks = 0
                max_no_trigger_weeks = 0

            if trigger == 0:

                no_trigger_weeks += 1

                max_no_trigger_weeks = max(
                    max_no_trigger_weeks,
                    no_trigger_weeks,
                )

            else:

                no_trigger_weeks = 0

        if run_start is not None:

            prev = g.iloc[-1]

            records.append(
                {
                    "Ticker": ticker,
                    "Direction": run_direction,
                    "Start": run_start,
                    "End": prev["Date"],
                    "TotalWeakWeeks": (
                        len(g) - run_start_pos
                    ),
                    "MaxConsecutiveWeeksTriggerZero":
                        max_no_trigger_weeks,
                    "ExitPhase":
                        "END_OF_DATA",
                    "ExitDate":
                        pd.NaT,
                }
            )

    result = pd.DataFrame(records)

    banner(
        "PERSISTENZA DEBOLEZZA - TUTTI GLI EPISODI"
    )

    print(
        result
        .sort_values(
            [
                "MaxConsecutiveWeeksTriggerZero",
                "TotalWeakWeeks",
            ],
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    banner(
        "EPISODI CON >= 3 SETTIMANE SENZA TRIGGER"
    )

    suspicious = result[
        result[
            "MaxConsecutiveWeeksTriggerZero"
        ] >= 3
    ].copy()

    print(
        suspicious
        .sort_values(
            "MaxConsecutiveWeeksTriggerZero",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    banner(
        "SUMMARY"
    )

    print(
        f"Episodi totali: {len(result):,}"
    )

    print(
        f"Episodi con >=3 settimane senza trigger: "
        f"{len(suspicious):,}"
    )

    if not result.empty:

        print(
            "Massimo numero consecutivo di settimane "
            "senza trigger:",
            int(
                result[
                    "MaxConsecutiveWeeksTriggerZero"
                ].max()
            ),
        )

    print(
        "\nNESSUN FILE DI PRODUZIONE MODIFICATO."
    )


if __name__ == "__main__":
    main()