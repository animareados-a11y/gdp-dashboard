import pandas as pd

import weekly_v40_7_weakness_severity_engine as v407


INPUT_FILE = "data/v40_35_full200_weekly.csv"


def main():

    print("Caricamento FULL200...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    df["Date"] = pd.to_datetime(df["Date"])

    tickers = df["Ticker"].dropna().unique()
    total = len(tickers)

    print(f"Ticker da analizzare: {total}")

    bull_rows = []
    bear_rows = []

    for n, ticker in enumerate(tickers, start=1):

        if n == 1 or n % 10 == 0 or n == total:
            print(f"Analisi {n}/{total}: {ticker}")

        g = (
            df.loc[df["Ticker"] == ticker]
            .sort_values("Date")
            .reset_index(drop=True)
        )

        a = v407.process_ticker(g.copy())

        bull_mask = (
            (a["PHASE"] == "INDECISIONE")
            &
            (a["SAR_SIDE"] == 1)
            &
            (a["WEAK_UP_TRIGGER"] == 1)
            &
            (a["LATERAL_SIGNAL"] == 0)
            &
            (a["BULL_RECOVERY_CONFIRMED"] == 0)
        )

        if bull_mask.any():

            z = a.loc[
                bull_mask,
                [
                    "Ticker",
                    "Date",
                    "Close",
                    "PHASE",
                    "SAR_SIDE",
                    "SAR_AGE",
                    "WEAK_UP_TRIGGER",
                    "LATERAL_SIGNAL",
                    "BULL_RECOVERY_RAW",
                    "BULL_RECOVERY_CONFIRMED",
                ],
            ].copy()

            bull_rows.append(z)

        bear_mask = (
            (a["PHASE"] == "INDECISIONE")
            &
            (a["SAR_SIDE"] == -1)
            &
            (a["WEAK_DOWN_TRIGGER"] == 1)
            &
            (a["LATERAL_SIGNAL"] == 0)
            &
            (a["BEAR_RECOVERY_CONFIRMED"] == 0)
        )

        if bear_mask.any():

            z = a.loc[
                bear_mask,
                [
                    "Ticker",
                    "Date",
                    "Close",
                    "PHASE",
                    "SAR_SIDE",
                    "SAR_AGE",
                    "WEAK_DOWN_TRIGGER",
                    "LATERAL_SIGNAL",
                    "BEAR_RECOVERY_RAW",
                    "BEAR_RECOVERY_CONFIRMED",
                ],
            ].copy()

            bear_rows.append(z)

    print()
    print("=" * 100)
    print("RISULTATO TEST INDECISIONE -> DEBOLEZZA")
    print("=" * 100)

    if bull_rows:

        bull = pd.concat(
            bull_rows,
            ignore_index=True
        )

        bull = (
            bull
            .sort_values(["Date", "Ticker"])
            .reset_index(drop=True)
        )

        print("\nLATO RIALZISTA")
        print(f"CASI TOTALI: {len(bull)}")
        print(f"TICKER COINVOLTI: {bull['Ticker'].nunique()}")

        print("\nTOP 20 TICKER PER NUMERO DI CASI:")
        print(
            bull.groupby("Ticker")
            .size()
            .sort_values(ascending=False)
            .head(20)
            .to_string()
        )

        latest_date = df["Date"].max()

        latest_bull = bull.loc[
            bull["Date"] == latest_date
        ]

        print(
            f"\nCASI ALL'ULTIMA DATA "
            f"{latest_date.date()}: {len(latest_bull)}"
        )

        if not latest_bull.empty:
            print(
                latest_bull
                .to_string(index=False)
            )

        fineco = bull.loc[
            bull["Ticker"] == "FBK.MI"
        ]

        print("\nFINECO:")

        if fineco.empty:
            print("Nessun caso Fineco.")
        else:
            print(
                fineco.tail(20)
                .to_string(index=False)
            )

    else:

        print("\nLATO RIALZISTA: NESSUN CASO")

    print()
    print("=" * 100)

    if bear_rows:

        bear = pd.concat(
            bear_rows,
            ignore_index=True
        )

        bear = (
            bear
            .sort_values(["Date", "Ticker"])
            .reset_index(drop=True)
        )

        print("\nLATO RIBASSISTA")
        print(f"CASI TOTALI: {len(bear)}")
        print(f"TICKER COINVOLTI: {bear['Ticker'].nunique()}")

        print("\nTOP 20 TICKER PER NUMERO DI CASI:")
        print(
            bear.groupby("Ticker")
            .size()
            .sort_values(ascending=False)
            .head(20)
            .to_string()
        )

        latest_date = df["Date"].max()

        latest_bear = bear.loc[
            bear["Date"] == latest_date
        ]

        print(
            f"\nCASI ALL'ULTIMA DATA "
            f"{latest_date.date()}: {len(latest_bear)}"
        )

        if not latest_bear.empty:
            print(
                latest_bear
                .to_string(index=False)
            )

    else:

        print("\nLATO RIBASSISTA: NESSUN CASO")


if __name__ == "__main__":
    main()