import time
import schedule

from config import MARKETS
from data import get_timeframe
from engine import analyze, breadth_score
from notifier import send_telegram


TIMEFRAMES = [
    "Weekly",
    "Daily",
    "4H",
    "2H",
    "1H"
]


def create_report(report_type):

    lines = [
        f"📈 MARKET SENTINEL — {report_type}"
    ]

    for market, config in MARKETS.items():

        frames_by_ticker = {}

        for ticker in config["tickers"]:

            frames_by_ticker[ticker] = {
                timeframe:
                    get_timeframe(
                        ticker,
                        timeframe
                    )

                for timeframe in TIMEFRAMES
            }

        breadth = {
            timeframe: 5.0
            for timeframe in TIMEFRAMES
        }

        # ----------------------------------------------------
        # MARKET BREADTH
        # ----------------------------------------------------

        for timeframe in TIMEFRAMES:

            advancing = 0
            declining = 0

            for frames in (
                frames_by_ticker.values()
            ):

                df = frames[
                    timeframe
                ]

                if len(df) < 3:
                    continue

                if (
                    df.Close.iloc[-2] >
                    df.Close.iloc[-3]
                ):

                    advancing += 1

                elif (
                    df.Close.iloc[-2] <
                    df.Close.iloc[-3]
                ):

                    declining += 1

            breadth[timeframe] = (
                breadth_score(
                    advancing,
                    declining
                )
            )

        # ----------------------------------------------------
        # ANALISI
        # ----------------------------------------------------

        analysed = []

        for ticker, frames in (
            frames_by_ticker.items()
        ):

            result = analyze(
                ticker,
                frames,
                breadth
            )

            if result:
                analysed.append(
                    result
                )

        analysed.sort(
            key=lambda x:
                x["global_score"],
            reverse=True
        )

        lines.append(
            f"\n{market}"
        )

        for result in analysed[:5]:

            lines.append(
                f"{result['ticker']}: "
                f"Trend {result['global_score']:.1f} | "
                f"Reversal↓ "
                f"{result['bearish_risk']:.1f} | "
                f"Reversal↑ "
                f"{result['bullish_reversal']:.1f}"
            )

            if (
                result["bearish_risk"]
                >= 6
            ):

                lines.append(
                    "🔴 "
                    +
                    " · ".join(
                        result[
                            "bearish_reasons"
                        ][:3]
                    )
                )

            if (
                result["bullish_reversal"]
                >= 6
            ):

                lines.append(
                    "🟢 "
                    +
                    " · ".join(
                        result[
                            "bullish_reasons"
                        ][:3]
                    )
                )

    return "\n".join(
        lines
    )


def send_report(report_type):

    text = create_report(
        report_type
    )

    send_telegram(
        text
    )


# ============================================================
# TRE REPORT GIORNALIERI
# ============================================================

schedule.every().day.at(
    "08:00"
).do(
    lambda:
        send_report(
            "PRE-MARKET"
        )
)

schedule.every().day.at(
    "12:00"
).do(
    lambda:
        send_report(
            "MIDDAY"
        )
)

schedule.every().day.at(
    "18:00"
).do(
    lambda:
        send_report(
            "CLOSURE"
        )
)


while True:

    schedule.run_pending()

    time.sleep(20)
    