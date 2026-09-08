import os
import numpy as np
import pandas as pd

from weekly_phase1_sequence_v1 import (
    TICKERS,
    get_weekly_data,
)


EVENTS_FILE = "data/weekly_phase1_sequence_v2_events.csv"

OUTPUT_FILE = "data/weekly_phase1_rsi_diagnostic_v1.csv"

RULE = "HA_BB_RETREAT_SAR_NEAR"

RSI_PERIOD = 14


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close, period=14):

    close = pd.to_numeric(
        close,
        errors="coerce"
    )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder RSI
    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


# ============================================================
# PREPARA RSI WEEKLY
# ============================================================

def prepare_rsi(ticker):

    weekly = get_weekly_data(
        ticker
    ).copy()

    weekly.index = pd.to_datetime(
        weekly.index,
        utc=True
    )

    weekly = (
        weekly
        .sort_index()
        .loc[
            ~weekly.index.duplicated(
                keep="last"
            )
        ]
    )

    weekly["RSI"] = calculate_rsi(
        weekly["Close"],
        RSI_PERIOD
    )

    # RSI precedenti
    weekly["RSI_1W_AGO"] = (
        weekly["RSI"].shift(1)
    )

    weekly["RSI_2W_AGO"] = (
        weekly["RSI"].shift(2)
    )

    weekly["RSI_3W_AGO"] = (
        weekly["RSI"].shift(3)
    )

    weekly["RSI_4W_AGO"] = (
        weekly["RSI"].shift(4)
    )

    # Movimento RSI
    weekly["RSI_CHANGE_1W"] = (
        weekly["RSI"]
        -
        weekly["RSI_1W_AGO"]
    )

    weekly["RSI_CHANGE_2W"] = (
        weekly["RSI"]
        -
        weekly["RSI_2W_AGO"]
    )

    # Massimo/minimo recente PRIMA
    # della settimana del segnale
    previous = weekly["RSI"].shift(1)

    weekly["RSI_PREV_MAX_4W"] = (
        previous
        .rolling(4)
        .max()
    )

    weekly["RSI_PREV_MIN_4W"] = (
        previous
        .rolling(4)
        .min()
    )

    weekly["RSI_PREV_MEAN_4W"] = (
        previous
        .rolling(4)
        .mean()
    )

    # Distanza dall'estremo precedente
    weekly["RSI_FROM_PREV_MAX"] = (
        weekly["RSI"]
        -
        weekly["RSI_PREV_MAX_4W"]
    )

    weekly["RSI_FROM_PREV_MIN"] = (
        weekly["RSI"]
        -
        weekly["RSI_PREV_MIN_4W"]
    )

    # Zone RSI
    weekly["RSI_ABOVE_70"] = (
        weekly["RSI"] >= 70
    )

    weekly["RSI_ABOVE_65"] = (
        weekly["RSI"] >= 65
    )

    weekly["RSI_ABOVE_60"] = (
        weekly["RSI"] >= 60
    )

    weekly["RSI_BELOW_30"] = (
        weekly["RSI"] <= 30
    )

    weekly["RSI_BELOW_35"] = (
        weekly["RSI"] <= 35
    )

    weekly["RSI_BELOW_40"] = (
        weekly["RSI"] <= 40
    )

    # Il contesto precedente ha raggiunto
    # una zona alta/bassa?
    weekly["RSI_PREV_MAX_ABOVE_70"] = (
        weekly["RSI_PREV_MAX_4W"] >= 70
    )

    weekly["RSI_PREV_MAX_ABOVE_65"] = (
        weekly["RSI_PREV_MAX_4W"] >= 65
    )

    weekly["RSI_PREV_MAX_ABOVE_60"] = (
        weekly["RSI_PREV_MAX_4W"] >= 60
    )

    weekly["RSI_PREV_MIN_BELOW_30"] = (
        weekly["RSI_PREV_MIN_4W"] <= 30
    )

    weekly["RSI_PREV_MIN_BELOW_35"] = (
        weekly["RSI_PREV_MIN_4W"] <= 35
    )

    weekly["RSI_PREV_MIN_BELOW_40"] = (
        weekly["RSI_PREV_MIN_4W"] <= 40
    )

    return weekly


# ============================================================
# CARICA I 33 SEGNALI V2
# ============================================================

def load_events():

    if not os.path.exists(
        EVENTS_FILE
    ):
        raise RuntimeError(
            f"File non trovato: {EVENTS_FILE}"
        )

    df = pd.read_csv(
        EVENTS_FILE
    )

    df = df[
        df["Rule"] == RULE
    ].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True
    )

    # Matched può essere bool oppure stringa
    if df["Matched"].dtype != bool:

        df["Matched"] = (
            df["Matched"]
            .astype(str)
            .str.lower()
            .isin(
                ["true", "1", "1.0"]
            )
        )

    return df


# ============================================================
# AGGIUNGE RSI AI SEGNALI
# ============================================================

def enrich_events(events):

    rows = []

    rsi_columns = [
        "RSI",
        "RSI_1W_AGO",
        "RSI_2W_AGO",
        "RSI_3W_AGO",
        "RSI_4W_AGO",
        "RSI_CHANGE_1W",
        "RSI_CHANGE_2W",
        "RSI_PREV_MAX_4W",
        "RSI_PREV_MIN_4W",
        "RSI_PREV_MEAN_4W",
        "RSI_FROM_PREV_MAX",
        "RSI_FROM_PREV_MIN",
        "RSI_ABOVE_70",
        "RSI_ABOVE_65",
        "RSI_ABOVE_60",
        "RSI_BELOW_30",
        "RSI_BELOW_35",
        "RSI_BELOW_40",
        "RSI_PREV_MAX_ABOVE_70",
        "RSI_PREV_MAX_ABOVE_65",
        "RSI_PREV_MAX_ABOVE_60",
        "RSI_PREV_MIN_BELOW_30",
        "RSI_PREV_MIN_BELOW_35",
        "RSI_PREV_MIN_BELOW_40",
    ]

    for ticker in TICKERS:

        ticker_events = events[
            events["Ticker"] == ticker
        ].copy()

        if ticker_events.empty:
            continue

        print(
            f"Calcolo RSI per {ticker}..."
        )

        weekly = prepare_rsi(
            ticker
        )

        for _, event in ticker_events.iterrows():

            date = event["Date"]

            if date not in weekly.index:

                # Cerca la settimana più vicina
                pos = weekly.index.get_indexer(
                    [date],
                    method="nearest"
                )[0]

                if pos < 0:
                    continue

                date_used = weekly.index[pos]

            else:
                date_used = date

            indicator_row = weekly.loc[
                date_used
            ]

            item = event.to_dict()

            item["RSIDate"] = date_used

            for column in rsi_columns:

                item[column] = (
                    indicator_row[column]
                )

            rows.append(
                item
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# UTILITY
# ============================================================

def mean_value(df, column):

    if df.empty:
        return np.nan

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).mean()


def percentage_true(df, column):

    if df.empty:
        return np.nan

    return (
        df[column]
        .fillna(False)
        .astype(bool)
        .mean()
        *
        100
    )


# ============================================================
# ANALISI NUMERICA
# ============================================================

def diagnostic(enriched):

    print()
    print("=" * 100)
    print("RSI DIAGNOSTIC - HA_BB_RETREAT_SAR_NEAR")
    print("=" * 100)

    for direction in [
        "ALL",
        "BUY",
        "SELL",
    ]:

        if direction == "ALL":

            x = enriched.copy()

        else:

            x = enriched[
                enriched["Direction"]
                ==
                direction
            ].copy()

        good = x[
            x["Matched"] == True
        ]

        bad = x[
            x["Matched"] == False
        ]

        print()
        print("=" * 80)
        print(direction)
        print("=" * 80)

        print(
            f"Segnali: {len(x)}"
        )

        print(
            f"Corretti: {len(good)}"
        )

        print(
            f"Falsi: {len(bad)}"
        )

        print()

        numeric_columns = [
            "RSI",
            "RSI_1W_AGO",
            "RSI_2W_AGO",
            "RSI_PREV_MAX_4W",
            "RSI_PREV_MIN_4W",
            "RSI_PREV_MEAN_4W",
            "RSI_CHANGE_1W",
            "RSI_CHANGE_2W",
            "RSI_FROM_PREV_MAX",
            "RSI_FROM_PREV_MIN",
        ]

        print(
            "MEDIE RSI:"
        )

        for column in numeric_columns:

            good_mean = mean_value(
                good,
                column
            )

            bad_mean = mean_value(
                bad,
                column
            )

            print(
                f"{column:22s} "
                f"CORRETTI "
                f"{good_mean:7.2f} | "
                f"FALSI "
                f"{bad_mean:7.2f}"
            )

        print()
        print(
            "ZONE RSI:"
        )

        boolean_columns = [
            "RSI_ABOVE_70",
            "RSI_ABOVE_65",
            "RSI_ABOVE_60",
            "RSI_BELOW_30",
            "RSI_BELOW_35",
            "RSI_BELOW_40",
            "RSI_PREV_MAX_ABOVE_70",
            "RSI_PREV_MAX_ABOVE_65",
            "RSI_PREV_MAX_ABOVE_60",
            "RSI_PREV_MIN_BELOW_30",
            "RSI_PREV_MIN_BELOW_35",
            "RSI_PREV_MIN_BELOW_40",
        ]

        for column in boolean_columns:

            good_pct = percentage_true(
                good,
                column
            )

            bad_pct = percentage_true(
                bad,
                column
            )

            print(
                f"{column:27s} "
                f"CORRETTI "
                f"{good_pct:6.1f}% | "
                f"FALSI "
                f"{bad_pct:6.1f}%"
            )


# ============================================================
# TEST CONCETTUALE BUY / SELL
# ============================================================

def contextual_tests(enriched):

    print()
    print("=" * 100)
    print("TEST FILTRI RSI - SOLO DIAGNOSTICA")
    print("=" * 100)

    tests = []

    # ========================================================
    # BUY
    # ========================================================

    buy = enriched[
        enriched["Direction"] == "BUY"
    ].copy()

    buy_tests = {

        "BUY RSI <= 40":
            buy["RSI"] <= 40,

        "BUY RSI <= 45":
            buy["RSI"] <= 45,

        "BUY RSI <= 50":
            buy["RSI"] <= 50,

        "BUY prev min <= 30":
            buy["RSI_PREV_MIN_4W"] <= 30,

        "BUY prev min <= 35":
            buy["RSI_PREV_MIN_4W"] <= 35,

        "BUY prev min <= 40":
            buy["RSI_PREV_MIN_4W"] <= 40,

        "BUY RSI rising":
            buy["RSI_CHANGE_1W"] > 0,

        "BUY prev min <=40 + rising":
            (
                (buy["RSI_PREV_MIN_4W"] <= 40)
                &
                (buy["RSI_CHANGE_1W"] > 0)
            ),
    }

    for name, mask in buy_tests.items():

        z = buy[
            mask.fillna(False)
        ]

        if z.empty:
            continue

        correct = int(
            z["Matched"].sum()
        )

        tests.append(
            {
                "Test": name,
                "Signals": len(z),
                "Correct": correct,
                "False": len(z) - correct,
                "PrecisionPct":
                    100 * correct / len(z),
            }
        )

    # ========================================================
    # SELL
    # ========================================================

    sell = enriched[
        enriched["Direction"] == "SELL"
    ].copy()

    sell_tests = {

        "SELL RSI >= 60":
            sell["RSI"] >= 60,

        "SELL RSI >= 55":
            sell["RSI"] >= 55,

        "SELL RSI >= 50":
            sell["RSI"] >= 50,

        "SELL prev max >= 70":
            sell["RSI_PREV_MAX_4W"] >= 70,

        "SELL prev max >= 65":
            sell["RSI_PREV_MAX_4W"] >= 65,

        "SELL prev max >= 60":
            sell["RSI_PREV_MAX_4W"] >= 60,

        "SELL RSI falling":
            sell["RSI_CHANGE_1W"] < 0,

        "SELL prev max >=60 + falling":
            (
                (sell["RSI_PREV_MAX_4W"] >= 60)
                &
                (sell["RSI_CHANGE_1W"] < 0)
            ),
    }

    for name, mask in sell_tests.items():

        z = sell[
            mask.fillna(False)
        ]

        if z.empty:
            continue

        correct = int(
            z["Matched"].sum()
        )

        tests.append(
            {
                "Test": name,
                "Signals": len(z),
                "Correct": correct,
                "False": len(z) - correct,
                "PrecisionPct":
                    100 * correct / len(z),
            }
        )

    result = pd.DataFrame(
        tests
    )

    if result.empty:

        print(
            "Nessun test disponibile."
        )

        return

    result = result.sort_values(
        [
            "PrecisionPct",
            "Signals",
        ],
        ascending=[
            False,
            False,
        ]
    )

    print()
    print(
        result.to_string(
            index=False
        )
    )


# ============================================================
# MOSTRA I 33 CASI
# ============================================================

def print_cases(enriched):

    print()
    print("=" * 100)
    print("TUTTI I SEGNALI CON RSI")
    print("=" * 100)

    columns = [
        "Ticker",
        "Date",
        "Direction",
        "Matched",
        "Score",
        "RSI",
        "RSI_1W_AGO",
        "RSI_PREV_MAX_4W",
        "RSI_PREV_MIN_4W",
        "RSI_CHANGE_1W",
        "RSI_CHANGE_2W",
    ]

    print(
        enriched[
            columns
        ]
        .sort_values(
            [
                "Ticker",
                "Date",
            ]
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("MARKET SENTINEL")
    print("WEEKLY PHASE 1 - RSI DIAGNOSTIC V1")
    print("=" * 100)

    events = load_events()

    print()
    print(
        f"Segnali caricati: {len(events)}"
    )

    print(
        f"Corretti: "
        f"{int(events['Matched'].sum())}"
    )

    print(
        f"Falsi: "
        f"{len(events) - int(events['Matched'].sum())}"
    )

    enriched = enrich_events(
        events
    )

    enriched.to_csv(
        OUTPUT_FILE,
        index=False
    )

    diagnostic(
        enriched
    )

    contextual_tests(
        enriched
    )

    print_cases(
        enriched
    )

    print()
    print("=" * 100)
    print("FILE SALVATO")
    print("=" * 100)
    print(
        OUTPUT_FILE
    )

    print()
    print(
        "TEST RSI COMPLETATO."
    )


if __name__ == "__main__":