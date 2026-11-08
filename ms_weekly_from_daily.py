
"""WEEKLY provvisorio da DAILY, senza alterare lo storico definitivo."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "data/marketsentinel_intraday/scientific_cache"
WEEKLY = ROOT / "data/operational_market"
EXPECTED = {"italy40":40, "europe40":40, "usa80":80, "usa40":39}
OHLCV = ["Open","High","Low","Close","Volume"]

def build_weekly_from_daily(as_of=None):
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    cutoff = cutoff.normalize()
    result = {}

    for market, expected in EXPECTED.items():
        snapshot = (
            ROOT
            / "data"
            / "marketsentinel_intraday"
            / "weekly_ohlcv_snapshot"
            / f"{market}_daily_master.csv"
        )

        daily_path = snapshot if snapshot.is_file() else (
            DAILY / f"{market}_daily_master.csv"
        )

        daily = pd.read_csv(daily_path)

        print(
            f"{market}: sorgente OHLCV = "
            f"{'SNAPSHOT AGGIORNATO' if snapshot.is_file() else 'CACHE SCIENTIFICA'}",
            flush=True,
        )
        weekly = pd.read_csv(WEEKLY / f"{market}_weekly_master.csv")

        if daily.Ticker.nunique() != expected or weekly.Ticker.nunique() != expected:
            raise RuntimeError(f"{market}: universo incompleto")
        if weekly.duplicated(["Ticker","Date"]).any():
            raise RuntimeError(f"{market}: duplicati definitivi")

        daily["Date"] = pd.to_datetime(daily["Date"], utc=True)
        weekly["Date"] = pd.to_datetime(weekly["Date"], utc=True)

        final_dates = weekly.groupby("Ticker")["Date"].max()
        if final_dates.nunique() != 1:
            raise RuntimeError(f"{market}: date definitive non allineate")

        last_final = final_dates.iloc[0]
        daily = daily[daily["Date"] <= cutoff].copy()
        daily = daily[daily["Date"] > last_final].copy()

        weekly["WEEKLY_STATUS"] = "DEFINITIVO"

        if daily.empty:
            result[market] = weekly
            print(f"{market}: storico definitivo conservato")
            continue

        last_daily = daily["Date"].max()
        friday = (last_daily + pd.Timedelta(days=4-last_daily.weekday())).normalize()
        monday = friday - pd.Timedelta(days=4)

        if friday <= last_final:
            raise RuntimeError(f"{market}: settimana non successiva")

        current = daily[
            (daily["Date"] >= monday) & (daily["Date"] <= friday)
        ].copy()

        # Le quotazioni possono arrivare in momenti differenti.
        # I titoli senza nuovi dati mantengono lo storico definitivo,
        # senza generare barre settimanali artificiali.
        if current.empty:
            result[market] = weekly
            print(f"{market}: nessuna nuova quotazione")
            continue

        if current.duplicated(["Ticker","Date"]).any():
            raise RuntimeError(f"{market}: duplicati DAILY")

        bars = (
            current.sort_values(["Ticker","Date"])
            .groupby("Ticker", as_index=False)
            .agg(
                Open=("Open","first"),
                High=("High","max"),
                Low=("Low","min"),
                Close=("Close","last"),
                Volume=("Volume","sum")
            )
        )
        bars["Date"] = friday
        bars["WEEKLY_STATUS"] = "PROVVISORIO"

        result[market] = (
            pd.concat([weekly,bars],ignore_index=True)
            .sort_values(["Ticker","Date"])
            .reset_index(drop=True)
        )
        if result[market].duplicated(["Ticker","Date"]).any():
            raise RuntimeError(f"{market}: duplicati risultanti")

        print(f"{market}: nuova settimana provvisoria {friday.date()}")

    return result

if __name__ == "__main__":
    build_weekly_from_daily()
