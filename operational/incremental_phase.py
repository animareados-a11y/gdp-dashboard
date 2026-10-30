from pathlib import Path
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weekly_v40_39_phase_canonical import process_ticker

import os

if os.environ.get("MS_PHASE_V4_FAST", "0") == "1":
    import ms_daily_engine
    import weekly_v40_5_phase_dynamics_shadow_v4 as v4
    from operational.phase_v4_fast import optimized
    v4._surgical_reclassification = optimized
    print("[PHASE] V4 FAST ATTIVO", flush=True)


BASE = Path("data/operational_market")


def calculate_one_ticker(g):
    """Calcolo PHASE canonico originale, senza modifiche scientifiche."""
    g = g.sort_values("Date").reset_index(drop=True)
    result = process_ticker(g)
    result["Ticker"] = g["Ticker"].iloc[0]

    h = result[
        ["Ticker", "Date", "PHASE_DYNAMICS_V5"]
    ].copy()

    h = h.rename(columns={"PHASE_DYNAMICS_V5": "PHASE"})

    run_id = h["PHASE"].ne(h["PHASE"].shift()).cumsum()
    run_age = h.groupby(run_id).cumcount() + 1
    signal = h["PHASE"].isin(["BUY", "SELL"])

    if (signal & (run_age > 3)).any():
        raise ValueError("BUY/SELL oltre tre settimane")

    h.loc[signal, "PHASE"] = (
        h.loc[signal, "PHASE"]
        + run_age.loc[signal].astype(str)
    )

    return h


def identify_changed_tickers(market):
    """Individua titoli con prezzi non presenti nello storico PHASE."""
    prices = pd.read_csv(
        BASE / f"{market}_weekly_master.csv",
        usecols=["Ticker", "Date"]
    )
    history_path = BASE / f"{market}_phase_history.csv"

    if not history_path.exists():
        return set(prices["Ticker"].unique())

    history = pd.read_csv(
        history_path,
        usecols=["Ticker", "Date"]
    )

    for df in (prices, history):
        df["Date"] = pd.to_datetime(df["Date"], utc=True)

    expected = prices.groupby("Ticker")["Date"].max()
    actual = history.groupby("Ticker")["Date"].max()

    return set(
        expected.index[
            expected.ne(actual.reindex(expected.index))
        ]
    )


if __name__ == "__main__":
    for market in ["italy40", "europe40", "usa80", "usa40"]:
        changed = identify_changed_tickers(market)
        print(market, "TITOLI DA RICALCOLARE:", len(changed))
    print("NESSUN ARCHIVIO SCIENTIFICO MODIFICATO")


def prepare_incremental_phase(market, dry_run=True):
    """Aggiorna PHASE per titolo preservando lo storico."""
    prices = pd.read_csv(BASE / f"{market}_weekly_master.csv")
    prices["Date"] = pd.to_datetime(prices["Date"])

    history_path = BASE / f"{market}_phase_history.csv"
    current_path = BASE / f"{market}_current_phase.csv"

    if history_path.exists():
        old = pd.read_csv(history_path)
        old["Date"] = pd.to_datetime(old["Date"])
    else:
        old = pd.DataFrame(columns=["Ticker", "Date", "PHASE"])

    if old.duplicated(["Ticker", "Date"]).any():
        raise ValueError(f"{market}: PHASE duplicate")

    changed = identify_changed_tickers(market)

    parts = []
    current = []

    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.sort_values("Date").copy()
        previous = old.loc[old["Ticker"] == ticker].copy()

        if set(g["Date"]) != set(previous["Date"]):
            changed.add(ticker)

        if ticker in changed:
            h = calculate_one_ticker(g)
        else:
            h = previous

        parts.append(h)

        last = h.sort_values("Date").iloc[-1]
        current.append({
            "Ticker": ticker,
            "Date": last["Date"],
            "PHASE": last["PHASE"],
            "Close": float(g.iloc[-1]["Close"])
        })

    combined = pd.concat(parts, ignore_index=True)
    current_df = pd.DataFrame(current)

    if len(combined) != len(prices):
        raise ValueError(f"{market}: numero righe non coerente")

    if combined.duplicated(["Ticker", "Date"]).any():
        raise ValueError(f"{market}: chiavi duplicate")

    price_keys = set(zip(prices["Ticker"], prices["Date"]))
    phase_keys = set(zip(combined["Ticker"], combined["Date"]))

    if price_keys != phase_keys:
        raise ValueError(f"{market}: chiavi non allineate")

    print(
        f"{market}: {len(current_df)} titoli, "
        f"{len(changed)} ricalcolati, "
        f"{len(combined)} righe storiche"
    )

    if dry_run:
        return combined, current_df, changed

    combined["Date"] = combined["Date"].dt.strftime("%Y-%m-%d")
    current_df["Date"] = current_df["Date"].dt.strftime("%Y-%m-%d")

    for destination, frame in [
        (history_path, combined),
        (current_path, current_df)
    ]:
        temporary = destination.with_suffix(".tmp.csv")
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)

    return combined, current_df, changed
