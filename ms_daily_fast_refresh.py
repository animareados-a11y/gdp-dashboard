"""Aggiornamento rapido OHLCV senza calcolo scientifico."""
from pathlib import Path
import os
import uuid
import pandas as pd

def publish(sandbox, root, markets):
    base = Path(sandbox) / "data" / "marketsentinel_daily"
    destination = Path(root) / "data" / "marketsentinel_intraday"
    destination.mkdir(parents=True, exist_ok=True)

    frames = []
    for prefix in markets.values():
        path = base / f"{prefix}_daily_master.csv"
        df = pd.read_csv(path)
        latest = (
            df.sort_values("Date")
              .groupby("Ticker", as_index=False)
              .tail(1)
              .copy()
        )
        latest["Market"] = prefix
        frames.append(latest)

    result = pd.concat(frames, ignore_index=True)
    if len(result) != 199 or result["Ticker"].nunique() != 199:
        raise RuntimeError("FAST REFRESH: universo incompleto")

    target = destination / "marketsentinel_latest_quotes.csv"
    temporary = destination / (target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        result.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        "[FAST REFRESH] Pubblicate",
        len(result),
        "quotazioni in",
        target,
        flush=True
    )
