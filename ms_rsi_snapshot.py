"""Riepiloghi RSI derivati dagli archivi scientifici, senza modificarli."""
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

MARKETS = ("italy40", "europe40", "usa80", "usa40")
COLUMNS = ["Ticker", "Date", "V4015_RSI"]
SNAPSHOT = "ms_rsi_snapshot.json"


def signature(base):
    base = Path(base)
    result = []
    for market in MARKETS:
        path = base / f"{market}_v160_features.csv"
        stat = path.stat()
        result.append([path.name, stat.st_size, stat.st_mtime_ns])
    return result


def build(base):
    base = Path(base)
    before = signature(base)
    values = {}

    for market in MARKETS:
        path = base / f"{market}_v160_features.csv"
        df = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        latest = df.sort_values("Date").drop_duplicates("Ticker", keep="last")

        for _, row in latest.iterrows():
            ticker = str(row["Ticker"]).strip().upper()
            value = pd.to_numeric(row["V4015_RSI"], errors="coerce")
            date = row["Date"]

            if pd.notna(value) and 0 <= value <= 100 and pd.notna(date):
                values[ticker] = {
                    "date": date.isoformat(),
                    "rsi": float(value),
                }

    if signature(base) != before:
        raise RuntimeError("Archivi RSI modificati durante la lettura")

    if len(values) != 199:
        raise ValueError(f"RSI incompleti: {len(values)}/199")

    payload = {"signature": before, "values": values}
    target = base / SNAPSHOT

    fd, temporary = tempfile.mkstemp(
        prefix=".ms_rsi_", suffix=".tmp", dir=base
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    return target


def load(base):
    base = Path(base)
    path = base / SNAPSHOT
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["signature"] != signature(base):
            return None
        if len(payload["values"]) != 199:
            return None
        return payload["values"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
