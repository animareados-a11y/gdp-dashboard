"""Controllo incrementale WEEKLY basato sugli OHLCV DAILY."""
from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAPSHOT = DATA / "marketsentinel_intraday" / "weekly_ohlcv_snapshot"
CACHE = DATA / "marketsentinel_intraday" / "scientific_cache"
DEFINITIVE = DATA / "operational_market"
MANIFEST = DATA / "marketsentinel_weekly_active.json"

EXPECTED = {"italy40": 40, "europe40": 40, "usa80": 80, "usa40": 39}
OHLCV = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]


def weekly_input_fingerprint():
    """Restituisce data massima e impronta delle sole righe post-definitiva."""
    signatures = []
    latest_dates = []
    definitive_dates = []

    for market, expected in EXPECTED.items():
        snapshot = SNAPSHOT / f"{market}_daily_master.csv"
        fallback = CACHE / f"{market}_daily_master.csv"
        source = snapshot if snapshot.is_file() else fallback

        daily = pd.read_csv(source, usecols=OHLCV)
        definitive = pd.read_csv(
            DEFINITIVE / f"{market}_weekly_master.csv",
            usecols=["Ticker", "Date"],
        )

        if daily["Ticker"].nunique() != expected:
            raise ValueError(f"{market}: universo DAILY incompleto")
        if definitive["Ticker"].nunique() != expected:
            raise ValueError(f"{market}: universo WEEKLY incompleto")

        if daily.duplicated(["Ticker", "Date"]).any():
            raise ValueError(f"{market}: duplicati DAILY")
        if definitive.duplicated(["Ticker", "Date"]).any():
            raise ValueError(f"{market}: duplicati WEEKLY")

        daily["Date"] = pd.to_datetime(
            daily["Date"], utc=True, errors="raise"
        )
        definitive["Date"] = pd.to_datetime(
            definitive["Date"], utc=True, errors="raise"
        )

        if daily["Date"].isna().any() or definitive["Date"].isna().any():
            raise ValueError(f"{market}: date non valide")

        last_per_ticker = daily.groupby("Ticker")["Date"].max()
        final_per_ticker = definitive.groupby("Ticker")["Date"].max()

        if final_per_ticker.nunique() != 1:
            raise ValueError(f"{market}: WEEKLY non allineata")

        if set(last_per_ticker.index) != set(final_per_ticker.index):
            raise ValueError(f"{market}: universo DAILY/WEEKLY differente")

        last_daily = last_per_ticker.max()
        last_final = final_per_ticker.iloc[0]

        latest_dates.append(last_daily)
        definitive_dates.append(last_final)

        incremental = daily.loc[daily["Date"] > last_final, OHLCV].copy()
        incremental = incremental.sort_values(
            ["Ticker", "Date"]
        ).reset_index(drop=True)

        payload = incremental.to_csv(
            index=False, columns=OHLCV, float_format="%.12g"
        ).encode("utf-8")

        signatures.append(
            market + ":" + hashlib.sha256(payload).hexdigest()
        )

    # Mercati con orari di chiusura differenti:
    # utilizza la data più recente realmente disponibile.
    if len(set(definitive_dates)) != 1:
        raise ValueError("Date WEEKLY definitive non coerenti")

    combined = hashlib.sha256(
        "\\n".join(signatures).encode("utf-8")
    ).hexdigest()

    return max(latest_dates), definitive_dates[0], combined


def check_weekly_refresh():
    """Ritorna (necessario, motivo, ultima_data_daily)."""
    last_daily, last_final, fingerprint = weekly_input_fingerprint()

    if last_daily <= last_final:
        return False, "Nessuna nuova quotazione dopo la definitiva", last_daily

    from ms_weekly_publication import weekly_science_fingerprint

    current_science = weekly_science_fingerprint()

    if MANIFEST.is_file():
        info = json.loads(MANIFEST.read_text(encoding="utf-8"))

        if info.get("weekly_science_sha256") != current_science:
            return (
                True,
                "Versione scientifica WEEKLY modificata: ricalcolo necessario",
                last_daily,
            )

    if MANIFEST.is_file():
        info = json.loads(MANIFEST.read_text(encoding="utf-8"))
        recorded = info.get("daily_ohlcv_fingerprint")
        if recorded == fingerprint:
            from ms_weekly_publication import (
                active_version,
                validate_build,
            )

            try:
                published = active_version()

                if published is None:
                    return (
                        True,
                        "Versione WEEKLY pubblicata assente",
                        last_daily,
                    )

                published_date = validate_build(published)

                if (
                    info.get("reference_date")
                    != published_date.date().isoformat()
                ):
                    return (
                        True,
                        "Data manifest WEEKLY non coerente",
                        last_daily,
                    )

                if published_date <= last_final:
                    return (
                        True,
                        "WEEKLY pubblicata non successiva alla definitiva",
                        last_daily,
                    )

            except (OSError, ValueError, KeyError, TypeError) as exc:
                return (
                    True,
                    "Versione WEEKLY non valida: " + str(exc),
                    last_daily,
                )

            return (
                False,
                "OHLCV DAILY già elaborati e WEEKLY valida",
                last_daily,
            )

    return True, "Nuovi OHLCV DAILY da elaborare", last_daily


def completed_friday_week():
    """Restituisce il venerdì consolidabile solo con 199 prezzi completi."""
    now = pd.Timestamp.now(tz="UTC")
    today = now.normalize()
    days_since_friday = (today.weekday() - 4) % 7
    friday = today - pd.Timedelta(days=days_since_friday)

    # Venerdì: attendere la chiusura di tutti i mercati.
    if days_since_friday == 0:
        return None

    for market, expected in EXPECTED.items():
        snapshot = SNAPSHOT / f"{market}_daily_master.csv"
        fallback = CACHE / f"{market}_daily_master.csv"
        source = snapshot if snapshot.is_file() else fallback

        if not source.is_file():
            return None

        daily = pd.read_csv(source, usecols=["Ticker", "Date"])
        daily["Date"] = pd.to_datetime(daily["Date"], utc=True)

        latest = daily.groupby("Ticker")["Date"].max()

        if len(latest) != expected:
            return None

        if not (latest.dt.normalize() == friday).all():
            return None

    return friday
