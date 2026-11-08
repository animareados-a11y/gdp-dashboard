"""Selezione sicura sorgente WEEKLY per MarketSentinel UNIFIED."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

DEFINITIVE = ROOT / "data/operational_market"
PROVISIONAL = ROOT / "data/marketsentinel_weekly_provisional"

EXPECTED = {
    "italy40": 40,
    "europe40": 40,
    "usa80": 80,
    "usa40": 39,
}


_VALIDATION_CACHE = {}


def _source_signature(base):
    """Firma dei file scientifici controllati dal selettore."""
    paths = [base / "marketsentinel_200_phase_nuova.csv"]

    for market in EXPECTED:
        for suffix in (
            "_weekly_master.csv",
            "_phase_history.csv",
            "_current_phase.csv",
            "_v160_features.csv",
        ):
            paths.append(base / f"{market}{suffix}")

    signature = []

    for path in paths:
        try:
            stat = path.stat()
            signature.append(
                (path.name, stat.st_size, stat.st_mtime_ns)
            )
        except FileNotFoundError:
            signature.append((path.name, None, None))

    return tuple(signature)


def validate_source(base):
    """Verifica completezza e coerenza temporale dei 199 titoli."""

    base = Path(base)
    signature = _source_signature(base)
    cached = _VALIDATION_CACHE.get(base)

    if cached is not None and cached[0] == signature:
        return cached[1]

    reference = _validate_source_full(base)

    # Non memorizzare una validazione se i file sono cambiati
    # mentre venivano letti.
    if _source_signature(base) == signature:
        _VALIDATION_CACHE[base] = (signature, reference)
    else:
        _VALIDATION_CACHE.pop(base, None)

    return reference


def _validate_source_full(base):
    """Valida 199 titoli, consentendo date diverse per ticker."""

    base = Path(base)
    ranking_path = base / "marketsentinel_200_phase_nuova.csv"

    if not ranking_path.is_file():
        raise ValueError("Classifica assente")

    ranking = pd.read_csv(ranking_path)

    required = {
        "Ticker", "Date", "PHASE_NUOVA", "RATING_V160"
    }

    if not required.issubset(ranking.columns):
        raise ValueError("Colonne classifica mancanti")

    if len(ranking) != 199 or not ranking["Ticker"].is_unique:
        raise ValueError("Universo classifica non valido")

    ranking["Date"] = pd.to_datetime(
        ranking["Date"], utc=True, errors="coerce"
    )

    if ranking["Date"].isna().any():
        raise ValueError("Date classifica non valide")

    reference = ranking["Date"].max()
    all_tickers = set()

    for market, expected in EXPECTED.items():

        latest_by_archive = {}

        for suffix in (
            "_weekly_master.csv",
            "_phase_history.csv",
            "_current_phase.csv",
            "_v160_features.csv",
        ):
            path = base / f"{market}{suffix}"

            if not path.is_file():
                raise ValueError(f"Archivio assente: {path.name}")

            frame = pd.read_csv(
                path,
                usecols=["Ticker", "Date"],
                low_memory=False,
            )

            frame["Date"] = pd.to_datetime(
                frame["Date"], utc=True, errors="coerce"
            )

            if frame["Date"].isna().any():
                raise ValueError(f"Date non valide: {path.name}")

            if frame.duplicated(["Ticker", "Date"]).any():
                raise ValueError(f"Duplicati: {path.name}")

            latest = frame.groupby("Ticker")["Date"].max()

            if len(latest) != expected:
                raise ValueError(
                    f"Universo incompleto: {path.name}"
                )

            latest_by_archive[suffix] = latest

        master = latest_by_archive["_weekly_master.csv"]

        for suffix, latest in latest_by_archive.items():
            if not latest.index.equals(master.index):
                raise ValueError(
                    f"Ticker incoerenti: {market}{suffix}"
                )

            if not latest.equals(master):
                raise ValueError(
                    f"Date incoerenti: {market}{suffix}"
                )

        if all_tickers.intersection(master.index):
            raise ValueError("Ticker presenti in più mercati")

        all_tickers.update(master.index)

        selected = ranking.set_index("Ticker").reindex(master.index)

        if selected["Date"].isna().any():
            raise ValueError(f"Classifica incompleta: {market}")

        if not selected["Date"].equals(master.rename("Date")):
            raise ValueError(
                f"Date classifica non coerenti: {market}"
            )

    if all_tickers != set(ranking["Ticker"]):
        raise ValueError("Classifica e archivi non corrispondenti")

    return reference


def select_weekly_source():
    """Restituisce percorso, stato e data della sorgente valida."""

    definitive_date = validate_source(DEFINITIVE)

    try:
        from ms_weekly_publication import (
            MANIFEST,
            active_version,
        )

        if MANIFEST.exists():
            provisional_base = active_version()
        else:
            provisional_base = PROVISIONAL

        # Controllo preliminare della data: non sostituisce
        # la validazione scientifica integrale.
        ranking_path = (
            provisional_base / "marketsentinel_200_phase_nuova.csv"
        )

        if ranking_path.is_file():
            preview = pd.read_csv(
                ranking_path,
                usecols=["Ticker", "Date"],
            )

            preview_dates = pd.to_datetime(
                preview["Date"],
                utc=True,
                errors="coerce",
            )

            # Se la data non è determinabile in modo sicuro,
            # eseguiamo comunque la validazione integrale.
            if (
                len(preview) != 199
                or preview["Ticker"].nunique() != 199
                or preview_dates.isna().any()
                or preview_dates.nunique() != 1
                or preview_dates.max() > definitive_date
            ):
                provisional_date = validate_source(provisional_base)

                if provisional_date > definitive_date:
                    return provisional_base, "PROVVISORIO", provisional_date

    except (ValueError, OSError, KeyError) as exc:
        print(
            "[WEEKLY] Provvisorio non utilizzabile:",
            exc,
            flush=True,
        )

    return DEFINITIVE, "DEFINITIVO", definitive_date


if __name__ == "__main__":
    base, status, date = select_weekly_source()

    print("SORGENTE:", base)
    print("STATO:", status)
    print("DATA:", date.date())
