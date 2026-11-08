"""Preparazione ambiente WEEKLY provvisorio isolato."""

from pathlib import Path
import shutil
import tempfile
import pandas as pd

from ms_weekly_from_daily import (
    build_weekly_from_daily,
    WEEKLY,
    EXPECTED,
)

ROOT = Path(__file__).resolve().parent

PROVISIONAL = (
    ROOT / "data" / "marketsentinel_weekly_provisional"
)

SCIENCE_SUFFIXES = (
    "_phase_history.csv",
    "_current_phase.csv",
    "_v160_features.csv",
)


def prepare_provisional(as_of=None, destination=None):
    destination = PROVISIONAL if destination is None else Path(destination)
    generated = build_weekly_from_daily(as_of=as_of)

    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(
        prefix=".weekly_provisional_",
        dir=destination.parent,
    ))

    try:
        for market, expected in EXPECTED.items():
            frame = generated[market].copy()

            if frame["Ticker"].nunique() != expected:
                raise RuntimeError(
                    f"{market}: universo incompleto"
                )

            if frame.duplicated(["Ticker", "Date"]).any():
                raise RuntimeError(
                    f"{market}: duplicati Ticker/Date"
                )

            frame = frame.drop(
                columns=["WEEKLY_STATUS"],
                errors="ignore",
            )

            frame["Date"] = pd.to_datetime(
                frame["Date"], utc=True
            ).dt.strftime("%Y-%m-%d")

            frame.to_csv(
                staging / f"{market}_weekly_master.csv",
                index=False,
            )

            for suffix in SCIENCE_SUFFIXES:
                source = WEEKLY / f"{market}{suffix}"

                if not source.exists():
                    raise FileNotFoundError(source)

                # Hard link iniziale: nessuna duplicazione dei dati.
                # I motori scientifici scrivono su file temporanei
                # e sostituiscono atomicamente il file destinazione.
                # L'archivio definitivo resta quindi separato.
                target = staging / source.name
                target.hardlink_to(source)

            print(
                market,
                "preparato:",
                expected,
                "titoli",
            )

        if destination.exists():
            backup = destination.with_name(
                destination.name + "_previous"
            )

            if backup.exists():
                shutil.rmtree(backup)

            destination.rename(backup)

            try:
                staging.rename(destination)
            except Exception:
                backup.rename(destination)
                raise

            shutil.rmtree(backup)
        else:
            staging.rename(destination)

    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print("\nAMBIENTE PROVVISORIO PRONTO")
    print("PERCORSO:", destination)
    print("ARCHIVI DEFINITIVI INVARIATI")


if __name__ == "__main__":
    prepare_provisional()
