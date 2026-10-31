"""Elaborazione e pubblicazione isolata WEEKLY MarketSentinel."""

import os
import subprocess
import sys
import uuid
from pathlib import Path

from ms_weekly_publication import (
    VERSIONS,
    publish_version,
)

from ms_weekly_refresh_guard import weekly_input_fingerprint

ROOT = Path(__file__).resolve().parent


def run():
    # Fotografia degli OHLCV prima del calcolo.
    input_before = weekly_input_fingerprint()

    from ms_weekly_publication import MANIFEST, weekly_science_fingerprint
    import json

    previous_science = None
    if MANIFEST.is_file():
        previous_science = json.loads(MANIFEST.read_text()).get("weekly_science_sha256")

    science_changed = previous_science != weekly_science_fingerprint()

    if input_before[0] <= input_before[1] and not science_changed:
        print(
            "WEEKLY SKIP: nessuna nuova quotazione DAILY",
            flush=True,
        )
        return None

    VERSIONS.mkdir(parents=True, exist_ok=True)

    version = VERSIONS / ("weekly_" + uuid.uuid4().hex)

    if version.exists():
        raise RuntimeError("Versione WEEKLY già esistente")

    print("WEEKLY BUILD:", version, flush=True)

    env = os.environ.copy()
    env["MS_WEEKLY_BUILD_DIR"] = str(version)

    science = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "ms_weekly_provisional_science.py"),
        ],
        cwd=ROOT,
        env=env,
        timeout=1800,
        check=False,
    )

    if science.returncode != 0:
        raise RuntimeError(
            "Calcolo scientifico WEEKLY fallito: "
            + str(science.returncode)
        )

    aggregate_script = (
        "import os; "
        "from pathlib import Path; "
        "import operational.run_marketsentinel as m; "
        "m.BASE=Path(os.environ['MS_WEEKLY_BUILD_DIR']).resolve(); "
        "m.main()"
    )

    aggregate = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            aggregate_script,
        ],
        cwd=ROOT,
        env=env,
        timeout=600,
        check=False,
    )

    if aggregate.returncode != 0:
        raise RuntimeError(
            "Classifica WEEKLY fallita: "
            + str(aggregate.returncode)
        )

    # Non pubblicare risultati costruiti con input cambiati.
    input_after = weekly_input_fingerprint()

    if input_before != input_after:
        raise RuntimeError(
            "WEEKLY NON PUBBLICATA: "
            "OHLCV DAILY o definitiva modificati durante il calcolo"
        )

    published, reference = publish_version(
        version,
        daily_ohlcv_fingerprint=input_before[2],
    )

    print("WEEKLY PUBBLICATA:", published, flush=True)
    print("DATA SCIENTIFICA:", reference, flush=True)
    print("TITOLI ATTESI: 199", flush=True)

    return published


if __name__ == "__main__":
    run()
