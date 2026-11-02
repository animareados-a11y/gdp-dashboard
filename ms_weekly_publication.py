"""Pubblicazione versionata WEEKLY MarketSentinel."""

from pathlib import Path
import json
import os
import tempfile
import shutil
import re

from ms_weekly_source_selector import validate_source

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

VERSIONS = DATA / "marketsentinel_weekly_versions"
MANIFEST = DATA / "marketsentinel_weekly_active.json"

def weekly_science_fingerprint():
    """Impronta della versione scientifica WEEKLY."""
    import hashlib

    sources = (
        "weekly_v40_39_phase_canonical.py",
        "weekly_v40_5_phase_dynamics_shadow.py",
        "weekly_v40_5_phase_dynamics_shadow_v5.py",
        "weekly_v40_5_phase_dynamics_shadow_v4.py",
        "operational/incremental_phase.py",
        "operational/incremental_v160.py",
        "operational/phase_v4_fast.py",
    )

    digest = hashlib.sha256()

    for name in sources:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)

        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())

    return digest.hexdigest()


REQUIRED_MARKETS = {
    "italy40": 40,
    "europe40": 40,
    "usa80": 80,
    "usa40": 39,
}


def active_version():
    """Restituisce soltanto una versione pubblicata valida."""
    if not MANIFEST.is_file():
        return None

    info = json.loads(MANIFEST.read_text(encoding="utf-8"))
    name = info.get("version")

    if not isinstance(name, str) or not name:
        raise ValueError("Manifest WEEKLY non valido")

    base = (VERSIONS / name).resolve()

    if base.parent != VERSIONS.resolve():
        raise ValueError("Percorso versione WEEKLY non valido")

    if not base.is_dir():
        raise FileNotFoundError(base)

    return base


def validate_build(folder):
    """Valida integralmente i 199 titoli prima della pubblicazione."""
    folder = Path(folder).resolve()

    if not folder.is_dir():
        raise FileNotFoundError(folder)

    return validate_source(folder)



def isolate_scientific_archives(folder):
    """Separa gli hard link scientifici prima della pubblicazione."""
    from ms_weekly_source_selector import DEFINITIVE

    suffixes = (
        "_phase_history.csv",
        "_current_phase.csv",
        "_v160_features.csv",
    )

    separated = 0
    bytes_copied = 0

    for market in REQUIRED_MARKETS:
        for suffix in suffixes:
            name = market + suffix
            source = DEFINITIVE / name
            target = folder / name

            if not source.is_file() or not target.is_file():
                raise FileNotFoundError(name)

            original = source.stat()
            candidate = target.stat()

            if (
                original.st_dev == candidate.st_dev
                and original.st_ino == candidate.st_ino
            ):
                temporary = target.with_name(
                    target.name + ".isolating.tmp"
                )

                if temporary.exists():
                    raise RuntimeError(
                        "File temporaneo già presente: " + str(temporary)
                    )

                try:
                    shutil.copy2(source, temporary)
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        temporary.unlink()

                separated += 1
                bytes_copied += original.st_size

    print(
        "ARCHIVI SCIENTIFICI ISOLATI:",
        separated,
        "SPAZIO COPIATO MB:",
        round(bytes_copied / 1024**2, 1),
        flush=True,
    )

def publish_version(folder, daily_ohlcv_fingerprint=None):
    """Pubblica una versione completa mediante manifest atomico."""
    folder = Path(folder).resolve()

    if folder.parent != VERSIONS.resolve():
        raise ValueError(
            "La versione deve trovarsi nella cartella versioni"
        )

    # Nessuna versione pubblicata deve condividere inode
    # scientifici con gli archivi WEEKLY definitivi.
    isolate_scientific_archives(folder)

    reference = validate_build(folder)

    # Nessuna regressione rispetto alla WEEKLY definitiva.
    from ms_weekly_source_selector import DEFINITIVE

    definitive_date = validate_build(DEFINITIVE)

    science_revision = (
        daily_ohlcv_fingerprint is not None
        and reference == definitive_date
        and (
            not MANIFEST.is_file()
            or json.loads(MANIFEST.read_text()).get("weekly_science_sha256")
            != weekly_science_fingerprint()
        )
    )

    if reference <= definitive_date and not science_revision:
        raise ValueError(
            "WEEKLY non più recente della definitiva: "
            f"{reference} <= {definitive_date}"
        )

    # Nessuna regressione rispetto alla versione già pubblicata.
    previous = active_version()

    if previous is not None:
        previous_date = validate_build(previous)

        if reference < previous_date:
            raise ValueError(
                "WEEKLY precedente alla versione pubblicata: "
                f"{reference} < {previous_date}"
            )

    from ms_weekly_refresh_guard import completed_friday_week

    completed_friday = completed_friday_week()
    is_definitive = (
        completed_friday is not None
        and reference.normalize() == completed_friday.normalize()
    )

    payload = {
        "version": folder.name,
        "reference_date": reference.date().isoformat(),
        "weekly_science_sha256": weekly_science_fingerprint(),
        "publication_status": (
            "DEFINITIVO" if is_definitive else "PROVVISORIO"
        ),
    }

    if daily_ohlcv_fingerprint is not None:
        fingerprint = daily_ohlcv_fingerprint

        if (
            not isinstance(fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
        ):
            raise ValueError("Impronta OHLCV DAILY non valida")

        payload["daily_ohlcv_fingerprint"] = fingerprint

    DATA.mkdir(parents=True, exist_ok=True)

    fd, temporary = tempfile.mkstemp(
        prefix=".weekly_manifest_",
        suffix=".tmp",
        dir=DATA,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, MANIFEST)

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    return folder, reference
