"""Promozione protetta degli archivi WEEKLY definitivi."""
import json
import os
from pathlib import Path

from ms_weekly_publication import (
    MANIFEST, VERSIONS, validate_build,
    weekly_science_fingerprint,
)
from ms_weekly_refresh_guard import completed_friday_week
from ms_weekly_source_selector import DEFINITIVE


def consolidate():
    if not MANIFEST.is_file():
        return False, "Manifest WEEKLY assente"

    info = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if info.get("publication_status") != "DEFINITIVO":
        return False, "Versione WEEKLY non definitiva"

    if info.get("weekly_science_sha256") != weekly_science_fingerprint():
        raise RuntimeError("Versione scientifica WEEKLY non coerente")

    friday = completed_friday_week()
    if friday is None:
        return False, "Quotazioni del venerdì incomplete"

    name = info.get("version", "")
    source = (VERSIONS / name).resolve()

    if source.parent != VERSIONS.resolve() or not source.is_dir():
        raise RuntimeError("Percorso versione non valido")

    reference = validate_build(source)
    previous = validate_build(DEFINITIVE)

    if reference.normalize() != friday.normalize():
        raise RuntimeError("Data WEEKLY non coerente con venerdì")

    if reference <= previous:
        return False, "WEEKLY già consolidato"

    if not DEFINITIVE.is_symlink():
        raise RuntimeError("Base definitiva non versionata")

    # Il backup della precedente definitiva rimane disponibile
    # nella struttura versionata e nell'archivio originale.
    old_target = DEFINITIVE.resolve()

    if not old_target.is_dir():
        raise RuntimeError("Vecchia definitiva non disponibile")

    # La nuova versione è già stata validata e pubblicata.
    # Prepariamo un symlink temporaneo nella stessa directory.
    temporary = DEFINITIVE.with_name(
        ".operational_market_atomic_next"
    )

    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("Collegamento temporaneo già presente")

    relative = os.path.relpath(source, DEFINITIVE.parent)
    temporary.symlink_to(relative, target_is_directory=True)

    try:
        if validate_build(temporary) != reference:
            raise RuntimeError("Validazione collegamento fallita")

        # Sostituzione atomica del solo collegamento.
        os.replace(temporary, DEFINITIVE)

    finally:
        if temporary.is_symlink():
            temporary.unlink()

    if DEFINITIVE.resolve() != source:
        raise RuntimeError("Promozione non coerente")

    return True, "WEEKLY definitivo consolidato: " + str(friday.date())


if __name__ == "__main__":
    print(*consolidate())
