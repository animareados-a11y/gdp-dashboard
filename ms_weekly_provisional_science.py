"""Calcolo scientifico WEEKLY isolato, senza download Yahoo."""

from pathlib import Path
import os
import json
import pandas as pd

from ms_weekly_provisional import (
    prepare_provisional,
    PROVISIONAL,
    EXPECTED,
)


def run_science():
    # Ambiente isolato opzionale per la nuova WEEKLY.
    requested = os.environ.get("MS_WEEKLY_BUILD_DIR")

    sandbox = (
        Path(requested).resolve()
        if requested
        else PROVISIONAL.resolve()
    )

    root = Path(__file__).resolve().parent
    data_root = (root / "data").resolve()

    protected = [
        (data_root / "operational_market").resolve(),
        (data_root / "marketsentinel_intraday").resolve(),
        PROVISIONAL.resolve(),
    ]

    if requested:
        if sandbox == data_root or data_root not in sandbox.parents:
            raise RuntimeError(
                "Destinazione WEEKLY fuori dall'area di lavoro consentita"
            )

        if any(
            sandbox == base or base in sandbox.parents
            for base in protected
        ):
            raise RuntimeError(
                "Destinazione WEEKLY protetta: " + str(sandbox)
            )

    if requested:
        prepare_provisional(destination=sandbox)
    else:
        prepare_provisional()

    import operational.incremental_phase as phase
    import operational.incremental_v160 as v160
    from ms_weekly_publication import (
        MANIFEST,
        weekly_science_fingerprint,
    )

    previous_science = None
    if MANIFEST.is_file():
        previous_science = json.loads(
            MANIFEST.read_text(encoding="utf-8")
        ).get("weekly_science_sha256")

    force_recalc = (
        previous_science != weekly_science_fingerprint()
    )

    if force_recalc:
        os.environ["MS_WEEKLY_FORCE_PHASE_RECALC"] = "1"
        print("WEEKLY V5: RICALCOLO SCIENTIFICO COMPLETO", flush=True)
    else:
        os.environ.pop("MS_WEEKLY_FORCE_PHASE_RECALC", None)
        print("WEEKLY: MODALITÀ INCREMENTALE", flush=True)

    phase.BASE = sandbox
    v160.BASE = sandbox

    results = {}
    total = 0

    for market, expected in EXPECTED.items():
        changed_phase = phase.identify_changed_tickers(market)
        changed_v160 = v160.identify_changed_tickers(market)

        changed = changed_phase | changed_v160

        if force_recalc:
            prices = pd.read_csv(
                sandbox / f"{market}_weekly_master.csv",
                usecols=["Ticker"],
            )
            changed = set(prices["Ticker"].unique())

            if len(changed) != expected:
                raise RuntimeError(
                    f"{market}: universo incompleto per ricalcolo V5"
                )

        if not changed:
            print(f"{market}: SCIENTIFICO GIÀ ALLINEATO", flush=True)
            results[market] = 0
            continue

        print(
            f"{market}: RICALCOLO PHASE + V160 "
            f"PER {len(changed)} TITOLI",
            flush=True,
        )

        _, current, calculated_phase = (
            phase.prepare_incremental_phase(
                market,
                dry_run=False,
            )
        )

        _, calculated_v160 = (
            v160.prepare_incremental_v160(
                market,
                changed_tickers=changed,
                dry_run=False,
            )
        )

        if len(current) != expected:
            raise RuntimeError(
                f"{market}: risultati PHASE incompleti"
            )

        results[market] = len(calculated_v160)
        total += len(calculated_v160)

    print("\nWEEKLY SCIENTIFICO COMPLETATO")
    print("TITOLI RICALCOLATI:", total)
    print("ARCHIVI DEFINITIVI INVARIATI")

    return results


if __name__ == "__main__":
    run_science()
