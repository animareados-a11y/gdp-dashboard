"""Worker WEEKLY isolato MarketSentinel.

Utilizza esclusivamente il runner scientifico versionato.
Non modifica direttamente gli archivi WEEKLY definitivi.
"""

from ms_weekly_isolated_runner import run


def run_weekly():
    result = run()

    if result is None:
        print(
            "[WEEKLY] Nessuna nuova quotazione da elaborare",
            flush=True,
        )
    else:
        print(
            "[WEEKLY] Versione pubblicata:",
            result,
            flush=True,
        )

    return result


if __name__ == "__main__":
    run_weekly()
