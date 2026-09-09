"""
MarketSentinel
V40.39 - PHASE CANONICAL ENGINE
===============================

SCOPO
-----
Esporre come motore PHASE canonico la logica V5 RESET
validata su FULL200.

La V5 ha superato:

- validazione Italia40;
- validazione FULL200;
- 200/200 ticker processati;
- 0 errori;
- 0 modifiche SELL rispetto alla V4;
- 0 modifiche fuori dai run BUY protetti;
- controlli semantici mirati;
- controllo BUY1 tardivi;
- reset causale su HA bearish decisiva.

IMPORTANTE
----------
Questo file NON modifica la logica V5.

È intenzionalmente un wrapper minimale:
la logica tecnica rimane in

    weekly_v40_5_phase_dynamics_shadow_v5.py

In questo modo possiamo:

1. stabilire un unico entry point PHASE canonico;
2. verificare equivalenza assoluta con V5;
3. collegare successivamente V40.12 / FULL200 alla nuova PHASE;
4. mantenere intatta la V5 validata come riferimento.

NON modifica:
- engine.py
- REVERSAL
- STRENGTH
- dataset FULL200
- V40.10
- V40.9
"""

from __future__ import annotations

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v5 as v5


VERSION = "V40.39"

SOURCE_ENGINE = "weekly_v40_5_phase_dynamics_shadow_v5"
SOURCE_VERSION = getattr(v5, "VERSION", "V5")


def process_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica la PHASE canonica.

    Al momento V40.39 è intenzionalmente identica alla
    V5 RESET validata: nessuna nuova regola viene introdotta.
    """
    return v5.process_ticker(df.copy())


def apply_phase_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alias esplicito per l'utilizzo come motore PHASE canonico.
    """
    return process_ticker(df)


# Alias compatibile con gli engine PHASE precedenti.
apply_phase_dynamics_shadow = process_ticker