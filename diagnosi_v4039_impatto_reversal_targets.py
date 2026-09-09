"""
MarketSentinel
DIAGNOSI IMPATTO PHASE V40.39 SUI TARGET REVERSAL

OBIETTIVO
---------
Confrontare i TRUE REVERSAL TARGET costruiti con:

    VECCHIA PHASE:
        data/v40_36b_full200_features.csv

    NUOVA PHASE CANONICAL:
        data/v40_39_full200_features.csv

La logica dei target replica quella usata da:

    weekly_v40_37c_full200_production_fit.py

NON:
- modifica PHASE
- modifica REVERSAL
- addestra modelli
- modifica V40.37c
- modifica dataset esistenti

Serve esclusivamente a misurare quanto il passaggio
V40.36b -> V40.39 cambia:

1. PHASE
2. PHASE_FAMILY
3. eleggibilità dei target
4. TRUE REVERSAL TARGET 1W / 2W / 3W
5. numero di positivi reversal

OUTPUT
------
data/diagnosi_v4039_reversal_target_summary.csv
data/diagnosi_v4039_phase_family_changes.csv
data/diagnosi_v4039_reversal_target_changes.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

OLD_FILE = Path("data/v40_36b_full200_features.csv")
NEW_FILE = Path("data/v40_39_full200_features.csv")

OUT_SUMMARY = Path(
    "data/diagnosi_v4039_reversal_target_summary.csv"
)

OUT_PHASE_CHANGES = Path(
    "data/diagnosi_v4039_phase_family_changes.csv"
)

OUT_TARGET_CHANGES = Path(
    "data/diagnosi_v4039_reversal_target_changes.csv"
)


TARGET_CONFIG = {
    "BEAR_TO_BULL_1W": {
        "family": "RIBASSISTA",
        "future_family": "RIALZISTA",
        "horizon": 1,
    },
    "BEAR_TO_BULL_2W": {
        "family": "RIBASSISTA",
        "future_family": "RIALZISTA",
        "horizon": 2,
    },
    "BEAR_TO_BULL_3W": {
        "family": "RIBASSISTA",
        "future_family": "RIALZISTA",
        "horizon": 3,
    },
    "BULL_TO_BEAR_1W": {
        "family": "RIALZISTA",
        "future_family": "RIBASSISTA",
        "horizon": 1,
    },
    "BULL_TO_BEAR_2W": {
        "family": "RIALZISTA",
        "future_family": "RIBASSISTA",
        "horizon": 2,
    },
    "BULL_TO_BEAR_3W": {
        "family": "RIALZISTA",
        "future_family": "RIBASSISTA",
        "horizon": 3,
    },
}


# ============================================================
# UTILITA'
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def find_column(columns, candidates):
    lookup = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    return None


# ============================================================
# REPLICA ESATTA NORMALIZZAZIONE V40.37c
# ============================================================

def normalize_phase(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .upper()
        .replace("À", "A")
        .replace("È", "E")
        .replace("É", "E")
        .replace("Ì", "I")
        .replace("Ò", "O")
        .replace("Ù", "U")
    )


def phase_family(value):
    phase = normalize_phase(value)

    bullish = {
        "BUY",
        "TREND_RIALZISTA",
        "DEBOLEZZA_RIALZISTA",
    }

    bearish = {
        "SELL",
        "TREND_RIBASSISTA",
        "DEBOLEZZA_RIBASSISTA",
    }

    if phase in bullish:
        return "RIALZISTA"

    if phase in bearish:
        return "RIBASSISTA"

    return "INDECISIONE"


# ============================================================
# LETTURA DATASET
# ============================================================

def load_phase_file(path: Path, suffix: str) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path}"
        )

    header = pd.read_csv(
        path,
        nrows=0,
    )

    columns = list(header.columns)

    ticker_col = find_column(
        columns,
        ["Ticker", "TICKER", "ticker"],
    )

    date_col = find_column(
        columns,
        ["Date", "DATE", "date"],
    )

    phase_col = find_column(
        columns,
        ["PHASE", "Phase", "phase"],
    )

    if ticker_col is None:
        raise RuntimeError(
            f"Ticker non trovato in {path}"
        )

    if date_col is None:
        raise RuntimeError(
            f"Date non trovata in {path}"
        )

    if phase_col is None:
        raise RuntimeError(
            f"PHASE non trovata in {path}"
        )

    df = pd.read_csv(
        path,
        usecols=[
            ticker_col,
            date_col,
            phase_col,
        ],
    )

    df = df.rename(
        columns={
            ticker_col: "TICKER",
            date_col: "DATE",
            phase_col: f"PHASE_{suffix}",
        }
    )

    df["DATE"] = pd.to_datetime(
        df["DATE"],
        utc=True,
        errors="coerce",
    ).dt.tz_convert(None)

    if df["DATE"].isna().any():
        raise RuntimeError(
            f"Date non valide in {path}"
        )

    duplicates = df.duplicated(
        subset=["TICKER", "DATE"]
    ).sum()

    if duplicates != 0:
        raise RuntimeError(
            f"Duplicati TICKER+DATE in {path}: "
            f"{duplicates}"
        )

    df[f"PHASE_FAMILY_{suffix}"] = (
        df[f"PHASE_{suffix}"]
        .apply(phase_family)
    )

    return df


# ============================================================
# COSTRUZIONE TARGET
# ============================================================

def build_targets(
    df: pd.DataFrame,
    family_col: str,
    suffix: str,
) -> pd.DataFrame:

    work = df.copy()

    work = work.sort_values(
        ["TICKER", "DATE"]
    ).reset_index(drop=True)

    for target_name, cfg in TARGET_CONFIG.items():

        horizon = cfg["horizon"]
        current_family = cfg["family"]
        future_family_target = cfg["future_family"]

        future_family = (
            work
            .groupby(
                "TICKER",
                sort=False,
            )[family_col]
            .shift(-horizon)
        )

        valid_future = future_family.notna()

        target = pd.Series(
            np.nan,
            index=work.index,
            dtype=float,
        )

        eligible = (
            (work[family_col] == current_family)
            & valid_future
        )

        target.loc[eligible] = (
            future_family.loc[eligible]
            == future_family_target
        ).astype(float)

        work[
            f"{target_name}_{suffix}"
        ] = target

    return work


# ============================================================
# MAIN
# ============================================================

section("CARICAMENTO DATASET")

old = load_phase_file(
    OLD_FILE,
    "OLD",
)

new = load_phase_file(
    NEW_FILE,
    "NEW",
)

print(
    f"V40.36b: {len(old):,} righe | "
    f"{old['TICKER'].nunique()} ticker"
)

print(
    f"V40.39:  {len(new):,} righe | "
    f"{new['TICKER'].nunique()} ticker"
)


# ============================================================
# ALLINEAMENTO
# ============================================================

section("ALLINEAMENTO V40.36b vs V40.39")

merged = old.merge(
    new,
    on=["TICKER", "DATE"],
    how="outer",
    indicator=True,
)

alignment_counts = (
    merged["_merge"]
    .value_counts()
    .to_dict()
)

print(
    "Righe presenti in entrambi:",
    f"{alignment_counts.get('both', 0):,}",
)

print(
    "Solo V40.36b:",
    f"{alignment_counts.get('left_only', 0):,}",
)

print(
    "Solo V40.39:",
    f"{alignment_counts.get('right_only', 0):,}",
)

if (
    alignment_counts.get("left_only", 0) != 0
    or alignment_counts.get("right_only", 0) != 0
):
    raise RuntimeError(
        "I due dataset non hanno identico "
        "universo TICKER+DATE."
    )

merged = merged.drop(
    columns="_merge"
)

merged = merged.sort_values(
    ["TICKER", "DATE"]
).reset_index(drop=True)


# ============================================================
# CONFRONTO PHASE
# ============================================================

section("CONFRONTO PHASE")

merged["PHASE_CHANGED"] = (
    merged["PHASE_OLD"].fillna("")
    != merged["PHASE_NEW"].fillna("")
)

merged["FAMILY_CHANGED"] = (
    merged["PHASE_FAMILY_OLD"]
    != merged["PHASE_FAMILY_NEW"]
)

phase_changed = int(
    merged["PHASE_CHANGED"].sum()
)

family_changed = int(
    merged["FAMILY_CHANGED"].sum()
)

print(
    f"PHASE modificata: "
    f"{phase_changed:,} / {len(merged):,} "
    f"({phase_changed / len(merged) * 100:.4f}%)"
)

print(
    f"PHASE_FAMILY modificata: "
    f"{family_changed:,} / {len(merged):,} "
    f"({family_changed / len(merged) * 100:.4f}%)"
)


# ============================================================
# SALVATAGGIO CAMBI FAMIGLIA
# ============================================================

family_changes = merged.loc[
    merged["FAMILY_CHANGED"],
    [
        "TICKER",
        "DATE",
        "PHASE_OLD",
        "PHASE_NEW",
        "PHASE_FAMILY_OLD",
        "PHASE_FAMILY_NEW",
    ],
].copy()

family_changes.to_csv(
    OUT_PHASE_CHANGES,
    index=False,
)


# ============================================================
# COSTRUZIONE TARGET OLD E NEW
# ============================================================

section("COSTRUZIONE TARGET REVERSAL")

old_targets = build_targets(
    merged[
        [
            "TICKER",
            "DATE",
            "PHASE_FAMILY_OLD",
        ]
    ],
    "PHASE_FAMILY_OLD",
    "OLD",
)

new_targets = build_targets(
    merged[
        [
            "TICKER",
            "DATE",
            "PHASE_FAMILY_NEW",
        ]
    ],
    "PHASE_FAMILY_NEW",
    "NEW",
)


comparison = old_targets.merge(
    new_targets,
    on=["TICKER", "DATE"],
    how="inner",
)


# ============================================================
# CONFRONTO DEI SEI TARGET
# ============================================================

summary_rows = []
changed_rows = []

for target_name in TARGET_CONFIG:

    old_col = f"{target_name}_OLD"
    new_col = f"{target_name}_NEW"

    old_valid = comparison[old_col].notna()
    new_valid = comparison[new_col].notna()

    both_valid = old_valid & new_valid

    eligibility_changed = (
        old_valid != new_valid
    )

    label_changed = (
        both_valid
        & (
            comparison[old_col]
            != comparison[new_col]
        )
    )

    any_definition_change = (
        eligibility_changed
        | label_changed
    )

    old_eligible_n = int(old_valid.sum())
    new_eligible_n = int(new_valid.sum())

    old_positive_n = int(
        (comparison[old_col] == 1.0).sum()
    )

    new_positive_n = int(
        (comparison[new_col] == 1.0).sum()
    )

    both_valid_n = int(
        both_valid.sum()
    )

    label_changed_n = int(
        label_changed.sum()
    )

    eligibility_changed_n = int(
        eligibility_changed.sum()
    )

    any_change_n = int(
        any_definition_change.sum()
    )

    label_change_pct = (
        label_changed_n
        / both_valid_n
        * 100
        if both_valid_n
        else np.nan
    )

    any_change_pct = (
        any_change_n
        / len(comparison)
        * 100
    )

    summary_rows.append(
        {
            "TARGET": target_name,
            "OLD_ELIGIBLE": old_eligible_n,
            "NEW_ELIGIBLE": new_eligible_n,
            "ELIGIBILITY_DIFF": (
                new_eligible_n
                - old_eligible_n
            ),
            "OLD_POSITIVE": old_positive_n,
            "NEW_POSITIVE": new_positive_n,
            "POSITIVE_DIFF": (
                new_positive_n
                - old_positive_n
            ),
            "BOTH_VALID": both_valid_n,
            "LABEL_CHANGED_BOTH_VALID": (
                label_changed_n
            ),
            "LABEL_CHANGE_PCT_BOTH_VALID": (
                label_change_pct
            ),
            "ELIGIBILITY_CHANGED": (
                eligibility_changed_n
            ),
            "ANY_TARGET_DEFINITION_CHANGE": (
                any_change_n
            ),
            "ANY_CHANGE_PCT_ALL_ROWS": (
                any_change_pct
            ),
        }
    )

    changed_part = comparison.loc[
        any_definition_change,
        [
            "TICKER",
            "DATE",
            old_col,
            new_col,
        ],
    ].copy()

    if not changed_part.empty:

        changed_part["TARGET"] = target_name

        changed_part[
            "ELIGIBILITY_CHANGED"
        ] = eligibility_changed.loc[
            changed_part.index
        ].values

        changed_part[
            "LABEL_CHANGED"
        ] = label_changed.loc[
            changed_part.index
        ].values

        changed_part = changed_part.rename(
            columns={
                old_col: "TARGET_OLD",
                new_col: "TARGET_NEW",
            }
        )

        changed_rows.append(
            changed_part[
                [
                    "TICKER",
                    "DATE",
                    "TARGET",
                    "TARGET_OLD",
                    "TARGET_NEW",
                    "ELIGIBILITY_CHANGED",
                    "LABEL_CHANGED",
                ]
            ]
        )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

if changed_rows:

    target_changes = pd.concat(
        changed_rows,
        ignore_index=True,
    )

else:

    target_changes = pd.DataFrame(
        columns=[
            "TICKER",
            "DATE",
            "TARGET",
            "TARGET_OLD",
            "TARGET_NEW",
            "ELIGIBILITY_CHANGED",
            "LABEL_CHANGED",
        ]
    )

target_changes.to_csv(
    OUT_TARGET_CHANGES,
    index=False,
)


# ============================================================
# OUTPUT COMPATTO
# ============================================================

section("RISULTATO")

display_cols = [
    "TARGET",
    "OLD_ELIGIBLE",
    "NEW_ELIGIBLE",
    "OLD_POSITIVE",
    "NEW_POSITIVE",
    "LABEL_CHANGED_BOTH_VALID",
    "LABEL_CHANGE_PCT_BOTH_VALID",
    "ELIGIBILITY_CHANGED",
    "ANY_TARGET_DEFINITION_CHANGE",
    "ANY_CHANGE_PCT_ALL_ROWS",
]

print(
    summary[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()
print(
    f"PHASE_CHANGED = {phase_changed:,}"
)

print(
    f"PHASE_FAMILY_CHANGED = {family_changed:,}"
)

print()
print("File salvati:")

print(
    f"  {OUT_SUMMARY}"
)

print(
    f"  {OUT_PHASE_CHANGES}"
)

print(
    f"  {OUT_TARGET_CHANGES}"
)

print()
print(
    "DIAGNOSI COMPLETATA - "
    "NESSUN MODELLO E' STATO ADDESTRATO."
)