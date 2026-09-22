"""
MarketSentinel
V40.62 — OPPORTUNITY TRUE-TREND TARGET AUDIT

PURPOSE
-------
Build the historical TRUE-TREND universe that will be used by the
continuous Opportunity architecture.

This script DOES NOT:
- train ML models
- create Opportunity scores
- modify PHASE V40.39
- modify REVERSAL V40.40
- modify STRENGTH V40.44
- modify V40.61
- impose score weights or score thresholds

FINAL Opportunity requirement remains:
    BUY Opportunity(T)
    SELL Opportunity(T)

for every ticker and every week T, independently of current PHASE.

V40.62 is the historical target-foundation audit.

----------------------------------------------------------------------
TRUE TREND / RETRACEMENT RULE
----------------------------------------------------------------------

Historical anti-retracement rule, recovered and validated exactly:

N = 8
    If a new SAR regime starts after >= 8 weeks of the opposite
    previous SAR regime, the new movement is initially considered
    a potential retracement.

K = 3
    The movement becomes a TRUE TREND if, before the SAR regime ends,
    it accumulates 3 decisive Heikin-Ashi confirmations in the new
    direction.

BUY decisive HA:
    V4015_HA_DIRECTION == +1
    V4015_HA_INDECISION == 0

SELL decisive HA:
    V4015_HA_DIRECTION == -1
    V4015_HA_INDECISION == 0

Confirmations are CUMULATIVE / NON-CONSECUTIVE:
- indecision does not count
- opposite HA does not count
- neither resets previous confirmations
- SAR flip ends the attempt

If K3 is reached:
    TRUE_TREND_N8_K3

The economic T0 remains the ORIGINAL SAR FLIP / BUY1 / SELL1.
K3 validates the movement; it DOES NOT move T0.

If K3 is not reached:
    RETRACEMENT_N8_K3_FAIL

For movements whose previous opposite SAR regime is < 8 weeks:
    TRUE_TREND_BASE

N8/K3 is specifically the anti-retracement safeguard for flips
following a long opposite SAR regime. It is not applied retroactively
as a universal K3 requirement to all ordinary PHASE movements.

----------------------------------------------------------------------
ECONOMIC ENDPOINT
----------------------------------------------------------------------

Economic endpoint semantics are inherited directly from V40.61.

BULL:
    active = BUY / TREND_RIALZISTA

BEAR:
    active = SELL / TREND_RIBASSISTA

Temporary weakness followed by same-side active resumption remains
inside the economic movement.

Terminal weakness after the last same-side active candle is excluded.

The endpoint is NOT selected by ex-post price maximum/minimum.

----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------

1) SAR movement audit
2) PHASE branch + TRUE/RETRACEMENT classification
3) TRUE-TREND economic outcomes T0 -> economic endpoint
4) structural audit
5) summaries
6) metadata

The next Opportunity stages will use these historical labels/outcomes
while still calculating BUY and SELL Opportunity continuously at every T.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import weekly_v40_61_opportunity_phase_branch_target_audit as v61


# =============================================================================
# CONFIG
# =============================================================================

VERSION = "V40.62"

PHASE_FILE = Path(
    "data/v40_39_full200_phase12.csv"
)

FEATURE_FILE = Path(
    "data/v40_39_full200_features.csv"
)

OUTDIR = Path(
    "data/v40_62_opportunity_true_trend_target_audit"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

SAR_MOVEMENT_FILE = (
    OUTDIR
    / "v40_62_sar_movements.csv"
)

BRANCH_FILE = (
    OUTDIR
    / "v40_62_true_trend_branches.csv"
)

TRUE_TREND_FILE = (
    OUTDIR
    / "v40_62_true_trend_outcomes.csv"
)

SUMMARY_FILE = (
    OUTDIR
    / "v40_62_true_trend_summary.csv"
)

AUDIT_FILE = (
    OUTDIR
    / "v40_62_structural_audit.csv"
)

METADATA_FILE = (
    OUTDIR
    / "v40_62_metadata.json"
)


N_PREV_SAR_MIN = 8
K_HA_CONFIRMATIONS = 3


# =============================================================================
# HELPERS
# =============================================================================

def safe_numeric(
    s: pd.Series,
    default: float = np.nan,
) -> pd.Series:

    x = pd.to_numeric(
        s,
        errors="coerce",
    )

    if np.isfinite(default):
        x = x.fillna(default)

    return x


def directional_realized_pct(
    entry: float,
    exit_: float,
    side: str,
) -> float:

    if (
        not np.isfinite(entry)
        or not np.isfinite(exit_)
        or entry == 0
    ):
        return np.nan

    if side == "BULL":
        return (
            (exit_ / entry) - 1.0
        ) * 100.0

    if side == "BEAR":
        return (
            (entry / exit_) - 1.0
        ) * 100.0

    return np.nan


def favorable_adverse(
    entry: float,
    path: np.ndarray,
    side: str,
) -> tuple[float, float]:

    if (
        not np.isfinite(entry)
        or entry == 0
    ):
        return np.nan, np.nan

    p = np.asarray(
        path,
        dtype=float,
    )

    p = p[
        np.isfinite(p)
    ]

    if len(p) == 0:
        return np.nan, np.nan

    if side == "BULL":

        favorable = np.max(
            (p - entry)
            / entry
        ) * 100.0

        adverse = np.max(
            (entry - p)
            / entry
        ) * 100.0

    elif side == "BEAR":

        favorable = np.max(
            (entry - p)
            / entry
        ) * 100.0

        adverse = np.max(
            (p - entry)
            / entry
        ) * 100.0

    else:
        return np.nan, np.nan

    return (
        max(float(favorable), 0.0),
        max(float(adverse), 0.0),
    )


def valid_ha_confirmation(
    ha_direction: float,
    ha_indecision: float,
    sar_side: int,
) -> bool:

    if not np.isfinite(ha_direction):
        ha_direction = 0

    if not np.isfinite(ha_indecision):
        ha_indecision = 0

    if int(ha_indecision) != 0:
        return False

    if sar_side == 1:
        return int(ha_direction) == 1

    if sar_side == -1:
        return int(ha_direction) == -1

    return False


# =============================================================================
# LOAD + 1:1 MERGE
# =============================================================================

def load_source() -> pd.DataFrame:

    phase_cols = [
        "Ticker",
        "Date",
        "PHASE",
        "V4012_PHASE_AGE",
        "V4012_PHASE_DIRECTION",
        "Close",
    ]

    feature_cols = [
        "Ticker",
        "Date",
        "V4012_SAR_SIDE",
        "V4012_SAR_AGE",
        "V4010_PREV_SAR_SIDE",
        "V4010_PREV_SAR_AGE",
        "V4015_HA_DIRECTION",
        "V4015_HA_INDECISION",
    ]

    phase = pd.read_csv(
        PHASE_FILE,
        usecols=phase_cols,
    )

    feat = pd.read_csv(
        FEATURE_FILE,
        usecols=feature_cols,
    )

    phase["Date"] = pd.to_datetime(
        phase["Date"]
    )

    feat["Date"] = pd.to_datetime(
        feat["Date"]
    )

    phase_dup = int(
        phase.duplicated(
            ["Ticker", "Date"]
        ).sum()
    )

    feat_dup = int(
        feat.duplicated(
            ["Ticker", "Date"]
        ).sum()
    )

    if phase_dup != 0:
        raise RuntimeError(
            "PHASE source has duplicate Ticker/Date keys: "
            f"{phase_dup}"
        )

    if feat_dup != 0:
        raise RuntimeError(
            "FEATURE source has duplicate Ticker/Date keys: "
            f"{feat_dup}"
        )

    merged = phase.merge(
        feat,
        on=["Ticker", "Date"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(phase):
        raise RuntimeError(
            "1:1 merge failed: "
            f"phase={len(phase):,} "
            f"merged={len(merged):,}"
        )

    if len(merged) != len(feat):
        raise RuntimeError(
            "1:1 merge failed: "
            f"features={len(feat):,} "
            f"merged={len(merged):,}"
        )

    merged = (
        merged
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    return merged


# =============================================================================
# SAR MOVEMENTS
# =============================================================================

def build_sar_movements(
    g: pd.DataFrame,
) -> list[dict]:

    g = (
        g
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    side = safe_numeric(
        g["V4012_SAR_SIDE"],
        0,
    ).to_numpy()

    age = safe_numeric(
        g["V4012_SAR_AGE"],
        0,
    ).to_numpy()

    prev_side = safe_numeric(
        g["V4010_PREV_SAR_SIDE"],
        0,
    ).to_numpy()

    prev_age = safe_numeric(
        g["V4010_PREV_SAR_AGE"],
        0,
    ).to_numpy()

    ha_dir = safe_numeric(
        g["V4015_HA_DIRECTION"],
        0,
    ).to_numpy()

    ha_ind = safe_numeric(
        g["V4015_HA_INDECISION"],
        0,
    ).to_numpy()

    flip_positions = np.flatnonzero(
        (age == 1)
        & np.isin(
            side,
            [-1, 1],
        )
    )

    rows = []

    for start_idx in flip_positions:

        current_side = int(
            side[start_idx]
        )

        direction = (
            "BULL"
            if current_side == 1
            else "BEAR"
        )

        previous_side = int(
            prev_side[start_idx]
        )

        previous_age = float(
            prev_age[start_idx]
        )

        run_end_idx = start_idx

        valid_count = 0
        k3_idx = None

        for i in range(
            start_idx,
            len(g),
        ):

            if int(side[i]) != current_side:
                break

            run_end_idx = i

            if valid_ha_confirmation(
                ha_dir[i],
                ha_ind[i],
                current_side,
            ):

                valid_count += 1

                if (
                    valid_count
                    == K_HA_CONFIRMATIONS
                    and k3_idx is None
                ):
                    k3_idx = i

        run_length = (
            run_end_idx
            - start_idx
            + 1
        )

        n8_applicable = (
            previous_age
            >= N_PREV_SAR_MIN
        )

        k3_reached = (
            k3_idx is not None
        )

        if n8_applicable:

            if k3_reached:
                classification = (
                    "TRUE_TREND_N8_K3"
                )
            else:
                classification = (
                    "RETRACEMENT_N8_K3_FAIL"
                )

        else:
            classification = (
                "TRUE_TREND_BASE"
            )

        rows.append(
            {
                "Ticker": (
                    g.at[
                        start_idx,
                        "Ticker",
                    ]
                ),
                "sar_start_idx": int(
                    start_idx
                ),
                "sar_start_date": (
                    g.at[
                        start_idx,
                        "Date",
                    ]
                ),
                "sar_start_close": float(
                    g.at[
                        start_idx,
                        "Close",
                    ]
                ),
                "direction": direction,
                "sar_side": current_side,
                "previous_sar_side": (
                    previous_side
                ),
                "previous_sar_age": (
                    previous_age
                ),
                "n8_applicable": bool(
                    n8_applicable
                ),
                "k3_reached": bool(
                    k3_reached
                ),
                "k3_confirmation_idx": (
                    int(k3_idx)
                    if k3_idx is not None
                    else None
                ),
                "k3_confirmation_date": (
                    g.at[
                        k3_idx,
                        "Date",
                    ]
                    if k3_idx is not None
                    else pd.NaT
                ),
                "k3_confirmation_week": (
                    int(
                        k3_idx
                        - start_idx
                        + 1
                    )
                    if k3_idx is not None
                    else np.nan
                ),
                "total_valid_ha_in_run": int(
                    valid_count
                ),
                "sar_run_end_idx": int(
                    run_end_idx
                ),
                "sar_run_end_date": (
                    g.at[
                        run_end_idx,
                        "Date",
                    ]
                ),
                "sar_run_length": int(
                    run_length
                ),
                "trend_classification": (
                    classification
                ),
                "is_true_trend": bool(
                    classification.startswith(
                        "TRUE_TREND"
                    )
                ),
                "is_retracement": bool(
                    classification.startswith(
                        "RETRACEMENT"
                    )
                ),
            }
        )

    return rows


# =============================================================================
# LINK V40.61 PHASE BRANCHES TO SAR MOVEMENTS
# =============================================================================

def build_branch_audit(
    g: pd.DataFrame,
    sar_movements: list[dict],
) -> list[dict]:

    branches = (
        v61.build_branches_for_ticker(
            g
        )
    )

    sar_by_start = {
        int(x["sar_start_idx"]): x
        for x in sar_movements
    }

    rows = []

    for b in branches:

        start_idx = int(
            b["start_idx"]
        )

        side = b["side"]

        sar = sar_by_start.get(
            start_idx
        )

        phase_start_is_sar_flip = (
            sar is not None
        )

        sar_direction_matches = False

        if sar is not None:
            sar_direction_matches = (
                sar["direction"]
                == side
            )

        if (
            sar is not None
            and sar_direction_matches
        ):
            classification = (
                sar[
                    "trend_classification"
                ]
            )

            is_true = bool(
                sar["is_true_trend"]
            )

            is_retracement = bool(
                sar["is_retracement"]
            )

        else:
            classification = (
                "UNRESOLVED_ALIGNMENT"
            )

            is_true = False
            is_retracement = False

        row = dict(b)

        row.update(
            {
                "phase_start_is_sar_flip": bool(
                    phase_start_is_sar_flip
                ),
                "sar_direction_matches_phase": bool(
                    sar_direction_matches
                ),
                "trend_classification": (
                    classification
                ),
                "is_true_trend": bool(
                    is_true
                ),
                "is_retracement": bool(
                    is_retracement
                ),
            }
        )

        if sar is not None:

            for key in [
                "sar_start_date",
                "sar_start_close",
                "sar_side",
                "previous_sar_side",
                "previous_sar_age",
                "n8_applicable",
                "k3_reached",
                "k3_confirmation_date",
                "k3_confirmation_week",
                "total_valid_ha_in_run",
                "sar_run_end_date",
                "sar_run_length",
            ]:
                row[key] = sar[key]

        else:

            row.update(
                {
                    "sar_start_date": pd.NaT,
                    "sar_start_close": np.nan,
                    "sar_side": np.nan,
                    "previous_sar_side": np.nan,
                    "previous_sar_age": np.nan,
                    "n8_applicable": False,
                    "k3_reached": False,
                    "k3_confirmation_date": pd.NaT,
                    "k3_confirmation_week": np.nan,
                    "total_valid_ha_in_run": np.nan,
                    "sar_run_end_date": pd.NaT,
                    "sar_run_length": np.nan,
                }
            )

        rows.append(row)

    return rows


# =============================================================================
# TRUE-TREND ECONOMIC OUTCOMES
# =============================================================================

def build_true_trend_outcomes(
    g: pd.DataFrame,
    branch_rows: list[dict],
) -> list[dict]:

    rows = []

    for b in branch_rows:

        if not bool(
            b["completed"]
        ):
            continue

        if not bool(
            b["is_true_trend"]
        ):
            continue

        endpoint_idx = (
            b["economic_endpoint_idx"]
        )

        if endpoint_idx is None:
            continue

        if pd.isna(endpoint_idx):
            continue

        start_idx = int(
            b["start_idx"]
        )

        endpoint_idx = int(
            endpoint_idx
        )

        if endpoint_idx < start_idx:
            raise RuntimeError(
                "Economic endpoint before T0."
            )

        side = b["side"]

        entry = float(
            g.at[
                start_idx,
                "Close",
            ]
        )

        exit_ = float(
            g.at[
                endpoint_idx,
                "Close",
            ]
        )

        realized = (
            directional_realized_pct(
                entry,
                exit_,
                side,
            )
        )

        path = (
            g.loc[
                start_idx:endpoint_idx,
                "Close",
            ]
            .to_numpy(
                dtype=float
            )
        )

        mfe, mae = favorable_adverse(
            entry,
            path,
            side,
        )

        row = {
            "Ticker": (
                g.at[
                    start_idx,
                    "Ticker",
                ]
            ),
            "side": side,
            "trend_classification": (
                b[
                    "trend_classification"
                ]
            ),
            "T0_idx": start_idx,
            "T0_date": (
                g.at[
                    start_idx,
                    "Date",
                ]
            ),
            "T0_phase": (
                g.at[
                    start_idx,
                    "PHASE",
                ]
            ),
            "T0_close": entry,
            "previous_sar_age": (
                b[
                    "previous_sar_age"
                ]
            ),
            "n8_applicable": (
                b[
                    "n8_applicable"
                ]
            ),
            "k3_reached": (
                b[
                    "k3_reached"
                ]
            ),
            "k3_confirmation_date": (
                b[
                    "k3_confirmation_date"
                ]
            ),
            "k3_confirmation_week": (
                b[
                    "k3_confirmation_week"
                ]
            ),
            "economic_endpoint_idx": (
                endpoint_idx
            ),
            "economic_endpoint_date": (
                g.at[
                    endpoint_idx,
                    "Date",
                ]
            ),
            "economic_endpoint_phase": (
                g.at[
                    endpoint_idx,
                    "PHASE",
                ]
            ),
            "economic_endpoint_close": (
                exit_
            ),
            "weeks_T0_to_endpoint": (
                endpoint_idx
                - start_idx
            ),
            "directional_realized_pct": (
                realized
            ),
            "mfe_pct": mfe,
            "mae_pct": mae,
            "temporary_weakness_rows": (
                b[
                    "temporary_weakness_rows"
                ]
            ),
            "terminal_weakness_rows": (
                b[
                    "terminal_weakness_rows"
                ]
            ),
            "resumed_after_weakness": (
                b[
                    "resumed_after_weakness"
                ]
            ),
        }

        rows.append(row)

    return rows


# =============================================================================
# AUDIT
# =============================================================================

def add_audit(
    rows: list[dict],
    name: str,
    value,
    expected,
    passed: bool,
):

    rows.append(
        {
            "check": name,
            "value": value,
            "expected": expected,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
        }
    )


def build_audit(
    source: pd.DataFrame,
    sar: pd.DataFrame,
    branches: pd.DataFrame,
    true_trends: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    add_audit(
        rows,
        "source_rows",
        len(source),
        141886,
        len(source) == 141886,
    )

    add_audit(
        rows,
        "source_tickers",
        source["Ticker"].nunique(),
        200,
        source["Ticker"].nunique()
        == 200,
    )

    source_dup = int(
        source.duplicated(
            ["Ticker", "Date"]
        ).sum()
    )

    add_audit(
        rows,
        "source_key_duplicates",
        source_dup,
        0,
        source_dup == 0,
    )

    # ---------------------------------------------------------
    # Historical N8 population must reproduce recovered audit.
    # ---------------------------------------------------------

    n8 = sar[
        sar["n8_applicable"]
        == True
    ].copy()

    n8_buy = n8[
        n8["direction"]
        == "BULL"
    ]

    n8_sell = n8[
        n8["direction"]
        == "BEAR"
    ]

    add_audit(
        rows,
        "n8_buy_events",
        len(n8_buy),
        3495,
        len(n8_buy) == 3495,
    )

    add_audit(
        rows,
        "n8_sell_events",
        len(n8_sell),
        4576,
        len(n8_sell) == 4576,
    )

    buy_k3 = int(
        n8_buy[
            "k3_reached"
        ].sum()
    )

    sell_k3 = int(
        n8_sell[
            "k3_reached"
        ].sum()
    )

    add_audit(
        rows,
        "n8_buy_k3_reached",
        buy_k3,
        2638,
        buy_k3 == 2638,
    )

    add_audit(
        rows,
        "n8_sell_k3_reached",
        sell_k3,
        2678,
        sell_k3 == 2678,
    )

    # ---------------------------------------------------------
    # N8 classification integrity
    # ---------------------------------------------------------

    bad_n8_true = int(
        (
            n8["is_true_trend"]
            &
            ~n8["k3_reached"]
        ).sum()
    )

    add_audit(
        rows,
        "n8_true_without_k3",
        bad_n8_true,
        0,
        bad_n8_true == 0,
    )

    bad_n8_retracement = int(
        (
            n8["is_retracement"]
            &
            n8["k3_reached"]
        ).sum()
    )

    add_audit(
        rows,
        "n8_retracement_with_k3",
        bad_n8_retracement,
        0,
        bad_n8_retracement == 0,
    )

    # ---------------------------------------------------------
    # Branch alignment
    # ---------------------------------------------------------

    completed = branches[
        branches["completed"]
        == True
    ].copy()

    unresolved = int(
        (
            branches[
                "trend_classification"
            ]
            == "UNRESOLVED_ALIGNMENT"
        ).sum()
    )

    add_audit(
        rows,
        "phase_branch_unresolved_alignment",
        unresolved,
        0,
        unresolved == 0,
    )

    wrong_direction = int(
        (
            branches[
                "phase_start_is_sar_flip"
            ]
            &
            ~branches[
                "sar_direction_matches_phase"
            ]
        ).sum()
    )

    add_audit(
        rows,
        "phase_sar_direction_mismatch",
        wrong_direction,
        0,
        wrong_direction == 0,
    )

    # ---------------------------------------------------------
    # Retracements must never enter true-trend outcomes.
    # ---------------------------------------------------------

    if len(true_trends):

        bad_retracement = int(
            true_trends[
                "trend_classification"
            ]
            .str.startswith(
                "RETRACEMENT",
                na=False,
            )
            .sum()
        )

    else:
        bad_retracement = 0

    add_audit(
        rows,
        "retracement_in_true_trend_outcomes",
        bad_retracement,
        0,
        bad_retracement == 0,
    )

    # ---------------------------------------------------------
    # Endpoint integrity
    # ---------------------------------------------------------

    if len(true_trends):

        bad_endpoint = int(
            (
                true_trends[
                    "economic_endpoint_date"
                ]
                <
                true_trends[
                    "T0_date"
                ]
            ).sum()
        )

    else:
        bad_endpoint = 0

    add_audit(
        rows,
        "true_trend_endpoint_before_T0",
        bad_endpoint,
        0,
        bad_endpoint == 0,
    )

    if len(true_trends):

        bad_mfe = int(
            (
                true_trends[
                    "mfe_pct"
                ]
                < 0
            ).sum()
        )

        bad_mae = int(
            (
                true_trends[
                    "mae_pct"
                ]
                < 0
            ).sum()
        )

    else:
        bad_mfe = 0
        bad_mae = 0

    add_audit(
        rows,
        "negative_mfe",
        bad_mfe,
        0,
        bad_mfe == 0,
    )

    add_audit(
        rows,
        "negative_mae",
        bad_mae,
        0,
        bad_mae == 0,
    )

    # ---------------------------------------------------------
    # K3 must NOT move T0.
    # ---------------------------------------------------------

    k3_true = true_trends[
        true_trends[
            "trend_classification"
        ]
        == "TRUE_TREND_N8_K3"
    ].copy()

    if len(k3_true):

        shifted_t0 = int(
            (
                k3_true[
                    "T0_date"
                ]
                ==
                k3_true[
                    "k3_confirmation_date"
                ]
            ).sum()
        )

        week1_k3 = int(
            (
                k3_true[
                    "k3_confirmation_week"
                ]
                == 1
            ).sum()
        )

        suspicious_shift = (
            shifted_t0
            - week1_k3
        )

    else:
        suspicious_shift = 0

    add_audit(
        rows,
        "k3_improperly_used_as_T0",
        suspicious_shift,
        0,
        suspicious_shift == 0,
    )

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARIES
# =============================================================================

def build_summary(
    sar: pd.DataFrame,
    branches: pd.DataFrame,
    true_trends: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for side in [
        "BULL",
        "BEAR",
    ]:

        s = sar[
            sar["direction"]
            == side
        ].copy()

        b = branches[
            branches["side"]
            == side
        ].copy()

        t = true_trends[
            true_trends["side"]
            == side
        ].copy()

        n8 = s[
            s["n8_applicable"]
            == True
        ]

        retr = s[
            s["is_retracement"]
            == True
        ]

        true_sar = s[
            s["is_true_trend"]
            == True
        ]

        row = {
            "side": side,
            "sar_movements": len(s),
            "n8_movements": len(n8),
            "n8_k3_reached": int(
                n8[
                    "k3_reached"
                ].sum()
            ),
            "retracements": len(retr),
            "true_sar_movements": (
                len(true_sar)
            ),
            "phase_branches": len(b),
            "completed_phase_branches": int(
                b[
                    "completed"
                ].sum()
            ),
            "true_trend_outcomes": len(t),
        }

        if len(t):

            row.update(
                {
                    "realized_mean": (
                        t[
                            "directional_realized_pct"
                        ].mean()
                    ),
                    "realized_median": (
                        t[
                            "directional_realized_pct"
                        ].median()
                    ),
                    "realized_positive_rate": (
                        (
                            t[
                                "directional_realized_pct"
                            ]
                            > 0
                        ).mean()
                    ),
                    "mfe_mean": (
                        t[
                            "mfe_pct"
                        ].mean()
                    ),
                    "mae_mean": (
                        t[
                            "mae_pct"
                        ].mean()
                    ),
                    "weeks_to_endpoint_mean": (
                        t[
                            "weeks_T0_to_endpoint"
                        ].mean()
                    ),
                    "weeks_to_endpoint_median": (
                        t[
                            "weeks_T0_to_endpoint"
                        ].median()
                    ),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("MARKETSENTINEL V40.62")
    print("OPPORTUNITY TRUE-TREND TARGET AUDIT")
    print("=" * 80)

    if not PHASE_FILE.exists():
        raise FileNotFoundError(
            f"Missing source: {PHASE_FILE}"
        )

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing source: {FEATURE_FILE}"
        )

    print("\nLoading and joining V40.39 sources...")

    df = load_source()

    print(
        f"ROWS: {len(df):,}"
    )

    print(
        f"TICKERS: "
        f"{df['Ticker'].nunique():,}"
    )

    print(
        "DATE:",
        df["Date"].min().date(),
        "->",
        df["Date"].max().date(),
    )

    all_sar = []
    all_branches = []
    all_true_trends = []

    ticker_groups = list(
        df.groupby(
            "Ticker",
            sort=True,
        )
    )

    total = len(
        ticker_groups
    )

    print(
        "\nReconstructing SAR movements, "
        "N8/K3 and V40.61 economic branches..."
    )

    for k, (
        ticker,
        g0,
    ) in enumerate(
        ticker_groups,
        start=1,
    ):

        g = (
            g0
            .sort_values("Date")
            .reset_index(drop=True)
            .copy()
        )

        sar_rows = (
            build_sar_movements(
                g
            )
        )

        branch_rows = (
            build_branch_audit(
                g,
                sar_rows,
            )
        )

        true_rows = (
            build_true_trend_outcomes(
                g,
                branch_rows,
            )
        )

        all_sar.extend(
            sar_rows
        )

        for x in branch_rows:
            x["Ticker"] = ticker

        all_branches.extend(
            branch_rows
        )

        all_true_trends.extend(
            true_rows
        )

        if (
            k % 20 == 0
            or k == total
        ):
            print(
                f"  {k:3d}/{total} tickers"
            )

    sar_df = pd.DataFrame(
        all_sar
    )

    branch_df = pd.DataFrame(
        all_branches
    )

    true_df = pd.DataFrame(
        all_true_trends
    )

    # =========================================================
    # AUDIT
    # =========================================================

    print(
        "\nRunning structural audit..."
    )

    audit_df = build_audit(
        df,
        sar_df,
        branch_df,
        true_df,
    )

    summary_df = build_summary(
        sar_df,
        branch_df,
        true_df,
    )

    # =========================================================
    # SAVE
    # =========================================================

    sar_df.to_csv(
        SAR_MOVEMENT_FILE,
        index=False,
    )

    branch_df.to_csv(
        BRANCH_FILE,
        index=False,
    )

    true_df.to_csv(
        TRUE_TREND_FILE,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    audit_df.to_csv(
        AUDIT_FILE,
        index=False,
    )

    all_pass = bool(
        (
            audit_df["status"]
            == "PASS"
        ).all()
    )

    metadata = {
        "version": VERSION,
        "phase_source": str(
            PHASE_FILE
        ),
        "feature_source": str(
            FEATURE_FILE
        ),
        "rows": int(
            len(df)
        ),
        "tickers": int(
            df[
                "Ticker"
            ].nunique()
        ),
        "date_min": str(
            df[
                "Date"
            ].min().date()
        ),
        "date_max": str(
            df[
                "Date"
            ].max().date()
        ),
        "n_prev_sar_min": (
            N_PREV_SAR_MIN
        ),
        "k_ha_confirmations": (
            K_HA_CONFIRMATIONS
        ),
        "k_mode": (
            "CUMULATIVE_NON_CONSECUTIVE"
        ),
        "k_moves_t0": False,
        "economic_endpoint_source": (
            "V40.61"
        ),
        "phase_modified": False,
        "reversal_modified": False,
        "strength_modified": False,
        "ml_trained": False,
        "opportunity_score_created": False,
        "final_score_requirement": (
            "BUY_AND_SELL_EVERY_T_INDEPENDENT_OF_PHASE"
        ),
        "structural_audit_pass": (
            all_pass
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================
    # REPORT
    # =========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SAR MOVEMENT CLASSIFICATION"
    )

    print(
        "=" * 80
    )

    print(
        sar_df[
            "trend_classification"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PHASE BRANCH CLASSIFICATION"
    )

    print(
        "=" * 80
    )

    print(
        branch_df[
            "trend_classification"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TRUE TREND ECONOMIC SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # =========================================================
    # RETRACEMENT ECONOMIC DIAGNOSTIC
    # =========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "N8/K3 FAIL — RETRACEMENT ECONOMIC DIAGNOSTIC"
    )

    print(
        "=" * 80
    )

    retracement_rows = []

    retracements = sar_df[
        sar_df["trend_classification"]
        == "RETRACEMENT_N8_K3_FAIL"
    ].copy()

    for ticker, rt in retracements.groupby(
        "Ticker",
        sort=True,
    ):

        g = (
            df[df["Ticker"] == ticker]
            .sort_values("Date")
            .reset_index(drop=True)
        )

        for _, r in rt.iterrows():

            start_date = pd.to_datetime(
                r["sar_start_date"]
            )

            end_date = pd.to_datetime(
                r["sar_run_end_date"]
            )

            z = g[
                (g["Date"] >= start_date)
                & (g["Date"] <= end_date)
            ]

            if z.empty:
                continue

            entry = float(
                z.iloc[0]["Close"]
            )

            exit_ = float(
                z.iloc[-1]["Close"]
            )

            side = r["direction"]

            realized = directional_realized_pct(
                entry,
                exit_,
                side,
            )

            mfe, mae = favorable_adverse(
                entry,
                z["Close"].to_numpy(dtype=float),
                side,
            )

            retracement_rows.append(
                {
                    "side": side,
                    "realized_pct": realized,
                    "mfe_pct": mfe,
                    "mae_pct": mae,
                    "weeks": len(z),
                }
            )

    retracement_econ = pd.DataFrame(
        retracement_rows
    )

    for side, x in retracement_econ.groupby(
        "side"
    ):

        print(
            f"\n{side}  N={len(x):,}"
        )

        print(
            x[
                [
                    "realized_pct",
                    "mfe_pct",
                    "mae_pct",
                    "weeks",
                ]
            ]
            .agg(
                ["mean", "median"]
            )
            .to_string()
        )

        print(
            "positive_rate="
            f"{(x['realized_pct'] > 0).mean():.4%}"
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STRUCTURAL AUDIT"
    )

    print(
        "=" * 80
    )

    print(
        audit_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    if all_pass:

        print(
            "FINAL RESULT: PASS"
        )

        print(
            "V40.62 TRUE-TREND TARGET "
            "FOUNDATION STRUCTURALLY VALID"
        )

    else:

        print(
            "FINAL RESULT: FAIL"
        )

        print(
            "DO NOT PROCEED TO ML / "
            "CONTINUOUS OPPORTUNITY YET"
        )

    print(
        "=" * 80
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  {SAR_MOVEMENT_FILE}"
    )

    print(
        f"  {BRANCH_FILE}"
    )

    print(
        f"  {TRUE_TREND_FILE}"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print(
        f"  {AUDIT_FILE}"
    )

    print(
        f"  {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()