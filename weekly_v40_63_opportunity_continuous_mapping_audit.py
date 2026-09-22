"""
MarketSentinel
V40.63 — OPPORTUNITY CONTINUOUS MAPPING AUDIT

PURPOSE
-------
Create a diagnostic point-in-time map for BOTH BUY and SELL at every
Ticker/Date, using the frozen V40.62 true-trend foundation.

This script DOES NOT:
- train ML
- create Opportunity scores
- create a final economic Y
- choose weights
- choose score thresholds
- use fixed forecast horizons
- modify PHASE / REVERSAL / STRENGTH
- modify V40.62

For every T and for each side independently, the script identifies
the temporal relation between T and the historical directional
structures established by V40.62.

RELATIONS
---------
CURRENT_TRUE_TREND
    T is between original T0 and economic endpoint, inclusive,
    of a completed TRUE trend of that side.

CURRENT_RETRACEMENT
    T is inside the SAR regime of a completed N8/K3 failed movement
    of that side.

AFTER_ECONOMIC_ENDPOINT
    T is after the economic endpoint but still inside the structural
    branch before the opposite PHASE anchor.

NEXT_TRUE_TREND
    None of the previous conditions applies and a later completed
    TRUE trend of that side exists.

NO_COMPLETED_TRUE_TREND_AHEAD
    No completed later TRUE trend of that side is available.

IMPORTANT
---------
NEXT_TRUE_TREND is only a temporal mapping here.
V40.63 does NOT assume that the economics of that future trend are
already the correct target for T.

That methodological decision belongs to the next target-design stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

VERSION = "V40.63"

SOURCE_FILE = Path(
    "data/v40_39_full200_phase12.csv"
)

V62_DIR = Path(
    "data/v40_62_opportunity_true_trend_target_audit"
)

BRANCH_FILE = (
    V62_DIR
    / "v40_62_true_trend_branches.csv"
)

TRUE_TREND_FILE = (
    V62_DIR
    / "v40_62_true_trend_outcomes.csv"
)

OUTDIR = Path(
    "data/v40_63_opportunity_continuous_mapping_audit"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAPPING_FILE = (
    OUTDIR
    / "v40_63_continuous_mapping.csv"
)

SUMMARY_FILE = (
    OUTDIR
    / "v40_63_mapping_summary.csv"
)

PHASE_SUMMARY_FILE = (
    OUTDIR
    / "v40_63_phase_mapping_summary.csv"
)

AUDIT_FILE = (
    OUTDIR
    / "v40_63_structural_audit.csv"
)

METADATA_FILE = (
    OUTDIR
    / "v40_63_metadata.json"
)


TRUE_CLASSES = {
    "TRUE_TREND_BASE",
    "TRUE_TREND_N8_K3",
}

RETRACEMENT_CLASS = (
    "RETRACEMENT_N8_K3_FAIL"
)


# =============================================================================
# HELPERS
# =============================================================================

def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


def add_check(
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


# =============================================================================
# LOAD
# =============================================================================

def load_data():

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Missing source: {SOURCE_FILE}"
        )

    if not BRANCH_FILE.exists():
        raise FileNotFoundError(
            f"Missing V40.62 branches: "
            f"{BRANCH_FILE}"
        )

    if not TRUE_TREND_FILE.exists():
        raise FileNotFoundError(
            f"Missing V40.62 true trends: "
            f"{TRUE_TREND_FILE}"
        )

    source = pd.read_csv(
        SOURCE_FILE,
        usecols=[
            "Ticker",
            "Date",
            "PHASE",
            "V4012_PHASE_AGE",
            "V4012_PHASE_DIRECTION",
            "Close",
        ],
    )

    branches = pd.read_csv(
        BRANCH_FILE
    )

    true_trends = pd.read_csv(
        TRUE_TREND_FILE
    )

    source["Date"] = pd.to_datetime(
        source["Date"]
    )

    date_cols_branch = [
        "start_date",
        "structural_end_date",
        "economic_endpoint_date",
        "sar_start_date",
        "k3_confirmation_date",
        "sar_run_end_date",
    ]

    for c in date_cols_branch:
        if c in branches.columns:
            branches[c] = pd.to_datetime(
                branches[c],
                errors="coerce",
            )

    date_cols_true = [
        "T0_date",
        "k3_confirmation_date",
        "economic_endpoint_date",
    ]

    for c in date_cols_true:
        if c in true_trends.columns:
            true_trends[c] = pd.to_datetime(
                true_trends[c],
                errors="coerce",
            )

    branches["completed"] = as_bool(
        branches["completed"]
    )

    branches["is_true_trend"] = as_bool(
        branches["is_true_trend"]
    )

    branches["is_retracement"] = as_bool(
        branches["is_retracement"]
    )

    source = (
        source
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    branches = (
        branches
        .sort_values(
            ["Ticker", "start_date"]
        )
        .reset_index(drop=True)
    )

    true_trends = (
        true_trends
        .sort_values(
            ["Ticker", "T0_date"]
        )
        .reset_index(drop=True)
    )

    return (
        source,
        branches,
        true_trends,
    )


# =============================================================================
# SIDE MAP FOR ONE TICKER
# =============================================================================

def build_side_mapping(
    g: pd.DataFrame,
    b: pd.DataFrame,
    tt: pd.DataFrame,
    side: str,
) -> pd.DataFrame:

    n = len(g)

    dates = g["Date"].to_numpy(
        dtype="datetime64[ns]"
    )

    relation = np.full(
        n,
        "",
        dtype=object,
    )

    current_start_date = np.full(
        n,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    current_endpoint_date = np.full(
        n,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    current_class = np.full(
        n,
        "",
        dtype=object,
    )

    current_realized_from_t = np.full(
        n,
        np.nan,
        dtype=float,
    )

    current_mfe_from_t = np.full(
        n,
        np.nan,
        dtype=float,
    )

    current_mae_from_t = np.full(
        n,
        np.nan,
        dtype=float,
    )

    weeks_to_current_endpoint = np.full(
        n,
        np.nan,
        dtype=float,
    )

    retracement_start_date = np.full(
        n,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    retracement_end_date = np.full(
        n,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    weeks_since_economic_endpoint = np.full(
        n,
        np.nan,
        dtype=float,
    )

    next_true_start_date = np.full(
        n,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    next_true_endpoint_date = np.full(
        n,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    next_true_class = np.full(
        n,
        "",
        dtype=object,
    )

    weeks_to_next_true_start = np.full(
        n,
        np.nan,
        dtype=float,
    )

    weeks_to_next_true_endpoint = np.full(
        n,
        np.nan,
        dtype=float,
    )

    # -----------------------------------------------------------------
    # Completed branches for this side.
    # -----------------------------------------------------------------

    sb = b[
        (b["side"] == side)
        & b["completed"]
    ].copy()

    true_b = sb[
        sb["is_true_trend"]
        & sb[
            "trend_classification"
        ].isin(TRUE_CLASSES)
    ].copy()

    retr_b = sb[
        sb["is_retracement"]
        & (
            sb[
                "trend_classification"
            ]
            == RETRACEMENT_CLASS
        )
    ].copy()

    # -----------------------------------------------------------------
    # 1. CURRENT TRUE TREND:
    #    T0 -> economic endpoint inclusive.
    #
    #    Economics are measured FROM CURRENT T, not from original T0.
    #    This is diagnostic only, not yet the final Y.
    # -----------------------------------------------------------------

    for _, r in true_b.iterrows():

        start = r["start_date"]
        endpoint = r[
            "economic_endpoint_date"
        ]

        if (
            pd.isna(start)
            or pd.isna(endpoint)
        ):
            continue

        mask = (
            (dates >= np.datetime64(start))
            & (
                dates
                <= np.datetime64(endpoint)
            )
        )

        idxs = np.flatnonzero(mask)

        endpoint_close = float(
            r["economic_endpoint_close"]
        )

        for i in idxs:

            if relation[i] != "":
                raise RuntimeError(
                    f"Overlapping mapping for "
                    f"{g.at[i, 'Ticker']} "
                    f"{g.at[i, 'Date']} "
                    f"{side}: "
                    f"{relation[i]} vs "
                    f"CURRENT_TRUE_TREND"
                )

            relation[i] = (
                "CURRENT_TRUE_TREND"
            )

            current_start_date[i] = (
                np.datetime64(start)
            )

            current_endpoint_date[i] = (
                np.datetime64(endpoint)
            )

            current_class[i] = (
                r[
                    "trend_classification"
                ]
            )

            entry = float(
                g.at[i, "Close"]
            )

            endpoint_idx_arr = np.flatnonzero(
                dates
                == np.datetime64(endpoint)
            )

            if len(endpoint_idx_arr) != 1:
                raise RuntimeError(
                    "Economic endpoint date "
                    "not uniquely found in source."
                )

            e = int(
                endpoint_idx_arr[0]
            )

            if e < i:
                raise RuntimeError(
                    "CURRENT TRUE endpoint "
                    "before T."
                )

            if side == "BULL":
                realized = (
                    endpoint_close
                    / entry
                    - 1.0
                ) * 100.0
            else:
                realized = (
                    entry
                    / endpoint_close
                    - 1.0
                ) * 100.0

            path = (
                g.loc[
                    i:e,
                    "Close",
                ]
                .to_numpy(
                    dtype=float
                )
            )

            if side == "BULL":
                mfe = np.max(
                    (path - entry)
                    / entry
                ) * 100.0

                mae = np.max(
                    (entry - path)
                    / entry
                ) * 100.0

            else:
                mfe = np.max(
                    (entry - path)
                    / entry
                ) * 100.0

                mae = np.max(
                    (path - entry)
                    / entry
                ) * 100.0

            current_realized_from_t[i] = (
                float(realized)
            )

            current_mfe_from_t[i] = max(
                float(mfe),
                0.0,
            )

            current_mae_from_t[i] = max(
                float(mae),
                0.0,
            )

            weeks_to_current_endpoint[i] = (
                e - i
            )

    # -----------------------------------------------------------------
    # 2. CURRENT RETRACEMENT:
    #    SAR start -> SAR run end inclusive.
    #
    #    We do NOT assign an artificial zero/penalty.
    # -----------------------------------------------------------------

    for _, r in retr_b.iterrows():

        start = r[
            "sar_start_date"
        ]

        end = r[
            "sar_run_end_date"
        ]

        if (
            pd.isna(start)
            or pd.isna(end)
        ):
            continue

        mask = (
            (dates >= np.datetime64(start))
            & (
                dates
                <= np.datetime64(end)
            )
        )

        idxs = np.flatnonzero(mask)

        for i in idxs:

            if relation[i] != "":
                raise RuntimeError(
                    f"Overlap TRUE/RETRACEMENT "
                    f"for {g.at[i, 'Ticker']} "
                    f"{g.at[i, 'Date']} "
                    f"{side}"
                )

            relation[i] = (
                "CURRENT_RETRACEMENT"
            )

            retracement_start_date[i] = (
                np.datetime64(start)
            )

            retracement_end_date[i] = (
                np.datetime64(end)
            )

    # -----------------------------------------------------------------
    # 3. AFTER ECONOMIC ENDPOINT:
    #    endpoint+1 -> structural end for TRUE branch.
    #
    #    This is terminal same-side structure. We deliberately do not
    #    reinterpret it as CURRENT opportunity.
    # -----------------------------------------------------------------

    for _, r in true_b.iterrows():

        endpoint = r[
            "economic_endpoint_date"
        ]

        structural_end = r[
            "structural_end_date"
        ]

        if (
            pd.isna(endpoint)
            or pd.isna(structural_end)
        ):
            continue

        mask = (
            (dates > np.datetime64(endpoint))
            & (
                dates
                <= np.datetime64(
                    structural_end
                )
            )
        )

        idxs = np.flatnonzero(mask)

        endpoint_idx_arr = np.flatnonzero(
            dates
            == np.datetime64(endpoint)
        )

        if len(endpoint_idx_arr) != 1:
            raise RuntimeError(
                "Endpoint not uniquely found "
                "for AFTER_ENDPOINT mapping."
            )

        e = int(
            endpoint_idx_arr[0]
        )

        for i in idxs:

            # A later same-side retracement mapping has priority if
            # structurally present. Do not overwrite it.
            if relation[i] != "":
                continue

            relation[i] = (
                "AFTER_ECONOMIC_ENDPOINT"
            )

            current_start_date[i] = (
                np.datetime64(
                    r["start_date"]
                )
            )

            current_endpoint_date[i] = (
                np.datetime64(endpoint)
            )

            current_class[i] = (
                r[
                    "trend_classification"
                ]
            )

            weeks_since_economic_endpoint[i] = (
                i - e
            )

    # -----------------------------------------------------------------
    # 4. NEXT TRUE TREND.
    #
    #    Pure mapping only.
    #    We DO NOT turn its future economics into T's target here.
    # -----------------------------------------------------------------

    tt_side = tt[
        tt["side"] == side
    ].copy()

    tt_side = (
        tt_side
        .sort_values("T0_date")
        .reset_index(drop=True)
    )

    future_starts = (
        tt_side["T0_date"]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )

    for i in range(n):

        if relation[i] != "":
            continue

        t = dates[i]

        j = np.searchsorted(
            future_starts,
            t,
            side="right",
        )

        if j < len(tt_side):

            nr = tt_side.iloc[j]

            start = nr["T0_date"]
            endpoint = nr[
                "economic_endpoint_date"
            ]

            relation[i] = (
                "NEXT_TRUE_TREND"
            )

            next_true_start_date[i] = (
                np.datetime64(start)
            )

            next_true_endpoint_date[i] = (
                np.datetime64(endpoint)
            )

            next_true_class[i] = (
                nr[
                    "trend_classification"
                ]
            )

            start_idx_arr = np.flatnonzero(
                dates
                == np.datetime64(start)
            )

            endpoint_idx_arr = (
                np.flatnonzero(
                    dates
                    == np.datetime64(endpoint)
                )
            )

            if len(start_idx_arr) != 1:
                raise RuntimeError(
                    "NEXT TRUE start date "
                    "not uniquely found."
                )

            if len(endpoint_idx_arr) != 1:
                raise RuntimeError(
                    "NEXT TRUE endpoint date "
                    "not uniquely found."
                )

            sidx = int(
                start_idx_arr[0]
            )

            eidx = int(
                endpoint_idx_arr[0]
            )

            next_true_start_date[i] = (
                np.datetime64(start)
            )

            weeks_to_next_true_start[i] = (
                sidx - i
            )

            weeks_to_next_true_endpoint[i] = (
                eidx - i
            )

        else:

            relation[i] = (
                "NO_COMPLETED_TRUE_TREND_AHEAD"
            )

    prefix = (
        "buy"
        if side == "BULL"
        else "sell"
    )

    out = pd.DataFrame(
        {
            f"{prefix}_relation": (
                relation
            ),
            f"{prefix}_current_start_date": (
                current_start_date
            ),
            f"{prefix}_current_endpoint_date": (
                current_endpoint_date
            ),
            f"{prefix}_current_class": (
                current_class
            ),
            f"{prefix}_current_realized_from_T_pct": (
                current_realized_from_t
            ),
            f"{prefix}_current_mfe_from_T_pct": (
                current_mfe_from_t
            ),
            f"{prefix}_current_mae_from_T_pct": (
                current_mae_from_t
            ),
            f"{prefix}_weeks_to_current_endpoint": (
                weeks_to_current_endpoint
            ),
            f"{prefix}_retracement_start_date": (
                retracement_start_date
            ),
            f"{prefix}_retracement_end_date": (
                retracement_end_date
            ),
            f"{prefix}_weeks_since_economic_endpoint": (
                weeks_since_economic_endpoint
            ),
            f"{prefix}_next_true_start_date": (
                next_true_start_date
            ),
            f"{prefix}_next_true_endpoint_date": (
                next_true_endpoint_date
            ),
            f"{prefix}_next_true_class": (
                next_true_class
            ),
            f"{prefix}_weeks_to_next_true_start": (
                weeks_to_next_true_start
            ),
            f"{prefix}_weeks_to_next_true_endpoint": (
                weeks_to_next_true_endpoint
            ),
        }
    )

    return out


# =============================================================================
# PROCESS TICKER
# =============================================================================

def process_ticker(
    g0: pd.DataFrame,
    b0: pd.DataFrame,
    tt0: pd.DataFrame,
) -> pd.DataFrame:

    g = (
        g0
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    buy = build_side_mapping(
        g,
        b0,
        tt0,
        "BULL",
    )

    sell = build_side_mapping(
        g,
        b0,
        tt0,
        "BEAR",
    )

    base = g[
        [
            "Ticker",
            "Date",
            "PHASE",
            "V4012_PHASE_AGE",
            "V4012_PHASE_DIRECTION",
            "Close",
        ]
    ].copy()

    return pd.concat(
        [
            base,
            buy,
            sell,
        ],
        axis=1,
    )


# =============================================================================
# SUMMARY
# =============================================================================

def make_summary(
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for side in [
        "buy",
        "sell",
    ]:

        counts = (
            mapping[
                f"{side}_relation"
            ]
            .value_counts(
                dropna=False
            )
        )

        for relation, n in counts.items():

            x = mapping[
                mapping[
                    f"{side}_relation"
                ]
                == relation
            ]

            row = {
                "side": side.upper(),
                "relation": relation,
                "N": int(n),
                "pct_rows": (
                    float(n)
                    / len(mapping)
                ),
            }

            if (
                relation
                == "CURRENT_TRUE_TREND"
            ):

                row.update(
                    {
                        "realized_from_T_mean": (
                            x[
                                f"{side}_current_realized_from_T_pct"
                            ].mean()
                        ),
                        "realized_from_T_median": (
                            x[
                                f"{side}_current_realized_from_T_pct"
                            ].median()
                        ),
                        "realized_from_T_positive_rate": (
                            (
                                x[
                                    f"{side}_current_realized_from_T_pct"
                                ]
                                > 0
                            ).mean()
                        ),
                        "mfe_from_T_mean": (
                            x[
                                f"{side}_current_mfe_from_T_pct"
                            ].mean()
                        ),
                        "mae_from_T_mean": (
                            x[
                                f"{side}_current_mae_from_T_pct"
                            ].mean()
                        ),
                        "weeks_to_endpoint_mean": (
                            x[
                                f"{side}_weeks_to_current_endpoint"
                            ].mean()
                        ),
                        "weeks_to_endpoint_median": (
                            x[
                                f"{side}_weeks_to_current_endpoint"
                            ].median()
                        ),
                    }
                )

            if (
                relation
                == "NEXT_TRUE_TREND"
            ):

                row.update(
                    {
                        "weeks_to_next_start_mean": (
                            x[
                                f"{side}_weeks_to_next_true_start"
                            ].mean()
                        ),
                        "weeks_to_next_start_median": (
                            x[
                                f"{side}_weeks_to_next_true_start"
                            ].median()
                        ),
                        "weeks_to_next_endpoint_mean": (
                            x[
                                f"{side}_weeks_to_next_true_endpoint"
                            ].mean()
                        ),
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def make_phase_summary(
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for phase, g in mapping.groupby(
        "PHASE",
        sort=True,
    ):

        for side in [
            "buy",
            "sell",
        ]:

            counts = (
                g[
                    f"{side}_relation"
                ]
                .value_counts(
                    dropna=False
                )
            )

            for relation, n in counts.items():

                rows.append(
                    {
                        "PHASE": phase,
                        "side": side.upper(),
                        "relation": relation,
                        "N": int(n),
                        "pct_within_phase": (
                            float(n)
                            / len(g)
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# AUDIT
# =============================================================================

def structural_audit(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    add_check(
        rows,
        "source_rows",
        len(source),
        141886,
        len(source) == 141886,
    )

    add_check(
        rows,
        "mapping_rows",
        len(mapping),
        len(source),
        len(mapping) == len(source),
    )

    add_check(
        rows,
        "source_tickers",
        source[
            "Ticker"
        ].nunique(),
        200,
        source[
            "Ticker"
        ].nunique()
        == 200,
    )

    add_check(
        rows,
        "mapping_tickers",
        mapping[
            "Ticker"
        ].nunique(),
        200,
        mapping[
            "Ticker"
        ].nunique()
        == 200,
    )

    dup = int(
        mapping.duplicated(
            ["Ticker", "Date"]
        ).sum()
    )

    add_check(
        rows,
        "mapping_key_duplicates",
        dup,
        0,
        dup == 0,
    )

    valid_relations = {
        "CURRENT_TRUE_TREND",
        "CURRENT_RETRACEMENT",
        "AFTER_ECONOMIC_ENDPOINT",
        "NEXT_TRUE_TREND",
        "NO_COMPLETED_TRUE_TREND_AHEAD",
    }

    for side in [
        "buy",
        "sell",
    ]:

        missing = int(
            mapping[
                f"{side}_relation"
            ]
            .isna()
            .sum()
        )

        blank = int(
            (
                mapping[
                    f"{side}_relation"
                ]
                .astype(str)
                .str.len()
                == 0
            ).sum()
        )

        bad_relation = int(
            (
                ~mapping[
                    f"{side}_relation"
                ]
                .isin(
                    valid_relations
                )
            ).sum()
        )

        add_check(
            rows,
            f"{side}_missing_relation",
            missing,
            0,
            missing == 0,
        )

        add_check(
            rows,
            f"{side}_blank_relation",
            blank,
            0,
            blank == 0,
        )

        add_check(
            rows,
            f"{side}_invalid_relation",
            bad_relation,
            0,
            bad_relation == 0,
        )

        current = mapping[
            mapping[
                f"{side}_relation"
            ]
            == "CURRENT_TRUE_TREND"
        ]

        bad_endpoint = int(
            (
                pd.to_datetime(
                    current[
                        f"{side}_current_endpoint_date"
                    ]
                )
                <
                current["Date"]
            ).sum()
        )

        add_check(
            rows,
            f"{side}_current_endpoint_before_T",
            bad_endpoint,
            0,
            bad_endpoint == 0,
        )

        negative_mfe = int(
            (
                current[
                    f"{side}_current_mfe_from_T_pct"
                ]
                < 0
            ).sum()
        )

        negative_mae = int(
            (
                current[
                    f"{side}_current_mae_from_T_pct"
                ]
                < 0
            ).sum()
        )

        add_check(
            rows,
            f"{side}_negative_current_mfe",
            negative_mfe,
            0,
            negative_mfe == 0,
        )

        add_check(
            rows,
            f"{side}_negative_current_mae",
            negative_mae,
            0,
            negative_mae == 0,
        )

        nxt = mapping[
            mapping[
                f"{side}_relation"
            ]
            == "NEXT_TRUE_TREND"
        ]

        bad_next = int(
            (
                pd.to_datetime(
                    nxt[
                        f"{side}_next_true_start_date"
                    ]
                )
                <=
                nxt["Date"]
            ).sum()
        )

        add_check(
            rows,
            f"{side}_next_true_not_future",
            bad_next,
            0,
            bad_next == 0,
        )

        bad_next_weeks = int(
            (
                nxt[
                    f"{side}_weeks_to_next_true_start"
                ]
                <= 0
            ).sum()
        )

        add_check(
            rows,
            f"{side}_next_true_nonpositive_distance",
            bad_next_weeks,
            0,
            bad_next_weeks == 0,
        )

        after = mapping[
            mapping[
                f"{side}_relation"
            ]
            == "AFTER_ECONOMIC_ENDPOINT"
        ]

        bad_after = int(
            (
                after[
                    f"{side}_weeks_since_economic_endpoint"
                ]
                <= 0
            ).sum()
        )

        add_check(
            rows,
            f"{side}_after_endpoint_nonpositive_distance",
            bad_after,
            0,
            bad_after == 0,
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "MARKETSENTINEL V40.63"
    )

    print(
        "OPPORTUNITY CONTINUOUS MAPPING AUDIT"
    )

    print(
        "=" * 80
    )

    (
        source,
        branches,
        true_trends,
    ) = load_data()

    print(
        f"\nSOURCE ROWS: "
        f"{len(source):,}"
    )

    print(
        f"TICKERS: "
        f"{source['Ticker'].nunique():,}"
    )

    print(
        "DATE:",
        source["Date"].min().date(),
        "->",
        source["Date"].max().date(),
    )

    print(
        f"\nV40.62 BRANCHES: "
        f"{len(branches):,}"
    )

    print(
        f"V40.62 TRUE TREND OUTCOMES: "
        f"{len(true_trends):,}"
    )

    all_rows = []

    source_groups = {
        ticker: g
        for ticker, g in source.groupby(
            "Ticker",
            sort=True,
        )
    }

    branch_groups = {
        ticker: g
        for ticker, g in branches.groupby(
            "Ticker",
            sort=True,
        )
    }

    true_groups = {
        ticker: g
        for ticker, g in true_trends.groupby(
            "Ticker",
            sort=True,
        )
    }

    tickers = sorted(
        source_groups.keys()
    )

    total = len(
        tickers
    )

    print(
        "\nBuilding BUY + SELL "
        "point-in-time mappings..."
    )

    for k, ticker in enumerate(
        tickers,
        start=1,
    ):

        g = source_groups[
            ticker
        ]

        b = branch_groups.get(
            ticker,
            branches.iloc[0:0].copy(),
        )

        tt = true_groups.get(
            ticker,
            true_trends.iloc[0:0].copy(),
        )

        x = process_ticker(
            g,
            b,
            tt,
        )

        all_rows.append(
            x
        )

        if (
            k % 20 == 0
            or k == total
        ):
            print(
                f"  {k:3d}/{total} tickers"
            )

    mapping = pd.concat(
        all_rows,
        ignore_index=True,
    )

    mapping = (
        mapping
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    print(
        "\nRunning structural audit..."
    )

    audit = structural_audit(
        source,
        mapping,
    )

    summary = make_summary(
        mapping
    )

    phase_summary = (
        make_phase_summary(
            mapping
        )
    )

    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------

    mapping.to_csv(
        MAPPING_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    phase_summary.to_csv(
        PHASE_SUMMARY_FILE,
        index=False,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    all_pass = bool(
        (
            audit["status"]
            == "PASS"
        ).all()
    )

    metadata = {
        "version": VERSION,
        "source": str(
            SOURCE_FILE
        ),
        "v40_62_branch_source": str(
            BRANCH_FILE
        ),
        "v40_62_true_trend_source": str(
            TRUE_TREND_FILE
        ),
        "rows": int(
            len(mapping)
        ),
        "tickers": int(
            mapping[
                "Ticker"
            ].nunique()
        ),
        "date_min": str(
            mapping[
                "Date"
            ].min().date()
        ),
        "date_max": str(
            mapping[
                "Date"
            ].max().date()
        ),
        "buy_and_sell_every_T": True,
        "fixed_horizon_target": False,
        "final_y_created": False,
        "ml_trained": False,
        "score_created": False,
        "weights_created": False,
        "phase_modified": False,
        "reversal_modified": False,
        "strength_modified": False,
        "v40_62_modified": False,
        "next_true_economics_assigned_to_T": False,
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

    # -----------------------------------------------------------------
    # REPORT
    # -----------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "GLOBAL MAPPING SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RELATION COUNTS"
    )

    print(
        "=" * 80
    )

    for side in [
        "buy",
        "sell",
    ]:

        print(
            f"\n{side.upper()}"
        )

        print(
            mapping[
                f"{side}_relation"
            ]
            .value_counts()
            .to_string()
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
        audit.to_string(
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
            "V40.63 CONTINUOUS MAPPING "
            "STRUCTURALLY VALID"
        )

    else:

        print(
            "FINAL RESULT: FAIL"
        )

        print(
            "DO NOT DESIGN FINAL "
            "CONTINUOUS TARGET Y YET"
        )

    print(
        "=" * 80
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  {MAPPING_FILE}"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print(
        f"  {PHASE_SUMMARY_FILE}"
    )

    print(
        f"  {AUDIT_FILE}"
    )

    print(
        f"  {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()