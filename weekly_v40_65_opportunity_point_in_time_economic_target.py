#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MarketSentinel
V40.65 - Opportunity Point-in-Time Economic Target

PURPOSE
-------
Build the point-in-time economic outcome dataset that will become the
historical target base for the final Opportunity model.

This script DOES NOT:
- train ML models;
- create an Opportunity score;
- define score weights;
- modify PHASE;
- modify REVERSAL;
- modify STRENGTH;
- modify V40.62 or V40.63;
- use future K3 information as a model feature.

The task here is exclusively historical target construction.

CORE QUESTION
-------------
For every ticker, every weekly observation T, and independently for BUY
and SELL:

    "If I enter in this direction at T, what is the economic outcome
     of the structurally relevant opportunity from T forward?"

The entry price is ALWAYS Close(T).

TARGET ENDPOINT RULES
---------------------
1. CURRENT_TRUE_TREND
   Endpoint = frozen V40.61/V40.62 economic endpoint of the current
   completed true trend.

2. CURRENT_RETRACEMENT
   Endpoint = end of the failed N8/K3 SAR run.
   No artificial zero or arbitrary penalty is assigned.

3. NEXT_TRUE_TREND
   Endpoint = economic endpoint of the next completed true trend.
   Economics are measured FROM CURRENT T, not from future T0.
   Therefore the path includes all movement between T and future T0.

4. AFTER_ECONOMIC_ENDPOINT
   The previous endpoint is already in the past and cannot be used as
   a prospective target.
   Endpoint = economic endpoint of the next completed true trend,
   when available.

5. NO_COMPLETED_TRUE_TREND_AHEAD
   There is no fully observed future true-trend endpoint in the
   historical sample.
   Target status = RIGHT_CENSORED.
   No artificial economic outcome is created.

IMPORTANT
---------
The future structural classification is used only to construct the
historical supervised target. It is NOT a causal model input.

Economic variables are calculated from T to target endpoint:
- directional_realized_pct
- mfe_pct
- mae_pct
- weeks_to_endpoint

For BUY:
    realized = endpoint / entry - 1
    MFE      = max(path / entry - 1)
    MAE      = max(1 - path / entry)

For SELL:
    realized = entry / endpoint - 1
    MFE      = max(1 - path / entry)
    MAE      = max(path / entry - 1)

All percentages are multiplied by 100.

V40.65 intentionally DOES NOT collapse these economic components into
one Y. That combination belongs to the subsequent model-building stage
and must be empirically justified rather than assigned arbitrary weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

VERSION = "V40.65"

SOURCE_FILE = Path(
    "data/v40_39_full200_features.csv"
)

MAPPING_FILE = Path(
    "data/v40_63_opportunity_continuous_mapping_audit/"
    "v40_63_continuous_mapping.csv"
)

OUTDIR = Path(
    "data/v40_65_opportunity_point_in_time_economic_target"
)

TARGET_FILE = OUTDIR / "v40_65_point_in_time_economic_target.csv"
SUMMARY_FILE = OUTDIR / "v40_65_target_summary.csv"
RELATION_FILE = OUTDIR / "v40_65_relation_summary.csv"
AUDIT_FILE = OUTDIR / "v40_65_structural_audit.csv"
METADATA_FILE = OUTDIR / "v40_65_metadata.json"

VALID_RELATIONS = {
    "CURRENT_TRUE_TREND",
    "CURRENT_RETRACEMENT",
    "AFTER_ECONOMIC_ENDPOINT",
    "NEXT_TRUE_TREND",
    "NO_COMPLETED_TRUE_TREND_AHEAD",
}

OBSERVED_STATUS = "OBSERVED"
CENSORED_STATUS = "RIGHT_CENSORED"


# =============================================================================
# HELPERS
# =============================================================================

def add_check(
    rows: list[dict],
    check: str,
    passed: bool,
    value,
    expected,
) -> None:
    rows.append(
        {
            "check": check,
            "passed": bool(passed),
            "value": value,
            "expected": expected,
        }
    )


def directional_economics(
    close: np.ndarray,
    start_idx: int,
    end_idx: int,
    side: str,
) -> tuple[float, float, float, int]:
    """
    Calculate prospective economics from T=start_idx through endpoint=end_idx.

    Both endpoints are inclusive.
    """

    if end_idx < start_idx:
        raise RuntimeError(
            f"Endpoint before T: start={start_idx}, end={end_idx}"
        )

    entry = float(close[start_idx])
    endpoint = float(close[end_idx])

    if (
        not np.isfinite(entry)
        or not np.isfinite(endpoint)
        or entry <= 0.0
        or endpoint <= 0.0
    ):
        raise RuntimeError(
            "Invalid entry/endpoint Close encountered."
        )

    path = close[start_idx : end_idx + 1].astype(float)

    if len(path) == 0:
        raise RuntimeError("Empty economic path.")

    if not np.isfinite(path).all():
        raise RuntimeError(
            "Non-finite Close inside economic path."
        )

    if side == "BUY":
        realized = (
            endpoint / entry - 1.0
        ) * 100.0

        mfe = np.max(
            (path - entry) / entry
        ) * 100.0

        mae = np.max(
            (entry - path) / entry
        ) * 100.0

    elif side == "SELL":
        realized = (
            entry / endpoint - 1.0
        ) * 100.0

        mfe = np.max(
            (entry - path) / entry
        ) * 100.0

        mae = np.max(
            (path - entry) / entry
        ) * 100.0

    else:
        raise ValueError(
            f"Unknown side: {side}"
        )

    return (
        float(realized),
        max(float(mfe), 0.0),
        max(float(mae), 0.0),
        int(end_idx - start_idx),
    )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 80)
    print(f"{VERSION} - OPPORTUNITY POINT-IN-TIME ECONOMIC TARGET")
    print("=" * 80)

    print("\nLoading V40.39 source...")

    source_cols = [
        "Ticker",
        "Date",
        "Close",
        "PHASE",
        "V4012_PHASE_AGE",
        "V4012_PHASE_DIRECTION",
    ]

    source = pd.read_csv(
        SOURCE_FILE,
        usecols=source_cols,
    )

    source["Date"] = pd.to_datetime(
        source["Date"],
        errors="raise",
    )

    source["Ticker"] = source["Ticker"].astype(str)

    source = (
        source
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    print(
        f"  rows={len(source):,} "
        f"tickers={source['Ticker'].nunique():,}"
    )

    print("\nLoading frozen V40.63 mapping...")

    mapping = pd.read_csv(
        MAPPING_FILE,
        low_memory=False,
    )

    mapping["Date"] = pd.to_datetime(
        mapping["Date"],
        errors="raise",
    )

    mapping["Ticker"] = mapping["Ticker"].astype(str)

    date_cols = [
        c
        for c in mapping.columns
        if c.endswith("_date")
    ]

    for c in date_cols:
        mapping[c] = pd.to_datetime(
            mapping[c],
            errors="coerce",
        )

    mapping = (
        mapping
        .sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )

    print(
        f"  rows={len(mapping):,} "
        f"tickers={mapping['Ticker'].nunique():,}"
    )

    if len(source) != len(mapping):
        raise RuntimeError(
            "Source/mapping row-count mismatch."
        )

    source_keys = source[
        ["Ticker", "Date"]
    ].reset_index(drop=True)

    mapping_keys = mapping[
        ["Ticker", "Date"]
    ].reset_index(drop=True)

    if not source_keys.equals(mapping_keys):
        raise RuntimeError(
            "Source/mapping Ticker-Date keys are not identical."
        )

    return source, mapping


# =============================================================================
# SIDE TARGET
# =============================================================================

def build_side_target(
    g: pd.DataFrame,
    m: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    """
    Build one complete point-in-time target for one ticker and one side.
    """

    if side not in {"BUY", "SELL"}:
        raise ValueError(side)

    prefix = (
        "buy"
        if side == "BUY"
        else "sell"
    )

    n = len(g)

    dates = g["Date"].to_numpy(
        dtype="datetime64[ns]"
    )

    close = g["Close"].to_numpy(
        dtype=float
    )

    relation = (
        m[f"{prefix}_relation"]
        .astype(str)
        .to_numpy()
    )

    target_status = np.full(
        n,
        "",
        dtype=object,
    )

    endpoint_source = np.full(
        n,
        "",
        dtype=object,
    )

    target_endpoint_date = np.full(
        n,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    target_true_start_date = np.full(
        n,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    target_true_class = np.full(
        n,
        "",
        dtype=object,
    )

    directional_realized = np.full(
        n,
        np.nan,
        dtype=float,
    )

    mfe = np.full(
        n,
        np.nan,
        dtype=float,
    )

    mae = np.full(
        n,
        np.nan,
        dtype=float,
    )

    weeks_to_endpoint = np.full(
        n,
        np.nan,
        dtype=float,
    )

    weeks_to_true_start = np.full(
        n,
        np.nan,
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Exact date -> positional index lookup for this ticker.
    # -------------------------------------------------------------------------

    date_to_idx = {
        pd.Timestamp(d): i
        for i, d in enumerate(g["Date"])
    }

    # -------------------------------------------------------------------------
    # Row-by-row target assignment.
    #
    # This is intentionally explicit. 141k x 2 rows is small enough, and the
    # clarity of the structural rules is more important than clever vectorized
    # code at this stage.
    # -------------------------------------------------------------------------

    for i in range(n):

        rel = relation[i]

        if rel not in VALID_RELATIONS:
            raise RuntimeError(
                f"Invalid V40.63 relation "
                f"{g.iloc[i]['Ticker']} "
                f"{g.iloc[i]['Date']} "
                f"{side}: {rel}"
            )

        endpoint = pd.NaT
        true_start = pd.NaT
        true_class = ""

        # ---------------------------------------------------------------------
        # 1. Current completed true trend.
        # ---------------------------------------------------------------------

        if rel == "CURRENT_TRUE_TREND":

            endpoint = m.iloc[i][
                f"{prefix}_current_endpoint_date"
            ]

            true_start = m.iloc[i][
                f"{prefix}_current_start_date"
            ]

            raw_class = m.iloc[i][
                f"{prefix}_current_class"
            ]

            true_class = (
                ""
                if pd.isna(raw_class)
                else str(raw_class)
            )

            endpoint_source[i] = (
                "CURRENT_TRUE_ECONOMIC_ENDPOINT"
            )

        # ---------------------------------------------------------------------
        # 2. Current N8/K3 failed retracement.
        # ---------------------------------------------------------------------

        elif rel == "CURRENT_RETRACEMENT":

            endpoint = m.iloc[i][
                f"{prefix}_retracement_end_date"
            ]

            endpoint_source[i] = (
                "RETRACEMENT_SAR_RUN_END"
            )

        # ---------------------------------------------------------------------
        # 3. Next true trend.
        #
        # This includes both ordinary NEXT_TRUE_TREND and the period after the
        # economic endpoint of an old same-side trend.
        #
        # V40.63 stores next-true fields for rows left unmapped after the first
        # three structural passes. For AFTER_ECONOMIC_ENDPOINT rows those fields
        # are not populated. Therefore we find the next completed true trend
        # causally in historical target construction by searching the ticker's
        # V40.63 next-true mapping forward.
        # ---------------------------------------------------------------------

        elif rel in {
            "NEXT_TRUE_TREND",
            "AFTER_ECONOMIC_ENDPOINT",
        }:

            if rel == "NEXT_TRUE_TREND":

                true_start = m.iloc[i][
                    f"{prefix}_next_true_start_date"
                ]

                endpoint = m.iloc[i][
                    f"{prefix}_next_true_endpoint_date"
                ]

                raw_class = m.iloc[i][
                    f"{prefix}_next_true_class"
                ]

                true_class = (
                    ""
                    if pd.isna(raw_class)
                    else str(raw_class)
                )

            else:
                # AFTER_ECONOMIC_ENDPOINT:
                # find the first later row whose next true trend is populated.
                future = m.iloc[i + 1 :]

                valid = future[
                    future[
                        f"{prefix}_next_true_start_date"
                    ].notna()
                    & (
                        future[
                            f"{prefix}_next_true_start_date"
                        ]
                        > g.iloc[i]["Date"]
                    )
                ]

                if not valid.empty:

                    nr = valid.iloc[0]

                    true_start = nr[
                        f"{prefix}_next_true_start_date"
                    ]

                    endpoint = nr[
                        f"{prefix}_next_true_endpoint_date"
                    ]

                    raw_class = nr[
                        f"{prefix}_next_true_class"
                    ]

                    true_class = (
                        ""
                        if pd.isna(raw_class)
                        else str(raw_class)
                    )

            endpoint_source[i] = (
                "NEXT_TRUE_ECONOMIC_ENDPOINT"
            )

        # ---------------------------------------------------------------------
        # 4. No completed true trend ahead.
        # ---------------------------------------------------------------------

        elif rel == "NO_COMPLETED_TRUE_TREND_AHEAD":

            target_status[i] = CENSORED_STATUS
            endpoint_source[i] = (
                "NO_COMPLETED_TRUE_TREND_AHEAD"
            )
            continue

        # ---------------------------------------------------------------------
        # Missing endpoint => historical right censoring.
        # ---------------------------------------------------------------------

        if pd.isna(endpoint):

            target_status[i] = CENSORED_STATUS
            endpoint_source[i] = (
                endpoint_source[i]
                or "MISSING_FUTURE_ENDPOINT"
            )
            continue

        endpoint = pd.Timestamp(endpoint)

        if endpoint not in date_to_idx:
            raise RuntimeError(
                f"Endpoint date not found "
                f"{g.iloc[i]['Ticker']} "
                f"{g.iloc[i]['Date']} "
                f"{side}: {endpoint}"
            )

        e = int(date_to_idx[endpoint])

        if e < i:
            raise RuntimeError(
                f"Prospective endpoint before T "
                f"{g.iloc[i]['Ticker']} "
                f"{g.iloc[i]['Date']} "
                f"{side}: endpoint={endpoint}"
            )

        # ---------------------------------------------------------------------
        # Economic path from CURRENT T.
        # ---------------------------------------------------------------------

        realized, this_mfe, this_mae, weeks = (
            directional_economics(
                close=close,
                start_idx=i,
                end_idx=e,
                side=side,
            )
        )

        target_status[i] = OBSERVED_STATUS

        target_endpoint_date[i] = (
            np.datetime64(endpoint)
        )

        directional_realized[i] = realized
        mfe[i] = this_mfe
        mae[i] = this_mae
        weeks_to_endpoint[i] = weeks

        if not pd.isna(true_start):

            true_start = pd.Timestamp(true_start)

            target_true_start_date[i] = (
                np.datetime64(true_start)
            )

            if true_start in date_to_idx:
                s = int(
                    date_to_idx[true_start]
                )
                weeks_to_true_start[i] = (
                    s - i
                )

        target_true_class[i] = true_class

    return pd.DataFrame(
        {
            f"{prefix}_target_status":
                target_status,

            f"{prefix}_endpoint_source":
                endpoint_source,

            f"{prefix}_target_endpoint_date":
                target_endpoint_date,

            f"{prefix}_target_true_start_date":
                target_true_start_date,

            f"{prefix}_target_true_class":
                target_true_class,

            f"{prefix}_directional_realized_pct":
                directional_realized,

            f"{prefix}_mfe_pct":
                mfe,

            f"{prefix}_mae_pct":
                mae,

            f"{prefix}_weeks_to_endpoint":
                weeks_to_endpoint,

            f"{prefix}_weeks_to_true_start":
                weeks_to_true_start,
        }
    )


# =============================================================================
# PROCESS ALL TICKERS
# =============================================================================

def build_targets(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    output_parts = []

    tickers = (
        source["Ticker"]
        .drop_duplicates()
        .tolist()
    )

    total = len(tickers)

    print("\nBuilding point-in-time economic targets...")

    for k, ticker in enumerate(
        tickers,
        start=1,
    ):

        g = (
            source[
                source["Ticker"] == ticker
            ]
            .copy()
            .reset_index(drop=True)
        )

        m = (
            mapping[
                mapping["Ticker"] == ticker
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(g) != len(m):
            raise RuntimeError(
                f"Ticker row mismatch: {ticker}"
            )

        if not g[
            ["Ticker", "Date"]
        ].equals(
            m[
                ["Ticker", "Date"]
            ]
        ):
            raise RuntimeError(
                f"Ticker-Date mismatch: {ticker}"
            )

        buy = build_side_target(
            g=g,
            m=m,
            side="BUY",
        )

        sell = build_side_target(
            g=g,
            m=m,
            side="SELL",
        )

        base = g[
            [
                "Ticker",
                "Date",
                "Close",
                "PHASE",
                "V4012_PHASE_AGE",
                "V4012_PHASE_DIRECTION",
            ]
        ].copy()

        base["buy_relation"] = (
            m["buy_relation"].to_numpy()
        )

        base["sell_relation"] = (
            m["sell_relation"].to_numpy()
        )

        part = pd.concat(
            [
                base.reset_index(drop=True),
                buy.reset_index(drop=True),
                sell.reset_index(drop=True),
            ],
            axis=1,
        )

        output_parts.append(part)

        if (
            k == 1
            or k % 20 == 0
            or k == total
        ):
            print(
                f"  ticker {k:3d}/{total}: "
                f"{ticker}"
            )

    out = pd.concat(
        output_parts,
        ignore_index=True,
    )

    return out


# =============================================================================
# SUMMARIES
# =============================================================================

def make_summary(
    target: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for side in ["buy", "sell"]:

        observed = target[
            target[
                f"{side}_target_status"
            ]
            == OBSERVED_STATUS
        ].copy()

        for relation, g in observed.groupby(
            f"{side}_relation",
            dropna=False,
        ):

            realized = g[
                f"{side}_directional_realized_pct"
            ]

            mfe = g[
                f"{side}_mfe_pct"
            ]

            mae = g[
                f"{side}_mae_pct"
            ]

            weeks = g[
                f"{side}_weeks_to_endpoint"
            ]

            rows.append(
                {
                    "side": side.upper(),
                    "relation": relation,
                    "n": int(len(g)),
                    "realized_mean_pct":
                        float(realized.mean()),
                    "realized_median_pct":
                        float(realized.median()),
                    "realized_positive_pct":
                        float(
                            (realized > 0.0).mean()
                            * 100.0
                        ),
                    "mfe_mean_pct":
                        float(mfe.mean()),
                    "mfe_median_pct":
                        float(mfe.median()),
                    "mae_mean_pct":
                        float(mae.mean()),
                    "mae_median_pct":
                        float(mae.median()),
                    "weeks_mean":
                        float(weeks.mean()),
                    "weeks_median":
                        float(weeks.median()),
                }
            )

    return pd.DataFrame(rows)


def make_relation_summary(
    target: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for side in ["buy", "sell"]:

        counts = (
            target.groupby(
                [
                    f"{side}_relation",
                    f"{side}_target_status",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="n")
        )

        counts.insert(
            0,
            "side",
            side.upper(),
        )

        counts = counts.rename(
            columns={
                f"{side}_relation":
                    "relation",
                f"{side}_target_status":
                    "target_status",
            }
        )

        rows.append(counts)

    return pd.concat(
        rows,
        ignore_index=True,
    )


# =============================================================================
# STRUCTURAL AUDIT
# =============================================================================

def structural_audit(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    add_check(
        rows,
        "source_rows",
        len(source) == 141886,
        len(source),
        141886,
    )

    add_check(
        rows,
        "source_tickers",
        source["Ticker"].nunique() == 200,
        source["Ticker"].nunique(),
        200,
    )

    add_check(
        rows,
        "mapping_rows_equal_source",
        len(mapping) == len(source),
        len(mapping),
        len(source),
    )

    add_check(
        rows,
        "target_rows_equal_source",
        len(target) == len(source),
        len(target),
        len(source),
    )

    dup = int(
        target.duplicated(
            ["Ticker", "Date"]
        ).sum()
    )

    add_check(
        rows,
        "target_ticker_date_duplicates",
        dup == 0,
        dup,
        0,
    )

    for side in ["buy", "sell"]:

        rel_col = f"{side}_relation"
        status_col = (
            f"{side}_target_status"
        )

        endpoint_col = (
            f"{side}_target_endpoint_date"
        )

        realized_col = (
            f"{side}_directional_realized_pct"
        )

        mfe_col = f"{side}_mfe_pct"
        mae_col = f"{side}_mae_pct"
        weeks_col = (
            f"{side}_weeks_to_endpoint"
        )

        invalid_rel = int(
            (
                ~target[rel_col]
                .isin(VALID_RELATIONS)
            ).sum()
        )

        add_check(
            rows,
            f"{side}_invalid_relations",
            invalid_rel == 0,
            invalid_rel,
            0,
        )

        invalid_status = int(
            (
                ~target[status_col]
                .isin(
                    {
                        OBSERVED_STATUS,
                        CENSORED_STATUS,
                    }
                )
            ).sum()
        )

        add_check(
            rows,
            f"{side}_invalid_target_status",
            invalid_status == 0,
            invalid_status,
            0,
        )

        observed = (
            target[status_col]
            == OBSERVED_STATUS
        )

        censored = (
            target[status_col]
            == CENSORED_STATUS
        )

        observed_missing_endpoint = int(
            (
                observed
                & target[endpoint_col].isna()
            ).sum()
        )

        add_check(
            rows,
            f"{side}_observed_missing_endpoint",
            observed_missing_endpoint == 0,
            observed_missing_endpoint,
            0,
        )

        observed_missing_economics = int(
            (
                observed
                & (
                    target[
                        [
                            realized_col,
                            mfe_col,
                            mae_col,
                            weeks_col,
                        ]
                    ]
                    .isna()
                    .any(axis=1)
                )
            ).sum()
        )

        add_check(
            rows,
            f"{side}_observed_missing_economics",
            observed_missing_economics == 0,
            observed_missing_economics,
            0,
        )

        censored_with_economics = int(
            (
                censored
                & (
                    target[
                        [
                            realized_col,
                            mfe_col,
                            mae_col,
                            weeks_col,
                        ]
                    ]
                    .notna()
                    .any(axis=1)
                )
            ).sum()
        )

        add_check(
            rows,
            f"{side}_censored_with_economics",
            censored_with_economics == 0,
            censored_with_economics,
            0,
        )

        negative_mfe = int(
            (
                observed
                & (
                    target[mfe_col]
                    < -1e-12
                )
            ).sum()
        )

        negative_mae = int(
            (
                observed
                & (
                    target[mae_col]
                    < -1e-12
                )
            ).sum()
        )

        negative_weeks = int(
            (
                observed
                & (
                    target[weeks_col]
                    < 0
                )
            ).sum()
        )

        add_check(
            rows,
            f"{side}_negative_mfe",
            negative_mfe == 0,
            negative_mfe,
            0,
        )

        add_check(
            rows,
            f"{side}_negative_mae",
            negative_mae == 0,
            negative_mae,
            0,
        )

        add_check(
            rows,
            f"{side}_negative_weeks",
            negative_weeks == 0,
            negative_weeks,
            0,
        )

        # -------------------------------------------------------------
        # Exact equivalence check for CURRENT_TRUE_TREND.
        #
        # V40.65 must reproduce V40.63 economics exactly for this
        # already-frozen case.
        # -------------------------------------------------------------

        current_true = (
            target[rel_col]
            == "CURRENT_TRUE_TREND"
        )

        for metric65, metric63 in [
            (
                realized_col,
                f"{side}_current_realized_from_T_pct",
            ),
            (
                mfe_col,
                f"{side}_current_mfe_from_T_pct",
            ),
            (
                mae_col,
                f"{side}_current_mae_from_T_pct",
            ),
            (
                weeks_col,
                f"{side}_weeks_to_current_endpoint",
            ),
        ]:

            a = target.loc[
                current_true,
                metric65,
            ].to_numpy(dtype=float)

            b = mapping.loc[
                current_true,
                metric63,
            ].to_numpy(dtype=float)

            if len(a) == 0:
                max_diff = np.nan
                passed = False
            else:
                diff = np.abs(a - b)
                max_diff = float(
                    np.nanmax(diff)
                )
                passed = bool(
                    np.allclose(
                        a,
                        b,
                        rtol=1e-10,
                        atol=1e-10,
                        equal_nan=True,
                    )
                )

            add_check(
                rows,
                (
                    f"{side}_current_true_exact_"
                    f"{metric65}"
                ),
                passed,
                max_diff,
                0.0,
            )

        # -------------------------------------------------------------
        # CURRENT_RETRACEMENT observed rows must terminate at the
        # frozen V40.63 retracement end date.
        # -------------------------------------------------------------

        retr = (
            target[rel_col]
            == "CURRENT_RETRACEMENT"
        )

        if retr.any():

            a = pd.to_datetime(
                target.loc[
                    retr,
                    endpoint_col,
                ]
            ).reset_index(drop=True)

            b = pd.to_datetime(
                mapping.loc[
                    retr,
                    f"{side}_retracement_end_date",
                ]
            ).reset_index(drop=True)

            mismatch = int(
                (
                    ~(
                        (a == b)
                        | (a.isna() & b.isna())
                    )
                ).sum()
            )

            add_check(
                rows,
                (
                    f"{side}_retracement_"
                    "endpoint_exact"
                ),
                mismatch == 0,
                mismatch,
                0,
            )

        # -------------------------------------------------------------
        # NEXT_TRUE_TREND exact endpoint equivalence.
        # -------------------------------------------------------------

        nxt = (
            target[rel_col]
            == "NEXT_TRUE_TREND"
        )

        if nxt.any():

            a = pd.to_datetime(
                target.loc[
                    nxt,
                    endpoint_col,
                ]
            ).reset_index(drop=True)

            b = pd.to_datetime(
                mapping.loc[
                    nxt,
                    f"{side}_next_true_endpoint_date",
                ]
            ).reset_index(drop=True)

            mismatch = int(
                (
                    ~(
                        (a == b)
                        | (a.isna() & b.isna())
                    )
                ).sum()
            )

            add_check(
                rows,
                (
                    f"{side}_next_true_"
                    "endpoint_exact"
                ),
                mismatch == 0,
                mismatch,
                0,
            )

        # -------------------------------------------------------------
        # NO_COMPLETED_TRUE_TREND_AHEAD must be censored.
        # -------------------------------------------------------------

        no_ahead = (
            target[rel_col]
            == "NO_COMPLETED_TRUE_TREND_AHEAD"
        )

        bad_no_ahead = int(
            (
                target.loc[
                    no_ahead,
                    status_col,
                ]
                != CENSORED_STATUS
            ).sum()
        )

        add_check(
            rows,
            (
                f"{side}_no_completed_"
                "true_ahead_censored"
            ),
            bad_no_ahead == 0,
            bad_no_ahead,
            0,
        )

    audit = pd.DataFrame(rows)

    return audit


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source, mapping = load_data()

    target = build_targets(
        source=source,
        mapping=mapping,
    )

    print("\nRunning structural audit...")

    audit = structural_audit(
        source=source,
        mapping=mapping,
        target=target,
    )

    failed = audit[
        ~audit["passed"]
    ]

    print(
        audit.to_string(
            index=False
        )
    )

    if not failed.empty:
        print("\nFAILED CHECKS:")
        print(
            failed.to_string(
                index=False
            )
        )
        raise RuntimeError(
            "V40.65 STRUCTURAL AUDIT FAILED."
        )

    summary = make_summary(target)

    relation_summary = (
        make_relation_summary(target)
    )

    # -----------------------------------------------------------------
    # Save only after structural PASS.
    # -----------------------------------------------------------------

    print("\nSaving outputs...")

    target.to_csv(
        TARGET_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    relation_summary.to_csv(
        RELATION_FILE,
        index=False,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    metadata = {
        "version": VERSION,
        "source_file": str(SOURCE_FILE),
        "mapping_file": str(MAPPING_FILE),
        "rows": int(len(target)),
        "tickers": int(
            target["Ticker"].nunique()
        ),
        "date_min": str(
            target["Date"].min().date()
        ),
        "date_max": str(
            target["Date"].max().date()
        ),
        "target_scope":
            "POINT_IN_TIME_BUY_AND_SELL",
        "entry_price":
            "CLOSE_AT_CURRENT_T",
        "current_true_endpoint":
            "FROZEN_ECONOMIC_ENDPOINT",
        "current_retracement_endpoint":
            "FAILED_N8_K3_SAR_RUN_END",
        "next_true_endpoint":
            "NEXT_COMPLETED_TRUE_TREND_ECONOMIC_ENDPOINT",
        "after_endpoint_rule":
            "NEXT_COMPLETED_TRUE_TREND_ECONOMIC_ENDPOINT",
        "no_completed_true_ahead":
            "RIGHT_CENSORED",
        "artificial_zero_or_penalty":
            False,
        "ml_trained":
            False,
        "opportunity_score_created":
            False,
        "weights_defined":
            False,
        "phase_modified":
            False,
        "reversal_modified":
            False,
        "strength_modified":
            False,
        "structural_audit":
            "PASS",
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

    print("\n" + "=" * 80)
    print("TARGET SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("RELATION / CENSORING SUMMARY")
    print("=" * 80)

    print(
        relation_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print("V40.65 STRUCTURAL AUDIT: PASS")
    print(
        f"ROWS: {len(target):,}"
    )
    print(
        f"TICKERS: "
        f"{target['Ticker'].nunique():,}"
    )

    print(
        "\nNo ML model, Opportunity score, "
        "economic weights or score thresholds "
        "have been created."
    )

    print("\nOutputs:")
    for p in [
        TARGET_FILE,
        SUMMARY_FILE,
        RELATION_FILE,
        AUDIT_FILE,
        METADATA_FILE,
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()