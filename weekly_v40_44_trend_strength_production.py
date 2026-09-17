#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MarketSentinel
V40.44 - FULL200 TREND STRENGTH PRODUCTION CANDIDATE
Derived from validated V40.43 methodology.

Semantics:
- PHASE determines direction; Strength measures quality/strength only.
- BULL/BEAR are kept separate; INDECISIONE has no primary Strength.
- AGE buckets are neutral: AGE1, AGE2, AGE3, AGE4_5, AGE6_8, AGE9_PLUS.
- Production representation: COMPACT_ROLLING only.
- No 1-10 score is created here.

Production anti-leakage policy:
- current as-of week is the last week in the FULL200 panel;
- training labels are restricted to episodes ending on/before a 5-market-week
  purge cutoff before the as-of week;
- therefore the current episode cannot train the model that scores itself;
- leave-one-ticker-out is NOT used for final production fitting: it was an OOS
  validation protocol in V40.43, not a production architecture rule.

This script does NOT modify PHASE V40.39, REVERSAL V40.40 or engine.py.
"""
from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
VERSION = "V40.44"
SOURCE_VALIDATION_VERSION = "V40.43"
PHASE_VERSION = "V40.39"
REVERSAL_VERSION = "V40.40"
STRENGTH_TARGET_VERSION = "V40.41"
VARIANT = "COMPACT_ROLLING"

DATA = Path("data")
FEATURE_DATA_FILE = DATA / "v40_39_full200_features.csv"
FEATURE_LIST_FILE = DATA / "v40_39_feature_list.csv"
EPISODE_FILE = DATA / "v40_41_strength_episode_audit" / "v40_41_strength_episodes.csv"
V43_DIR = DATA / "v40_43_trend_strength_oos"
V43_SOURCE_FEATURES = V43_DIR / "v40_43_source_features.csv"

OUT = DATA / "v40_44_trend_strength_production"
OUT_SOURCE_FEATURES = OUT / "v40_44_source_features.csv"
OUT_SELECTED = OUT / "v40_44_selected_features.csv"
OUT_CURRENT = OUT / "v40_44_current_strength.csv"
OUT_TRAIN_AUDIT = OUT / "v40_44_training_audit.csv"
OUT_METADATA = OUT / "v40_44_metadata.json"

CLASS_ORDER = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
CLASS_TO_NUM = {c: i for i, c in enumerate(CLASS_ORDER)}
AGE_BUCKETS = ["AGE1", "AGE2", "AGE3", "AGE4_5", "AGE6_8", "AGE9_PLUS"]

PURGE_WEEKS = 5
MIN_HISTORY_FOR_Z = 20
TOP_MODEL_FEATURES = 120
MIN_TRAIN_ROWS = 160
N_TREES = 320
RANDOM_STATE = 42
N_JOBS = 1
CHUNK_SIZE = 10_000
MIN_SOURCE_FEATURES = 20

# ============================================================
# UTILITIES
# ============================================================
def banner(text):
    print()
    print("=" * 88)
    print(text)
    print("=" * 88)


def age_bucket(age):
    age = int(age)
    if age == 1:
        return "AGE1"
    if age == 2:
        return "AGE2"
    if age == 3:
        return "AGE3"
    if age <= 5:
        return "AGE4_5"
    if age <= 8:
        return "AGE6_8"
    return "AGE9_PLUS"


PHASE_TO_DIRECTION = {
    "BUY": "BULL",
    "TREND_RIALZISTA": "BULL",
    "DEBOLEZZA_RIALZISTA": "BULL",
    "SELL": "BEAR",
    "TREND_RIBASSISTA": "BEAR",
    "DEBOLEZZA_RIBASSISTA": "BEAR",
    "INDECISIONE": None,
}


def phase_to_direction(value):
    if pd.isna(value):
        return None
    return PHASE_TO_DIRECTION.get(str(value).strip().upper())


def safe_spearman_array(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    xx = x[mask]
    yy = y[mask]
    if np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return np.nan
    try:
        return float(spearmanr(xx, yy).statistic)
    except Exception:
        return np.nan


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=N_TREES,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )),
    ])


def expected_from_proba(model, proba):
    classes = model.named_steps["rf"].classes_
    out = np.zeros(len(proba), dtype=np.float32)
    for k, cls in enumerate(classes):
        cls = str(cls)
        if cls in CLASS_TO_NUM:
            out += proba[:, k].astype(np.float32) * np.float32(CLASS_TO_NUM[cls])
    return out


def production_purge_cutoff(calendar, as_of):
    previous = [pd.Timestamp(x) for x in calendar if pd.Timestamp(x) < as_of]
    if len(previous) <= PURGE_WEEKS:
        return pd.NaT
    return previous[-(PURGE_WEEKS + 1)]

# ============================================================
# INPUTS
# ============================================================
def load_feature_list():
    if not FEATURE_LIST_FILE.exists():
        raise FileNotFoundError(FEATURE_LIST_FILE)
    x = pd.read_csv(FEATURE_LIST_FILE, low_memory=False)
    candidates = ["feature", "FEATURE", "Feature", "column", "COLUMN"]
    col = next((c for c in candidates if c in x.columns), None)
    if col is None:
        if len(x.columns) == 1:
            col = x.columns[0]
        else:
            raise RuntimeError("Non individuo la colonna delle feature in v40_39_feature_list.csv")
    return x[col].dropna().astype(str).tolist()


def read_header():
    if not FEATURE_DATA_FILE.exists():
        raise FileNotFoundError(FEATURE_DATA_FILE)
    return pd.read_csv(FEATURE_DATA_FILE, nrows=0)


def get_identity_columns(header):
    ticker_col = next((c for c in ["Ticker", "TICKER", "ticker"] if c in header.columns), None)
    date_col = next((c for c in ["Date", "DATE", "date"] if c in header.columns), None)
    if ticker_col is None:
        raise RuntimeError("Colonna Ticker non trovata.")
    if date_col is None:
        raise RuntimeError("Colonna Date non trovata.")
    return ticker_col, date_col


def load_validated_sources(valid_features):
    banner("CARICAMENTO SOURCE FEATURE VALIDATE V40.43")
    if not V43_SOURCE_FEATURES.exists():
        raise FileNotFoundError(
            f"Manca {V43_SOURCE_FEATURES}. V40.44 usa intenzionalmente le source feature "
            "già validate e salvate da V40.43."
        )
    s = pd.read_csv(V43_SOURCE_FEATURES, low_memory=False)
    required = {"direction", "source_feature"}
    if not required.issubset(s.columns):
        raise RuntimeError(f"Colonne mancanti in {V43_SOURCE_FEATURES}: {sorted(required - set(s.columns))}")
    s["direction"] = s["direction"].astype(str).str.upper().str.strip()
    s["source_feature"] = s["source_feature"].astype(str).str.strip()
    if "rank" in s.columns:
        s["rank"] = pd.to_numeric(s["rank"], errors="coerce")
        s = s.sort_values(["direction", "rank", "source_feature"])
    valid = set(valid_features)
    out = {}
    for direction in ["BULL", "BEAR"]:
        vals = []
        for f in s.loc[s["direction"] == direction, "source_feature"]:
            if f in valid and f not in vals:
                vals.append(f)
        if len(vals) < MIN_SOURCE_FEATURES:
            raise RuntimeError(f"Source {direction} validate insufficienti: {len(vals)}")
        out[direction] = vals
        print(f"{direction} source validate: {len(vals)}")
    save = pd.concat([
        pd.DataFrame({"direction": d, "rank": np.arange(1, len(fs) + 1), "source_feature": fs})
        for d, fs in out.items()
    ], ignore_index=True)
    save["source_policy"] = "frozen_from_V40.43_validation_output"
    save.to_csv(OUT_SOURCE_FEATURES, index=False)
    return out


def load_episodes():
    banner("CARICAMENTO EPISODI V40.41")
    if not EPISODE_FILE.exists():
        raise FileNotFoundError(EPISODE_FILE)
    ep = pd.read_csv(EPISODE_FILE, low_memory=False)
    required = {"episode_id", "Ticker", "direction", "start_date", "end_date", "strength_class"}
    missing = required - set(ep.columns)
    if missing:
        raise RuntimeError(f"Colonne V40.41 mancanti: {sorted(missing)}")
    ep["Ticker"] = ep["Ticker"].astype(str).str.strip()
    ep["direction"] = ep["direction"].astype(str).str.upper().str.strip()
    ep["start_date"] = pd.to_datetime(ep["start_date"], errors="coerce")
    ep["end_date"] = pd.to_datetime(ep["end_date"], errors="coerce")
    ep = ep[ep["strength_class"].notna()].copy()
    ep["strength_class"] = ep["strength_class"].astype(str)
    ep = ep[ep["strength_class"].isin(CLASS_ORDER)].copy()
    ep["y_num"] = ep["strength_class"].map(CLASS_TO_NUM).astype(np.int8)
    print(f"Episodi etichettati: {len(ep):,}")
    print(f"BULL: {(ep['direction'] == 'BULL').sum():,}")
    print(f"BEAR: {(ep['direction'] == 'BEAR').sum():,}")
    return ep

# ============================================================
# PANEL + CAUSAL Z (same methodology as V40.43)
# ============================================================
def load_direction_panel(direction, sources):
    banner(f"CARICAMENTO PANEL {direction} - CHUNKED")
    header = read_header()
    ticker_col, date_col = get_identity_columns(header)
    usable = [f for f in sources if f in header.columns]
    missing = [f for f in sources if f not in header.columns]
    print(f"Source richieste: {len(sources)}")
    print(f"Source disponibili: {len(usable)}")
    if missing:
        print(f"Source mancanti: {len(missing)}")
    if len(usable) < MIN_SOURCE_FEATURES:
        raise RuntimeError(f"Troppo poche feature {direction}: {len(usable)}")
    phase_col = next((c for c in ["PHASE", "Phase", "phase"] if c in header.columns), None)
    if phase_col is None:
        raise RuntimeError("Colonna PHASE non trovata nel FULL200 V40.39.")
    usecols = [ticker_col, date_col, phase_col] + usable
    chunks = []
    for n, chunk in enumerate(pd.read_csv(
        FEATURE_DATA_FILE, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False
    ), 1):
        chunk = chunk.rename(columns={ticker_col: "Ticker", date_col: "Date", phase_col: "PHASE"})
        chunk["PHASE"] = chunk["PHASE"].astype(str).str.strip().str.upper()
        chunk["Ticker"] = chunk["Ticker"].astype(str).str.strip()
        chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce")
        chunk = chunk.dropna(subset=["Ticker", "Date"])
        for c in usable:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype(np.float32)
        chunks.append(chunk)
        print(f"  chunk {n}: {len(chunk):,} righe")
        del chunk
        gc.collect()
    if not chunks:
        raise RuntimeError(f"Nessun dato caricato per {direction}.")
    df = pd.concat(chunks, ignore_index=True, copy=False)
    del chunks
    gc.collect()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    print(f"Righe panel: {len(df):,}")
    print(f"Ticker: {df['Ticker'].nunique()}")
    print(f"Periodo: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    return df, usable


def build_causal_z_matrix(panel, sources, direction):
    banner(f"NORMALIZZAZIONE CAUSALE WITHIN-TICKER {direction}")
    print(f"MIN_HISTORY_FOR_Z: {MIN_HISTORY_FOR_Z}")
    print("Formula: expanding storico + shift(1) + std(ddof=0) + clip[-8,+8]")
    n_rows = len(panel)
    n_features = len(sources)
    z_matrix = np.full((n_rows, n_features), np.nan, dtype=np.float32)
    grouped = list(panel.groupby("Ticker", sort=False).groups.items())
    total_tickers = len(grouped)
    for ti, (_, positions) in enumerate(grouped, 1):
        pos = np.asarray(positions, dtype=np.int64)
        for j, feature in enumerate(sources):
            s = pd.Series(
                pd.to_numeric(panel.loc[pos, feature], errors="coerce").to_numpy(dtype=np.float64),
                index=np.arange(len(pos)), dtype=np.float64,
            )
            hist_mean = s.expanding(min_periods=MIN_HISTORY_FOR_Z).mean().shift(1)
            hist_std = s.expanding(min_periods=MIN_HISTORY_FOR_Z).std(ddof=0).shift(1)
            z = ((s - hist_mean) / hist_std.replace(0, np.nan)).clip(-8, 8)
            z_matrix[pos, j] = pd.to_numeric(z, errors="coerce").to_numpy(dtype=np.float32)
        if ti == 1 or ti % 20 == 0 or ti == total_tickers:
            print(f"  ticker {ti:,}/{total_tickers:,}")
        gc.collect()
    valid_pct = float(np.isfinite(z_matrix).mean() * 100.0)
    print(f"Copertura valori Z finiti: {valid_pct:.2f}%")
    return z_matrix

# ============================================================
# EPISODE-WEEK INDEX (same mechanics as V40.43)
# ============================================================
def build_episode_week_index(panel, episodes, direction):
    banner(f"COSTRUZIONE INDICE EPISODI {direction}")
    groups = {
        str(ticker): g.index.to_numpy(dtype=np.int32)
        for ticker, g in panel.groupby("Ticker", sort=False)
    }
    date_values = panel["Date"].to_numpy(dtype="datetime64[ns]")
    rows = []
    eps = episodes[episodes["direction"] == direction].copy()
    used = 0
    for _, ep in eps.iterrows():
        ticker = str(ep["Ticker"])
        if ticker not in groups:
            continue
        positions = groups[ticker]
        ticker_dates = date_values[positions]
        start = np.datetime64(pd.Timestamp(ep["start_date"]))
        end = np.datetime64(pd.Timestamp(ep["end_date"]))
        s = np.where(ticker_dates == start)[0]
        e = np.where(ticker_dates == end)[0]
        if len(s) == 0 or len(e) == 0:
            continue
        local_start = int(s[0])
        local_end = int(e[-1])
        if local_end < local_start:
            continue
        global_start = int(positions[local_start])
        cls = str(ep["strength_class"])
        y_num = int(CLASS_TO_NUM[cls])
        for local_pos in range(local_start, local_end + 1):
            global_pos = int(positions[local_pos])
            prev_pos = global_start if local_pos == local_start else int(positions[local_pos - 1])
            age = local_pos - local_start + 1
            rows.append({
                "episode_id": ep["episode_id"],
                "Ticker": ticker,
                "direction": direction,
                "week_date": pd.Timestamp(panel.at[global_pos, "Date"]),
                "episode_start_date": pd.Timestamp(ep["start_date"]),
                "episode_end_date": pd.Timestamp(ep["end_date"]),
                "strength_class": cls,
                "y_num": y_num,
                "trend_age_w": age,
                "age_bucket": age_bucket(age),
                "pos": global_pos,
                "start_pos": global_start,
                "prev_pos": prev_pos,
            })
        used += 1
        if used == 1 or used % 500 == 0 or used == len(eps):
            print(f"  episodi: {used:,}/{len(eps):,}")
    idx = pd.DataFrame(rows)
    print(f"Righe indice {direction}: {len(idx):,}")
    return idx

# ============================================================
# CURRENT EPISODE FROM V40.39 PHASE
# ============================================================
def build_current_episode_index(panel, as_of, direction):
    """Ricostruisce il run BULL/BEAR corrente direttamente da PHASE V40.39."""
    rows = []
    as_of = pd.Timestamp(as_of)

    for ticker, g in panel.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index()
        if g.empty or pd.Timestamp(g.iloc[-1]["Date"]) != as_of:
            continue

        current_direction = phase_to_direction(g.iloc[-1]["PHASE"])
        if current_direction != direction:
            continue

        dirs = g["PHASE"].map(phase_to_direction).to_numpy(dtype=object)
        end_local = len(g) - 1
        start_local = end_local
        while start_local > 0 and dirs[start_local - 1] == direction:
            start_local -= 1

        global_current = int(g.iloc[end_local]["index"])
        global_start = int(g.iloc[start_local]["index"])
        if end_local == start_local:
            global_prev = global_start
        else:
            global_prev = int(
                g.iloc[end_local - 1]["index"]
            )
        age = end_local - start_local + 1

        rows.append({
            "episode_id": f"CURRENT_{direction}_{ticker}_{as_of.strftime('%Y%m%d')}",
            "Ticker": str(ticker),
            "direction": direction,
            "week_date": as_of,
            "episode_start_date": pd.Timestamp(g.iloc[start_local]["Date"]),
            "episode_end_date": as_of,
            "strength_class": "CURRENT_UNKNOWN",
            "y_num": np.nan,
            "trend_age_w": age,
            "age_bucket": age_bucket(age),
            "pos": global_current,
            "start_pos": global_start,
            "prev_pos": global_prev,
        })

    out = pd.DataFrame(rows)
    print(f"Current episodes ricostruiti da PHASE {direction}: {len(out):,}")
    return out


# ============================================================
# COMPACT_ROLLING + TRAIN-ONLY SCREENING
# ============================================================
def feature_names_compact(sources):
    names = []
    for prefix in ["CUR__Z__", "START__Z__", "D_START__Z__", "D_PREV__Z__"]:
        for f in sources:
            names.append(prefix + f)
    return names


def build_compact_matrix(z_matrix, index_rows):
    if len(index_rows) == 0:
        return np.empty((0, z_matrix.shape[1] * 4), dtype=np.float32)
    pos = index_rows["pos"].to_numpy(dtype=np.int64)
    start_pos = index_rows["start_pos"].to_numpy(dtype=np.int64)
    prev_pos = index_rows["prev_pos"].to_numpy(dtype=np.int64)
    cur = z_matrix[pos]
    start = z_matrix[start_pos]
    prev = z_matrix[prev_pos]
    d_start = (cur - start).astype(np.float32, copy=False)
    d_prev = (cur - prev).astype(np.float32, copy=False)
    return np.concatenate([cur, start, d_start, d_prev], axis=1).astype(np.float32, copy=False)


def screen_features(x_train, y_train, feature_names):
    scored = []
    for j, feature in enumerate(feature_names):
        sp = safe_spearman_array(x_train[:, j], y_train)
        if np.isfinite(sp):
            scored.append((feature, float(sp), abs(float(sp)), j))
    scored.sort(key=lambda x: (-x[2], x[0]))
    selected = scored[:TOP_MODEL_FEATURES]
    return selected, [x[3] for x in selected]

# ============================================================
# PRODUCTION FIT + CURRENT SCORE
# ============================================================
def run_direction_production(direction, sources, episodes, all_calendar, as_of, cutoff):
    banner(f"V40.44 PRODUCTION - {direction}")
    panel, sources = load_direction_panel(direction, sources)
    panel_as_of = pd.Timestamp(panel["Date"].max())
    if panel_as_of != as_of:
        raise RuntimeError(
            f"As-of incoerente per {direction}: panel={panel_as_of.date()} globale={as_of.date()}"
        )
    z_matrix = build_causal_z_matrix(panel, sources, direction)
    idx = build_episode_week_index(panel, episodes, direction)
    current_idx = build_current_episode_index(panel, as_of, direction)
    feature_names = feature_names_compact(sources)

    selected_rows = []
    current_rows = []
    audit_rows = []

    for bucket in AGE_BUCKETS:
        banner(f"{direction} - {bucket} - {VARIANT}")
        bucket_idx = idx[idx["age_bucket"] == bucket].copy().reset_index(drop=True)
        current_bucket = current_idx[current_idx["age_bucket"] == bucket].copy().reset_index(drop=True)
        print(f"Righe TRAIN bucket: {len(bucket_idx):,} | CURRENT bucket: {len(current_bucket):,}")
        if bucket_idx.empty:
            audit_rows.append({
                "direction": direction, "age_bucket": bucket, "variant": VARIANT,
                "as_of": as_of, "purge_cutoff": cutoff, "train_rows": 0,
                "train_episodes": 0, "train_tickers": 0, "current_rows": len(current_bucket),
                "selected_features": 0, "max_train_episode_end": pd.NaT,
                "PURGE_OK": False, "CURRENT_EPISODE_EXCLUDED": True,
                "status": "SKIP_EMPTY_BUCKET",
            })
            continue

        matrix = build_compact_matrix(z_matrix, bucket_idx)
        current_matrix = build_compact_matrix(z_matrix, current_bucket)
        end_dates = pd.to_datetime(bucket_idx["episode_end_date"], errors="coerce")

        train_mask = (end_dates <= cutoff).to_numpy()
        train_pos = np.flatnonzero(train_mask)
        current_pos = np.arange(len(current_bucket), dtype=np.int64)
        n_train = len(train_pos)
        n_current = len(current_bucket)

        train_classes = bucket_idx.iloc[train_pos]["strength_class"].astype(str)
        max_train_end = end_dates.iloc[train_pos].max() if n_train else pd.NaT
        purge_ok = bool(n_train > 0 and pd.notna(max_train_end) and max_train_end <= cutoff)

        # L'episodio CURRENT è ricostruito separatamente da PHASE V40.39.
        # Il training usa solo episodi V40.41 terminati entro cutoff:
        # l'episodio aperto corrente non può quindi entrare nel training.
        current_episode_excluded = True

        audit = {
            "direction": direction,
            "age_bucket": bucket,
            "variant": VARIANT,
            "as_of": as_of,
            "purge_cutoff": cutoff,
            "train_rows": n_train,
            "train_episodes": bucket_idx.iloc[train_pos]["episode_id"].nunique() if n_train else 0,
            "train_tickers": bucket_idx.iloc[train_pos]["Ticker"].nunique() if n_train else 0,
            "current_rows": n_current,
            "selected_features": 0,
            "max_train_episode_end": max_train_end,
            "PURGE_OK": purge_ok,
            "CURRENT_EPISODE_EXCLUDED": current_episode_excluded,
            "status": "PENDING",
        }

        if n_current == 0:
            audit["status"] = "NO_CURRENT_ROWS"
            audit_rows.append(audit)
            del matrix, bucket_idx
            gc.collect()
            continue

        if n_train < MIN_TRAIN_ROWS or train_classes.nunique() < 3:
            audit["status"] = "SKIP_INSUFFICIENT_TRAIN"
            audit_rows.append(audit)
            for _, rr in current_bucket.iterrows():
                current_rows.append({
                    "Ticker": str(rr["Ticker"]), "Date": as_of, "direction": direction,
                    "trend_age_w": int(rr["trend_age_w"]), "age_bucket": bucket,
                    "variant": VARIANT, "strength_expected_0_3": np.nan,
                    "strength_class": "N/A", "p_LOW": np.nan, "p_MEDIUM": np.nan,
                    "p_HIGH": np.nan, "p_VERY_HIGH": np.nan,
                    "status": "INSUFFICIENT_TRAIN",
                })
            del matrix, bucket_idx
            gc.collect()
            continue

        x_train = matrix[train_pos]
        y_train_num = bucket_idx.iloc[train_pos]["y_num"].to_numpy(dtype=np.float32)
        selected, selected_indices = screen_features(x_train, y_train_num, feature_names)
        audit["selected_features"] = len(selected_indices)

        if len(selected_indices) < 10:
            audit["status"] = "SKIP_FEW_FEATURES"
            audit_rows.append(audit)
            for _, rr in current_bucket.iterrows():
                current_rows.append({
                    "Ticker": str(rr["Ticker"]), "Date": as_of, "direction": direction,
                    "trend_age_w": int(rr["trend_age_w"]), "age_bucket": bucket,
                    "variant": VARIANT, "strength_expected_0_3": np.nan,
                    "strength_class": "N/A", "p_LOW": np.nan, "p_MEDIUM": np.nan,
                    "p_HIGH": np.nan, "p_VERY_HIGH": np.nan,
                    "status": "FEW_FEATURES",
                })
            del x_train, y_train_num, matrix, current_matrix, bucket_idx, current_bucket
            gc.collect()
            continue

        for rank, item in enumerate(selected, 1):
            feature, feat_sp, abs_sp, _ = item
            selected_rows.append({
                "direction": direction, "age_bucket": bucket, "variant": VARIANT,
                "rank": rank, "feature": feature,
                "train_spearman": feat_sp, "train_abs_spearman": abs_sp,
                "as_of": as_of, "purge_cutoff": cutoff,
            })

        x_train_sel = x_train[:, selected_indices].astype(np.float32, copy=True)
        labels_train = train_classes.to_numpy()
        model = make_model()
        model.fit(x_train_sel, labels_train)

        x_current_sel = current_matrix[:, selected_indices].astype(np.float32, copy=True)
        proba = model.predict_proba(x_current_sel)
        pred_class = model.predict(x_current_sel)
        pred_expected = expected_from_proba(model, proba)
        model_classes = [str(x) for x in model.named_steps["rf"].classes_]

        def class_probability(row_idx, cls):
            if cls not in model_classes:
                return 0.0
            return float(proba[row_idx, model_classes.index(cls)])

        current_subset = current_bucket.reset_index(drop=True)
        for k, rr in current_subset.iterrows():
            current_rows.append({
                "Ticker": str(rr["Ticker"]),
                "Date": as_of,
                "direction": direction,
                "trend_age_w": int(rr["trend_age_w"]),
                "age_bucket": bucket,
                "variant": VARIANT,
                "strength_expected_0_3": float(pred_expected[k]),
                "strength_class": str(pred_class[k]),
                "p_LOW": class_probability(k, "LOW"),
                "p_MEDIUM": class_probability(k, "MEDIUM"),
                "p_HIGH": class_probability(k, "HIGH"),
                "p_VERY_HIGH": class_probability(k, "VERY_HIGH"),
                "status": "READY",
            })

        audit["status"] = "OK"
        audit_rows.append(audit)
        print(
            f"TRAIN={n_train:,} | CURRENT={n_current:,} | "
            f"FEATURE={len(selected_indices)} | purge_ok={purge_ok} | "
            f"current_episode_excluded={current_episode_excluded}"
        )

        del x_train, y_train_num, x_train_sel, x_current_sel
        del labels_train, model, proba, pred_class, pred_expected
        del matrix, current_matrix, bucket_idx, current_bucket
        gc.collect()

    del idx, current_idx, z_matrix, panel
    gc.collect()
    return current_rows, selected_rows, audit_rows

# ============================================================
# OUTPUT / AUDIT
# ============================================================
def save_metadata(as_of, cutoff, sources, current_df, audit_df):
    ok_audit = audit_df[audit_df["status"] == "OK"].copy() if not audit_df.empty else pd.DataFrame()
    metadata = {
        "version": VERSION,
        "source_validation_version": SOURCE_VALIDATION_VERSION,
        "purpose": "FULL200 Trend Strength production candidate from validated V40.43 methodology",
        "production_model": True,
        "frozen_production": False,
        "phase_version": PHASE_VERSION,
        "phase_modified": False,
        "reversal_version": REVERSAL_VERSION,
        "reversal_modified": False,
        "strength_target_version": STRENGTH_TARGET_VERSION,
        "engine_modified": False,
        "opportunity_model": False,
        "directions": ["BULL", "BEAR"],
        "indecision_strength": "N/A",
        "age_buckets": AGE_BUCKETS,
        "variant": VARIANT,
        "buy_sell_age_names_used": False,
        "as_of": str(as_of.date()),
        "purge_weeks": PURGE_WEEKS,
        "purge_cutoff": str(cutoff.date()),
        "production_training_policy": "all historical V40.41 labeled rows with episode_end_date <= purge_cutoff; current episode reconstructed from V40.39 PHASE",
        "leave_one_ticker_out": False,
        "leave_one_ticker_out_role": "V40.43 validation protocol only; not final production fitting rule",
        "current_episode_training_excluded": bool(
            len(ok_audit) > 0
            and ok_audit["CURRENT_EPISODE_EXCLUDED"].astype(bool).all()
        ),
        "causal_within_ticker_z": {
            "enabled": True, "min_history": MIN_HISTORY_FOR_Z,
            "expanding": True, "shift": 1, "std_ddof": 0,
            "clip_min": -8, "clip_max": 8, "dtype": "float32",
        },
        "current_episode_policy": "reconstruct open current BULL/BEAR run directly from V40.39 PHASE; INDECISIONE has no Strength",
        "source_feature_policy": "load exact validated V40.43 source-feature output",
        "bull_source_features": len(sources["BULL"]),
        "bear_source_features": len(sources["BEAR"]),
        "top_model_features": TOP_MODEL_FEATURES,
        "feature_selection": "absolute Spearman on eligible production TRAIN only, separately by direction and AGE bucket",
        "min_train_rows": MIN_TRAIN_ROWS,
        "random_forest": {
            "n_estimators": N_TREES, "max_features": "sqrt",
            "min_samples_leaf": 5, "class_weight": "balanced_subsample",
            "random_state": RANDOM_STATE, "runtime_n_jobs": N_JOBS,
        },
        "current_rows": int(len(current_df)),
        "current_ready": int((current_df["status"] == "READY").sum()) if not current_df.empty else 0,
        "score_1_10_created": False,
        "status": "production candidate - requires runtime audit before freeze",
    }
    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def final_audit(current_df, audit_df, as_of, cutoff):
    banner("VERDETTO STRUTTURALE V40.44 PRODUCTION CANDIDATE")
    ok = audit_df[audit_df["status"] == "OK"].copy()
    purge_ok = bool(len(ok) > 0 and ok["PURGE_OK"].astype(bool).all())
    current_excluded = bool(len(ok) > 0 and ok["CURRENT_EPISODE_EXCLUDED"].astype(bool).all())
    duplicates = int(current_df.duplicated(["Ticker", "Date"]).sum()) if not current_df.empty else 0
    ready = int((current_df["status"] == "READY").sum()) if not current_df.empty else 0
    print(f"As-of:                         {as_of.date()}")
    print(f"Purge cutoff:                  {cutoff.date()}")
    print(f"Bucket production OK:          {len(ok)}/12")
    print(f"Purging corretto:              {purge_ok}")
    print(f"Episodio corrente fuori train: {current_excluded}")
    print(f"Duplicati ticker/data:          {duplicates}")
    print(f"Ticker Strength READY:         {ready}")
    structural_pass = purge_ok and current_excluded and duplicates == 0 and ready > 0
    print()
    if structural_pass:
        print("OK - V40.44 PRODUCTION CANDIDATE COMPLETATA SENZA VIOLAZIONI STRUTTURALI.")
        print("NON E' ANCORA CONGELATA: prima vanno esaminati output e audit del run reale.")
    else:
        print("ATTENZIONE - V40.44 PRESENTA UNO O PIU' PROBLEMI STRUTTURALI.")
        print("NON CONGELARE E NON INTEGRARE IN HOME.")

# ============================================================
# MAIN
# ============================================================
def main():
    banner(
        "MARKETSENTINEL V40.44 - FULL200 TREND STRENGTH PRODUCTION CANDIDATE "
        "- COMPACT_ROLLING - CAUSAL Z"
    )
    OUT.mkdir(parents=True, exist_ok=True)

    # Overwrite only V40.44 outputs, never V40.43.
    for p in [OUT_SOURCE_FEATURES, OUT_SELECTED, OUT_CURRENT, OUT_TRAIN_AUDIT, OUT_METADATA]:
        if p.exists():
            p.unlink()

    feature_list = load_feature_list()
    sources = load_validated_sources(feature_list)
    episodes = load_episodes()

    banner("CARICAMENTO CALENDARIO / AS-OF")
    header = read_header()
    _, date_col = get_identity_columns(header)
    dates = pd.read_csv(FEATURE_DATA_FILE, usecols=[date_col])
    dates[date_col] = pd.to_datetime(dates[date_col], errors="coerce")
    all_calendar = sorted(dates[date_col].dropna().unique())
    if not all_calendar:
        raise RuntimeError("Calendario FULL200 vuoto.")
    as_of = pd.Timestamp(all_calendar[-1])
    cutoff = production_purge_cutoff(all_calendar, as_of)
    if pd.isna(cutoff):
        raise RuntimeError("Impossibile calcolare il purge cutoff production.")
    print(f"Settimane calendario: {len(all_calendar):,}")
    print(f"As-of production: {as_of.date()}")
    print(f"Purge cutoff ({PURGE_WEEKS} settimane): {cutoff.date()}")
    del dates
    gc.collect()

    all_current = []
    all_selected = []
    all_audit = []

    for direction in ["BULL", "BEAR"]:
        current_rows, selected_rows, audit_rows = run_direction_production(
            direction=direction,
            sources=sources[direction],
            episodes=episodes,
            all_calendar=all_calendar,
            as_of=as_of,
            cutoff=cutoff,
        )
        all_current.extend(current_rows)
        all_selected.extend(selected_rows)
        all_audit.extend(audit_rows)
        gc.collect()

    current_df = pd.DataFrame(all_current)
    selected_df = pd.DataFrame(all_selected)
    audit_df = pd.DataFrame(all_audit)

    if not current_df.empty:
        current_df = current_df.sort_values(["direction", "age_bucket", "Ticker"]).reset_index(drop=True)
    if not selected_df.empty:
        selected_df = selected_df.sort_values(["direction", "age_bucket", "rank"]).reset_index(drop=True)
    if not audit_df.empty:
        audit_df = audit_df.sort_values(["direction", "age_bucket"]).reset_index(drop=True)

    current_df.to_csv(OUT_CURRENT, index=False)
    selected_df.to_csv(OUT_SELECTED, index=False)
    audit_df.to_csv(OUT_TRAIN_AUDIT, index=False)
    save_metadata(as_of, cutoff, sources, current_df, audit_df)
    final_audit(current_df, audit_df, as_of, cutoff)

    banner("V40.44 COMPLETATA")
    print(f"Output: {OUT}")
    print(f"Current Strength: {OUT_CURRENT}")
    print(f"Training audit:   {OUT_TRAIN_AUDIT}")
    print(f"Selected feature: {OUT_SELECTED}")
    print(f"Metadata:         {OUT_METADATA}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 88)
        print("ERRORE V40.44")
        print("=" * 88)
        print(f"{type(exc).__name__}: {exc}")
        raise
