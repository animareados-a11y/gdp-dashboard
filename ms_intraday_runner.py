from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import os
os.environ.setdefault("MS_DAILY_PHASE_V4_FAST", "1")
os.environ.setdefault("MS_DAILY_WORKERS", "2")
os.environ.setdefault("MS_DAILY_WINDOW_520", "0")
import uuid

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data" / "marketsentinel_daily"
MARKETS = {
    "ITALIA": "italy40",
    "EUROPA": "europe40",
    "USA80": "usa80",
    "USA40": "usa40",
}
EXPECTED = {
    "italy40": 40,
    "europe40": 40,
    "usa80": 80,
    "usa40": 39,
}


# Registro persistente dei soli cloni presenti nella cache scientifica pubblicata.
GAP_FILE = ROOT / "data" / "marketsentinel_intraday" / "yahoo_gap_registry.json"

def read_gaps():
    import json
    try:
        return json.loads(GAP_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

def write_gaps(gaps):
    import json
    GAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = GAP_FILE.with_name(GAP_FILE.name + ".tmp")
    tmp.write_text(json.dumps(gaps, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, GAP_FILE)

def download_latest(ticker, last_date, pending_dates=()):
    """Recupera nuove sedute e ricontrolla tutti i cloni ancora pendenti."""
    dates = [pd.Timestamp(last_date)] + [pd.Timestamp(x) for x in pending_dates]
    start = min(dates).strftime("%Y-%m-%d")
    data = yf.download(
        ticker, start=start, interval="1d", auto_adjust=False,
        progress=False, threads=False, timeout=25,
    )
    if data.empty:
        raise RuntimeError(f"{ticker}: Yahoo non ha restituito dati")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    dates = pd.to_datetime(data["Date"], errors="coerce")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    data["Date"] = dates.dt.strftime("%Y-%m-%d")
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    # Recupero mirato delle sedute Yahoo con OHLCV incompleti.
    incomplete = data.loc[data[required].isna().any(axis=1), "Date"].dropna().unique()
    for day in incomplete:
        try:
            begin = pd.Timestamp(day)
            finish = begin + pd.Timedelta(days=1)
            retry = yf.download(
                ticker,
                start=begin.strftime("%Y-%m-%d"),
                end=finish.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=25,
            )
            if isinstance(retry.columns, pd.MultiIndex):
                retry.columns = retry.columns.get_level_values(0)
            if retry.empty:
                continue
            retry = retry.reset_index()
            retry["Date"] = pd.to_datetime(retry["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
            retry = retry.dropna(subset=required)
            if not retry.empty:
                data = pd.concat([data.loc[data["Date"] != day], retry], ignore_index=True)
                print(f"[YAHOO RECOVERY] {ticker} {day}: OHLCV recuperati", flush=True)
        except Exception as exc:
            print(f"[YAHOO RECOVERY] {ticker} {day}: {exc}", flush=True)
    data = data.dropna(subset=required)
    data = data.sort_values("Date").drop_duplicates("Date", keep="last")
    if data.empty:
        raise RuntimeError(f"{ticker}: dati Yahoo non validi")
    return data.loc[data["Date"] >= start].copy()


def merge_quotes_and_gaps(history, quotes, pending_dates, market):
    """I cloni ripetono integralmente OHLCV/Adj Close; i reali prevalgono sempre."""
    history = history.copy()
    history["Date"] = pd.to_datetime(history["Date"]).dt.strftime("%Y-%m-%d")
    previous = history.sort_values("Date").drop_duplicates("Date", keep="last")
    real_dates = set(quotes["Date"])
    existing = {str(r["Date"]): r.copy() for _, r in previous.iterrows()}
    pending = set(pending_dates)
    # Per una data precedentemente clonata, il valore Yahoo ha sempre priorita'.
    for _, quote in quotes.iterrows():
        day = str(quote["Date"])
        template = existing.get(day, previous.iloc[-1]).copy()
        for col in ("Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"):
            if col in template.index and col in quote.index:
                template[col] = quote[col]
        existing[day] = template
        pending.discard(day)
    all_days = sorted(existing)
    # Riempire soltanto i giorni di negoziazione compresi fra due sedute note.
    # Il calendario di borsa evita di clonare festivita' e fine settimana.
    import exchange_calendars as xcals
    calendar_name = {"italy40": "XMIL", "europe40": "XETR", "usa80": "XNYS", "usa40": "XNYS"}[market]
    cal = xcals.get_calendar(calendar_name)
    # Non alterare buchi antichi preesistenti: lavorare solo dalla data
    # iniziale richiesta, incluse le date del registro da riparare.
    first = min(pd.Timestamp(previous["Date"].max()), *[pd.Timestamp(d) for d in pending_dates]) if pending_dates else pd.Timestamp(previous["Date"].max())
    last = pd.Timestamp(all_days[-1])
    sessions = cal.sessions_in_range(first, last)
    for date in sessions:
        day = date.strftime("%Y-%m-%d")
        if day in existing:
            continue
        prior = [d for d in existing if d < day]
        future = [d for d in existing if d > day]
        if not prior or not future:
            continue
        template = existing[max(prior)].copy()
        template["Date"] = day
        existing[day] = template
        pending.add(day)
    # Le sedute clonate in precedenza restano marcate fino al dato Yahoo reale.
    pending -= real_dates
    merged = pd.DataFrame([existing[d] for d in sorted(existing)])
    return merged, sorted(pending)



_RUN_GAPS = {}

def prepare_sandbox(sandbox):
    """Prepara la sandbox senza modificare i master DAILY consolidati."""
    destination = sandbox / "data" / "marketsentinel_daily"
    shutil.copytree(BASE, destination)

    # Preferire lo storico DAILY scientifico pubblicato,
    # se completo e non meno recente dei master consolidati.
    cache = (
        ROOT / "data" / "marketsentinel_intraday"
        / "scientific_cache"
    )
    published = (
        ROOT / "data" / "marketsentinel_intraday"
        / "marketsentinel_intraday.csv"
    )

    prefixes = tuple(MARKETS.values())
    suffixes = (
        "_daily_master.csv",
        "_phase_history.csv",
        "_current_phase.csv",
        "_v160_features.csv",
    )

    complete = (
        published.is_file()
        and cache.is_dir()
        and all(
            (cache / f"{prefix}{suffix}").is_file()
            for prefix in prefixes
            for suffix in suffixes
        )
    )

    if complete:
        for prefix in prefixes:
            original = pd.read_csv(
                destination / f"{prefix}_daily_master.csv",
                usecols=["Ticker", "Date"],
            )
            cached = pd.read_csv(
                cache / f"{prefix}_daily_master.csv",
                usecols=["Ticker", "Date"],
            )

            if (
                original["Ticker"].nunique() != EXPECTED[prefix]
                or cached["Ticker"].nunique() != EXPECTED[prefix]
                or pd.to_datetime(cached["Date"], utc=True).max()
                < pd.to_datetime(original["Date"], utc=True).max()
            ):
                complete = False
                break

    if complete:
        for prefix in prefixes:
            shutil.copy2(
                cache / f"{prefix}_daily_master.csv",
                destination / f"{prefix}_daily_master.csv",
            )
        print(
            "[DAILY] BASE: cache scientifica validata",
            flush=True,
        )
    else:
        print(
            "[DAILY] BASE: master consolidati",
            flush=True,
        )

    # Moduli PHASE V5 necessari al calcolo DAILY isolato.
    # La soglia DAILY viene impostata a 12 dal motore sandbox.
    # I sorgenti WEEKLY originali restano invariati.
    for name in (
        "weekly_v40_5_phase_dynamics_shadow_v5.py",
        "weekly_v40_39_phase_canonical.py",
    ):
        shutil.copy2(ROOT / name, sandbox / name)

    target_daily = sandbox / "operational"
    target_daily.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "operational/phase_v5_daily.py",
        target_daily / "phase_v5_daily.py"
    )
    shutil.copy2(ROOT / "ms_daily_engine.py", sandbox / "ms_daily_engine.py")
    shutil.copy2(ROOT / "ms_pr_filter.py", sandbox / "ms_pr_filter.py")
    shutil.copy2(ROOT / "ms_pr_unified.py", sandbox / "ms_pr_unified.py")
    shutil.copy2(ROOT / "ms_daily_window520.py", sandbox / "ms_daily_window520.py")
    if os.environ.get("MS_PROFILE_FINALIZE", "0") == "1":
        shutil.copy2(
            ROOT / "ms_daily_finalize_profiler.py",
            sandbox / "ms_daily_finalize_profiler.py"
        )
    shutil.copy2(ROOT / "ms_v160_incremental.py", sandbox / "ms_v160_incremental.py")  # MS_WINDOW520_INSTALL_V1
    if os.environ.get("MS_DAILY_PHASE_V4_FAST", "0") == "1":
        for name in ("weekly_v40_5_phase_dynamics_shadow.py", "weekly_v40_5_phase_dynamics_shadow_v4.py"):
            shutil.copy2(ROOT / name, sandbox / name)
        target = sandbox / "operational"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "operational/phase_v4_fast.py", target / "phase_v4_fast.py")
    for folder in (
        "v40_160_buy1_technical_analysis",
        "v40_160_buy2_technical_analysis",
        "v40_160_buy3_technical_analysis",
    ):
        source = ROOT / "data" / folder
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, sandbox / "data" / folder)

    total_updated = 0
    failures = []
    for market, prefix in MARKETS.items():
        path = destination / f"{prefix}_daily_master.csv"
        df = pd.read_csv(path)
        if df["Ticker"].nunique() != EXPECTED[prefix]:
            raise RuntimeError(f"Numero titoli non valido: {prefix}")
        if prefix == "usa40":
            df = df.loc[df["Ticker"] != "WBD"].copy()
        updates = []
        market_failures = []
        # Download concorrente; scrittura master solo a download
        # completamente riuscito per tutti i ticker del mercato.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        gaps_before = read_gaps()
        gaps_after = dict(gaps_before)

        def fetch_one(item):
            ticker, history = item
            pending = gaps_before.get(prefix, {}).get(ticker, [])
            quotes = download_latest(ticker, history["Date"].max(), pending)
            merged, new_pending = merge_quotes_and_gaps(history, quotes, pending, prefix)
            return ticker, merged, new_pending

        items = list(df.groupby("Ticker", sort=False))

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(fetch_one, item): item[0]
                for item in items
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    _, merged, new_pending = future.result()
                    updates.append(merged)
                    gaps_after.setdefault(prefix, {})[ticker] = new_pending
                    total_updated += 1
                except Exception as exc:
                    market_failures.append(f"{ticker}: {exc}")
        if market_failures:
            failures.extend(market_failures)
            raise RuntimeError(
                f"{prefix}: download incompleto; calcolo bloccato:\n"
                + "\n".join(market_failures)
            )
        df = pd.concat(updates, ignore_index=True)
        df = df.drop_duplicates(["Ticker", "Date"], keep="last")
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        dates = df.groupby("Ticker")["Date"].max()
        if dates.nunique() != 1:
            print(
                f"{prefix}: ultime quotazioni con date differenti "
                f"{dates.value_counts().to_dict()}; "
                "mantenute le ultime quotazioni reali disponibili",
                flush=True,
            )
        df.to_csv(path, index=False)
        _RUN_GAPS[prefix] = gaps_after.get(prefix, {})
        print(prefix, "titoli:", len(dates), "data:", dates.max(), flush=True)
    return total_updated, failures




def calculate_in_sandbox(sandbox, changed_markets):
    script = r'''
import os
import ms_daily_engine as engine

os.chdir(os.path.dirname(__file__))

import time
import traceback


from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

def calculate_market(prefix):
    import time
    import traceback

    started = time.perf_counter()
    if os.environ.get("MS_DAILY_PHASE_V4_FAST", "0") == "1":
        import weekly_v40_5_phase_dynamics_shadow_v4 as v4
        from operational.phase_v4_fast import optimized
        v4._surgical_reclassification = optimized
        print("[DAILY] V4 FAST ATTIVO", flush=True)
    print(f"[SCIENTIFIC] INIZIO {prefix}", flush=True)

    try:
        name, phase_seconds, total_seconds = (
            engine._calculate_one_market(prefix)
        )
        print(
            f"[SCIENTIFIC] OK {name} "
            f"PHASE={phase_seconds:.1f}s "
            f"TOTALE={total_seconds:.1f}s",
            flush=True
        )
        return prefix
    except BaseException:
        print(f"[SCIENTIFIC] ERRORE {prefix}", flush=True)
        traceback.print_exc()
        raise

markets = []

for prefix in engine.MARKETS.values():
    if prefix not in CHANGED_MARKETS:
        base = Path("data/marketsentinel_daily")
        required = [
            base / f"{prefix}_daily_master.csv",
            base / f"{prefix}_phase_history.csv",
            base / f"{prefix}_current_phase.csv",
            base / f"{prefix}_v160_features.csv",
        ]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise RuntimeError(
                f"CACHE SCIENTIFICA INCOMPLETA {prefix}: {missing}"
            )
        print(
            f"[SCIENTIFIC] CACHE RIUTILIZZATA {prefix}",
            flush=True
        )
    else:
        markets.append(prefix)

def available_memory():
    try:
        limit = Path("/sys/fs/cgroup/memory.max")
        used = Path("/sys/fs/cgroup/memory.current")
        if limit.exists() and used.exists():
            maximum = limit.read_text().strip()
            if maximum != "max":
                return int(maximum) - int(used.read_text())
    except Exception:
        pass
    return 0

free_memory = available_memory()
requested_workers = int(os.environ.get("MS_DAILY_WORKERS", "0"))

if requested_workers < 0:
    raise RuntimeError("MS_DAILY_WORKERS non valido")

if requested_workers:
    workers = min(requested_workers, max(1, len(markets)))
else:
    workers = (
        2 if len(markets) >= 2
        and free_memory >= 4 * 1024**3
        else 1
    )

print(
    f"[SCIENTIFIC] MERCATI DA CALCOLARE: {markets}",
    flush=True
)
print(
    f"[SCIENTIFIC] PROCESSI PARALLELI: {workers}",
    flush=True
)

t_markets = time.perf_counter()

if workers == 1:
    for prefix in markets:
        calculate_market(prefix)
else:
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context
    ) as executor:
        futures = {
            executor.submit(calculate_market, prefix): prefix
            for prefix in markets
        }
        for future in as_completed(futures):
            future.result()

print(f"[TIMING] MERCATI={time.perf_counter()-t_markets:.1f}s", flush=True)
t_finalize = time.perf_counter()

import ms_daily_window520
try:
    if os.environ.get("MS_PROFILE_FINALIZE", "0") == "1":
        from ms_daily_finalize_profiler import profile_finalize
        profile_finalize(
            ms_daily_window520.finalize,
            Path(__file__).resolve().parent
        )
    else:
        ms_daily_window520.finalize(Path(__file__).resolve().parent)
except Exception as exc:
    print("[WINDOW520] VALIDAZIONE FALLITA:", repr(exc), flush=True)
    raise RuntimeError("WINDOW520 non validata: pubblicazione bloccata") from exc

print(f"[TIMING] FINALIZE={time.perf_counter()-t_finalize:.1f}s", flush=True)
t_aggregation = time.perf_counter()
print("[SCIENTIFIC] INIZIO AGGREGAZIONE", flush=True)

try:
    from operational.run_marketsentinel import main
    main()
    print("[SCIENTIFIC] AGGREGAZIONE OK", flush=True)
    print(f"[TIMING] AGGREGAZIONE={time.perf_counter()-t_aggregation:.1f}s", flush=True)
except BaseException:
    print("[SCIENTIFIC] ERRORE AGGREGAZIONE", flush=True)
    traceback.print_exc()
    raise
'''

    # WBD escluso: adattamento dei soli controlli
    # di numerosità USA40 nella copia scientifica isolata.
    engine_file = sandbox / "ms_daily_engine.py"
    engine_src = engine_file.read_text()

    import ast
    tree = ast.parse(engine_src)
    assignment = next(
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_EMBEDDED"
            for t in n.targets
        )
    )
    # Gli algoritmi restano incorporati: non alterare
    # direttamente il loro contenuto codificato.
    # La modifica avviene tramite il loader isolato.
    loader_patch = """
_original_run_embedded = run_embedded

def run_embedded(name):
    if name == 'update_phase_usa40':
        _configure_daily_phase()
        key = 'operational/' + name + '.py'
        code = _SOURCES[key]
        assert 'assert len(current_df) == 40' in code
        assert 'assert current_df["Ticker"].nunique() == 40' in code
        code = code.replace(
            'assert len(current_df) == 40',
            'assert len(current_df) == 39'
        ).replace(
            'assert current_df["Ticker"].nunique() == 40',
            'assert current_df["Ticker"].nunique() == 39'
        )
        scope = {
            '__name__': '__main__',
            '__file__': str(ROOT / key),
            '__package__': 'operational'
        }
        exec(compile(code, str(ROOT / key), 'exec'), scope)
        return
    return _original_run_embedded(name)
"""
    marker = "def last_completed_session("
    assert marker in engine_src
    engine_src = engine_src.replace(
        marker, loader_patch + "\n" + marker, 1
    )
    engine_file.write_text(engine_src)

    engine_src = engine_file.read_text(encoding="utf-8")
    v160_key = "operational/v160_fast.py"
    v160_original = "def calculate_all_ratings(df):"
    v160_wrapper = '\ndef calculate_all_ratings(df):\n    import numpy as np\n\n    rating_cols = [\n        "RATING_V160_BUY1",\n        "RATING_V160_BUY2",\n        "RATING_V160_BUY3",\n    ]\n\n    ratings = np.full(\n        (len(df), len(rating_cols)),\n        np.nan,\n        dtype=float\n    )\n\n    positions = pd.Series(\n        np.arange(len(df)),\n        index=df.index\n    )\n\n    for ticker, part in df.groupby("Ticker", sort=False):\n        result = _calculate_all_ratings_original(part)\n\n        assert len(result) == len(part), ticker\n        assert all(c in result.columns for c in rating_cols)\n\n        idx = positions.loc[part.index].to_numpy()\n\n        ratings[idx, :] = result[\n            rating_cols\n        ].to_numpy(dtype=float)\n\n    original_length = len(df)\n\n    for j, col in enumerate(rating_cols):\n        df[col] = ratings[:, j]\n\n    assert len(df) == original_length\n    return df\n'

    # PHASE_NUOVA FAST: stessa logica, elaborazione su array.
    phase_path = sandbox / "operational" / "phase_nuova.py"
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    phase_path.write_text(
        __import__("ms_daily_engine")._SOURCES["operational/phase_nuova.py"],
        encoding="utf-8",
    )
    phase_source = phase_path.read_text(encoding="utf-8")

    phase_fast = """
import numpy as np
import pandas as pd

VERSION = "PHASE_NUOVA_V1"
MIN_RATING = 7.0

def calculate_phase_nuova(
    df,
    phase_col="PHASE",
    sar_col="SAR_BULLISH_FLIP",
    ha_col="HA_DIRECTION",
    rating_col="RATING_V160_BUY1",
):
    required = [
        "Ticker", "Date", phase_col,
        sar_col, ha_col, rating_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti: {missing}")

    result = df
    if not isinstance(result.index, pd.RangeIndex):
        raise ValueError("Indice non canonico")

    if not result[["Ticker", "Date"]].equals(
        result[["Ticker", "Date"]].sort_values(
            ["Ticker", "Date"]
        )
    ):
        raise ValueError("Dati non ordinati")

    phases = result[phase_col].to_numpy()
    flips = result[sar_col].to_numpy()
    has = pd.to_numeric(
        result[ha_col], errors="coerce"
    ).to_numpy()
    ratings = pd.to_numeric(
        result[rating_col], errors="coerce"
    ).to_numpy()
    tickers = result["Ticker"].to_numpy()
    sar_sides = pd.to_numeric(
        result["SAR_SIDE"], errors="coerce"
    ).to_numpy()
    ha_indecisions = pd.to_numeric(
        result["V4015_HA_INDECISION"], errors="coerce"
    ).to_numpy()

    n = len(result)
    new_phases = phases.astype(object, copy=True)
    early = np.zeros(n, dtype=bool)
    early_ratings = np.full(n, np.nan)
    reasons = np.full(n, "CANONICAL", dtype=object)

    stage = 0
    previous_ticker = None

    for i in range(n):
        ticker = tickers[i]

        if ticker != previous_ticker:
            stage = 0
            previous_ticker = ticker

        if stage in (1, 2):
            compatible = phases[i] in (
                "BUY1", "BUY2", "BUY3",
                "TREND_RIALZISTA", "DEBOLEZZA_RIALZISTA"
            )
            confirmed = (
                compatible
                and sar_sides[i] == 1
                and has[i] == 1
                and ha_indecisions[i] == 0
            )
            if not confirmed:
                stage = 0
                continue
            stage += 1
            new_phases[i] = f"BUY{stage}"
            early[i] = True
            reasons[i] = "EARLY_BUY_PROGRESS"
            if stage == 3:
                stage = 0
            continue

        flip = flips[i]
        bullish_flip = (
            pd.notna(flip)
            and str(flip).lower()
            in ("true", "1", "1.0")
        )

        if (
            bullish_flip
            and has[i] == 1
            and pd.notna(ratings[i])
            and ratings[i] >= MIN_RATING
            and phases[i] not in ("BUY1", "BUY2", "BUY3")
        ):
            stage = 1
            new_phases[i] = "BUY1"
            early[i] = True
            early_ratings[i] = float(ratings[i])
            reasons[i] = "SAR_HA_RATING_CONFIRM"

    result["PHASE_NUOVA"] = new_phases
    result["EARLY_BUY"] = early
    result["EARLY_BUY_RATING"] = early_ratings
    result["PHASE_NUOVA_REASON"] = reasons
    return result
"""

    ast.parse(phase_fast)
    phase_path.write_text(
        phase_fast, encoding="utf-8"
    )

    v160_patch = (
        "_v160_key = " + repr(v160_key) + "\n"
        + "_v160_code = _SOURCES[_v160_key]\n"
        + "assert _v160_code.count(" + repr(v160_original) + ") == 1\n"
        + "_v160_code = _v160_code.replace("
        + repr(v160_original) + ", "
        + repr("def _calculate_all_ratings_original(df):")
        + ", 1)\n"
        + "_v160_code += " + repr("\n" + v160_wrapper) + "\n"
        + "_SOURCES[_v160_key] = _v160_code\n"
    )
    insertion = "def last_completed_session("
    assert engine_src.count(insertion) == 1
    engine_src = engine_src.replace(
        insertion, v160_patch + "\n" + insertion, 1
    )
    engine_file.write_text(engine_src, encoding="utf-8")


    # Correzione date ISO miste nell'aggregatore sandbox.
    engine_src = engine_file.read_text(encoding="utf-8")
    aggregate_date_patch = (
        "\n_agg_key = 'operational/run_marketsentinel.py'\n"
        "_agg_code = _SOURCES[_agg_key]\n"
        "_old_dt = 'pd.to_datetime('\n"
        "_new_dt = '_ms_parse_datetime('\n"
        "assert _old_dt in _agg_code\n"
        "_agg_code = _agg_code.replace(_old_dt, _new_dt)\n"
        "_agg_code = _agg_code.replace(\n"
        "    'def main(',\n"
        "    'def _ms_parse_datetime(*args, **kwargs):\\n"
        "    kwargs.setdefault(\\\"format\\\", \\\"mixed\\\")\\n"
        "    return pd.to_datetime(*args, **kwargs)\\n\\n"
        "def main(',\n"
        "    1,\n"
        ")\n"
        "_SOURCES[_agg_key] = _agg_code\n"
    )
    marker = "def last_completed_session("
    assert engine_src.count(marker) == 1
    engine_src = engine_src.replace(
        marker, aggregate_date_patch + "\n" + marker, 1
    )
    engine_file.write_text(engine_src, encoding="utf-8")

    # Ottimizzazione della sola copia scientifica temporanea.
    engine_src = engine_file.read_text(encoding="utf-8")
    memory_patch = '\n# Riduzione memoria nell\'aggregatore sandbox.\n_aggregate_key = "operational/run_marketsentinel.py"\n_aggregate_code = _SOURCES[_aggregate_key]\n_old_load = "        df = pd.read_csv(path, low_memory=False)"\n_new_load = """        required_columns = {\n            "Ticker", "Date", "PHASE", "Close", "SAR",\n            "SAR_FLIP_UP", "V4015_HA_DIRECTION",\n            "V4015_HA_OPEN", "V4015_HA_CLOSE", "V4015_BB_MID",\n            "V4015_ATR_PCT", "V4012_SAR_DISTANCE_PCT",\n            "V4015_BB_WIDTH_PCT",\n            "V4012_DISTANCE_TO_BB_LOWER_PCT",\n            "V4015_DIST_26W_LOW_PCT",\n        }\n        df = pd.read_csv(\n            path,\n            usecols=lambda c: c in required_columns,\n            low_memory=False,\n        )"""\nif _aggregate_code.count(_old_load) == 1:\n    _aggregate_code = _aggregate_code.replace(\n        _old_load, _new_load, 1\n    )\nelif _aggregate_code.count(_new_load) != 1:\n    raise RuntimeError("Patch memoria: stato non riconosciuto")\n_SOURCES[_aggregate_key] = _aggregate_code\n'
    insertion = "def last_completed_session("
    assert engine_src.count(insertion) == 1
    engine_src = engine_src.replace(
        insertion, memory_patch + "\n" + insertion, 1
    )
    engine_file.write_text(engine_src, encoding="utf-8")



    runner = sandbox / "_run_scientific.py"
    script = (
        "CHANGED_MARKETS = " + repr(tuple(changed_markets))
        + "\n" + script
    )
    runner.write_text(script, encoding="utf-8")


    # Patch esclusiva della sandbox: date finali differenti consentite.
    engine_src = engine_file.read_text(encoding="utf-8")
    date_patch = """
for _name in (
    'operational/update_phase.py',
    'operational/update_phase_europe40.py',
    'operational/update_phase_usa80.py',
):
    _code = _SOURCES[_name]
    _old = 'assert current_df["Date"].nunique() == 1'
    _new = 'assert current_df["Date"].notna().all()'
    assert _code.count(_old) == 1, _name
    _SOURCES[_name] = _code.replace(_old, _new, 1)
"""
    marker_date = "def last_completed_session("
    assert engine_src.count(marker_date) == 1
    engine_src = engine_src.replace(
        marker_date, date_patch + "\n" + marker_date, 1
    )
    engine_file.write_text(engine_src, encoding="utf-8")

    return _execute_prepared_scientific(sandbox)


def _execute_prepared_scientific(sandbox):
    runner = sandbox / "_run_scientific.py"
    if not runner.is_file():
        raise RuntimeError("Script scientifico preparato assente")
    import time

    command = [sys.executable, "-u", str(runner)]
    output_tail = []
    started = time.monotonic()

    process = subprocess.Popen(
        command,
        cwd=str(sandbox),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    import threading
    import signal

    timed_out = threading.Event()

    def enforce_timeout():
        timed_out.set()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(2700, enforce_timeout)
    timer.daemon = True
    timer.start()

    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            output_tail.append(line)
            if len(output_tail) > 200:
                output_tail.pop(0)

        returncode = process.wait(timeout=30)

    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise

    finally:
        timer.cancel()

    if timed_out.is_set():
        raise TimeoutError(
            "Timeout scientifico DAILY: superati 45 minuti"
        )

    elapsed = time.monotonic() - started
    print(
        f"[SCIENTIFIC] PROCESSO TERMINATO "
        f"exit={returncode} durata={elapsed:.1f}s",
        flush=True,
    )

    if returncode:
        raise RuntimeError(
            "Ricalcolo scientifico non riuscito "
            f"(exit code {returncode}):\\n"
            + "".join(output_tail)[-12000:]
        )



# MS_PROVISIONAL_GUARD_V1
def provisional_markets(scientific_base):
    """Do not finalize today's daily candle, even if Yahoo reports it.

    Uses each exchange's local calendar day, deliberately conservative.
    Final closing prices still require independent verification.
    """
    from zoneinfo import ZoneInfo
    now_utc = datetime.now(timezone.utc)
    pending = []
    for prefix in MARKETS.values():
        tz = ZoneInfo('America/New_York') if prefix.startswith('usa') else ZoneInfo('Europe/Rome')
        today = now_utc.astimezone(tz).date()
        path = scientific_base / f'{prefix}_daily_master.csv'
        latest = pd.read_csv(path, usecols=['Date'])['Date'].max()
        if pd.Timestamp(latest).date() >= today:
            pending.append(prefix)
    return pending

def publish_provisional_only(destination, result):
    """Publish UI output atomically, without touching historical scientific cache."""
    output_file = destination / 'marketsentinel_intraday.csv'
    temporary_file = destination / ('marketsentinel_intraday.' + uuid.uuid4().hex + '.tmp')
    try:
        result.to_csv(temporary_file, index=False)
        os.replace(temporary_file, output_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()
    print('[DAILY] PUBBLICATO SOLO OUTPUT PROVVISORIO; CACHE SCIENTIFICA INVARIATA', flush=True)

def publish_scientific_cache(sandbox, destination, result):
    """Conserva gli intermedi DAILY senza alterare i dati consolidati.

    Pubblica una nuova directory completa solo dopo aver verificato i file.
    Non implementa ancora un aggiornamento scientifico incrementale.
    """
    scientific_base = sandbox / "data" / "marketsentinel_daily"
    prefixes = tuple(MARKETS.values())
    suffixes = (
        "_daily_master.csv",
        "_phase_history.csv",
        "_current_phase.csv",
        "_v160_features.csv",
    )
    required = [
        scientific_base / f"{prefix}{suffix}"
        for prefix in prefixes
        for suffix in suffixes
    ]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(
            "Cache DAILY incompleta; nessun risultato pubblicato: "
            + ", ".join(missing)
        )

    staging = destination / (".scientific_cache_" + uuid.uuid4().hex)
    cache = destination / "scientific_cache"
    previous = destination / ".scientific_cache_previous"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for source in required:
            shutil.copy2(source, staging / source.name)

        (staging / "daily_science_sha256.txt").write_text(
            daily_science_fingerprint() + "\n"
        )

        if previous.exists():
            shutil.rmtree(previous)
        if cache.exists():
            os.replace(cache, previous)
        try:
            os.replace(staging, cache)
        except Exception:
            if previous.exists() and not cache.exists():
                os.replace(previous, cache)
            raise

        output_file = destination / "marketsentinel_intraday.csv"
        temporary_file = destination / "marketsentinel_intraday.csv.tmp"
        result.to_csv(temporary_file, index=False)
        os.replace(temporary_file, output_file)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print("CACHE SCIENTIFICA DAILY SALVATA:", cache)


def daily_science_fingerprint():
    """SHA-256 dei sorgenti scientifici DAILY."""
    import hashlib
    sources = (
        "operational/phase_v5_daily.py",
        "weekly_v40_39_phase_canonical.py",
        "weekly_v40_5_phase_dynamics_shadow_v5.py",
        "ms_daily_engine.py",
        "operational/phase_v4_fast.py",
    )
    h = hashlib.sha256()
    for name in sources:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()


def daily_cache_science_valid(cache):
    marker = cache / "daily_science_sha256.txt"
    return marker.is_file() and marker.read_text().strip() == daily_science_fingerprint()


def same_ohlcv(path_a, path_b):
    cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
    if not path_a.is_file() or not path_b.is_file():
        return False
    a = pd.read_csv(path_a, usecols=lambda c: c in cols)
    b = pd.read_csv(path_b, usecols=lambda c: c in cols)
    if set(cols) - set(a.columns) or set(cols) - set(b.columns):
        return False
    a = a[cols].sort_values(["Ticker", "Date"]).reset_index(drop=True)
    b = b[cols].sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return a.equals(b)


def publish_weekly_ohlcv_snapshot(sandbox, destination):
    """Pubblica OHLCV DAILY separati dalla cache scientifica consolidata."""
    import pandas as pd

    source_dir = sandbox / "data" / "marketsentinel_daily"
    target = destination / "weekly_ohlcv_snapshot"
    staging = destination / (".weekly_ohlcv_" + uuid.uuid4().hex)
    previous = destination / ".weekly_ohlcv_previous"

    expected = {
        "italy40": 40,
        "europe40": 40,
        "usa80": 80,
        "usa40": 39,
    }

    staging.mkdir(parents=True, exist_ok=False)

    try:
        for market, count in expected.items():
            name = f"{market}_daily_master.csv"
            src = source_dir / name
            df = pd.read_csv(src)

            required = {
                "Ticker", "Date", "Open", "High",
                "Low", "Close", "Volume"
            }

            if not required.issubset(df.columns):
                raise RuntimeError(f"{market}: colonne OHLCV mancanti")

            if df["Ticker"].nunique() != count:
                raise RuntimeError(f"{market}: universo incompleto")

            if df.duplicated(["Ticker", "Date"]).any():
                raise RuntimeError(f"{market}: duplicati OHLCV")

            dates = pd.to_datetime(df["Date"], utc=True, errors="raise")
            if dates.isna().any():
                raise RuntimeError(f"{market}: date non valide")

            shutil.copy2(src, staging / name)

        if previous.exists():
            shutil.rmtree(previous)

        if target.exists():
            target.rename(previous)

        try:
            staging.rename(target)
        except Exception:
            if previous.exists() and not target.exists():
                previous.rename(target)
            raise

    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(
        "[WEEKLY] SNAPSHOT OHLCV DAILY PUBBLICATO:",
        target,
        flush=True,
    )


def _ms_weekend_cache_fast_path():
    """Salta il lavoro DAILY nel weekend solo con cache integralmente valida."""
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("Europe/Rome"))
    if now.weekday() < 5:
        return False

    root = Path(__file__).resolve().parent
    destination = root / "data" / "marketsentinel_intraday"
    cache = destination / "scientific_cache"
    published = destination / "marketsentinel_intraday.csv"
    snapshot = destination / "weekly_ohlcv_snapshot"

    if not published.is_file() or not cache.is_dir():
        return False

    if not daily_cache_science_valid(cache):
        return False

    # Non saltare mai il lavoro se esistono quotazioni da recuperare.
    try:
        gaps = read_gaps()
        if not isinstance(gaps, dict):
            return False
        for market_gaps in gaps.values():
            if not isinstance(market_gaps, dict):
                return False
            if any(bool(dates) for dates in market_gaps.values()):
                return False
    except Exception:
        return False

    suffixes = (
        "_daily_master.csv",
        "_phase_history.csv",
        "_current_phase.csv",
        "_v160_features.csv",
    )

    for prefix in MARKETS.values():
        if not all(
            (cache / f"{prefix}{suffix}").is_file()
            for suffix in suffixes
        ):
            return False

        base_master = BASE / f"{prefix}_daily_master.csv"
        cached_master = cache / f"{prefix}_daily_master.csv"
        snapshot_master = snapshot / f"{prefix}_daily_master.csv"

        if not snapshot_master.is_file():
            return False

        try:
            base_df = pd.read_csv(base_master, usecols=["Ticker", "Date"])
            cache_df = pd.read_csv(cached_master, usecols=["Ticker", "Date"])

            if (
                base_df["Ticker"].nunique() != EXPECTED[prefix]
                or cache_df["Ticker"].nunique() != EXPECTED[prefix]
            ):
                return False

            base_dates = pd.to_datetime(base_df["Date"], utc=True, errors="raise")
            cache_dates = pd.to_datetime(cache_df["Date"], utc=True, errors="raise")

            if cache_dates.max() < base_dates.max():
                return False

            # Cache e snapshot devono invece coincidere rigorosamente.
            if not same_ohlcv(cached_master, snapshot_master):
                return False
        except Exception:
            return False

    print(
        "[DAILY FAST PATH] Weekend: cache scientifica e snapshot "
        "coerenti; nessun download o ricalcolo necessario.",
        flush=True,
    )
    return True


def main():
    if _ms_weekend_cache_fast_path():
        return
    with tempfile.TemporaryDirectory(
        prefix="marketsentinel_intraday_"
    ) as temp:
        sandbox = Path(temp)

        updated, failures = prepare_sandbox(sandbox)

        if updated == 0:
            raise RuntimeError(
                "Nessuna quotazione aggiornata disponibile."
            )

        if os.environ.get("MS_DAILY_FAST_REFRESH", "0") == "1":
            from ms_daily_fast_refresh import publish
            publish(sandbox, ROOT, MARKETS)
            print("[FAST REFRESH] Motore scientifico non avviato", flush=True)
            return

        # Nessun ricalcolo se i 199 master OHLCV sono identici alla
        # precedente esecuzione scientifica pubblicata.
        destination = ROOT / "data" / "marketsentinel_intraday"
        cache = destination / "scientific_cache"
        published = destination / "marketsentinel_intraday.csv"
        science_valid = daily_cache_science_valid(cache)
        unchanged = cache.is_dir() and published.is_file() and science_valid
        market_unchanged = {}
        if unchanged:
            market_unchanged = {
                prefix: same_ohlcv(
                    cache / f"{prefix}_daily_master.csv",
                    sandbox / "data" / "marketsentinel_daily" / f"{prefix}_daily_master.csv",
                )
                for prefix in MARKETS.values()
            }
            unchanged = all(market_unchanged.values())
        if unchanged:
            print("NESSUNA VARIAZIONE OHLCV: cache scientifica DAILY riutilizzata; nessun ricalcolo.", flush=True)
            return

        changed_markets = []
        required_suffixes = (
            "_phase_history.csv",
            "_current_phase.csv",
            "_v160_features.csv",
        )

        for prefix in MARKETS.values():
            old_master = cache / f"{prefix}_daily_master.csv"
            new_master = (
                sandbox / "data" / "marketsentinel_daily"
                / f"{prefix}_daily_master.csv"
            )

            reusable = (
                published.is_file()
                and old_master.is_file()
                and all(
                    (cache / f"{prefix}{suffix}").is_file()
                    for suffix in required_suffixes
                )
            )

            reusable = reusable and science_valid
            if reusable:
                reusable = market_unchanged.get(prefix)
                if reusable is None:
                    reusable = same_ohlcv(old_master, new_master)

            if reusable:
                target = (
                    sandbox / "data" / "marketsentinel_daily"
                )
                for suffix in required_suffixes:
                    shutil.copy2(
                        cache / f"{prefix}{suffix}",
                        target / f"{prefix}{suffix}"
                    )
                print(
                    f"[SCIENTIFIC] RIUSO CACHE {prefix}",
                    flush=True
                )
            else:
                changed_markets.append(prefix)
                print(
                    f"[SCIENTIFIC] RICALCOLO {prefix}",
                    flush=True
                )

        if (
            os.environ.get("MS_DAILY_WINDOW_520") == "1"
            and changed_markets
        ):
            import ms_daily_window520
            try:
                ms_daily_window520.prepare(
                    sandbox, changed_markets, cache, rows=520
                )
                print(
                    "[SCIENTIFIC] WINDOW520 ATTIVATA",
                    flush=True
                )
            except RuntimeError as exc:
                message = str(exc)
                fallback_reasons = (
                    "Storico OHLCV modificato:",
                    "Sedute storiche mancanti:",
                    "Universo titoli cambiato:",
                    "Backfill storico:",
                )
                if not message.startswith(fallback_reasons):
                    raise
                print(
                    "[SCIENTIFIC] CACHE NON COMPATIBILE:",
                    message,
                    flush=True
                )
                print(
                    "[SCIENTIFIC] RICALCOLO COMPLETO AUTOMATICO",
                    flush=True
                )

        try:
            calculate_in_sandbox(sandbox, changed_markets)
        except Exception as exc:
            marker = sandbox / "_ms_window520_state.json"
            if not marker.exists():
                raise

            print(
                "[WINDOW520] FALLIMENTO: ripristino e ricalcolo completo",
                repr(exc),
                flush=True,
            )

            import json
            import ms_daily_window520

            state = json.loads(marker.read_text())
            archive = sandbox / "_ms_window520_full"
            base = sandbox / "data" / "marketsentinel_daily"

            # Ripristina tutti gli input e gli output scientifici
            # dalla cache precedente, prima del ricalcolo completo.
            for market in state["markets"]:
                name = f"{market}_daily_master.csv"
                shutil.copy2(archive / name, base / name)

                for suffix in (
                    "_phase_history.csv",
                    "_current_phase.csv",
                    "_v160_features.csv",
                ):
                    source = cache / f"{market}{suffix}"
                    target = base / f"{market}{suffix}"
                    if not source.is_file():
                        raise RuntimeError(
                            f"Fallback impossibile: cache assente {source}"
                        )
                    shutil.copy2(source, target)

            marker.unlink()

            _execute_prepared_scientific(sandbox)

        output = (
            sandbox
            / "data"
            / "marketsentinel_daily"
            / "marketsentinel_200_phase_nuova.csv"
        )

        result = pd.read_csv(output)

        if len(result) != 199:
            raise RuntimeError(
                f"Risultato incompleto: {len(result)}/199 titoli."
            )

        if os.environ.get("MS_DAILY_DRY_RUN", "0") == "1":
            if os.environ.get("MS_DAILY_AUDIT", "0") == "1":
                audit = ROOT / "data" / "ms_daily_scientific_audit"
                if audit.exists():
                    raise RuntimeError(
                        "Cartella audit già esistente: " + str(audit)
                    )

                audit.mkdir(parents=True)

                try:
                    base = sandbox / "data" / "marketsentinel_daily"

                    names = [
                        f"{market}{suffix}"
                        for market in (
                            "italy40", "europe40", "usa80", "usa40"
                        )
                        for suffix in (
                            "_v160_features.csv",
                            "_phase_history.csv",
                            "_current_phase.csv",
                        )
                    ]

                    names.append(
                        "marketsentinel_200_phase_nuova.csv"
                    )

                    for name in names:
                        src = base / name
                        if not src.is_file():
                            raise RuntimeError(
                                "File audit mancante: " + str(src)
                            )
                        shutil.copy2(src, audit / name)

                    print(
                        "[AUDIT] FILE SCIENTIFICI CONSERVATI:",
                        len(names),
                        flush=True
                    )
                    print(
                        "[AUDIT] CARTELLA:",
                        audit,
                        flush=True
                    )
                except Exception:
                    shutil.rmtree(audit)
                    raise

            print(
                "[DRY RUN] CALCOLO SCIENTIFICO COMPLETATO",
                flush=True
            )
            print(
                "[DRY RUN] TITOLI VERIFICATI:",
                len(result),
                flush=True
            )
            print(
                "[DRY RUN] PUBBLICAZIONE DISABILITATA",
                flush=True
            )
            print(
                "[DRY RUN] CACHE OPERATIVA NON MODIFICATA",
                flush=True
            )
            return

        destination = (
            ROOT
            / "data"
            / "marketsentinel_intraday"
        )
        destination.mkdir(parents=True, exist_ok=True)

        publish_weekly_ohlcv_snapshot(sandbox, destination)

        pending = provisional_markets(sandbox / 'data' / 'marketsentinel_daily')
        if pending:
            print('[DAILY] SEDUTE PROVVISORIE: ' + ', '.join(pending), flush=True)
            publish_provisional_only(destination, result)
        else:
            publish_scientific_cache(sandbox, destination, result)
            write_gaps(_RUN_GAPS)
            print("[YAHOO] Registro cloni pubblicato:", sum(len(v) for market in _RUN_GAPS.values() for v in market.values()), flush=True)

        print(
            "INTRADAY COMPLETATO:",
            len(result),
            "titoli;",
            updated,
            "quotazioni recuperate;",
            len(failures),
            "errori",
        )


if __name__ == "__main__":
    import fcntl

    lock_path = ROOT / "data" / "marketsentinel_intraday" / ".daily_refresh.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+") as lock_file:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            print(
                "AGGIORNAMENTO DAILY GIÀ IN CORSO: "
                "nessuna seconda esecuzione avviata.",
                flush=True
            )
            sys.exit(3)

        try:
            main()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
