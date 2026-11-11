import shutil
"""Safe 520-session scientific cache reconstruction.

The full cached OHLCV history must be append-only. Any historical correction
requires the original full-history calculation instead of this optimization.
"""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

MARKER = '_ms_window520_state.json'
CUM = ('V4015_ADL', 'V4015_OBV')
SUFFIXES = ('_phase_history.csv', '_current_phase.csv', '_v160_features.csv')
KEYS = ['Ticker', 'Date']
OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


def _read(path):
    d = pd.read_csv(path, low_memory=False)
    if not set(KEYS).issubset(d.columns):
        raise RuntimeError(f'Ticker/Date assenti: {path}')
    d['Date'] = pd.to_datetime(d['Date'], utc=True, errors='raise', format='mixed').dt.tz_localize(None)
    if d.duplicated(KEYS).any() or d['Date'].isna().any():
        raise RuntimeError(f'Chiavi duplicate o date nulle: {path}')
    return d.sort_values(KEYS).reset_index(drop=True)


def _write(d, path):
    d = d.copy()
    d['Date'] = pd.to_datetime(d['Date']).dt.strftime('%Y-%m-%d')
    d.to_csv(path, index=False)


def _equal(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-6, atol=1e-8, equal_nan=True))
    except (ValueError, TypeError):
        return str(a) == str(b)


def _validate_append_only(old, new, market):
    if not set(OHLCV).issubset(old.columns) or not set(OHLCV).issubset(new.columns):
        raise RuntimeError(f'OHLCV incompleto: {market}')
    old = old.set_index(KEYS)
    new = new.set_index(KEYS)
    if not old.index.isin(new.index).all():
        raise RuntimeError(f'Sedute storiche mancanti: {market}; usare ricalcolo completo')
    if set(old.index.get_level_values(0)) != set(new.index.get_level_values(0)):
        raise RuntimeError(f'Universo titoli cambiato: {market}; usare ricalcolo completo')
    aligned = new.loc[old.index, OHLCV]
    changed = np.zeros(len(old), dtype=bool)

    for col in OHLCV:
        a = pd.to_numeric(old[col], errors='coerce').to_numpy(dtype=float)
        b = pd.to_numeric(aligned[col], errors='coerce').to_numpy(dtype=float)
        changed |= ~np.isclose(
            a, b, rtol=1e-9, atol=1e-9, equal_nan=True
        )

    if changed.any():
        dates = old.index.get_level_values('Date')
        changed_dates = dates[changed]
        recent_sessions = sorted(
            set(new.index.get_level_values('Date'))
        )[-10:]

        if changed_dates.min() < recent_sessions[0]:
            raise RuntimeError(
                f'Storico OHLCV modificato: {market}; '
                'rettifiche oltre 10 sedute'
            )

        print(
            f'[WINDOW520] {market}: '
            f'{changed.sum()} righe OHLCV rettificate, '
            f'dal {changed_dates.min().date()}',
            flush=True
        )
        raise RuntimeError(
            f'Storico OHLCV modificato: {market}; '
            'ricalcolo incrementale delle rettifiche non ancora abilitato'
        )
    for ticker, g in new.groupby(level=0):
        prev_end = old.loc[ticker].index.max()
        added = g.index.get_level_values('Date').difference(old.loc[ticker].index)
        if len(added) and added.min() <= prev_end:
            raise RuntimeError(f'Backfill storico: {market}/{ticker}; usare ricalcolo completo')


def prepare(sandbox, changed_markets, cache, rows=520):
    if not changed_markets:
        return
    if rows < 260:
        raise RuntimeError('Finestra troppo corta')
    sandbox, cache = Path(sandbox), Path(cache)
    base = sandbox / 'data/marketsentinel_daily'
    archive = sandbox / '_ms_window520_full'
    if archive.exists() or (sandbox / MARKER).exists():
        raise RuntimeError('Sandbox WINDOW520 gia preparata')
    # Verify *all* inputs before changing any master in the sandbox.
    inputs = {}
    for market in changed_markets:
        master = base / f'{market}_daily_master.csv'
        full = _read(master)
        old = _read(cache / master.name)
        _validate_append_only(old, full, market)
        for suffix in SUFFIXES:
            if not (cache / f'{market}{suffix}').is_file():
                raise RuntimeError(f'Cache scientifica incompleta: {market}{suffix}')
        counts = full.groupby('Ticker').size()
        if (counts < rows + 1).any():
            raise RuntimeError(f'Storico insufficiente: {market}')
        inputs[market] = (master, full)
    archive.mkdir(exist_ok=False)
    for market, (master, full) in inputs.items():
        _write(full, archive / master.name)
        recent = full.groupby('Ticker', sort=False, group_keys=False).tail(rows)
        _write(recent, master)
        print(f'[WINDOW520] {market}: {len(full)} -> {len(recent)} righe', flush=True)
    (sandbox / MARKER).write_text(json.dumps({
        'markets': list(changed_markets), 'cache': str(cache.resolve())
    }))


def _merge(old, recent):
    """Keep every published historical observation; append only new dates."""
    old_keys = pd.MultiIndex.from_frame(old[KEYS])
    new_keys = pd.MultiIndex.from_frame(recent[KEYS])
    additions = recent.loc[~new_keys.isin(old_keys)]
    combined = pd.concat([old, additions], ignore_index=True)
    if combined.duplicated(KEYS).any():
        raise RuntimeError('Chiavi duplicate durante merge')
    return combined.sort_values(KEYS).reset_index(drop=True)



def _verify_overlap(old, new, market, label, *, cols=None):
    old_i = old.set_index(KEYS)
    new_i = new.set_index(KEYS)

    if cols is None:
        if set(old_i.columns) != set(new_i.columns):
            raise RuntimeError(f'Schema {label} cambiato: {market}')
        cols = list(old_i.columns)

    old_tickers = set(old_i.index.get_level_values(0))
    new_tickers = set(new_i.index.get_level_values(0))

    if old_tickers != new_tickers:
        raise RuntimeError(f'Universo {label} cambiato: {market}')

    old_last = old_i.groupby(level=0, sort=False).tail(1)
    old_last_dates = old_last.index.get_level_values('Date')

    new_last_common = new_i.loc[
        new_i.index.isin(old_i.index)
    ].groupby(level=0, sort=False).tail(1)

    if len(new_last_common) != len(old_tickers):
        raise RuntimeError(f'Overlap {label} mancante: {market}')

    keys = new_last_common.index

    a = old_i.loc[keys, cols]
    b = new_i.loc[keys, cols]

    for col in cols:
        av = a[col]
        bv = b[col]

        an = pd.to_numeric(av, errors='coerce')
        bn = pd.to_numeric(bv, errors='coerce')

        numeric = an.notna() & bn.notna()

        equal = pd.Series(False, index=av.index)

        if numeric.any():
            equal.loc[numeric] = np.isclose(
                an.loc[numeric].to_numpy(dtype=float),
                bn.loc[numeric].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            )

        remaining = ~numeric

        if remaining.any():
            equal.loc[remaining] = (
                (av.loc[remaining].isna() & bv.loc[remaining].isna())
                | (
                    av.loc[remaining].astype(str)
                    == bv.loc[remaining].astype(str)
                )
            )

        if not equal.all():
            bad = equal.index[~equal][0]
            raise RuntimeError(
                f'DRIFT AL CONFINE {label}: '
                f'{market}/{bad[0]}/{bad[1].date()}/{col}'
            )

    old_max = old_i.groupby(level=0).apply(
        lambda g: g.index.get_level_values('Date').max()
    )

    new_extra = new_i.loc[~new_i.index.isin(old_i.index)]

    if not new_extra.empty:
        extra_tickers = new_extra.index.get_level_values(0)
        extra_dates = new_extra.index.get_level_values('Date')

        for ticker, date in zip(extra_tickers, extra_dates):
            if date <= old_max[ticker]:
                raise RuntimeError(
                    f'Backfill {label}: {market}/{ticker}'
                )

def finalize(sandbox):
    sandbox = Path(sandbox)
    state_file = sandbox / MARKER
    if not state_file.exists():
        return
    state = json.loads(state_file.read_text())
    cache = Path(state['cache'])
    base = sandbox / 'data/marketsentinel_daily'
    archive = sandbox / '_ms_window520_full'

    # Stage all outputs in memory first. No partial reconstruction is accepted.
    pending = {}
    for market in state['markets']:
        full = _read(archive / f'{market}_daily_master.csv')
        reference = _read(cache / f'{market}_v160_features.csv')
        path = base / f'{market}_v160_features.csv'
        fresh = _read(path)
        if not set(CUM).issubset(reference.columns) or not set(CUM).issubset(fresh.columns):
            raise RuntimeError(f'Indicatori cumulativi mancanti: {market}')
        ref_i = reference.set_index(KEYS)
        adjusted = []
        for ticker, g in fresh.groupby('Ticker', sort=False):
            ref_ticker = ref_i.loc[ticker] if ticker in ref_i.index.get_level_values(0) else None
            if ref_ticker is None:
                raise RuntimeError(f'Ticker V160 assente: {market}/{ticker}')
            overlap = g['Date'].loc[g['Date'].isin(ref_ticker.index)]
            if overlap.empty:
                raise RuntimeError(f'Overlap V160 assente: {market}/{ticker}')
            anchor = overlap.min()
            # Anchor cumulative series on an already-validated cached date.
            # This also handles non-zero starting offsets and OBV first-row rules.
            row = g.loc[g['Date'].eq(anchor)].iloc[0]
            g = g.copy()
            for col in CUM:
                delta = float(ref_ticker.at[anchor, col]) - float(row[col])
                g[col] = pd.to_numeric(g[col], errors='raise') + delta
            adjusted.append(g)
        recent = pd.concat(adjusted, ignore_index=True)
        _verify_overlap(reference, recent, market, 'V160')
        combined_v160 = _merge(reference, recent)

        hpath = base / f'{market}_phase_history.csv'
        old_phase = _read(cache / hpath.name)
        new_phase = _read(hpath)
        _verify_overlap(old_phase, new_phase, market, 'PHASE')
        combined_phase = _merge(old_phase, new_phase)

        # Current-phase output is generated by the scientific engine from the window.
        current = _read(base / f'{market}_current_phase.csv')
        if current['Ticker'].nunique() != full['Ticker'].nunique():
            raise RuntimeError(f'Current phase incompleta: {market}')
        same_v160_keys = (
            len(reference) == len(combined_v160)
            and reference[KEYS].equals(combined_v160[KEYS])
            and reference.columns.equals(combined_v160.columns)
        )

        old_keys = pd.MultiIndex.from_frame(reference[KEYS])
        recent_keys = pd.MultiIndex.from_frame(recent[KEYS])

        no_new_rows = bool(recent_keys.isin(old_keys).all())

        fastcopy = (
            os.environ.get('MS_DAILY_FASTCOPY', '0') == '1'
            and same_v160_keys
            and no_new_rows
        )

        if fastcopy:
            ref_overlap = reference.set_index(KEYS)
            new_overlap = recent.set_index(KEYS)

            common = new_overlap.index.intersection(ref_overlap.index)

            if len(common) != len(new_overlap):
                raise RuntimeError(
                    f'[FASTCOPY] Date non validate: {market}'
                )

            for col in reference.columns:
                if col in KEYS:
                    continue

                a = ref_overlap.loc[common, col]
                b = new_overlap.loc[common, col]

                if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                    equal = np.isclose(
                        a.to_numpy(dtype=float),
                        b.to_numpy(dtype=float),
                        rtol=1e-6,
                        atol=1e-8,
                        equal_nan=True
                    )
                else:
                    equal = (
                        a.fillna('<NULL>').astype(str).to_numpy()
                        == b.fillna('<NULL>').astype(str).to_numpy()
                    )

                if not equal.all():
                    raise RuntimeError(
                        f'[FASTCOPY] Differenza scientifica: {market}/{col}'
                    )

            print(
                f'[FASTCOPY] {market}: overlap completo validato',
                flush=True
            )

        pending[market] = (
            full, combined_v160, combined_phase, fastcopy, reference
        )

    for market, (full, v160, phase, fastcopy, reference) in pending.items():
        v160_path = base / f'{market}_v160_features.csv'

        if fastcopy:
            shutil.copy2(cache / v160_path.name, v160_path)
            print(
                f'[FASTCOPY] {market}: storico V160 riutilizzato',
                flush=True
            )
        elif (
            os.environ.get('MS_DAILY_INCREMENTAL', '0') == '1'
            and os.environ.get('MS_DAILY_WINDOW_520', '0') == '1'
        ):
            from ms_v160_incremental import incremental_write_v4

            reference_path = cache / v160_path.name

            staged_path = v160_path.with_name(
                v160_path.name + '.incremental_v4'
            )

            if staged_path.exists():
                raise RuntimeError(
                    f'[FASTCSV V4] File temporaneo già presente: {market}'
                )

            try:
                result = incremental_write_v4(
                    reference_path,
                    reference,
                    v160,
                    staged_path
                )

                staged_path.replace(v160_path)

                print(
                    f'[FASTCSV V4] {market}: '
                    f'{result["new_rows"]} nuove righe validate',
                    flush=True
                )
            finally:
                if staged_path.exists():
                    staged_path.unlink()
        else:
            _write(v160, v160_path)
        _write(phase, base / f'{market}_phase_history.csv')
        _write(full, base / f'{market}_daily_master.csv')
        print(f'[WINDOW520] {market}: storico completo ripristinato', flush=True)
    state_file.unlink()
