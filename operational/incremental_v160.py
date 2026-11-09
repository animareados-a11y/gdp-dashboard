from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from operational.incremental_phase import BASE
import weekly_v40_15_score_feature_dataset as v15
import weekly_v40_12_phase_strength_dataset as v12
from weekly_v40_1_core_phase_engine import add_psar


def normalize_dates(df):
    df["Date"] = pd.to_datetime(
        df["Date"], utc=True
    ).dt.tz_localize(None)
    return df


def calculate_one_ticker(raw, phase):
    """Riproduce la pipeline V160 originale per un singolo titolo."""
    raw = normalize_dates(raw.copy())
    phase = normalize_dates(phase.copy())

    raw = raw.sort_values("Date").reset_index(drop=True)

    g = v15.calculate_features(raw)
    g = normalize_dates(g)
    g = g.drop(columns=["PHASE"], errors="ignore")

    g = g.merge(
        phase[["Ticker", "Date", "PHASE"]],
        on=["Ticker", "Date"],
        how="left",
        validate="one_to_one"
    )

    if g["PHASE"].isna().any():
        raise ValueError("PHASE mancanti")

    sar = normalize_dates(add_psar(raw.copy()))

    g = g.merge(
        sar[[
            "Date", "SAR", "SAR_SIDE", "SAR_AGE",
            "SAR_FLIP_UP", "SAR_FLIP_DOWN"
        ]],
        on="Date",
        how="left",
        validate="one_to_one"
    )

    close = pd.to_numeric(g["Close"], errors="coerce")
    middle = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std(ddof=0)

    g["BB_MIDDLE"] = middle
    g["BB_MID"] = middle
    g["BB_UPPER"] = middle + 2 * sd
    g["BB_LOWER"] = middle - 2 * sd

    g = v12.normalize_base_columns(g)
    g = v12.add_phase_strength_features(g)

    return g


def identify_changed_tickers(market):
    """Confronta le chiavi Ticker/Date di prezzi e V160."""
    prices = normalize_dates(pd.read_csv(
        BASE / f"{market}_weekly_master.csv",
        usecols=["Ticker", "Date"]
    ))

    path = BASE / f"{market}_v160_features.csv"

    if not path.exists():
        return set(prices["Ticker"].unique())

    saved = normalize_dates(pd.read_csv(
        path, usecols=["Ticker", "Date"]
    ))

    changed = set()

    for ticker, g in prices.groupby("Ticker"):
        expected = set(g["Date"])
        actual = set(
            saved.loc[saved["Ticker"] == ticker, "Date"]
        )

        if expected != actual:
            changed.add(ticker)

    return changed


if __name__ == "__main__":
    total = 0

    for market in ["italy40", "europe40", "usa80", "usa40"]:
        changed = identify_changed_tickers(market)
        print(
            market,
            "TITOLI DA RICALCOLARE:",
            len(changed)
        )
        total += len(changed)

    print("TOTALE:", total)
    print("ARCHIVI SCIENTIFICI INVARIATI")


def prepare_incremental_v160(market, changed_tickers=None, dry_run=True):
    """
    Ricompone V160:
    - ricalcola l'intera storia dei soli ticker modificati;
    - riutilizza i dati salvati degli altri ticker;
    - verifica la coerenza prima di qualsiasi scrittura.
    """
    raw = normalize_dates(pd.read_csv(
        BASE / f"{market}_weekly_master.csv"
    ))
    phase = normalize_dates(pd.read_csv(
        BASE / f"{market}_phase_history.csv"
    ))

    output_path = BASE / f"{market}_v160_features.csv"

    if output_path.exists():
        saved = normalize_dates(pd.read_csv(output_path))
    else:
        saved = pd.DataFrame()

    changed = identify_changed_tickers(market)
    changed.update(changed_tickers or [])

    raw_tickers = set(raw["Ticker"].unique())
    saved_tickers = (
        set(saved["Ticker"].unique()) if not saved.empty else set()
    )

    changed.update(raw_tickers - saved_tickers)

    # Le PHASE possono essere cambiate anche senza nuove date.
    if not saved.empty:
        comparison = saved[["Ticker", "Date", "PHASE"]].merge(
            phase[["Ticker", "Date", "PHASE"]],
            on=["Ticker", "Date"],
            how="inner",
            suffixes=("_saved", "_current"),
            validate="one_to_one"
        )

        mismatch = (
            comparison["PHASE_saved"].fillna("").astype(str)
            != comparison["PHASE_current"].fillna("").astype(str)
        )

        changed.update(
            comparison.loc[mismatch, "Ticker"].unique()
        )

    if not changed:
        print(market, "NESSUN RICALCOLO NECESSARIO")
        return saved, set()

    unknown = changed - raw_tickers
    if unknown:
        raise ValueError(f"Ticker sconosciuti: {sorted(unknown)}")

    frames = []

    for ticker in sorted(raw_tickers):
        if ticker in changed:
            ticker_raw = raw.loc[raw["Ticker"] == ticker]
            ticker_phase = phase.loc[phase["Ticker"] == ticker]

            result = calculate_one_ticker(
                ticker_raw, ticker_phase
            )
            frames.append(result)
        else:
            frames.append(
                saved.loc[saved["Ticker"] == ticker].copy()
            )

    combined = pd.concat(
        frames, ignore_index=True, sort=False
    )

    combined = normalize_dates(combined)
    combined = combined.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    if combined.duplicated(["Ticker", "Date"]).any():
        raise ValueError("Duplicati nell'archivio V160")

    if raw.duplicated(["Ticker", "Date"]).any():
        raise ValueError("Duplicati nell'archivio prezzi")

    expected = pd.MultiIndex.from_frame(
        raw[["Ticker", "Date"]]
    )
    actual = pd.MultiIndex.from_frame(
        combined[["Ticker", "Date"]]
    )

    if len(combined) != len(raw):
        raise ValueError("Numero righe V160 non coerente")

    if not expected.difference(actual).empty:
        raise ValueError("Date mancanti in V160")

    if not actual.difference(expected).empty:
        raise ValueError("Date extra in V160")

    phase_check = combined[
        ["Ticker", "Date", "PHASE"]
    ].merge(
        phase[["Ticker", "Date", "PHASE"]],
        on=["Ticker", "Date"],
        suffixes=("_v160", "_original"),
        validate="one_to_one"
    )

    if len(phase_check) != len(combined):
        raise ValueError("Copertura PHASE incompleta")

    if not (
        phase_check["PHASE_v160"].fillna("").astype(str).to_numpy()
        == phase_check["PHASE_original"].fillna("").astype(str).to_numpy()
    ).all():
        raise ValueError("PHASE differenti")

    if not saved.empty:
        missing_columns = set(saved.columns) - set(combined.columns)
        if missing_columns:
            raise ValueError(
                f"Colonne V160 mancanti: {sorted(missing_columns)}"
            )
        combined = combined.reindex(columns=saved.columns)

    print(
        market,
        "RICALCOLATI:", len(changed),
        "RIUTILIZZATI:", len(raw_tickers - changed),
        "RIGHE:", len(combined)
    )

    if not dry_run:
        temp = output_path.with_suffix(".tmp.csv")
        combined.to_csv(temp, index=False)
        temp.replace(output_path)
        print(market, "ARCHIVIO V160 AGGIORNATO")

    return combined, changed
