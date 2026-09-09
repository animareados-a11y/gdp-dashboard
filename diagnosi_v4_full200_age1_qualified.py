from __future__ import annotations

from pathlib import Path

import pandas as pd

import weekly_v40_5_phase_dynamics_shadow_v4 as v4


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/v40_35_full200_weekly.csv")

CHECKPOINT_FILE = Path("data/diagnosi_v4_full200_age1_checkpoint.csv")
OUTPUT_FILE = Path("data/diagnosi_v4_full200_age1_qualified.csv")
SUMMARY_FILE = Path("data/diagnosi_v4_full200_age1_summary.csv")

SAVE_EVERY = 1


# ============================================================
# HELPERS
# ============================================================

def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_completed_tickers() -> set[str]:
    if not CHECKPOINT_FILE.exists():
        return set()

    try:
        ck = pd.read_csv(CHECKPOINT_FILE, usecols=["Ticker"])
        return set(ck["Ticker"].dropna().astype(str).unique())
    except Exception:
        return set()


def append_checkpoint(df: pd.DataFrame) -> None:
    if df.empty:
        return

    header = not CHECKPOINT_FILE.exists()

    df.to_csv(
        CHECKPOINT_FILE,
        mode="a",
        header=header,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 72)
    print("DIAGNOSI FULL200 - RUN SAR AGE1 QUALIFICATI VS NON QUALIFICATI")
    print("VERSIONE CON CHECKPOINT / RIPRESA AUTOMATICA")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    tickers = sorted(df["Ticker"].dropna().astype(str).unique())

    completed = load_completed_tickers()

    print(f"Ticker totali: {len(tickers)}")
    print(f"Ticker già completati: {len(completed)}")
    print(f"Ticker ancora da elaborare: {len(tickers) - len(completed)}")
    print()

    keep_cols = [
        "Ticker",
        "Date",
        "SAR_SIDE",
        "SAR_AGE",
        "V4010_PREV_SAR_AGE",
        "V4010_PREV_SAR_SIDE",
        "V4010_PREV_PHASE",
        "V4010_PREVIOUS_STRUCTURE_CONFIRMED",
        "V4010_QUALIFIED_FLIP",
        "V4010_RUN_QUALIFIED",
    ]

    process_errors = 0

    for i, ticker in enumerate(tickers, start=1):
        if ticker in completed:
            continue

        g = df[df["Ticker"].astype(str) == ticker].copy()

        if "Date" in g.columns:
            g = g.sort_values("Date")

        g = g.reset_index(drop=True)

        try:
            out = v4.process_ticker(g)

        except Exception as exc:
            process_errors += 1
            print(f"ERRORE {ticker}: {exc}")
            continue

        age = safe_num(out["SAR_AGE"])
        x = out[age.eq(1)].copy()

        available = [c for c in keep_cols if c in x.columns]
        x = x[available].copy()

        append_checkpoint(x)

        completed.add(ticker)

        done = len(completed)

        if (
            done == 1
            or done % 10 == 0
            or done == len(tickers)
        ):
            print(f"Completati {done}/{len(tickers)} ticker")

    print()
    print("=" * 72)
    print("ELABORAZIONE CHECKPOINT COMPLETATA")
    print("=" * 72)

    if not CHECKPOINT_FILE.exists():
        print("Nessun checkpoint disponibile.")
        return

    result = pd.read_csv(CHECKPOINT_FILE)

    if "Date" in result.columns:
        result["Date"] = pd.to_datetime(result["Date"], errors="coerce")

    result = result.drop_duplicates(
        subset=["Ticker", "Date"],
        keep="last",
    ).reset_index(drop=True)

    result.to_csv(OUTPUT_FILE, index=False)

    q = safe_num(result["V4010_RUN_QUALIFIED"])
    prev_age = safe_num(result["V4010_PREV_SAR_AGE"])

    total = len(result)
    qualified = int(q.eq(1).sum())
    non_qualified = int(q.eq(0).sum())

    print()
    print("=" * 72)
    print("RISULTATO")
    print("=" * 72)

    print(f"RUN AGE1 TOTALI: {total}")
    print(f"QUALIFICATI: {qualified}")
    print(f"NON QUALIFICATI: {non_qualified}")
    print(f"PROCESS ERRORS: {process_errors}")

    summary_rows = []

    for label, value in [
        ("QUALIFICATI", 1),
        ("NON_QUALIFICATI", 0),
    ]:
        z = prev_age[q.eq(value)].dropna()

        n = len(z)
        mean = z.mean() if n else float("nan")
        median = z.median() if n else float("nan")
        minimum = z.min() if n else float("nan")
        maximum = z.max() if n else float("nan")

        print()
        print(label)
        print(f"  N       = {n}")
        print(f"  media   = {round(mean, 2) if n else 'NA'}")
        print(f"  mediana = {round(median, 2) if n else 'NA'}")
        print(f"  min     = {minimum if n else 'NA'}")
        print(f"  max     = {maximum if n else 'NA'}")
        print("  distribuzione PREV_SAR_AGE:")

        if n:
            print(
                z.value_counts()
                .sort_index()
                .head(20)
                .to_string()
            )
        else:
            print("  nessun dato")

        summary_rows.append(
            {
                "GROUP": label,
                "N": n,
                "MEAN_PREV_SAR_AGE": mean,
                "MEDIAN_PREV_SAR_AGE": median,
                "MIN_PREV_SAR_AGE": minimum,
                "MAX_PREV_SAR_AGE": maximum,
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 72)
    print("FILE SALVATI")
    print("=" * 72)
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()