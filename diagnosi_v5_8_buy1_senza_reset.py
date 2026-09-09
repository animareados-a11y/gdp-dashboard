import pandas as pd

FILE = "data/validazione_phase_dynamics_v5_full200_rows.csv"

df = pd.read_csv(FILE, low_memory=False)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

targets = df[
    (df["V5_BUY_STAGE"] == "BUY1")
    & (pd.to_numeric(df["SAR_AGE"], errors="coerce") > 10)
    & (
        pd.to_numeric(
            df["V5_BUY_RESET_COUNT"],
            errors="coerce"
        ).fillna(0) == 0
    )
][["Ticker", "Date"]].copy()

cols = [
    "Ticker",
    "Date",
    "SAR_SIDE",
    "SAR_AGE",
    "HA_DIRECTION",
    "HA_INDECISION",
    "PHASE_DYNAMICS_V4",
    "PHASE_DYNAMICS_V5",
    "V5_BUY_CONFIRM_COUNT",
    "V5_BUY_CONFIRM_RESET",
    "V5_BUY_RESET_COUNT",
    "V5_BUY_STAGE",
]

cols = [c for c in cols if c in df.columns]

for _, target in targets.iterrows():

    ticker = target["Ticker"]
    date = target["Date"]

    g = (
        df[df["Ticker"] == ticker]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    idx_list = g.index[g["Date"] == date].tolist()

    if not idx_list:
        continue

    i = idx_list[0]

    start = max(0, i - 10)
    end = min(len(g), i + 2)

    x = g.iloc[start:end][cols].copy()

    print()
    print("=" * 140)
    print(f"{ticker}  BUY1 V5 = {date.date()}")
    print("=" * 140)

    print(x.to_string(index=False))

print()
print("=" * 140)
print(f"CASI ANALIZZATI: {len(targets)}")
print("=" * 140)