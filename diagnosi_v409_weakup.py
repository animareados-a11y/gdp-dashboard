import pandas as pd
import weekly_v40_7_weakness_severity_engine as v407

df = pd.read_csv(
    "data/v40_35_full200_weekly.csv",
    low_memory=False
)

df["Date"] = pd.to_datetime(df["Date"])

g = (
    df[df["Ticker"] == "FBK.MI"]
    .sort_values("Date")
    .reset_index(drop=True)
)

a = v407.process_ticker(g.copy())

wanted = [
    "Date",
    "Close",
    "PHASE",
    "SAR_SIDE",
    "SAR_AGE",
    "WEAK_UP_TRIGGER",
    "WEAK_UP_BB_PATH",
    "WEAK_UP_HA_PATH",
    "HA_STRUCTURAL_WEAK_UP",
    "BB_DETACH_UPPER",
    "MOM_WEAK_UP_2OF3",
    "LATERAL_SIGNAL",
    "BULL_RECOVERY_RAW",
    "BULL_RECOVERY_CONFIRMED",
]

cols = [c for c in wanted if c in a.columns]

mask = a["Date"] >= pd.Timestamp("2026-07-01")

print("\nCOLONNE DISPONIBILI:")
print(cols)

print("\nFINECO - V40.7:")
print(
    a.loc[mask, cols]
    .to_string(index=False)
)
