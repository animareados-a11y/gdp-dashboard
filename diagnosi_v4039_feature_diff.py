from pathlib import Path
import pandas as pd
import numpy as np

OLD_FILE = Path("data/v40_36b_full200_features.csv")
NEW_FILE = Path("data/v40_39_full200_features.csv")

OLD_LIST = Path("data/v40_36b_feature_list.csv")
NEW_LIST = Path("data/v40_39_feature_list.csv")


def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def read_feature_list(path):
    df = pd.read_csv(path)

    if len(df.columns) == 1:
        return df.iloc[:, 0].astype(str).tolist()

    for candidate in ["FEATURE", "feature", "Feature"]:
        if candidate in df.columns:
            return df[candidate].astype(str).tolist()

    raise RuntimeError(f"Impossibile leggere feature list: {path}")


section("FEATURE LIST")

old_features = read_feature_list(OLD_LIST)
new_features = read_feature_list(NEW_LIST)

print("Feature V40.36b:", len(old_features))
print("Feature V40.39: ", len(new_features))

only_old = sorted(set(old_features) - set(new_features))
only_new = sorted(set(new_features) - set(old_features))

print("Solo V40.36b:", len(only_old))
print("Solo V40.39: ", len(only_new))

if only_old:
    print("Esempi solo OLD:", only_old[:20])

if only_new:
    print("Esempi solo NEW:", only_new[:20])


section("CARICAMENTO")

common_features = [
    f for f in old_features
    if f in set(new_features)
]

header_old = pd.read_csv(OLD_FILE, nrows=0)
header_new = pd.read_csv(NEW_FILE, nrows=0)

ticker_old = next(
    c for c in header_old.columns
    if c.lower() == "ticker"
)

date_old = next(
    c for c in header_old.columns
    if c.lower() == "date"
)

ticker_new = next(
    c for c in header_new.columns
    if c.lower() == "ticker"
)

date_new = next(
    c for c in header_new.columns
    if c.lower() == "date"
)

use_old = [ticker_old, date_old] + common_features
use_new = [ticker_new, date_new] + common_features

old = pd.read_csv(
    OLD_FILE,
    usecols=use_old
)

new = pd.read_csv(
    NEW_FILE,
    usecols=use_new
)

old = old.rename(
    columns={
        ticker_old: "TICKER",
        date_old: "DATE",
    }
)

new = new.rename(
    columns={
        ticker_new: "TICKER",
        date_new: "DATE",
    }
)

old["DATE"] = pd.to_datetime(old["DATE"])
new["DATE"] = pd.to_datetime(new["DATE"])

old = old.sort_values(
    ["TICKER", "DATE"]
).reset_index(drop=True)

new = new.sort_values(
    ["TICKER", "DATE"]
).reset_index(drop=True)

if len(old) != len(new):
    raise RuntimeError("Numero righe diverso.")

if not old[["TICKER", "DATE"]].equals(
    new[["TICKER", "DATE"]]
):
    raise RuntimeError(
        "TICKER/DATE non perfettamente allineati."
    )

print("Righe:", len(old))
print("Feature comuni:", len(common_features))


section("CONFRONTO VALORI")

summary = []

for feature in common_features:

    a = pd.to_numeric(
        old[feature],
        errors="coerce"
    )

    b = pd.to_numeric(
        new[feature],
        errors="coerce"
    )

    same_nan = a.isna() & b.isna()

    both_valid = a.notna() & b.notna()

    diff = pd.Series(
        False,
        index=a.index
    )

    diff.loc[both_valid] = ~np.isclose(
        a.loc[both_valid],
        b.loc[both_valid],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True
    )

    diff |= (
        a.isna() ^ b.isna()
    )

    changed = int(diff.sum())

    max_abs_diff = np.nan

    if both_valid.any():
        max_abs_diff = float(
            np.nanmax(
                np.abs(
                    a.loc[both_valid]
                    - b.loc[both_valid]
                )
            )
        )

    summary.append({
        "FEATURE": feature,
        "CHANGED_ROWS": changed,
        "CHANGED_PCT": changed / len(old) * 100,
        "MAX_ABS_DIFF": max_abs_diff,
    })


summary_df = pd.DataFrame(summary)

changed_features = summary_df[
    summary_df["CHANGED_ROWS"] > 0
].copy()

changed_features = changed_features.sort_values(
    ["CHANGED_ROWS", "FEATURE"],
    ascending=[False, True]
)


section("RISULTATO")

print(
    "Feature identiche:",
    int(
        (summary_df["CHANGED_ROWS"] == 0).sum()
    )
)

print(
    "Feature con almeno una differenza:",
    len(changed_features)
)

print()

if len(changed_features) > 0:

    print(
        changed_features.head(40).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

else:
    print(
        "TUTTE LE FEATURE COMUNI SONO IDENTICHE."
    )


OUT = Path(
    "data/diagnosi_v4039_feature_diff.csv"
)

summary_df.to_csv(
    OUT,
    index=False
)

print()
print("Salvato:")
print(OUT)