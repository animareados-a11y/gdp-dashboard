"""Gestione storica delle transizioni PR -> BUY ordinario.

Non modifica gli indicatori scientifici né genera nuovi segnali BUY.
Una ripartenza ordinaria è ammessa soltanto quando il motore
scientifico conferma un BUY valido.
"""
import pandas as pd

BUY = ("BUY1", "BUY2", "BUY3")
PR = tuple(f"{p}(PR)" for p in BUY)


def apply_pr_transition_guard(df, phase_col="PHASE_NUOVA"):
    required = {"Ticker", "Date", phase_col, "PHASE_PRE_PR"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {sorted(missing)}")

    out = df.copy()
    out["Date"] = pd.to_datetime(
        out["Date"], utc=True, format="mixed"
    )
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    out["PR_TRANSITION_RESET"] = False
    out["PR_TRANSITION_BLOCKED"] = False

    for _, positions in out.groupby("Ticker", sort=False).groups.items():
        previous = None
        ordinary_stage = 0

        for pos in positions:
            current = str(out.at[pos, phase_col])
            scientific = str(out.at[pos, "PHASE_PRE_PR"])

            if current in PR:
                ordinary_stage = 0

            elif current in BUY:
                stage = BUY.index(current) + 1

                if previous in PR:
                    if scientific in BUY:
                        out.at[pos, phase_col] = "BUY1"
                        out.at[pos, "PR_TRANSITION_RESET"] = True
                        ordinary_stage = 1
                    else:
                        out.at[pos, "PR_TRANSITION_BLOCKED"] = True
                        ordinary_stage = 0

                elif ordinary_stage and stage > ordinary_stage + 1:
                    out.at[pos, "PR_TRANSITION_BLOCKED"] = True

                else:
                    ordinary_stage = stage

            else:
                ordinary_stage = 0

            previous = str(out.at[pos, phase_col])

    return out
