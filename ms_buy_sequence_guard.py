"""Controllo di continuità BUY dopo EARLY BUY e PR."""
import pandas as pd

BUY = ("BUY1", "BUY2", "BUY3")
BLOCKED = "INDECISIONE"

def apply_buy_sequence_guard(history):
    out = history.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)

    required = [
        "Ticker", "Date", "PHASE", "PHASE_NUOVA",
        "SAR_SIDE", "V4015_HA_DIRECTION", "V4015_HA_INDECISION",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"BUY SEQUENCE: colonne mancanti: {missing}")

    out["BUY_SEQUENCE_GUARD"] = False
    out["BUY_SEQUENCE_PRE_GUARD"] = out["PHASE_NUOVA"]

    for _, indices in out.groupby("Ticker", sort=False).groups.items():
        previous = ""
        interrupted = False

        for i in indices:
            original = str(out.at[i, "PHASE_NUOVA"])
            current = original.replace("(PR)", "")
            reason = str(out.at[i, "PHASE_NUOVA_REASON"]) if "PHASE_NUOVA_REASON" in out else ""

            sar = pd.to_numeric(out.at[i, "SAR_SIDE"], errors="coerce")
            ha = pd.to_numeric(out.at[i, "V4015_HA_DIRECTION"], errors="coerce")
            indecision = pd.to_numeric(out.at[i, "V4015_HA_INDECISION"], errors="coerce")
            confirmed = sar == 1 and ha == 1 and indecision == 0

            target = original

            if (
                (previous == "BUY1" and current in ("BUY3", "TREND_RIALZISTA"))
                or (previous == "BUY2" and current == "TREND_RIALZISTA")
            ):
                if confirmed:
                    target = "BUY2" if previous == "BUY1" else "BUY3"
                else:
                    target = BLOCKED

            if target != original:
                out.at[i, "PHASE_NUOVA"] = target
                out.at[i, "BUY_SEQUENCE_GUARD"] = True
                if "PHASE_NUOVA_REASON" in out:
                    out.at[i, "PHASE_NUOVA_REASON"] = (
                        "BUY_SEQUENCE_INTERRUPTED"
                        if target == BLOCKED else "BUY_SEQUENCE_PROGRESS"
                    )

            if target == BLOCKED and original != BLOCKED:
                interrupted = True
            elif current not in BUY + ("TREND_RIALZISTA",):
                interrupted = False
            elif current == "BUY1" and reason in (
                "PR_EXIT_RESTART_BUY1",
                "SAR_HA_RATING_CONFIRM",
                "EARLY_BUY_START",
            ):
                interrupted = False

            previous = target.replace("(PR)", "")

    return out
