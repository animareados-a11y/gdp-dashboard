import pandas as pd

def compare_daily_weekly(daily, root):
    from ms_weekly_source_selector import select_weekly_source
    weekly_base, weekly_status, weekly_date = select_weekly_source()
    w = pd.read_csv(weekly_base / "marketsentinel_200_phase_nuova.csv")
    w = w[["Ticker", "PHASE_NUOVA", "RATING_V160"]]
    w = w.rename(columns={"PHASE_NUOVA": "PHASE WEEKLY", "RATING_V160": "RATING WEEKLY"})
    d = daily.merge(w, left_on="NOME TITOLO", right_on="Ticker", how="left", validate="one_to_one")
    d["PHASE DAILY"] = d["PHASE NUOVA"]
    d["RATING DAILY"] = d["RATING"]
    return d
