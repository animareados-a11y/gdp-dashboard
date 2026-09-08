import pandas as pd

import weekly_v40_4_core_phase_engine as v404_old
import weekly_v40_4_core_phase_engine_shadow as v404_shadow
import weekly_v40_5_core_phase_engine as v405
import weekly_v40_10_sar_phase_fix as v4010


INPUT_FILE = "data/v40_35_full200_weekly.csv"


def run_chain(g, base_v404):

    # V40.5 usa V40.4 attraverso la variabile "base".
    # Qui scegliamo se far passare la catena
    # attraverso l'originale oppure attraverso lo shadow.
    v405.base = base_v404

    return v4010.process_ticker(
        g.copy()
    )


def main():

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
    )

    g = (
        df[
            df["Ticker"] == "FBK.MI"
        ]
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    # =========================================================
    # CATENA ORIGINALE
    # =========================================================

    old_out = run_chain(
        g,
        v404_old,
    )

    # =========================================================
    # CATENA SHADOW
    # =========================================================

    shadow_out = run_chain(
        g,
        v404_shadow,
    )

    # =========================================================
    # PERIODO DA APRILE 2026
    # =========================================================

    mask = (
        old_out["Date"]
        >= pd.Timestamp(
            "2026-04-01",
            tz="UTC",
        )
    )

    out = pd.DataFrame({

        "Date":
            old_out.loc[
                mask,
                "Date",
            ]
            .dt.strftime(
                "%Y-%m-%d"
            )
            .values,

        "Close":
            old_out.loc[
                mask,
                "Close",
            ].values,

        "SAR_SIDE":
            old_out.loc[
                mask,
                "SAR_SIDE",
            ].values,

        "SAR_AGE":
            old_out.loc[
                mask,
                "SAR_AGE",
            ].values,

        "WEAK_UP_TRIGGER":
            old_out.loc[
                mask,
                "WEAK_UP_TRIGGER",
            ].values,

        "LATERAL_SIGNAL":
            old_out.loc[
                mask,
                "LATERAL_SIGNAL",
            ].values,

        "BULL_RECOVERY_RAW":
            old_out.loc[
                mask,
                "BULL_RECOVERY_RAW",
            ].values,

        "BULL_RECOVERY_CONFIRMED":
            old_out.loc[
                mask,
                "BULL_RECOVERY_CONFIRMED",
            ].values,

        "PHASE_FINAL_OLD":
            old_out.loc[
                mask,
                "PHASE",
            ].values,

        "PHASE_FINAL_SHADOW":
            shadow_out.loc[
                mask,
                "PHASE",
            ].values,

    })

    print(
        "\n"
        + "=" * 150
    )

    print(
        "FINECO - CONFRONTO PHASE FINALE V40.10"
    )

    print(
        "=" * 150
    )

    print(
        out.to_string(
            index=False
        )
    )

    # =========================================================
    # SOLO DIFFERENZE
    # =========================================================

    diff = out[
        out["PHASE_FINAL_OLD"]
        !=
        out["PHASE_FINAL_SHADOW"]
    ].copy()

    print(
        "\n"
        + "=" * 150
    )

    print(
        "SETTIMANE MODIFICATE DALLO SHADOW"
    )

    print(
        "=" * 150
    )

    if diff.empty:

        print(
            "NESSUNA DIFFERENZA"
        )

    else:

        print(
            diff.to_string(
                index=False
            )
        )


if __name__ == "__main__":

    main()