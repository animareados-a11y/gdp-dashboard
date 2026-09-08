import warnings
warnings.filterwarnings("ignore")

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import MARKETS

from data import (
    clean_universe,
    download_market_timeframes,
    get_info,
)

from engine import (
    analyze,
    breadth_score,
    final_buy_score,
)

from backtest import (
    BACKTEST_PERIODS,
    FORECAST_HORIZONS,
    STEP_SESSIONS,

    universe_tickers,
    random_ticker,
    multi_sample_tickers,

    run_backtest,
    run_multi_backtest,

    score_calibration,
    predictive_correlations,
    score_extremes_analysis,
    predictive_diagnosis,

    market_predictive_summary,
    build_ticker_summary,
    multi_predictive_report,

    key_moments,
)


# ============================================================
# MARKET SENTINEL V5.0
#
# BUY
# PORTFOLIO
# SINGLE STOCK BACKTEST
# MULTI STOCK BACKTEST
# ============================================================

st.set_page_config(
    page_title="Market Sentinel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.html(
    """
<style>

.stApp {
    background: linear-gradient(180deg, #06131e 0%, #071824 100%);
    color: #eef4f7;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.15rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    color: #f4f8fa !important;
}

p, label {
    color: #a8bac4;
}


/* =========================================================
   BRAND
   ========================================================= */

.ms-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.ms-logo {
    width: 46px;
    height: 46px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 11px;

    background: linear-gradient(135deg, #e5e4f0, #8d7ccd);

    font-size: 26px;
}

.ms-brand-name {
    font-size: 31px;
    font-weight: 900;
    color: white;
    letter-spacing: -1px;
}

.ms-green {
    color: #20d66b;
}

.ms-subtitle {
    color: #78909c;
    font-size: 13px;
    margin-top: 6px;
    margin-bottom: 22px;
}

.mode-title {
    color: #8196a2;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .8px;
    margin-bottom: 4px;
}


/* =========================================================
   BUTTONS
   ========================================================= */

div[data-testid="stButton"] > button {

    background: linear-gradient(
        135deg,
        #146edb,
        #0d55b5
    ) !important;

    color: #ffffff !important;

    border: 1px solid rgba(
        125,
        197,
        255,
        .70
    ) !important;

    border-radius: 9px !important;

    min-height: 42px;

    padding: .45rem 1rem;

    font-weight: 800 !important;

    box-shadow:
        0 4px 12px
        rgba(0, 0, 0, .18);
}

div[data-testid="stButton"] > button p {
    color: #ffffff !important;
    font-weight: 800 !important;
}

div[data-testid="stButton"] > button:hover {

    background: linear-gradient(
        135deg,
        #1880ff,
        #1267d6
    ) !important;

    border-color: #a8d7ff !important;
}

div[data-testid="stButton"] > button:disabled {

    background: #17303e !important;

    color: #75909f !important;

    border: 1px solid #284756 !important;
}


/* =========================================================
   RADIO
   ========================================================= */

div[role="radiogroup"] {
    gap: 10px;
}

div[role="radiogroup"] label {

    background: #0c2230;

    border: 1px solid #1b4355;

    border-radius: 11px;

    padding: 8px 18px;

    min-height: 42px;

    display: flex;

    align-items: center;

    font-weight: 850;
}

div[role="radiogroup"] label:hover {
    border-color: #20d66b;
    background: #0e2b37;
}


/* =========================================================
   SELECT / INPUT
   ========================================================= */

div[data-baseweb="select"] > div {

    background: #edf2f6 !important;

    color: #10202b !important;

    border-radius: 8px !important;
}

div[data-testid="stTextInput"] input {

    background: #edf2f6 !important;

    color: #10202b !important;

    border-radius: 8px !important;
}


/* =========================================================
   INFO BAR
   ========================================================= */

.info-bar {

    background: #0b1e2a;

    border: 1px solid #183747;

    border-radius: 12px;

    padding: 11px 14px;

    color: #90a4ae;

    font-size: 12px;

    margin: 8px 0 19px 0;
}


/* =========================================================
   SUMMARY
   ========================================================= */

.summary-grid {

    display: grid;

    grid-template-columns:
        repeat(
            4,
            minmax(0, 1fr)
        );

    gap: 12px;

    margin: 18px 0 28px 0;
}

.summary-card {

    background:
        linear-gradient(
            145deg,
            #0d202c,
            #091a25
        );

    border:
        1px solid
        #183646;

    border-radius:
        15px;

    padding:
        17px 18px;
}

.summary-label {

    color:
        #7c919d;

    font-size:
        10px;

    text-transform:
        uppercase;

    letter-spacing:
        .7px;
}

.summary-value {

    margin-top:
        7px;

    font-size:
        31px;

    line-height:
        1;

    font-weight:
        950;
}

.green {
    color: #20d66b;
}

.yellow {
    color: #ffb522;
}

.red {
    color: #ff525a;
}

.blue {
    color: #4ca7ff;
}


/* =========================================================
   STOCK CARDS
   ========================================================= */

.stock-card {

    display: grid;

    grid-template-columns:
        minmax(260px, 1.8fr)
        minmax(190px, 1fr)
        minmax(230px, 1.15fr);

    gap: 20px;

    align-items: center;

    background:
        linear-gradient(
            135deg,
            #0d202d,
            #091a25
        );

    border:
        1px solid
        #173644;

    border-radius:
        16px;

    padding:
        18px 20px;

    margin-top:
        10px;

    box-shadow:
        0 8px 22px
        rgba(0, 0, 0, .12);
}

.card-green {
    border-left: 5px solid #20d66b;
}

.card-yellow {
    border-left: 5px solid #ffb522;
}

.card-red {
    border-left: 5px solid #ff525a;
}

.stock-name {

    font-size: 20px;

    font-weight: 900;

    color: white;
}

.stock-market {

    color: #738995;

    font-size: 11px;

    margin-top: 3px;
}

.stock-price {

    color: #c3cfd5;

    font-size: 13px;

    margin-top: 8px;
}

.expected-up {
    color: #20d66b;
    font-weight: 850;
}

.expected-down {
    color: #ff525a;
    font-weight: 850;
}


/* =========================================================
   BUY
   ========================================================= */

.buy-label {

    color: #20d66b;

    font-size: 10px;

    font-weight: 900;

    letter-spacing: .8px;
}

.buy-score {

    color: #20d66b;

    font-size: 45px;

    line-height: 1;

    font-weight: 950;

    letter-spacing: -2px;
}

.buy-den {

    font-size: 13px;

    color: #81939d;

    font-weight: 500;
}

.phase {

    margin-top: 7px;

    color: #94a5ad;

    font-size: 11px;
}


/* =========================================================
   ENTRY
   ========================================================= */

.entry-badge {

    display: inline-block;

    padding: 4px 9px;

    margin-top: 9px;

    border-radius: 20px;

    font-size: 10px;

    font-weight: 850;
}

.entry-green {

    color: #20d66b;

    background:
        rgba(
            32,
            214,
            107,
            .10
        );

    border:
        1px solid
        rgba(
            32,
            214,
            107,
            .28
        );
}

.entry-yellow {

    color: #ffb522;

    background:
        rgba(
            255,
            181,
            34,
            .10
        );

    border:
        1px solid
        rgba(
            255,
            181,
            34,
            .28
        );
}

.entry-red {

    color: #ff6268;

    background:
        rgba(
            255,
            82,
            90,
            .10
        );

    border:
        1px solid
        rgba(
            255,
            82,
            90,
            .28
        );
}


/* =========================================================
   SELL
   ========================================================= */

.sell-card {

    background:
        linear-gradient(
            135deg,
            #0c202b,
            #081923
        );

    border:
        1px solid
        #183846;

    border-radius:
        16px;

    padding:
        18px 20px;

    margin:
        11px 0 0 0;
}

.sell-low {
    border-left: 5px solid #20d66b;
}

.sell-watch {
    border-left: 5px solid #ffb522;
}

.sell-high {
    border-left: 5px solid #ff525a;
}

.sell-score {

    font-size: 39px;

    line-height: 1;

    font-weight: 950;
}

.sell-green {
    color: #20d66b;
}

.sell-yellow {
    color: #ffb522;
}

.sell-red {
    color: #ff525a;
}

.sell-action {

    font-size: 12px;

    font-weight: 900;

    margin-top: 6px;
}


/* =========================================================
   BACKTEST
   ========================================================= */

.backtest-box {

    background:
        linear-gradient(
            135deg,
            #0d202d,
            #091a25
        );

    border:
        1px solid
        #193b4a;

    border-radius:
        16px;

    padding:
        18px 20px;

    margin:
        10px 0 18px 0;
}

.backtest-title {

    color: white;

    font-size: 18px;

    font-weight: 900;
}

.backtest-note {

    color: #8ea2ad;

    font-size: 12px;

    margin-top: 6px;
}

.predictive-box {

    background:
        linear-gradient(
            145deg,
            #102635,
            #091b27
        );

    border:
        1px solid
        #1f4b5f;

    border-radius:
        18px;

    padding:
        22px;

    margin:
        14px 0 20px 0;
}

.predictive-title {

    color: white;

    font-size: 18px;

    font-weight: 900;
}

.predictive-value {

    font-size: 40px;

    line-height: 1;

    font-weight: 950;

    margin-top: 9px;
}

.predictive-label {

    font-size: 12px;

    margin-top: 7px;

    color: #91a6b1;
}

.predictive-note {

    font-size: 12px;

    color: #879ca7;

    margin-top: 8px;
}


/* =========================================================
   STREAMLIT
   ========================================================= */

div[data-testid="stMetric"] {

    background: #0b1e2a;

    border: 1px solid #173543;

    padding: 12px;

    border-radius: 11px;
}

div[data-testid="stExpander"] {

    background: #091a25;

    border: 1px solid #173543;

    border-radius: 11px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 760px) {

    .block-container {
        padding-left: .65rem;
        padding-right: .65rem;
    }

    .ms-brand-name {
        font-size: 24px;
    }

    .summary-grid {

        grid-template-columns:
            repeat(
                2,
                minmax(0, 1fr)
            );
    }

    .stock-card {

        grid-template-columns:
            1fr;

        gap: 12px;

        padding: 15px;
    }

    .buy-score {
        font-size: 40px;
    }

    .sell-score {
        font-size: 35px;
    }
}

</style>
    """
)


# ============================================================
# CACHE / PORTFOLIO
# ============================================================

CACHE_DIR = os.path.join(
    "data",
    "market_sentinel_cache_v35"
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)

PORTFOLIO_FILE = (
    "portafoglio_autonomo.csv"
)


def snapshot_path(
    market_name
):

    safe_name = (
        market_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )

    return os.path.join(
        CACHE_DIR,
        f"{safe_name}.pkl"
    )


# ============================================================
# PORTFOLIO
# ============================================================

def load_portfolio():

    if not os.path.exists(
        PORTFOLIO_FILE
    ):

        return pd.DataFrame(
            columns=[
                "ticker",
                "prezzo_carico",
            ]
        )

    try:

        df = pd.read_csv(
            PORTFOLIO_FILE
        )

        if "ticker" not in df.columns:
            raise ValueError

        if (
            "prezzo_carico"
            not in df.columns
        ):

            df[
                "prezzo_carico"
            ] = np.nan

        return df[
            [
                "ticker",
                "prezzo_carico",
            ]
        ]

    except Exception:

        return pd.DataFrame(
            columns=[
                "ticker",
                "prezzo_carico",
            ]
        )


def save_portfolio(
    df
):

    df.to_csv(
        PORTFOLIO_FILE,
        index=False
    )


def in_portfolio(
    ticker,
    df
):

    return (
        ticker
        in
        df[
            "ticker"
        ]
        .astype(str)
        .tolist()
    )


def add_portfolio(
    ticker,
    df
):

    if in_portfolio(
        ticker,
        df
    ):

        return df

    new_row = pd.DataFrame(
        [
            {
                "ticker":
                    ticker,

                "prezzo_carico":
                    np.nan,
            }
        ]
    )

    df = pd.concat(
        [
            df,
            new_row,
        ],
        ignore_index=True
    )

    save_portfolio(
        df
    )

    return df


def remove_portfolio(
    ticker,
    df
):

    df = (
        df[
            df[
                "ticker"
            ]
            !=
            ticker
        ]
        .reset_index(
            drop=True
        )
    )

    save_portfolio(
        df
    )

    return df


portfolio = load_portfolio()


# ============================================================
# HELPERS
# ============================================================

def flag_for_market(
    market
):

    return {

        "Italia":
            "🇮🇹",

        "Europa":
            "🇪🇺",

        "USA":
            "🇺🇸",

    }.get(
        market,
        "🌍"
    )


def safe_float(
    value,
    default=np.nan
):

    try:

        value = float(
            value
        )

        if np.isfinite(
            value
        ):

            return value

    except Exception:
        pass

    return default


def fmt_number(
    value,
    decimals=2
):

    value = safe_float(
        value
    )

    if pd.isna(
        value
    ):

        return "n/d"

    return (
        f"{value:.{decimals}f}"
    )


def expected_html(
    value
):

    value = safe_float(
        value
    )

    if pd.isna(
        value
    ):

        return "n/d"

    if value >= 0:

        return (
            "<span class='expected-up'>"
            f"+{value:.1f}%"
            "</span>"
        )

    return (
        "<span class='expected-down'>"
        f"{value:.1f}%"
        "</span>"
    )


def card_class(
    score
):

    if score >= 7:
        return "card-green"

    if score >= 5.5:
        return "card-yellow"

    return "card-red"


def entry_class(
    text
):

    text = str(
        text
    ).upper()

    if (
        "OTTIMO"
        in text
        or
        "BUON"
        in text
    ):

        return "entry-green"

    if (
        "DISCRETO"
        in text
        or
        "ATTENDERE"
        in text
    ):

        return "entry-yellow"

    return "entry-red"


def sell_style(
    score
):

    if score >= 6.5:

        return (
            "sell-high",
            "sell-red"
        )

    if score >= 3.5:

        return (
            "sell-watch",
            "sell-yellow"
        )

    return (
        "sell-low",
        "sell-green"
    )


def round_dataframe(
    df,
    decimals=3
):

    if (
        df is None
        or
        df.empty
    ):

        return df

    result = df.copy()

    for column in result.columns:

        if pd.api.types.is_numeric_dtype(
            result[
                column
            ]
        ):

            result[
                column
            ] = result[
                column
            ].round(
                decimals
            )

    return result


# ============================================================
# BREADTH
# ============================================================

def calculate_breadth(
    frames_by_ticker
):

    timeframes = [
        "Monthly",
        "Weekly",
        "Daily",
        "4H",
        "2H",
        "1H",
    ]

    result = {
        timeframe:
            5.0

        for timeframe
        in timeframes
    }

    for timeframe in timeframes:

        up = 0
        down = 0

        for frames in (
            frames_by_ticker
            .values()
        ):

            df = frames.get(
                timeframe
            )

            if (
                df is None
                or
                len(
                    df
                ) < 3
            ):

                continue

            try:

                latest = float(
                    df[
                        "Close"
                    ].iloc[-1]
                )

                previous = float(
                    df[
                        "Close"
                    ].iloc[-2]
                )

                if latest > previous:
                    up += 1

                elif latest < previous:
                    down += 1

            except Exception:
                continue

        result[
            timeframe
        ] = breadth_score(
            up,
            down
        )

    return result


# ============================================================
# PRICE TARGET
# ============================================================

def enrich_candidate(
    master,
    index
):

    ticker = master.at[
        index,
        "Ticker"
    ]

    try:

        info = (
            get_info(
                ticker
            )
            or
            {}
        )

    except Exception:

        info = {}

    target = info.get(
        "target",
        np.nan
    )

    beta = info.get(
        "beta",
        np.nan
    )

    master.at[
        index,
        "Target"
    ] = target

    master.at[
        index,
        "Beta"
    ] = beta

    price = master.at[
        index,
        "Price"
    ]

    expected = np.nan

    if (
        not pd.isna(
            price
        )
        and
        not pd.isna(
            target
        )
        and
        float(
            price
        ) != 0
    ):

        expected = (
            (
                float(
                    target
                )
                /
                float(
                    price
                )
            )
            -
            1
        ) * 100

    master.at[
        index,
        "Expected"
    ] = expected

    technical = safe_float(
        master.at[
            index,
            "Technical Buy Score"
        ],
        0
    )

    final_score, adjustment = (
        final_buy_score(
            technical,
            expected
        )
    )

    master.at[
        index,
        "Buy Score"
    ] = final_score

    master.at[
        index,
        "Target Adjustment"
    ] = adjustment


# ============================================================
# MARKET SNAPSHOT
# ============================================================

def build_market_snapshot(
    market_name,
    progress_widget=None
):

    tickers = clean_universe(
        MARKETS[
            market_name
        ][
            "tickers"
        ]
    )

    def callback(
        step,
        total,
        label
    ):

        if progress_widget is None:
            return

        value = int(
            step
            /
            max(
                total,
                1
            )
            *
            55
        )

        progress_widget.progress(
            min(
                value,
                55
            ),
            text=(
                f"{market_name}: "
                f"{label}..."
            )
        )

    frames_by_ticker = (
        download_market_timeframes(
            tickers,
            progress_callback=
                callback
        )
    )

    if not frames_by_ticker:

        raise RuntimeError(
            "Nessun dato ricevuto."
        )

    breadth = calculate_breadth(
        frames_by_ticker
    )

    rows = []

    total = max(
        len(
            frames_by_ticker
        ),
        1
    )

    for number, (
        ticker,
        frames
    ) in enumerate(
        frames_by_ticker.items(),
        start=1
    ):

        if progress_widget:

            value = (
                55
                +
                int(
                    number
                    /
                    total
                    *
                    30
                )
            )

            progress_widget.progress(
                min(
                    value,
                    85
                ),
                text=(
                    f"{market_name}: "
                    f"analisi {ticker}..."
                )
            )

        try:

            analysis = analyze(
                ticker,
                frames,
                breadth
            )

        except Exception:

            analysis = None

        if not analysis:
            continue

        price = np.nan

        daily = frames.get(
            "Daily"
        )

        if (
            daily is not None
            and
            not daily.empty
        ):

            try:

                price = float(
                    daily[
                        "Close"
                    ].iloc[-1]
                )

            except Exception:
                pass

        technical = safe_float(
            analysis.get(
                "technical_buy_score"
            ),
            0
        )

        rows.append(
            {

                "Market":
                    market_name,

                "Ticker":
                    ticker,

                "Buy Score":
                    technical,

                "Technical Buy Score":
                    technical,

                "Target Adjustment":
                    0.0,

                "Sell Score":
                    analysis.get(
                        "sell_score",
                        0
                    ),

                "Sell Action":
                    analysis.get(
                        "sell_text",
                        ""
                    ),

                "Trend Score":
                    analysis.get(
                        "trend_score",
                        0
                    ),

                "Entry":
                    analysis.get(
                        "entry_text",
                        ""
                    ),

                "Entry Numeric":
                    analysis.get(
                        "entry_score",
                        0
                    ),

                "Rev Down":
                    analysis.get(
                        "bearish_risk",
                        0
                    ),

                "Rev Up":
                    analysis.get(
                        "bullish_reversal",
                        0
                    ),

                "Confidence":
                    analysis.get(
                        "confidence",
                        0
                    ),

                "Phase":
                    analysis.get(
                        "phase",
                        ""
                    ),

                "Price":
                    price,

                "Target":
                    np.nan,

                "Beta":
                    np.nan,

                "Expected":
                    np.nan,

                "analysis_raw":
                    analysis,

                "frames_raw":
                    frames,
            }
        )

    master = pd.DataFrame(
        rows
    )

    if master.empty:

        raise RuntimeError(
            f"Analisi vuota per "
            f"{market_name}."
        )

    master = (
        master
        .sort_values(
            "Technical Buy Score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    candidate_indices = (
        master.index[
            :min(
                len(
                    master
                ),
                30
            )
        ]
        .tolist()
    )

    for number, index in enumerate(
        candidate_indices,
        start=1
    ):

        if progress_widget:

            value = (
                85
                +
                int(
                    number
                    /
                    max(
                        len(
                            candidate_indices
                        ),
                        1
                    )
                    *
                    14
                )
            )

            progress_widget.progress(
                min(
                    value,
                    99
                ),
                text=(
                    f"{market_name}: "
                    "Price Target / Beta..."
                )
            )

        enrich_candidate(
            master,
            index
        )

    master = (
        master
        .sort_values(
            "Buy Score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    snapshot = {

        "market":
            market_name,

        "updated_at":
            datetime.now(),

        "master":
            master,
    }

    with open(
        snapshot_path(
            market_name
        ),
        "wb"
    ) as file:

        pickle.dump(
            snapshot,
            file,
            protocol=
                pickle.HIGHEST_PROTOCOL
        )

    if progress_widget:

        progress_widget.progress(
            100,
            text=(
                f"{market_name}: "
                "completato."
            )
        )

    return snapshot


def load_snapshot(
    market_name
):

    path = snapshot_path(
        market_name
    )

    if not os.path.exists(
        path
    ):

        return None

    try:

        with open(
            path,
            "rb"
        ) as file:

            snapshot = pickle.load(
                file
            )

        return snapshot

    except Exception:

        return None


AVAILABLE_MARKETS = [
    market

    for market
    in [
        "Italia",
        "Europa",
        "USA",
    ]

    if market
    in MARKETS
]


# ============================================================
# HEADER
# ============================================================

st.html(
    """
<div class="ms-brand">

    <div class="ms-logo">
        📈
    </div>

    <div class="ms-brand-name">
        Market<span class="ms-green">Sentinel</span>
    </div>

</div>

<div class="ms-subtitle">
    BUY SCORE • SELL SCORE • Trend & Reversal Engine • Portfolio Monitor • Predictive Research
</div>
    """
)


st.html(
    """
<div class="mode-title">
    SEZIONE
</div>
    """
)


view_mode = st.radio(
    "Sezione",
    [
        "🟢 BUY",
        "🛡️ IN PORTAFOGLIO",
        "🧪 BACKTEST",
    ],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# LOAD SNAPSHOTS
# ============================================================

snapshots = {}

for current_market in (
    AVAILABLE_MARKETS
):

    snap = load_snapshot(
        current_market
    )

    if snap:

        snapshots[
            current_market
        ] = snap


all_frames = []

for current_market in (
    AVAILABLE_MARKETS
):

    snap = snapshots.get(
        current_market
    )

    if (
        snap is not None
        and
        "master"
        in snap
    ):

        all_frames.append(
            snap[
                "master"
            ]
        )


all_master = (
    pd.concat(
        all_frames,
        ignore_index=True
    )

    if all_frames

    else

    pd.DataFrame()
)


def update_markets(
    markets
):

    for current_market in markets:

        progress = st.progress(
            0,
            text=(
                f"Aggiornamento "
                f"{current_market}..."
            )
        )

        try:

            build_market_snapshot(
                current_market,
                progress_widget=
                    progress
            )

            progress.empty()

            st.success(
                f"✅ {current_market} aggiornato."
            )

        except Exception as exc:

            progress.empty()

            st.error(
                f"{current_market}: "
                f"{exc}"
            )


# ============================================================
# BUY
# ============================================================

if view_mode == "🟢 BUY":

    col_market, col_info = st.columns(
        [
            1.3,
            2.7
        ]
    )

    with col_market:

        market = st.selectbox(
            "Mercato",
            [
                "World",
                *AVAILABLE_MARKETS,
            ],
            key="buy_market"
        )

    with col_info:

        st.caption(
            "BUY SCORE = qualità complessiva "
            "dell'opportunità. World confronta "
            "tutti i titoli dei tre mercati."
        )

    c1, c2, c3 = st.columns(
        [
            1.4,
            1.4,
            4
        ]
    )

    with c1:

        update_selected = st.button(
            "🔄 AGGIORNA MERCATO",
            key="buy_update_market",
            width="stretch"
        )

    with c2:

        update_all = st.button(
            "🌍 AGGIORNA TUTTO",
            key="buy_update_all",
            width="stretch"
        )

    if update_selected:

        if market == "World":

            update_markets(
                AVAILABLE_MARKETS
            )

        else:

            update_markets(
                [
                    market
                ]
            )

        st.rerun()

    if update_all:

        update_markets(
            AVAILABLE_MARKETS
        )

        st.rerun()

    if market == "World":

        master = all_master.copy()

    else:

        if market in snapshots:

            master = (
                snapshots[
                    market
                ][
                    "master"
                ]
                .copy()
            )

        else:

            master = pd.DataFrame()

    if master.empty:

        st.warning(
            "Non ci sono dati disponibili."
        )

        st.stop()

    master = (
        master
        .sort_values(
            "Buy Score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    top10 = master.head(
        10
    ).copy()

    strong_buy = int(
        (
            top10[
                "Buy Score"
            ]
            >=
            8
        ).sum()
    )

    top_opportunities = int(
        (
            top10[
                "Buy Score"
            ]
            >=
            7
        ).sum()
    )

    watch_count = int(
        (
            (
                top10[
                    "Buy Score"
                ]
                >=
                5.5
            )
            &
            (
                top10[
                    "Buy Score"
                ]
                <
                7
            )
        ).sum()
    )

    reversal_count = int(
        (
            top10[
                "Rev Down"
            ]
            >=
            5
        ).sum()
    )

    st.html(
        f"""
<div class="summary-grid">

    <div class="summary-card">
        <div class="summary-label">
            BUY ≥ 8
        </div>
        <div class="summary-value green">
            {strong_buy}
        </div>
    </div>

    <div class="summary-card">
        <div class="summary-label">
            TOP ≥ 7
        </div>
        <div class="summary-value green">
            {top_opportunities}
        </div>
    </div>

    <div class="summary-card">
        <div class="summary-label">
            DA OSSERVARE
        </div>
        <div class="summary-value yellow">
            {watch_count}
        </div>
    </div>

    <div class="summary-card">
        <div class="summary-label">
            REVERSAL ↓
        </div>
        <div class="summary-value red">
            {reversal_count}
        </div>
    </div>

</div>
        """
    )

    st.subheader(
        f"🎯 Migliori opportunità — "
        f"{market}"
    )

    for position, row in (
        top10.iterrows()
    ):

        ticker = str(
            row[
                "Ticker"
            ]
        )

        buy_score = safe_float(
            row[
                "Buy Score"
            ],
            0
        )

        st.html(
            f"""
<div class="stock-card {card_class(buy_score)}">

    <div>

        <div class="stock-name">
            {flag_for_market(row["Market"])}
            {ticker}
        </div>

        <div class="stock-market">
            #{position + 1}
            •
            {row["Market"]}
        </div>

        <div class="stock-price">

            Prezzo
            <b>
                {fmt_number(row.get("Price"))}
            </b>

            &nbsp; → &nbsp;

            Target
            <b>
                {fmt_number(row.get("Target"))}
            </b>

        </div>

        <div class="stock-price">

            Rend. atteso
            {expected_html(row.get("Expected"))}

            &nbsp; • &nbsp;

            Beta
            <b>
                {fmt_number(row.get("Beta"))}
            </b>

        </div>

        <span class="
            entry-badge
            {entry_class(row.get("Entry",""))}
        ">
            {row.get("Entry","")}
        </span>

    </div>


    <div>

        <div class="buy-label">
            BUY SCORE
        </div>

        <div class="buy-score">

            {buy_score:.1f}

            <span class="buy-den">
                /10
            </span>

        </div>

        <div class="phase">
            {row.get("Phase","")}
        </div>

    </div>


    <div>

        <div class="stock-price">
            Trend:
            <b>
                {safe_float(row.get("Trend Score"),0):.1f}
            </b>
        </div>

        <div class="stock-price">
            Confidence:
            <b>
                {safe_float(row.get("Confidence"),0):.1f}
            </b>
        </div>

        <div class="stock-price">
            Reversal ↓:
            <b>
                {safe_float(row.get("Rev Down"),0):.1f}
            </b>
        </div>

    </div>

</div>
            """
        )

        b1, b2, b3 = st.columns(
            [
                1.3,
                1.2,
                5
            ]
        )

        with b1:

            already = in_portfolio(
                ticker,
                portfolio
            )

            if st.button(
                (
                    "✓ IN PORTAFOGLIO"

                    if already

                    else

                    "＋ PORTAFOGLIO"
                ),
                key=(
                    f"buy_port_"
                    f"{ticker}"
                ),
                disabled=
                    already,
                width="stretch"
            ):

                portfolio = add_portfolio(
                    ticker,
                    portfolio
                )

                st.rerun()

        with b2:

            if st.button(
                "🔎 ANALIZZA",
                key=(
                    f"buy_an_"
                    f"{ticker}"
                ),
                width="stretch"
            ):

                st.session_state[
                    "detail_ticker"
                ] = ticker


# ============================================================
# PORTFOLIO
# ============================================================

elif view_mode == "🛡️ IN PORTAFOGLIO":

    st.subheader(
        "🛡️ Portafoglio monitorato"
    )

    c1, c2 = st.columns(
        [
            1.4,
            4
        ]
    )

    with c1:

        selected_market = st.selectbox(
            "Mercato da aggiornare",
            AVAILABLE_MARKETS,
            key="portfolio_market"
        )

    b1, b2, b3 = st.columns(
        [
            1.4,
            1.4,
            4
        ]
    )

    with b1:

        if st.button(
            "🔄 AGGIORNA MERCATO",
            key="portfolio_update_market",
            width="stretch"
        ):

            update_markets(
                [
                    selected_market
                ]
            )

            st.rerun()

    with b2:

        if st.button(
            "🌍 AGGIORNA TUTTO",
            key="portfolio_update_all",
            width="stretch"
        ):

            update_markets(
                AVAILABLE_MARKETS
            )

            st.rerun()

    if not all_master.empty:

        st.subheader(
            "🔎 Aggiungi titolo"
        )

        search = st.text_input(
            "Cerca ticker",
            placeholder=(
                "Es. LDO.MI, PRY.MI, MRNA..."
            ),
            key="portfolio_search"
        )

        options = []

        option_map = {}

        searchable = (
            all_master[
                [
                    "Ticker",
                    "Market",
                ]
            ]
            .drop_duplicates(
                "Ticker"
            )
        )

        for _, row in (
            searchable.iterrows()
        ):

            label = (
                f"{row['Ticker']} "
                f"— {row['Market']}"
            )

            if (
                not search.strip()
                or
                search.upper()
                in label.upper()
            ):

                options.append(
                    label
                )

                option_map[
                    label
                ] = row[
                    "Ticker"
                ]

        if options:

            chosen = st.selectbox(
                "Titolo",
                options,
                key="portfolio_add_select"
            )

            ticker_to_add = (
                option_map[
                    chosen
                ]
            )

            if in_portfolio(
                ticker_to_add,
                portfolio
            ):

                st.success(
                    f"✓ {ticker_to_add} "
                    "è già nel portafoglio."
                )

            else:

                if st.button(
                    f"＋ AGGIUNGI {ticker_to_add}",
                    key="portfolio_add_button"
                ):

                    portfolio = add_portfolio(
                        ticker_to_add,
                        portfolio
                    )

                    st.rerun()

    portfolio_tickers = (
        portfolio[
            "ticker"
        ]
        .astype(str)
        .tolist()
    )

    if all_master.empty:

        portfolio_rows = (
            pd.DataFrame()
        )

    else:

        portfolio_rows = (
            all_master[
                all_master[
                    "Ticker"
                ]
                .isin(
                    portfolio_tickers
                )
            ]
        )

    if portfolio_rows.empty:

        st.info(
            "Nessun titolo disponibile "
            "nel portafoglio."
        )

    else:

        portfolio_rows = (
            portfolio_rows
            .sort_values(
                "Sell Score",
                ascending=False
            )
        )

        for _, row in (
            portfolio_rows.iterrows()
        ):

            ticker = str(
                row[
                    "Ticker"
                ]
            )

            sell_score = safe_float(
                row[
                    "Sell Score"
                ],
                0
            )

            sell_box, sell_color = (
                sell_style(
                    sell_score
                )
            )

            st.html(
                f"""
<div class="sell-card {sell_box}">

    <div style="
        display:grid;
        grid-template-columns:
            2fr 1fr;
        gap:25px;
        align-items:center;
    ">

        <div>

            <div class="stock-name">
                {flag_for_market(row["Market"])}
                {ticker}
            </div>

            <div class="stock-price">

                Prezzo
                <b>
                    {fmt_number(row.get("Price"))}
                </b>

                &nbsp; → &nbsp;

                Target
                <b>
                    {fmt_number(row.get("Target"))}
                </b>

            </div>

            <div class="stock-price">

                Trend:
                <b>
                    {safe_float(row.get("Trend Score"),0):.1f}
                </b>

                &nbsp; • &nbsp;

                Reversal ↓:
                <b>
                    {safe_float(row.get("Rev Down"),0):.1f}
                </b>

                &nbsp; • &nbsp;

                Confidence:
                <b>
                    {safe_float(row.get("Confidence"),0):.1f}
                </b>

            </div>

        </div>

        <div style="
            text-align:right;
        ">

            <div class="summary-label">
                SELL SCORE
            </div>

            <div class="
                sell-score
                {sell_color}
            ">
                {sell_score:.1f}/10
            </div>

            <div class="
                sell-action
                {sell_color}
            ">
                {row.get("Sell Action","")}
            </div>

        </div>

    </div>

</div>
                """
            )

            c1, c2, c3 = st.columns(
                [
                    1.2,
                    1.2,
                    5
                ]
            )

            with c1:

                st.button(
                    "🔎 ANALIZZA",
                    key=(
                        f"port_an_"
                        f"{ticker}"
                    ),
                    width="stretch"
                )

            with c2:

                if st.button(
                    "🗑️ RIMUOVI",
                    key=(
                        f"port_rem_"
                        f"{ticker}"
                    ),
                    width="stretch"
                ):

                    portfolio = remove_portfolio(
                        ticker,
                        portfolio
                    )

                    st.rerun()


# ============================================================
# BACKTEST
# ============================================================

else:

    st.subheader(
        "🧪 Predictive Walk-Forward Backtest"
    )

    st.html(
        f"""
<div class="backtest-box">

    <div class="backtest-title">
        Test della capacità predittiva reale
    </div>

    <div class="backtest-note">

        Ogni {STEP_SESSIONS} sedute
        MarketSentinel viene riportato nel passato.

        I dati successivi alla data simulata
        vengono completamente nascosti.

        Il motore calcola BUY SCORE,
        SELL SCORE, Trend, Reversal e timing.

        Solo successivamente vengono aperte
        le 10 e 15 sedute future.

        La modalità MULTI-TITOLO applica
        lo stesso identico test a molti titoli
        e aggrega tutte le osservazioni.

    </div>

</div>
        """
    )

    # ========================================================
    # SINGLE / MULTI
    # ========================================================

    test_type = st.radio(
        "Tipo di Backtest",
        [
            "📌 SINGOLO TITOLO",
            "🌍 MULTI-TITOLO",
        ],
        horizontal=True,
        key="backtest_type"
    )

    period_label = st.selectbox(
        "Periodo di test",
        list(
            BACKTEST_PERIODS.keys()
        ),
        index=1,
        key="backtest_period"
    )

    # ========================================================
    # SINGLE
    # ========================================================

    if test_type == "📌 SINGOLO TITOLO":

        col1, col2 = st.columns(
            2
        )

        with col1:

            single_mode = st.selectbox(
                "Modalità",
                [
                    "Scegli titolo",
                    "Titolo casuale",
                ],
                key="single_mode"
            )

        selected_ticker = None

        items = universe_tickers()

        label_map = {}

        labels = []

        for item in items:

            label = (
                f"{flag_for_market(item['market'])} "
                f"{item['ticker']} "
                f"— {item['market']}"
            )

            labels.append(
                label
            )

            label_map[
                label
            ] = item[
                "ticker"
            ]

        if single_mode == "Scegli titolo":

            with col2:

                selected_label = st.selectbox(
                    "Titolo",
                    labels,
                    key="single_ticker"
                )

            selected_ticker = (
                label_map[
                    selected_label
                ]
            )

        else:

            if (
                "random_backtest_ticker"
                not in
                st.session_state
            ):

                st.session_state[
                    "random_backtest_ticker"
                ] = random_ticker()

            with col2:

                if st.button(
                    "🎲 NUOVO CASUALE",
                    key="new_random_single",
                    width="stretch"
                ):

                    old = (
                        st.session_state.get(
                            "random_backtest_ticker"
                        )
                    )

                    st.session_state[
                        "random_backtest_ticker"
                    ] = random_ticker(
                        exclude=old
                    )

                    st.session_state.pop(
                        "single_backtest_result",
                        None
                    )

                    st.rerun()

            selected_ticker = (
                st.session_state.get(
                    "random_backtest_ticker"
                )
            )

            st.info(
                f"Titolo estratto: "
                f"**{selected_ticker}**"
            )

        c1, c2, c3 = st.columns(
            [
                1.5,
                1.1,
                4
            ]
        )

        with c1:

            run_single = st.button(
                "▶ AVVIA WALK-FORWARD",
                key="run_single_bt",
                width="stretch"
            )

        with c2:

            clear_single = st.button(
                "🧹 PULISCI",
                key="clear_single_bt",
                width="stretch"
            )

        if clear_single:

            st.session_state.pop(
                "single_backtest_result",
                None
            )

            st.session_state.pop(
                "single_backtest_ticker",
                None
            )

            st.session_state.pop(
                "single_backtest_period",
                None
            )

            st.rerun()

        if run_single:

            progress = st.progress(
                0,
                text=(
                    f"Preparazione "
                    f"{selected_ticker}..."
                )
            )

            def single_progress_callback(
                value,
                text
            ):

                percent = int(
                    max(
                        0,
                        min(
                            float(
                                value
                            ),
                            1
                        )
                    )
                    *
                    100
                )

                progress.progress(
                    percent,
                    text=text
                )

            try:

                result = run_backtest(
                    selected_ticker,
                    period_label=
                        period_label,
                    progress_callback=
                        single_progress_callback
                )

                st.session_state[
                    "single_backtest_result"
                ] = result

                st.session_state[
                    "single_backtest_ticker"
                ] = selected_ticker

                st.session_state[
                    "single_backtest_period"
                ] = period_label

                progress.empty()

                st.success(
                    f"✅ Walk-forward completato: "
                    f"{selected_ticker}."
                )

            except Exception as exc:

                progress.empty()

                st.error(
                    f"Errore nel backtest: "
                    f"{exc}"
                )

        result = st.session_state.get(
            "single_backtest_result"
        )

        result_ticker = (
            st.session_state.get(
                "single_backtest_ticker"
            )
        )

        result_period = (
            st.session_state.get(
                "single_backtest_period"
            )
        )

        if (
            isinstance(
                result,
                pd.DataFrame
            )
            and
            not result.empty
        ):

            st.divider()

            st.subheader(
                f"📊 Risultati — "
                f"{result_ticker}"
            )

            st.caption(
                f"Periodo: {result_period}. "
                f"{len(result)} previsioni "
                "walk-forward indipendenti."
            )

            moments = key_moments(
                result
            )

            max_buy = moments.get(
                "max_buy",
                {}
            )

            max_sell = moments.get(
                "max_sell",
                {}
            )

            m1, m2, m3, m4 = st.columns(
                4
            )

            m1.metric(
                "BUY massimo",
                f"{safe_float(max_buy.get('Buy Score'),0):.1f}/10"
            )

            m2.metric(
                "SELL massimo",
                f"{safe_float(max_sell.get('Sell Score'),0):.1f}/10"
            )

            m3.metric(
                "BUY medio",
                f"{result['Buy Score'].mean():.1f}/10"
            )

            m4.metric(
                "SELL medio",
                f"{result['Sell Score'].mean():.1f}/10"
            )

            # =================================================
            # GRAPH
            # =================================================

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=.07,
                row_heights=[
                    .58,
                    .42
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=result[
                        "Date"
                    ],
                    y=result[
                        "Price"
                    ],
                    mode="lines",
                    name="Prezzo"
                ),
                row=1,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=result[
                        "Date"
                    ],
                    y=result[
                        "Buy Score"
                    ],
                    mode="lines+markers",
                    name="BUY Score"
                ),
                row=2,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=result[
                        "Date"
                    ],
                    y=result[
                        "Sell Score"
                    ],
                    mode="lines+markers",
                    name="SELL Score"
                ),
                row=2,
                col=1
            )

            fig.update_yaxes(
                title_text="Prezzo",
                row=1,
                col=1
            )

            fig.update_yaxes(
                title_text="Score",
                range=[
                    0,
                    10
                ],
                row=2,
                col=1
            )

            fig.update_layout(
                height=700,
                margin=dict(
                    l=20,
                    r=20,
                    t=40,
                    b=20
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            horizon = st.selectbox(
                "Orizzonte da analizzare",
                FORECAST_HORIZONS,
                index=1,
                format_func=lambda value:
                    f"{value} sedute",
                key="single_horizon"
            )

            diagnosis = (
                predictive_diagnosis(
                    result,
                    horizon=
                        horizon
                )
            )

            buy_corr = safe_float(
                diagnosis.get(
                    "buy_corr"
                )
            )

            sell_corr = safe_float(
                diagnosis.get(
                    "sell_corr"
                )
            )

            col_buy, col_sell = st.columns(
                2
            )

            with col_buy:

                st.html(
                    f"""
<div class="predictive-box">

    <div class="predictive-title">
        🟢 BUY SCORE
    </div>

    <div class="predictive-value green">
        {
            "n/d"
            if pd.isna(buy_corr)
            else f"{buy_corr:.3f}"
        }
    </div>

    <div class="predictive-label">
        Correlazione con rialzo futuro
    </div>

    <div class="predictive-note">
        {diagnosis.get("buy_label","")}
    </div>

</div>
                    """
                )

            with col_sell:

                st.html(
                    f"""
<div class="predictive-box">

    <div class="predictive-title">
        🔴 SELL SCORE
    </div>

    <div class="predictive-value red">
        {
            "n/d"
            if pd.isna(sell_corr)
            else f"{sell_corr:.3f}"
        }
    </div>

    <div class="predictive-label">
        Correlazione con ribasso futuro
    </div>

    <div class="predictive-note">
        {diagnosis.get("sell_label","")}
    </div>

</div>
                    """
                )

            st.subheader(
                "⏱️ Previsione a 10 e 15 sedute"
            )

            st.dataframe(
                round_dataframe(
                    predictive_correlations(
                        result
                    )
                ),
                width="stretch",
                hide_index=True
            )

            st.subheader(
                "🟢 Calibrazione BUY SCORE"
            )

            st.dataframe(
                round_dataframe(
                    score_calibration(
                        result,
                        "Buy Score",
                        horizon=
                            horizon,
                        direction=
                            "buy"
                    ),
                    2
                ),
                width="stretch",
                hide_index=True
            )

            st.subheader(
                "🔴 Calibrazione SELL SCORE"
            )

            st.dataframe(
                round_dataframe(
                    score_calibration(
                        result,
                        "Sell Score",
                        horizon=
                            horizon,
                        direction=
                            "sell"
                    ),
                    2
                ),
                width="stretch",
                hide_index=True
            )

            st.subheader(
                "⚖️ Score alti vs Score bassi"
            )

            st.dataframe(
                round_dataframe(
                    score_extremes_analysis(
                        result,
                        horizon=
                            horizon
                    ),
                    2
                ),
                width="stretch",
                hide_index=True
            )

            with st.expander(
                "🗓️ Tutte le osservazioni Walk-Forward",
                expanded=False
            ):

                raw = result.copy()

                if "Date" in raw.columns:

                    raw[
                        "Date"
                    ] = pd.to_datetime(
                        raw[
                            "Date"
                        ]
                    ).dt.strftime(
                        "%d/%m/%Y"
                    )

                st.dataframe(
                    round_dataframe(
                        raw,
                        2
                    ),
                    width="stretch",
                    hide_index=True
                )

    # ========================================================
    # MULTI STOCK
    # ========================================================

    else:

        st.subheader(
            "🌍 Backtest Multi-Titolo"
        )

        st.caption(
            "Lo stesso motore viene testato "
            "su più titoli mantenendo identica "
            "la metodologia walk-forward."
        )

        sample_option = st.selectbox(
            "Dimensione del campione",
            [
                "20 titoli",
                "40 titoli",
                "Tutto il paniere",
            ],
            index=0,
            key="multi_sample"
        )

        all_items = universe_tickers()

        total_universe = len(
            all_items
        )

        if sample_option == "20 titoli":

            sample_size = min(
                20,
                total_universe
            )

        elif sample_option == "40 titoli":

            sample_size = min(
                40,
                total_universe
            )

        else:

            sample_size = (
                total_universe
            )

        selected_multi_tickers = (
            multi_sample_tickers(
                sample_size=
                    sample_size
            )
        )

        st.info(
            f"Campione selezionato: "
            f"**{len(selected_multi_tickers)} titoli**. "
            f"Periodo: **{period_label}**."
        )

        with st.expander(
            "📋 Mostra i titoli del campione",
            expanded=False
        ):

            sample_df = pd.DataFrame(
                {
                    "Ticker":
                        selected_multi_tickers,
                }
            )

            market_map = {
                item[
                    "ticker"
                ]:
                    item[
                        "market"
                    ]

                for item in
                all_items
            }

            sample_df[
                "Mercato"
            ] = sample_df[
                "Ticker"
            ].map(
                market_map
            )

            st.dataframe(
                sample_df,
                width="stretch",
                hide_index=True
            )

        c1, c2, c3 = st.columns(
            [
                1.7,
                1.1,
                4
            ]
        )

        with c1:

            run_multi = st.button(
                "▶ AVVIA MULTI-BACKTEST",
                key="run_multi_backtest",
                width="stretch"
            )

        with c2:

            clear_multi = st.button(
                "🧹 PULISCI",
                key="clear_multi_backtest",
                width="stretch"
            )

        if clear_multi:

            st.session_state.pop(
                "multi_backtest_result",
                None
            )

            st.session_state.pop(
                "multi_backtest_ticker_summary",
                None
            )

            st.session_state.pop(
                "multi_backtest_failures",
                None
            )

            st.session_state.pop(
                "multi_backtest_period",
                None
            )

            st.rerun()

        if run_multi:

            progress = st.progress(
                0,
                text=(
                    "Preparazione "
                    "Multi-Backtest..."
                )
            )

            def multi_progress_callback(
                value,
                text
            ):

                percent = int(
                    max(
                        0,
                        min(
                            float(
                                value
                            ),
                            1
                        )
                    )
                    *
                    100
                )

                progress.progress(
                    percent,
                    text=text
                )

            try:

                (
                    combined,
                    ticker_summary,
                    failures,
                ) = run_multi_backtest(
                    tickers=
                        selected_multi_tickers,
                    sample_size=
                        sample_size,
                    period_label=
                        period_label,
                    progress_callback=
                        multi_progress_callback
                )

                st.session_state[
                    "multi_backtest_result"
                ] = combined

                st.session_state[
                    "multi_backtest_ticker_summary"
                ] = ticker_summary

                st.session_state[
                    "multi_backtest_failures"
                ] = failures

                st.session_state[
                    "multi_backtest_period"
                ] = period_label

                progress.empty()

                st.success(
                    "✅ Multi-Backtest completato."
                )

            except Exception as exc:

                progress.empty()

                st.error(
                    f"Errore nel Multi-Backtest: "
                    f"{exc}"
                )

        combined = (
            st.session_state.get(
                "multi_backtest_result"
            )
        )

        ticker_summary = (
            st.session_state.get(
                "multi_backtest_ticker_summary"
            )
        )

        failures = (
            st.session_state.get(
                "multi_backtest_failures"
            )
        )

        multi_period = (
            st.session_state.get(
                "multi_backtest_period"
            )
        )

        if (
            isinstance(
                combined,
                pd.DataFrame
            )
            and
            not combined.empty
        ):

            st.divider()

            st.subheader(
                "📊 Risultati aggregati"
            )

            total_observations = len(
                combined
            )

            total_tickers = (
                combined[
                    "Ticker"
                ]
                .nunique()
            )

            total_italy = (
                combined.loc[
                    combined[
                        "Market"
                    ]
                    ==
                    "Italia",
                    "Ticker"
                ]
                .nunique()
            )

            total_europe = (
                combined.loc[
                    combined[
                        "Market"
                    ]
                    ==
                    "Europa",
                    "Ticker"
                ]
                .nunique()
            )

            total_usa = (
                combined.loc[
                    combined[
                        "Market"
                    ]
                    ==
                    "USA",
                    "Ticker"
                ]
                .nunique()
            )

            st.html(
                f"""
<div class="summary-grid">

    <div class="summary-card">

        <div class="summary-label">
            OSSERVAZIONI
        </div>

        <div class="summary-value blue">
            {total_observations}
        </div>

    </div>


    <div class="summary-card">

        <div class="summary-label">
            TITOLI
        </div>

        <div class="summary-value green">
            {total_tickers}
        </div>

    </div>


    <div class="summary-card">

        <div class="summary-label">
            ITA / EUR / USA
        </div>

        <div class="summary-value yellow"
             style="font-size:23px;">
            {total_italy}
            /
            {total_europe}
            /
            {total_usa}
        </div>

    </div>


    <div class="summary-card">

        <div class="summary-label">
            PERIODO
        </div>

        <div class="summary-value blue"
             style="font-size:21px;">
            {multi_period}
        </div>

    </div>

</div>
                """
            )

            # =================================================
            # HORIZON
            # =================================================

            horizon = st.selectbox(
                "Orizzonte predittivo aggregato",
                FORECAST_HORIZONS,
                index=1,
                format_func=lambda value:
                    f"{value} sedute",
                key="multi_horizon"
            )

            report = multi_predictive_report(
                combined,
                horizon=
                    horizon
            )

            diagnosis = (
                report[
                    "diagnosis"
                ]
            )

            buy_corr = safe_float(
                diagnosis.get(
                    "buy_corr"
                )
            )

            sell_corr = safe_float(
                diagnosis.get(
                    "sell_corr"
                )
            )

            # =================================================
            # MAIN RESULTS
            # =================================================

            st.subheader(
                "🧠 Capacità predittiva aggregata"
            )

            col_buy, col_sell = (
                st.columns(
                    2
                )
            )

            with col_buy:

                st.html(
                    f"""
<div class="predictive-box">

    <div class="predictive-title">
        🟢 BUY SCORE
    </div>

    <div class="predictive-value green">
        {
            "n/d"
            if pd.isna(buy_corr)
            else f"{buy_corr:.3f}"
        }
    </div>

    <div class="predictive-label">
        Correlazione BUY → rialzo futuro
    </div>

    <div class="predictive-note">
        {diagnosis.get("buy_label","")}
    </div>

</div>
                    """
                )

            with col_sell:

                st.html(
                    f"""
<div class="predictive-box">

    <div class="predictive-title">
        🔴 SELL SCORE
    </div>

    <div class="predictive-value red">
        {
            "n/d"
            if pd.isna(sell_corr)
            else f"{sell_corr:.3f}"
        }
    </div>

    <div class="predictive-label">
        Correlazione SELL → ribasso futuro
    </div>

    <div class="predictive-note">
        {diagnosis.get("sell_label","")}
    </div>

</div>
                    """
                )

            st.caption(
                "Questi due numeri sono molto più importanti "
                "del risultato di un singolo titolo. "
                "Il campione aggrega tutte le osservazioni "
                "walk-forward disponibili."
            )

            # =================================================
            # 10 / 15 DAYS
            # =================================================

            st.subheader(
                "⏱️ Previsione a 10 e 15 sedute"
            )

            st.dataframe(
                round_dataframe(
                    report[
                        "correlations"
                    ],
                    3
                ),
                width="stretch",
                hide_index=True
            )

            # =================================================
            # BUY CALIBRATION
            # =================================================

            st.subheader(
                "🟢 Calibrazione BUY SCORE — Aggregata"
            )

            st.caption(
                "Se il BUY SCORE contiene vera capacità predittiva, "
                "salendo nelle fasce dovrebbe aumentare "
                "la probabilità e/o l'intensità dei rialzi futuri."
            )

            st.dataframe(
                round_dataframe(
                    report[
                        "buy_calibration"
                    ],
                    2
                ),
                width="stretch",
                hide_index=True
            )

            # =================================================
            # SELL CALIBRATION
            # =================================================

            st.subheader(
                "🔴 Calibrazione SELL SCORE — Aggregata"
            )

            st.caption(
                "All'aumentare del SELL SCORE "
                "dovremmo osservare più frequentemente "
                "rendimenti futuri negativi."
            )

            st.dataframe(
                round_dataframe(
                    report[
                        "sell_calibration"
                    ],
                    2
                ),
                width="stretch",
                hide_index=True
            )

            # =================================================
            # EXTREMES
            # =================================================

            st.subheader(
                "⚖️ Score alti vs Score bassi"
            )

            st.caption(
                "Confronto fra il 25% delle osservazioni "
                "con score più elevato e il 25% con score più basso."
            )

            st.dataframe(
                round_dataframe(
                    report[
                        "extremes"
                    ],
                    2
                ),
                width="stretch",
                hide_index=True
            )

            # =================================================
            # MARKET
            # =================================================

            st.subheader(
                "🌍 Capacità predittiva per mercato"
            )

            st.dataframe(
                round_dataframe(
                    report[
                        "market_summary"
                    ],
                    3
                ),
                width="stretch",
                hide_index=True
            )

            # =================================================
            # TICKERS
            # =================================================

            st.subheader(
                "📌 Risultato per singolo titolo"
            )

            st.caption(
                "Qui possiamo vedere quali titoli "
                "sono più o meno adatti al modello attuale."
            )

            ticker_report = (
                build_ticker_summary(
                    combined,
                    horizon=
                        horizon
                )
            )

            st.dataframe(
                round_dataframe(
                    ticker_report,
                    3
                ),
                width="stretch",
                hide_index=True
            )

            # =================================================
            # DISTRIBUTION
            # =================================================

            st.subheader(
                "📈 Distribuzione della capacità predittiva"
            )

            if (
                ticker_report is not None
                and
                not ticker_report.empty
            ):

                chart_df = (
                    ticker_report
                    .copy()
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=chart_df[
                            "Ticker"
                        ],
                        y=chart_df[
                            "BUY Corr"
                        ],
                        name="BUY Corr"
                    )
                )

                fig.add_trace(
                    go.Bar(
                        x=chart_df[
                            "Ticker"
                        ],
                        y=chart_df[
                            "SELL Corr"
                        ],
                        name="SELL Corr"
                    )
                )

                fig.update_layout(
                    barmode="group",
                    height=520,
                    xaxis_title=
                        "Titolo",
                    yaxis_title=
                        "Correlazione predittiva",
                    margin=dict(
                        l=20,
                        r=20,
                        t=40,
                        b=20
                    )
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

            # =================================================
            # FAILURES
            # =================================================

            if (
                isinstance(
                    failures,
                    pd.DataFrame
                )
                and
                not failures.empty
            ):

                with st.expander(
                    "⚠️ Titoli non elaborati",
                    expanded=False
                ):

                    st.dataframe(
                        failures,
                        width="stretch",
                        hide_index=True
                    )

            # =================================================
            # RAW
            # =================================================

            with st.expander(
                "🗓️ Tutte le osservazioni aggregate",
                expanded=False
            ):

                raw = combined.copy()

                if "Date" in raw.columns:

                    raw[
                        "Date"
                    ] = pd.to_datetime(
                        raw[
                            "Date"
                        ]
                    ).dt.strftime(
                        "%d/%m/%Y"
                    )

                st.dataframe(
                    round_dataframe(
                        raw,
                        2
                    ),
                    width="stretch",
                    hide_index=True
                )

            st.info(
                "Questo è il test che useremo per decidere "
                "se modificare l'engine. "
                "Prima guardiamo il comportamento aggregato "
                "su molti titoli; soltanto dopo interveniamo "
                "sui pesi e sulla logica BUY/SELL."
            )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Market Sentinel è uno strumento personale "
    "di analisi tecnica, monitoraggio e ricerca. "
    "I backtest storici non garantiscono risultati futuri. "
    "L'app non esegue automaticamente ordini di mercato."
)