from backtest import (
    download_backtest_frames,
    neutral_breadth,
)

from engine import analyze
from ml_features_v2 import extract_features_v2


# ============================================================
# MARKET SENTINEL
# TEST FEATURE ENGINE V2.1
# ============================================================

TICKER = "ISP.MI"


print()
print("============================================")
print(" TEST FEATURE ENGINE V2.1")
print("============================================")
print()

print(f"Scarico dati: {TICKER}")

frames = download_backtest_frames(
    TICKER,
    progress_callback=None
)

print("Analisi tecnica...")

analysis = analyze(
    TICKER,
    frames,
    neutral_breadth()
)

if not analysis:
    raise RuntimeError(
        "Analisi non disponibile."
    )

print("Estrazione nuove feature...")

features = extract_features_v2(
    analysis
)


# ============================================================
# FEATURE DA CONTROLLARE
# ============================================================

FEATURES_TO_CHECK = [

    # --------------------------------------------------------
    # SAR
    # --------------------------------------------------------

    "daily_sar_bullish",
    "daily_sar_flip_up_now",
    "daily_sar_flip_down_now",
    "daily_sar_flip_up_age",
    "daily_sar_flip_down_age",
    "daily_sar_flip_up_freshness",
    "daily_sar_flip_down_freshness",
    "daily_sar_distance_atr",
    "daily_sar_distance_change_3",
    "daily_sar_approaching_price_1",
    "daily_sar_approaching_price_3",

    # --------------------------------------------------------
    # HEIKIN ASHI
    # --------------------------------------------------------

    "daily_ha_green",
    "daily_ha_red",
    "daily_ha_body_ratio",
    "daily_ha_upper_wick_ratio",
    "daily_ha_lower_wick_ratio",
    "daily_ha_uncertainty",
    "daily_ha_uncertainty_age",
    "daily_ha_uncertainty_freshness",
    "daily_ha_flip_green_now",
    "daily_ha_flip_red_now",
    "daily_ha_flip_green_age",
    "daily_ha_flip_red_age",
    "daily_ha_flip_green_freshness",
    "daily_ha_flip_red_freshness",
    "daily_ha_color_streak",
    "daily_ha_body_shrinking",

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    "daily_bb_position",
    "daily_bb_above_upper",
    "daily_bb_below_lower",

    "daily_bb_max_upper_detach_5",
    "daily_bb_max_lower_detach_5",

    "daily_bb_upper_detach_age",
    "daily_bb_lower_detach_age",

    "daily_bb_upper_detach_freshness",
    "daily_bb_lower_detach_freshness",

    "daily_bb_reentry_from_upper",
    "daily_bb_reentry_from_lower",

    "daily_bb_reentry_upper_age",
    "daily_bb_reentry_lower_age",

    "daily_bb_reentry_upper_freshness",
    "daily_bb_reentry_lower_freshness",

    "daily_bb_mid_cross_up_age",
    "daily_bb_mid_cross_down_age",

    "daily_bb_mid_cross_up_freshness",
    "daily_bb_mid_cross_down_freshness",

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    "daily_macd",
    "daily_macd_signal",
    "daily_macd_hist",
    "daily_macd_spread",

    "daily_macd_cross_up_age",
    "daily_macd_cross_down_age",

    "daily_macd_cross_up_freshness",
    "daily_macd_cross_down_freshness",

    "daily_macd_recent_bull_cross_below_zero",
    "daily_macd_recent_bear_cross_above_zero",

    "daily_macd_hist_slope_3",
    "daily_macd_hist_acceleration",

    "daily_macd_converging",

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    "daily_rsi",
    "daily_rsi_percentile",
    "daily_rsi_zscore",

    "daily_rsi_slope_3",
    "daily_rsi_acceleration",

    "daily_rsi_turn_down",
    "daily_rsi_turn_up",

    "daily_rsi_turn_down_age",
    "daily_rsi_turn_up_age",

    "daily_rsi_turn_down_freshness",
    "daily_rsi_turn_up_freshness",

    "daily_rsi_overbought_turn_down",
    "daily_rsi_oversold_turn_up",

    "daily_rsi_overbought_turn_down_freshness",
    "daily_rsi_oversold_turn_up_freshness",

    # --------------------------------------------------------
    # CHAIKIN
    # --------------------------------------------------------

    "daily_chaikin",
    "daily_chaikin_percentile",
    "daily_chaikin_zscore",

    "daily_chaikin_slope_3",
    "daily_chaikin_acceleration",

    "daily_chaikin_inflated",
    "daily_chaikin_very_inflated",

    "daily_chaikin_deflated",
    "daily_chaikin_very_deflated",

    "daily_chaikin_inflation_score",
    "daily_chaikin_deflation_score",

    "daily_chaikin_decompression",
    "daily_chaikin_pressure_loss",
    "daily_chaikin_reinflation",

    "daily_chaikin_decompression_age",
    "daily_chaikin_reinflation_age",

    "daily_chaikin_decompression_freshness",
    "daily_chaikin_reinflation_freshness",

    # --------------------------------------------------------
    # ACCUMULATION / DISTRIBUTION
    # --------------------------------------------------------

    "daily_ad",
    "daily_ad_percentile",
    "daily_ad_zscore",

    "daily_ad_slope_3",
    "daily_ad_acceleration",

    "daily_ad_inflated",
    "daily_ad_very_inflated",

    "daily_ad_deflated",
    "daily_ad_very_deflated",

    "daily_ad_inflation_score",
    "daily_ad_deflation_score",

    "daily_ad_decompression",
    "daily_ad_pressure_loss",
    "daily_ad_reinflation",

    "daily_ad_decompression_age",
    "daily_ad_reinflation_age",

    "daily_ad_decompression_freshness",
    "daily_ad_reinflation_freshness",

    # --------------------------------------------------------
    # SETUP DAILY
    # --------------------------------------------------------

    "setup_early_sell_strength_daily",
    "setup_early_buy_strength_daily",

    "setup_early_sell_daily",
    "setup_early_buy_daily",

    "setup_sell_with_sar_confirmation_strength",
    "setup_buy_with_sar_confirmation_strength",

    "setup_sell_with_sar_confirmation",
    "setup_buy_with_sar_confirmation",

    # --------------------------------------------------------
    # SEQUENZA TEMPORALE
    # Bollinger -> HA -> MACD -> SAR
    # --------------------------------------------------------

    "setup_sell_sequence_boll_ha_macd_sar",
    "setup_buy_sequence_boll_ha_macd_sar",

    # --------------------------------------------------------
    # 4H -> DAILY
    # --------------------------------------------------------

    "setup_4h_early_sell_vs_daily",
    "setup_4h_early_buy_vs_daily",
]


print()
print("============================================")
print(" FEATURE PRINCIPALI")
print("============================================")
print()

for name in FEATURES_TO_CHECK:

    value = features.get(
        name,
        "MANCANTE"
    )

    print(
        f"{name:<55} {value}"
    )


# ============================================================
# RIEPILOGO FINALE
# ============================================================

print()
print("============================================")
print(" RIEPILOGO SEGNALI")
print("============================================")
print()

print(
    "EARLY SELL strength:       ",
    features.get(
        "setup_early_sell_strength_daily",
        "MANCANTE"
    )
)

print(
    "EARLY BUY strength:        ",
    features.get(
        "setup_early_buy_strength_daily",
        "MANCANTE"
    )
)

print()

print(
    "SELL + SAR strength:       ",
    features.get(
        "setup_sell_with_sar_confirmation_strength",
        "MANCANTE"
    )
)

print(
    "BUY + SAR strength:        ",
    features.get(
        "setup_buy_with_sar_confirmation_strength",
        "MANCANTE"
    )
)

print()

print(
    "SELL sequence strength:    ",
    features.get(
        "setup_sell_sequence_boll_ha_macd_sar",
        "MANCANTE"
    )
)

print(
    "BUY sequence strength:     ",
    features.get(
        "setup_buy_sequence_boll_ha_macd_sar",
        "MANCANTE"
    )
)

print()

print(
    "4H early SELL vs Daily:    ",
    features.get(
        "setup_4h_early_sell_vs_daily",
        "MANCANTE"
    )
)

print(
    "4H early BUY vs Daily:     ",
    features.get(
        "setup_4h_early_buy_vs_daily",
        "MANCANTE"
    )
)

print()

print(
    "EARLY SELL binary:         ",
    features.get(
        "setup_early_sell_daily",
        "MANCANTE"
    )
)

print(
    "EARLY BUY binary:          ",
    features.get(
        "setup_early_buy_daily",
        "MANCANTE"
    )
)

print(
    "SELL confermato SAR:       ",
    features.get(
        "setup_sell_with_sar_confirmation",
        "MANCANTE"
    )
)

print(
    "BUY confermato SAR:        ",
    features.get(
        "setup_buy_with_sar_confirmation",
        "MANCANTE"
    )
)


print()
print("============================================")
print(" TOTALE FEATURE")
print("============================================")
print()

print(
    len(
        features
    )
)

print()
print("TEST COMPLETATO.")
print()