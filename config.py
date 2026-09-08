# ============================================================
# CONFIGURAZIONE STRATEGICA ESPANSA
#
# PANIERE:
# - ITALIA: 40 titoli
# - EUROPA: 40 titoli
# - USA: 80 titoli
#
# WORLD = migliori titoli fra tutti i panieri
# ============================================================


MARKETS = {

    # ========================================================
    # ITALIA - 40
    # ========================================================

    "Italia": {

        "benchmark": "FTSEMIB.MI",

        "tickers": [

            "A2A.MI",
            "AMP.MI",
            "AZM.MI",
            "BAMI.MI",
            "BCA.MI",

            "BMED.MI",
            "BMPS.MI",
            "BPE.MI",
            "BRE.MI",
            "BZU.MI",

            "CPR.MI",
            "DIA.MI",
            "ENEL.MI",
            "ENI.MI",
            "ERG.MI",

            "FBK.MI",
            "G.MI",
            "HER.MI",
            "INW.MI",
            "ISP.MI",

            "LDO.MI",
            "MB.MI",
            "MONC.MI",
            "NEXI.MI",
            "PIR.MI",

            "PST.MI",
            "PRY.MI",
            "RACE.MI",
            "REC.MI",
            "SFER.MI",

            "SRG.MI",
            "SPM.MI",
            "STLAM.MI",
            "STMMI.MI",
            "TEN.MI",

            "TIT.MI",
            "TRN.MI",
            "UCG.MI",
            "UNI.MI",
            "BPSO.MI",
        ],
    },


    # ========================================================
    # EUROPA - 40
    # ========================================================

    "Europa": {

        "benchmark": "^STOXX50E",

        "tickers": [

            "SAP.DE",
            "SIE.DE",
            "MC.PA",
            "AIR.PA",
            "ASML.AS",

            "ALV.DE",
            "TTE.PA",
            "OR.PA",
            "SAN.MC",
            "SU.PA",

            "DBK.DE",
            "BMW.DE",
            "BAYN.DE",
            "BAS.DE",
            "ADS.DE",

            "DHL.DE",
            "MUV2.DE",
            "RWE.DE",
            "DTG.DE",
            "HEIA.AS",

            "INGA.AS",
            "ADYEN.AMS",
            "PRX.AMS",
            "CRH",
            "FLTR.ID",

            "BBVA.MC",
            "ITX.MC",
            "TEF.MC",
            "IBE.MC",
            "RMS.PA",

            "KER.PA",
            "BNP.PA",
            "SAN.PA",
            "AI.PA",
            "CS.PA",

            "DG.PA",
            "VIV.PA",
            "EL.PA",
            "ABN.AS",
            "SAF.PA",
        ],
    },


    # ========================================================
    # USA - 80 TITOLI LIQUIDI
    # ========================================================

    "USA": {

        "benchmark": "^GSPC",

        "tickers": [

            # ------------------------------------------------
            # MEGA CAP / BIG TECH
            # ------------------------------------------------

            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",

            "META",
            "TSLA",
            "AVGO",
            "NFLX",
            "ORCL",

            # ------------------------------------------------
            # SEMICONDUCTORS / TECHNOLOGY
            # ------------------------------------------------

            "AMD",
            "QCOM",
            "TXN",
            "INTC",
            "MU",

            "AMAT",
            "LRCX",
            "KLAC",
            "ARM",
            "SMCI",

            # ------------------------------------------------
            # SOFTWARE / CLOUD / CYBERSECURITY
            # ------------------------------------------------

            "CRM",
            "ADBE",
            "NOW",
            "PLTR",
            "PANW",

            "CRWD",
            "SNOW",
            "UBER",
            "ABNB",
            "SHOP",

            # ------------------------------------------------
            # FINANCIALS
            # ------------------------------------------------

            "JPM",
            "BAC",
            "GS",
            "MS",
            "WFC",

            "C",
            "V",
            "MA",
            "AXP",
            "BLK",

            # ------------------------------------------------
            # HEALTHCARE / BIOTECH
            # ------------------------------------------------

            "LLY",
            "UNH",
            "JNJ",
            "ABBV",
            "MRK",

            "PFE",
            "AMGN",
            "TMO",
            "GILD",
            "MRNA",

            # ------------------------------------------------
            # CONSUMER / RETAIL
            # ------------------------------------------------

            "WMT",
            "COST",
            "HD",
            "LOW",
            "NKE",

            "MCD",
            "SBUX",
            "TGT",
            "PG",
            "KO",

            # ------------------------------------------------
            # MEDIA / COMMUNICATION
            # ------------------------------------------------

            "DIS",
            "CMCSA",
            "TMUS",
            "T",
            "VZ",

            # ------------------------------------------------
            # ENERGY
            # ------------------------------------------------

            "XOM",
            "CVX",
            "COP",
            "SLB",
            "OXY",

            # ------------------------------------------------
            # INDUSTRIAL / AEROSPACE
            # ------------------------------------------------

            "GE",
            "CAT",
            "HON",
            "BA",
            "RTX",

            # ------------------------------------------------
            # TRANSPORT / INDUSTRIAL / MATERIALS
            # ------------------------------------------------

            "UPS",
            "FDX",
            "DE",
            "MMM",
            "LIN",
        ],
    },
}


# ============================================================
# PESI ORIGINALI
#
# Manteniamo WEIGHTS per compatibilità con eventuali
# script paralleli / versioni precedenti.
# ============================================================

WEIGHTS = {

    "SAR Weekly":
        0.15,

    "SAR Daily":
        0.15,

    "SAR 4H":
        0.05,

    "Bollinger":
        0.10,

    "MACD":
        0.10,

    "Chaikin":
        0.10,

    "Advance/Decline":
        0.10,

    "Volume Oscillator":
        0.10,

    "RSI":
        0.05,

    "Stochastic RSI":
        0.05,

    "ADX":
        0.05,
}


# ============================================================
# PESI INDICATORI
#
# Manteniamo anche questa struttura per compatibilità
# con indicators.py e altri script.
# ============================================================

PESI_INDICATORI = {

    "sar_weekly":
        0.15,

    "sar_daily":
        0.15,

    "sar_4h":
        0.05,

    "bollinger_ha":
        0.10,

    "macd":
        0.10,

    "chaikin":
        0.10,

    "a_d":
        0.10,

    "volume_osc":
        0.10,

    "rsi":
        0.05,

    "stoch_rsi":
        0.05,

    "adx":
        0.05,
}


# ============================================================
# TIMEFRAME
# ============================================================

TIMEFRAMES = [
    "Weekly",
    "Daily",
    "4H",
    "2H",
    "1H",
]


# ============================================================
# ORARI NOTIFICHE
# ============================================================

PREMARKET_HOUR = 8
PREMARKET_MINUTE = 0

MIDDAY_HOUR = 12
MIDDAY_MINUTE = 0

CLOSE_HOUR = 18
CLOSE_MINUTE = 0