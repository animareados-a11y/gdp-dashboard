import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import pandas as pd
import numpy as np

PANIERI_MERCATO = {
    "Milano": ["ENI.MI", "ISP.MI", "UCG.MI"],
    "USA": ["AAPL", "MSFT", "NVDA"]
}

WEIGHTS = {
    "SAR Weekly": 0.15, "SAR Daily": 0.15, "SAR 4H": 0.05,
    "Bollinger": 0.10, "MACD": 0.10, "RSI": 0.05, "Stochastic RSI": 0.05,
    "Chaikin": 0.10, "Advance/Decline": 0.10, "Volume Oscillator": 0.10, "ADX": 0.05
}

def download_daily_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except: return pd.DataFrame()

def download_intraday_data(ticker):
    try:
        df = yf.download(ticker, period="60d", interval="1h", auto_adjust=False, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except: return pd.DataFrame()

def resample_ohlcv(df, rule):
    if df.empty: return df
    result = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    result.dropna(inplace=True)
    return result

def get_timeframe_data(ticker, timeframe):
    if timeframe == "Daily": return download_daily_data(ticker)
    if timeframe == "Weekly":
        daily = download_daily_data(ticker)
        return resample_ohlcv(daily, "W-FRI")
    intraday = download_intraday_data(ticker)
    if timeframe == "4H": return resample_ohlcv(intraday, "4h")
    return pd.DataFrame()

def calculate_psar(df, step=0.02, max_step=0.20):
    if len(df) < 14: 
        return pd.Series(index=df.index, data=df['Close'].values)
    
    high = df['High'].to_numpy().astype(float)
    low = df['Low'].to_numpy().astype(float)
    
    psar = np.zeros(len(df))
    bull = True
    af = step
    ep = high[0] # CORRETTO: Estrae il singolo valore scalare iniziale
    psar[0] = low[0]
    
    for i in range(1, len(df)):
        prev_psar = psar[i - 1]
        psar[i] = prev_psar + af * (ep - prev_psar)
        
        if bull:
            if i >= 2: psar[i] = min(psar[i], low[i - 1], low[i - 2])
            else: psar[i] = min(psar[i], low[i - 1])
            
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            if i >= 2: psar[i] = max(psar[i], high[i - 1], high[i - 2])
            else: psar[i] = max(psar[i], high[i - 1])
            
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
                    
    return pd.Series(psar, index=df.index)

def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    
    ha_open = np.zeros(len(df))
    if len(df) > 0:
        ha_open[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2 # CORRETTO: Uso dell'indice numerico esplicito
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2
            
    ha["HA_Open"] = ha_open
    ha["HA_High"] = pd.concat([df["High"], ha["HA_Open"], ha["HA_Close"]], axis=1).max(axis=1)
    ha["HA_Low"] = pd.concat([df["Low"], ha["HA_Open"], ha["HA_Close"]], axis=1).min(axis=1)
    ha["HA_Indecision"] = (abs(ha["HA_Close"] - ha["HA_Open"]) / (ha["HA_High"] - ha["HA_Low"]).replace(0,1)) < 0.25
    return ha

def compute_technical_indicators(df):
    indicators = {}
    close = df['Close']
    mid = close.rolling(window=20).mean()
    stdev = close.rolling(window=20).std()
    indicators['bb_upper'], indicators['bb_lower'] = mid + (stdev * 2), mid - (stdev * 2)
    indicators['macd'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    rs = delta.clip(lower=0).ewm(com=13, adjust=False).mean() / (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean().replace(0,1)
    indicators['rsi'] = 100 - (100 / (1 + rs))
    lowest_rsi, highest_rsi = indicators['rsi'].rolling(window=14).min(), indicators['rsi'].rolling(window=14).max()
    indicators['stoch_rsi'] = 100 * ((indicators['rsi'] - lowest_rsi) / (highest_rsi - lowest_rsi).replace(0,1))
    denom = (df['High'] - df['Low']).replace(0, 1)
    indicators['cmf'] = ((close - df['Low']) - (df['High'] - close)) / denom * df['Volume']
    indicators['ad_line'] = indicators['cmf'].cumsum()
    indicators['vol_osc'] = 100 * ((df['Volume'].ewm(span=5, adjust=False).mean() - df['Volume'].ewm(span=20, adjust=False).mean()) / df['Volume'].ewm(span=20, adjust=False).mean().replace(0,1))
    return indicators

def calculate_v1_trend_score(t):
    try:
        df_w = get_timeframe_data(t, "Weekly")
        df_d = get_timeframe_data(t, "Daily")
        df_4h = get_timeframe_data(t, "4H")
        if df_w.empty or df_d.empty or df_4h.empty or len(df_d) < 30: 
            return "Dati insufficienti"
        
        ind = compute_technical_indicators(df_d)
        sar_w = calculate_psar(df_w)
        sar_d = calculate_psar(df_d)
        sar_4h = calculate_psar(df_4h)
        
        score = 0.0
        
        if float(df_w['Close'].dropna().iloc[-2]) > float(sar_w.dropna().iloc[-2]): score += WEIGHTS["SAR Weekly"] * 100
        if float(df_d['Close'].dropna().iloc[-2]) > float(sar_d.dropna().iloc[-2]): score += WEIGHTS["SAR Daily"] * 100
        if float(df_4h['Close'].dropna().iloc[-2]) > float(sar_4h.dropna().iloc[-2]): score += WEIGHTS["SAR 4H"] * 100
        
        ha_d = heikin_ashi(df_d)
        if not ha_d["HA_Indecision"].iloc[-2] and ha_d["HA_Close"].iloc[-2] > ha_d["HA_Open"].iloc[-2]: score += WEIGHTS["Bollinger"] * 100
        if ind['macd'].iloc[-2] > ind['macd_signal'].iloc[-2]: score += WEIGHTS["MACD"] * 100
        if 45.0 <= ind['rsi'].iloc[-2] <= 68.0: score += WEIGHTS["RSI"] * 100
        if ind['stoch_rsi'].iloc[-2] > 50.0: score += WEIGHTS["Stochastic RSI"] * 100
        if ind['cmf'].iloc[-2] > 0: score += WEIGHTS["Chaikin"] * 100
        if ind['ad_line'].iloc[-2] > ind['ad_line'].iloc[-6]: score += WEIGHTS["Advance/Decline"] * 100
        if ind['vol_osc'].iloc[-2] > 0: score += WEIGHTS["Volume Oscillator"] * 100
        
        final_score = max(1.0, min(10.0, round(score / 10.0, 1)))
        return f"{final_score}/10"
    except Exception as e:
        return f"CRASH INTERNO ({e})"

print("\n=== RUNNING FULL SCORING MOTOR ===")
for mercato, tickers in PANIERI_MERCATO.items():
    print(f"\nMercato: {mercato}")
    for t in tickers:
        score = calculate_v1_trend_score(t)
        print(f"  -> {t} | Algoritmo Score Risultato: {score}")
print("\n===============================\n")
