import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as gr
from plotly.subplots import make_subplots
import os

# Configurazione iniziale Layout Wide professionale per Cruscotto
st.set_page_config(page_title="Analisi Tecnica Borsa AI", page_icon="📈", layout="wide")

PORTFOLIO_FILE = "portafoglio_autonomo.csv"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try: 
            return pd.read_csv(PORTFOLIO_FILE)
        except: 
            return pd.DataFrame(columns=["Ticker", "Prezzo Carico"])
    return pd.DataFrame(columns=["Ticker", "Prezzo Carico"])

def save_portfolio(df):
    df.to_csv(PORTFOLIO_FILE, index=False)

# Universo dei Titoli Integrato Ufficiale
PANIERI_MERCATO = {
    "🇮🇹 Milano (FTSE MIB)": ["ENI.MI", "ISP.MI", "SPM.MI", "ENEL.MI", "LDO.MI", "G.MI", "STLAM.MI", "UCG.MI"],
    "🇪🇺 Europa (STOXX 50)": ["SAP.DE", "SIE.DE", "MC.PA", "AIR.PA", "ASML.AS", "ALV.DE", "TTE.PA"],
    "🇺🇸 USA (S&P 500)": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"]
}

# Pesi percentuali della Strategia Quantitativa (Somma = 100%)
WEIGHTS = {
    "SAR Weekly": 0.15, "SAR Daily": 0.15, "SAR 4H": 0.05,
    "Bollinger": 0.10, "MACD": 0.10, "RSI": 0.05, "Stochastic RSI": 0.05,
    "Chaikin": 0.10, "Advance/Decline": 0.10, "Volume Oscillator": 0.10, "ADX": 0.05
}

@st.cache_data(ttl=300)
def download_daily_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
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
    ep = high[0]
    psar[0] = low[0]
    
    for i in range(1, len(df)):
        prev_psar = psar[i - 1]
        psar[i] = prev_psar + af * (ep - prev_psar)
        
        if bull:
            if i >= 2: psar[i] = min(psar[i], low[i - 1], low[i - 2])
            else: psar[i] = min(psar[i], low[i - 1])
            
            if low[i] < psar[i]:
                bull = False; psar[i] = ep; ep = low[i]; af = step
            else:
                if high[i] > ep: ep = high[i]; af = min(af + step, max_step)
        else:
            if i >= 2: psar[i] = max(psar[i], high[i - 1], high[i - 2])
            else: psar[i] = max(psar[i], high[i - 1])
            
            if high[i] > psar[i]:
                bull = True; psar[i] = ep; ep = high[i]; af = step
            else:
                if low[i] < ep: ep = low[i]; af = min(af + step, max_step)
                    
    return pd.Series(psar, index=df.index)

def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    
    ha_open = np.zeros(len(df))
    if len(df) > 0:
        ha_open = (df["Open"].to_numpy() + df["Close"].to_numpy()) / 2
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
        if df_w.empty or df_d.empty or df_4h.empty or len(df_d) < 30: return None
        
        ind = compute_technical_indicators(df_d)
        sar_w = calculate_psar(df_w)
        sar_d = calculate_psar(df_d)
        sar_4h = calculate_psar(df_4h)
        
        score = 0.0
        motivi = []
        
        if float(df_w['Close'].dropna().iloc[-2]) > float(sar_w.dropna().iloc[-2]): score += WEIGHTS["SAR Weekly"] * 100; motivi.append("✅ SAR Weekly Bullish")
        if float(df_d['Close'].dropna().iloc[-2]) > float(sar_d.dropna().iloc[-2]): score += WEIGHTS["SAR Daily"] * 100; motivi.append("✅ SAR Daily Bullish")
        if float(df_4h['Close'].dropna().iloc[-2]) > float(sar_4h.dropna().iloc[-2]): score += WEIGHTS["SAR 4H"] * 100; motivi.append("✅ SAR 4H Bullish")
        
        ha_d = heikin_ashi(df_d)
        if ha_d["HA_Indecision"].iloc[-2]: motivi.append("⚪ Candele HA in Indecisione")
        elif ha_d["HA_Close"].iloc[-2] > ha_d["HA_Open"].iloc[-2]: score += WEIGHTS["Bollinger"] * 100; motivi.append("✅ Heikin-Ashi Rialzista")
        
        if ind['macd'].iloc[-2] > ind['macd_signal'].iloc[-2]: score += WEIGHTS["MACD"] * 100; motivi.append("✅ MACD Incrocio Rialzista")
        if 45.0 <= ind['rsi'].iloc[-2] <= 68.0: score += WEIGHTS["RSI"] * 100; motivi.append("✅ RSI in Zona Impulsiva")
        if ind['stoch_rsi'].iloc[-2] > 50.0: score += WEIGHTS["Stochastic RSI"] * 100; motivi.append("✅ Stochastic RSI Positivo")
        if ind['cmf'].iloc[-2] > 0: score += WEIGHTS["Chaikin"] * 100; motivi.append("✅ Flussi Chaikin Positivi")
        if ind['ad_line'].iloc[-2] > ind['ad_line'].iloc[-6]: score += WEIGHTS["Advance/Decline"] * 100; motivi.append("✅ Linea A/D in crescita")
        if ind['vol_osc'].iloc[-2] > 0: score += WEIGHTS["Volume Oscillator"] * 100; motivi.append("✅ Oscillatore Volumi Positivo")
        
        final_score = max(1.0, min(10.0, round(score / 10.0, 1)))
        label = "🔥 Trend Forte" if final_score >= 7.5 else "🟢 Trend Moderato" if final_score >= 5.5 else "🟡 Trend Laterale" if final_score >= 4.0 else "🔴 Trend Contrario"
        
        return {"ticker": t, "name": t, "price": float(df_d['Close'].dropna().iloc[-1]), "score": final_score, "label": label, "motivi": " | ".join(motivi), "sar_status": "KEEP" if df_d['Close'].dropna().iloc[-1] > sar_d.dropna().iloc[-1] else "EXIT"}
    except: return None

def process_single_ticker(t):
    return calculate_v1_trend_score(t)

@st.cache_data(ttl=600)
def run_strategic_screener():
    results = []
    for mercato, tickers in PANIERI_MERCATO.items():
        for t in tickers:
            res = process_single_ticker(t)
            if res is not None:
                res["Mercato"] = mercato
                results.append(res)
    return pd.DataFrame(results)

df_global = run_strategic_screener()
df_portfolio = load_portfolio()

st.title("📊 Analisi Tecnica Borsa AI & Automated Screener")
st.caption("Motore Quantitativo Ibrido Ottimizzato - Integrazione Architetturale Avanzata")

tab_top10, tab_allerta, tab_analisi = st.tabs(["🌟 TOP 10 STRATEGICA (BUY)", "🔴 ALLERTA PORTAFOGLIO & GESTIONE", "🔎 Focus Singolo Titolo"])

with tab_top10:
    st.subheader("📋 Classifica Top 10 Internazionale con Esplicazione dei Motivi")
    if not df_global.empty:
        df_top10 = df_global.sort_values(by="score", ascending=False).head(10)
        df_display_top = df_top10[['ticker', 'Mercato', 'price', 'score', 'label', 'motivi']].copy()
        df_display_top.columns = ['Ticker', 'Listino', 'Prezzo', 'Trend Score', 'Stato Categoria', 'Motivazioni Algoritmo']
        st.dataframe(df_display_top.style.format({'Prezzo': '{:.2f}', 'Trend Score': '{:.1f}/10'}).background_gradient(subset=['Trend Score'], cmap='RdYlGn', vmin=1, vmax=10), use_container_width=True, hide_index=True)
    else: st.info("Nessun titolo idoneo rilevato al momento o caricamento dei mercati in corso...")

with tab_allerta:
    st.subheader("🛠️ Gestione Titoli in Portafoglio")
    with st.form("add_stock_form", clear_on_submit=True):
        c1, col2 = st.columns(2)
        with c1: new_ticker = st.text_input("Inserisci il Ticker esatto:").upper().strip()
        with col2: new_price = st.number_input("Prezzo medio di acquisto:", min_value=0.0, step=0.01)
        if st.form_submit_button("🛒 Salva e Monitora Posizione") and new_ticker:
            if new_ticker not in df_portfolio['Ticker'].tolist():
                new_row = pd.DataFrame([{"Ticker": new_ticker, "Prezzo Carico": new_price}])
                df_portfolio = pd.concat([df_portfolio, new_row], ignore_index=True)
                save_portfolio(df_portfolio)
                st.success(f"Posizione registrata su {new_ticker}!")
                st.rerun()

    st.write("---")
    st.subheader("🚨 Monitoraggio Posizioni Attive e Segnali d'Inversione (SELL)")
    if not df_portfolio.empty:
        portfolio_alerts = []
        for idx, row_p in df_portfolio.iterrows():
            tk = row_p['Ticker']
            p_c = row_p['Prezzo Carico']
            match = df_global[df_global['ticker'] == tk] if not df_global.empty else pd.DataFrame()
            if not match.empty:
                s_score = float(match['score'].iloc); s_status = str(match['sar_status'].iloc); c_pr = float(match['price'].iloc)
            else:
                try:
                    df_d_p = get_timeframe_data(tk, "Daily"); sar_d_p = calculate_psar(df_d_p)
                    c_pr = float(df_d_p['Close'].iloc[-1]); s_status = "KEEP" if c_pr > sar_d_p.iloc[-1] else "EXIT"; s_score = 5.0
                except: c_pr, s_status, s_score = np.nan, "KEEP", 5.0
            azione = "🔴 ⚠️ SUGGERITO VENDERE IMMEDIATAMENTE" if (s_status == "EXIT" or s_score < 4.5) else "🟢 MANTIENI POSIZIONE"
            perf = ((c_pr / p_c) - 1) * 100 if pd.notna(c_pr) and p_c > 0 else 0
            portfolio_alerts.append({"Ticker": tk, "Prezzo Carico": p_c, "Prezzo Attuale": c_pr, "Performance": perf, "Algoritmo Score": s_score, "Azione Trend": azione})
        st.dataframe(pd.DataFrame(portfolio_alerts).style.format({'Prezzo Carico': '{:.2f}', 'Prezzo Attuale': '{:.2f}', 'Performance': '{:+.2f}%', 'Algoritmo Score': '{:.1f}/10'}), use_container_width=True, hide_index=True)
    else: st.info("Il portafoglio monitorato è vuoto.")

with tab_analisi:
    tickers_disponibili = df_global['ticker'].tolist() if not df_global.empty else ["AAPL"]
    ticker_scelto = st.selectbox("Seleziona un titolo per l'analisi visiva:", tickers_disponibili)
    ch_hist = get_timeframe_data(ticker_scelto, "Daily")
    if not ch_hist.empty and len(ch_hist) >= 26:
        ind = compute_technical_indicators(ch_hist); ha_df = heikin_ashi(ch_hist); sar_l = calculate_psar(ch_hist)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2])
        fig.add_trace(gr.Candlestick(x=ha_df.index, open=ha_df['HA_Open'], high=ha_df['HA_High'], low=ha_df['HA_Low'], close=ha_df['HA_Close'], name='Heikin-Ashi'), row=1, col=1)
        fig.add_trace(gr.Scatter(x=ind['bb_upper'].index, y=ind['bb_upper'], name='BB Sup', line=dict(color='#4a5759', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(gr.Scatter(x=ind['bb_lower'].index, y=ind['bb_lower'], name='BB Inf', line=dict(color='#4a5759', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(0,180,216,0.02)'), row=1, col=1)
        fig.add_trace(gr.Scatter(x=sar_l.index, y=sar_l, name='SAR', mode='markers', marker=dict(color='#ff7f0e', size=3)), row=1, col=1)
        fig.add_trace(gr.Bar(x=ch_hist.index, y=ch_hist['Volume'], name='Volumi', marker_color='rgba(31,119,180,0.4)'), row=2, col=1)
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
        y_min = float(ind['bb_lower'].dropna().min() * 0.97) if not ind['bb_lower'].dropna().empty else float(ch_hist['Close'].min() * 0.95)
        y_max = float(ind['bb_upper'].dropna().max() * 1.03) if not ind['bb_upper'].dropna().empty else float(ch_hist['Close'].max() * 1.05)
        fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)
