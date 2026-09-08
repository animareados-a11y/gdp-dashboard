import yfinance as yf
import pandas as pd
import numpy as np

PANIERI = {
    "Milano": ["ENI.MI", "ISP.MI", "UCG.MI"],
    "USA": ["AAPL", "MSFT", "NVDA"]
}

print("\n=== AVVIO DIAGNOSTICA SCREENER DA TERMINALE ===")

for mercato, tickers in PANIERI.items():
    print(f"\n🌍 Scansione Mercato: {mercato}")
    for t in tickers:
        print(f"  -> Analisi titolo: {t}...", end="", flush=True)
        try:
            # 1. Download Giornaliero
            df_d = yf.download(t, period="1y", interval="1d", auto_adjust=False, progress=False)
            if df_d.empty:
                print(" ❌ ERRORE: Dati Daily vuoti!")
                continue
                
            if isinstance(df_d.columns, pd.MultiIndex):
                df_d.columns = df_d.columns.get_level_values(0)
                
            # 2. Download Orario
            df_h = yf.download(t, period="60d", interval="1h", auto_adjust=False, progress=False)
            if df_h.empty:
                print(" ❌ ERRORE: Dati Orari vuoti!")
                continue
                
            if isinstance(df_h.columns, pd.MultiIndex):
                df_h.columns = df_h.columns.get_level_values(0)
                
            # 3. Estrazione Prezzo e Volumi
            prezzo = float(df_d['Close'].dropna().iloc[-1])
            vol_medio = df_d['Volume'].dropna().iloc[-10:].mean()
            
            print(f" ✅ OK | Prezzo: {prezzo:.2f} | Vol Medio: {vol_medio:.0f}")
            
        except Exception as e:
            print(f" ❌ CRASH: {e}")

print("\n=== DIAGNOSTICA COMPLETATA ===\n")
