"""Gestione persistente del PORTFOLIO MarketSentinel."""
import json
import os
from pathlib import Path

PORTFOLIO_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "marketsentinel_portfolio"
    / "portfolio.json"
)

def load_portfolio():
    if not PORTFOLIO_FILE.exists():
        return []
    data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Formato PORTFOLIO non valido")
    return sorted(set(str(t).strip().upper() for t in data if str(t).strip()))

def save_portfolio(tickers):
    PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    tickers = sorted(set(str(t).strip().upper() for t in tickers if str(t).strip()))
    tmp = PORTFOLIO_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(tickers, indent=2), encoding="utf-8")
    os.replace(tmp, PORTFOLIO_FILE)

def add_ticker(ticker):
    ticker = str(ticker).strip().upper()
    if not ticker:
        raise ValueError("Ticker vuoto")
    tickers = load_portfolio()
    if ticker not in tickers:
        tickers.append(ticker)
        save_portfolio(tickers)
    return sorted(set(tickers))

def remove_ticker(ticker):
    ticker = str(ticker).strip().upper()
    tickers = [t for t in load_portfolio() if t != ticker]
    save_portfolio(tickers)
    return tickers
