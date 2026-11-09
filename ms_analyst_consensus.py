"""Consensus analisti Yahoo Finance - modulo informativo indipendente."""
import math
import yfinance as yf
import streamlit as st

@st.cache_data(ttl=21600, show_spinner=False)
def get_analyst_consensus(ticker):
    result = {
        "consensus": "N/D",
        "analysts": 0,
        "target": None,
        "upside": None,
    }
    try:
        stock = yf.Ticker(str(ticker).strip())
        targets = stock.analyst_price_targets or {}
        rec = stock.recommendations

        if rec is not None and not rec.empty:
            row = rec.iloc[0]
            counts = [
                int(row.get(k, 0) or 0)
                for k in ("strongBuy", "buy", "hold", "sell", "strongSell")
            ]
            total = sum(counts)
            result["analysts"] = total

            if total:
                score = sum((i + 1) * n for i, n in enumerate(counts)) / total
                if score < 1.5:
                    result["consensus"] = "STRONG BUY"
                elif score < 2.5:
                    result["consensus"] = "BUY"
                elif score < 3.5:
                    result["consensus"] = "HOLD"
                elif score < 4.5:
                    result["consensus"] = "SELL"
                else:
                    result["consensus"] = "STRONG SELL"

        target = targets.get("mean")
        current = targets.get("current")

        if target is not None and math.isfinite(float(target)):
            result["target"] = float(target)

        if (
            result["target"] is not None
            and current is not None
            and math.isfinite(float(current))
            and float(current) > 0
        ):
            result["upside"] = (
                result["target"] / float(current) - 1
            ) * 100

    except Exception:
        pass

    return result
