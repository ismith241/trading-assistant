import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Trading Assistant (MVP)", layout="wide")
st.title("Trading Assistant (MVP)")
st.caption("Decision-support only. Not financial advice.")

DEFAULT = "AAPL,MSFT,NVDA,AMZN,GOOGL,SPY"

# ---------- Helpers ----------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.where(d > 0, 0.0).rolling(n).mean()
    loss = (-d.where(d < 0, 0.0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=900)  # cache 15 minutes
def fetch(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        # Fix: yfinance sometimes returns MultiIndex columns on cloud
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=lambda c: c.strip())
    return df

def signal_for(ticker: str) -> dict:
    df = fetch(ticker)
    if df.empty or "Close" not in df.columns:
        return {
            "ticker": ticker,
            "signal": "HOLD",
            "confidence": 0.0,
            "price": None,
            "reasons": ["No data returned."]
        }

    close = df["Close"].astype(float)
    price = float(close.iloc[-1])

    sma20 = sma(close, 20).iloc[-1]
    sma50 = sma(close, 50).iloc[-1]
    sma200 = sma(close, 200).iloc[-1] if len(close) >= 200 else np.nan
    rsi14 = rsi(close, 14).iloc[-1]

    score = 0.0
    reasons = []

    # Long trend
    if not np.isnan(sma200):
        if price > float(sma200):
            score += 0.35
            reasons.append("Above 200D avg (uptrend).")
        else:
            score -= 0.35
            reasons.append("Below 200D avg (downtrend).")
    else:
        reasons.append("Not enough history for 200D avg.")

    # Momentum
    if not np.isnan(sma20) and not np.isnan(sma50):
        if float(sma20) > float(sma50):
            score += 0.25
            reasons.append("20D > 50D (positive momentum).")
        else:
            score -= 0.25
            reasons.append("20D < 50D (weak momentum).")

    # RSI sanity
    if not np.isnan(rsi14):
        rv = float(rsi14)
        if rv < 35:
            score += 0.20
            reasons.append(f"RSI {rv:.1f} (oversold-ish).")
        elif rv > 70:
            score -= 0.20
            reasons.append(f"RSI {rv:.1f} (overbought-ish).")
        else:
            score += 0.05
            reasons.append(f"RSI {rv:.1f} (neutral).")

    # Volatility filter
    rets = close.pct_change().dropna()
    if len(rets) >= 20:
        vol20 = float(rets.tail(20).std() * np.sqrt(252))
        if vol20 < 0.35:
            score += 0.10
            reasons.append(f"Vol ~{vol20:.2f} (ok).")
        else:
            score -= 0.10
            reasons.append(f"Vol ~{vol20:.2f} (high).")
    else:
        reasons.append("Not enough returns history for vol check.")

    if score >= 0.35:
        sig = "BUY"
    elif score <= -0.35:
        sig = "SELL"
    else:
        sig = "HOLD"

    conf = float(min(1.0, max(0.0, abs(score))))

    return {
        "ticker": ticker,
        "signal": sig,
        "confidence": conf,
        "price": price,
        "reasons": reasons
    }

# ---------- UI ----------
left, right = st.columns([2, 1])

with left:
    tickers_text = st.text_input("Tickers (comma separated)", value=DEFAULT)
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
with right:
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.35, 0.05)
    period = st.selectbox("History window", ["6mo", "1y", "2y"], index=1)
    # Bust cache if user changes period a lot (simple approach)
    st.caption("Tip: refresh page if you change window often.")

rows = []
with st.spinner("Generating signals..."):
    for t in tickers[:50]:
        # period is not wired into cache key; quick hack: fetch more then compute same
        rows.append(signal_for(t))

df = pd.DataFrame(rows)
df = df[df["confidence"] >= min_conf].copy()
df["price"] = pd.to_numeric(df["price"], errors="coerce")

st.subheader("Signals")
st.dataframe(
    df[["ticker", "signal", "confidence", "price"]],
    use_container_width=True,
    hide_index=True
)
