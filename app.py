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


# ---------- Data ----------
@st.cache_data(ttl=900)  # cache 15 minutes
def fetch(ticker: str, period: str = "1y") -> pd.DataFrame:
    ticker = ticker.strip().upper()

    # 1) Try yfinance
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if df is None or df.empty:
            raise ValueError("yfinance returned empty")

        # yfinance sometimes returns MultiIndex columns on cloud
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=lambda c: c.strip())
        return df
    except Exception:
        pass

    # 2) Fallback: Stooq daily CSV (often works when yfinance is blocked/rate-limited)
    try:
        # For US stocks/ETFs, Stooq uses e.g. aapl.us, spy.us
        stooq_symbol = f"{ticker.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        df = pd.read_csv(url)

        if df is None or df.empty:
            return pd.DataFrame()

        # Normalize headers -> Date, Open, High, Low, Close, Volume
        df.columns = [c.strip().title() for c in df.columns]

        if "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

        # Roughly trim to match selected period (keeps app fast)
        if period == "6mo":
            df = df.tail(140)
        elif period == "1y":
            df = df.tail(260)
        elif period == "2y":
            df = df.tail(520)

        return df
    except Exception:
        return pd.DataFrame()


# ---------- Signal ----------
def signal_for(ticker: str, period: str) -> dict:
    df = fetch(ticker, period)
    if df.empty or "Close" not in df.columns:
        return {
            "ticker": ticker.upper(),
            "signal": "HOLD",
            "confidence": 0.0,
            "price": None,
            "reasons": ["No data returned (data source blocked or rate-limited)."],
        }

    close = df["Close"].astype(float)
    price = float(close.iloc[-1])

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    rsi14 = rsi(close, 14)

    score = 0.0
    reasons: list[str] = []

    # Long trend
    if len(close) >= 200 and not np.isnan(sma200.iloc[-1]):
        if price > float(sma200.iloc[-1]):
            score += 0.35
            reasons.append("Above 200D avg (uptrend).")
        else:
            score -= 0.35
            reasons.append("Below 200D avg (downtrend).")
    else:
        reasons.append("Not enough history for 200D avg.")

    # Momentum
    if not np.isnan(sma20.iloc[-1]) and not np.isnan(sma50.iloc[-1]):
        if float(sma20.iloc[-1]) > float(sma50.iloc[-1]):
            score += 0.25
            reasons.append("20D > 50D (positive momentum).")
        else:
            score -= 0.25
            reasons.append("20D < 50D (weak momentum).")
    else:
        reasons.append("Not enough history for 20/50D averages.")

    # RSI sanity
    if not np.isnan(rsi14.iloc[-1]):
        rv = float(rsi14.iloc[-1])
        if rv < 35:
            score += 0.20
            reasons.append(f"RSI {rv:.1f} (oversold-ish).")
        elif rv > 70:
            score -= 0.20
            reasons.append(f"RSI {rv:.1f} (overbought-ish).")
        else:
            score += 0.05
            reasons.append(f"RSI {rv:.1f} (neutral).")
    else:
        reasons.append("RSI unavailable.")

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
        "ticker": ticker.upper(),
        "signal": sig,
        "confidence": conf,
        "price": price,
        "reasons": reasons,
    }


# ---------- UI ----------
left, right = st.columns([2, 1])

with left:
    tickers_text = st.text_input("Tickers (comma separated)", value=DEFAULT)
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

with right:
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.20, 0.05)
    period = st.selectbox("History window", ["6mo", "1y", "2y"], index=1)
    st.caption("Tip: raise Min confidence to hide weak signals.")

rows = []
with st.spinner("Generating signals..."):
    for t in tickers[:50]:
        rows.append(signal_for(t, period))

df = pd.DataFrame(rows)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df[df["confidence"] >= float(min_conf)].copy()

st.subheader("Signals")
st.dataframe(
    df[["ticker", "signal", "confidence", "price"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Details")
choice = st.selectbox("Pick a ticker", options=df["ticker"].tolist() if not df.empty else tickers)
if choice:
    one = next((r for r in rows if r["ticker"] == choice), None)
    if one:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Signal", one["signal"])
            st.metric("Confidence", f'{one["confidence"]:.2f}')
            st.metric("Price", "—" if one["price"] is None else f'{float(one["price"]):.2f}')
            st.write("Chart:")
            st.write(f"https://finance.yahoo.com/quote/{choice}")
        with c2:
            st.write("Why:")
            for r in one["reasons"]:
                st.write(f"• {r}")
