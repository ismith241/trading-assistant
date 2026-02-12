import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Trading Assistant (MVP)", layout="wide")
st.title("Trading Assistant (MVP)")
st.caption("Decision-support only. Not financial advice.")

DEFAULT_TICKERS = "AAPL,MSFT,NVDA,AMZN,GOOGL,SPY"


# ---------------- Helpers ----------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.where(d > 0, 0.0).rolling(n).mean()
    loss = (-d.where(d < 0, 0.0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int)):
            return float(x)
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure columns are single-level and trimmed."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).strip())
    return df


def _trim_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    # Rough trading-day trims
    if df is None or df.empty:
        return pd.DataFrame()
    if period == "6mo":
        return df.tail(140)
    if period == "1y":
        return df.tail(260)
    if period == "2y":
        return df.tail(520)
    return df


def _is_plain_us_equity_ticker(ticker: str) -> bool:
    """
    Heuristic: plain US equity/ETF tickers are typically letters/numbers with optional - or .
    We EXCLUDE symbols like futures (=F), FX (=X), indexes (^), etc.
    """
    t = ticker.upper().strip()
    if any(ch in t for ch in ["=", "^", "/"]):
        return False
    return True


# ---------------- Data Fetch ----------------
@st.cache_data(ttl=900)  # cache 15 minutes
def fetch_prices(ticker: str, period: str) -> pd.DataFrame:
    """
    Fetch daily adjusted prices.
    - Primary: yfinance (broad coverage, including many special symbols like SI=F, XAGUSD=X)
    - Fallback: Stooq CSV for plain US equity/ETF tickers if yfinance returns empty
    """
    ticker = ticker.strip().upper()

    # 1) yfinance first
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        df = _normalize_df(df)
        if not df.empty and "Close" in df.columns:
            return df
    except Exception:
        pass

    # 2) Fallback for plain US tickers: Stooq
    if not _is_plain_us_equity_ticker(ticker):
        return pd.DataFrame()

    # Stooq generally uses lower-case with .us for US equities/ETFs
    candidates = [f"{ticker.lower()}.us", ticker.lower()]

    for sym in candidates:
        try:
            url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
            sdf = pd.read_csv(url)
            if sdf is None or sdf.empty:
                continue

            sdf.columns = [c.strip().title() for c in sdf.columns]
            if "Date" not in sdf.columns or "Close" not in sdf.columns:
                continue

            sdf["Date"] = pd.to_datetime(sdf["Date"], errors="coerce")
            sdf = sdf.dropna(subset=["Date"]).set_index("Date").sort_index()

            sdf = _trim_period(sdf, period)

            # Ensure columns match expected names
            # (Stooq typically has Open/High/Low/Close/Volume)
            return sdf
        except Exception:
            continue

    return pd.DataFrame()


# ---------------- Signal Engine ----------------
def make_signal(ticker: str, period: str) -> dict:
    df = fetch_prices(ticker, period)

    if df.empty or "Close" not in df.columns:
        return {
            "ticker": ticker.upper().strip(),
            "signal": "HOLD",
            "confidence": 0.0,
            "price": None,
            "status": "NO DATA",
            "reasons": ["No data returned (ticker unsupported by free sources or temporarily blocked)."],
        }

    close = df["Close"].astype(float)
    price = float(close.iloc[-1])

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    rsi14 = rsi(close, 14)

    score = 0.0
    reasons: list[str] = []

    # Long trend (200D)
    if len(close) >= 200 and not np.isnan(sma200.iloc[-1]):
        if price > float(sma200.iloc[-1]):
            score += 0.35
            reasons.append("Above 200D avg (uptrend).")
        else:
            score -= 0.35
            reasons.append("Below 200D avg (downtrend).")
    else:
        reasons.append("Not enough history for 200D avg.")

    # Momentum (20 vs 50)
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
        "ticker": ticker.upper().strip(),
        "signal": sig,
        "confidence": conf,
        "price": price,
        "status": "OK",
        "reasons": reasons,
    }


# ---------------- UI ----------------
left, right = st.columns([2, 1])

with left:
    tickers_text = st.text_input("Tickers (comma separated)", value=DEFAULT_TICKERS)
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

with right:
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.20, 0.05)
    period = st.selectbox("History window", ["6mo", "1y", "2y"], index=1)
    show = st.multiselect("Show", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD", "SELL"])
    st.caption("Tip: raise Min confidence to hide weak signals.")

# ---- Main signals (your input tickers) ----
rows = []
with st.spinner("Generating signals..."):
    for t in tickers[:200]:  # allow large lists; limit prevents accidental huge runs
        rows.append(make_signal(t, period))

df = pd.DataFrame(rows)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

# Keep NO DATA rows visible unless user filters them out by confidence/show
df_view = df[df["signal"].isin(show)].copy()
df_view = df_view[(df_view["confidence"] >= float(min_conf)) | (df_view["status"] != "OK")]

st.subheader("Signals")
st.dataframe(
    df_view[["ticker", "signal", "confidence", "price", "status"]],
    use_container_width=True,
    hide_index=True,
)

# ---- Metals section (separate area) ----
st.subheader("Metals")
st.caption("Silver trackers (separate from your equity watchlist).")

METALS = [
    ("Silver Futures (COMEX)", "SI=F"),
    ("Silver Spot (XAGUSD)", "XAGUSD=X"),
    ("SLV ETF", "SLV"),
]

met_rows = []
with st.spinner("Updating metals..."):
    for label, sym in METALS:
        s = make_signal(sym, period)
        s["name"] = label
        met_rows.append(s)

met_df = pd.DataFrame(met_rows)
met_df["price"] = pd.to_numeric(met_df["price"], errors="coerce")
met_df["confidence"] = pd.to_numeric(met_df["confidence"], errors="coerce").fillna(0.0)

st.dataframe(
    met_df[["name", "ticker", "signal", "confidence", "price", "status"]],
    use_container_width=True,
    hide_index=True,
)

# ---- Details ----
st.subheader("Details")
choices = df_view["ticker"].dropna().tolist()
default_choice = choices[0] if choices else (tickers[0] if tickers else "SPY")
choice = st.selectbox("Pick a ticker", options=choices if choices else [default_choice])

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

# Metals details quick pick
with st.expander("Metals details"):
    met_choice = st.selectbox("Pick a metals tracker", options=[m[0] for m in METALS])
    met_sym = next((m[1] for m in METALS if m[0] == met_choice), None)
    met_one = next((r for r in met_rows if r["ticker"] == met_sym), None)
    if met_one:
        st.write(f"**{met_choice}** ({met_sym})")
        st.write(f"Signal: **{met_one['signal']}** | Confidence: **{met_one['confidence']:.2f}** | Status: **{met_one['status']}**")
        st.write("Why:")
        for r in met_one["reasons"]:
            st.write(f"• {r}")
