import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from io import StringIO
from urllib.request import Request, urlopen

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


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).strip())
    return df


def _trim_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if period == "6mo":
        return df.tail(140)
    if period == "1y":
        return df.tail(260)
    if period == "2y":
        return df.tail(520)
    return df


def _stooq_read(symbol: str, period: str) -> pd.DataFrame:
    """
    Stooq CSV downloader (cloud-safe).
    Very reliable for US equities/ETFs (aapl.us, spy.us).
    """
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame()

    head = raw.lstrip()[:200].lower()
    if head.startswith("<!doctype") or head.startswith("<html"):
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO(raw))
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = [c.strip().title() for c in df.columns]
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = _trim_period(df, period)
    return df


def _yahoo_get_daily(symbol: str, period: str) -> pd.DataFrame:
    """
    Yahoo fetch (robust):
      - yf.download()
      - then yf.Ticker().history() fallback
    """
    symbol = symbol.strip()

    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        df = _normalize_df(df)
        if not df.empty and "Close" in df.columns:
            return df
    except Exception:
        pass

    try:
        t = yf.Ticker(symbol)
        h = t.history(period=period, interval="1d", auto_adjust=False)
        h = _normalize_df(h)
        if not h.empty and "Close" in h.columns:
            return h
    except Exception:
        pass

    return pd.DataFrame()


@st.cache_data(ttl=900)  # cache 15 minutes
def fetch_prices(ticker: str, period: str) -> pd.DataFrame:
    """
    Fetch daily prices.

    Strategy:
      - Try Yahoo first (best general coverage for tickers)
      - If empty, fall back to Stooq for US equities/ETFs (ticker.us, with BRK-B -> brk.b.us)
    """
    ticker = ticker.strip().upper()

    # A) Yahoo first
    df = _yahoo_get_daily(ticker, period)

    # Yahoo alias for spot silver
    if df.empty and ticker == "XAGUSD":
        df = _yahoo_get_daily("XAGUSD=X", period)

    if not df.empty and "Close" in df.columns:
        return df

    # B) Stooq fallback for US equities/ETFs
    t_low = ticker.lower()
    candidates = [f"{t_low}.us"]
    if "-" in t_low:
        candidates.append(f"{t_low.replace('-', '.')}.us")

    for sym in candidates:
        sdf = _stooq_read(sym, period)
        if not sdf.empty and "Close" in sdf.columns:
            return sdf

    return pd.DataFrame()


# ---------------- Signal Engine ----------------
def make_signal(ticker: str, period: str, display_ticker: str | None = None) -> dict:
    df = fetch_prices(ticker, period)

    if df.empty or "Close" not in df.columns:
        return {
            "ticker": (display_ticker or ticker).upper().strip(),
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
        "ticker": (display_ticker or ticker).upper().strip(),
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


# ---- Main watchlist ----
rows = []
with st.spinner("Generating signals..."):
    for t in tickers[:200]:
        rows.append(make_signal(t, period))

df = pd.DataFrame(rows)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

df_view = df[df["signal"].isin(show)].copy()
df_view = df_view[(df_view["confidence"] >= float(min_conf)) | (df_view["status"] != "OK")]

st.subheader("Signals")
st.dataframe(
    df_view[["ticker", "signal", "confidence", "price", "status"]],
    use_container_width=True,
    hide_index=True,
)


# ---- Metals (fixed + stable) ----
st.subheader("Metals")
st.caption("We use Spot Silver as a stable proxy for silver futures (they move very closely).")

# Silver spot:
#   Yahoo: XAGUSD=X
# SLV:
#   Yahoo: SLV
# Futures proxy:
#   We simply label spot as 'SIW00 (proxy)' for tracking purposes.
met_rows = []
with st.spinner("Updating metals..."):
    # Spot silver (the clean source)
    spot = make_signal("XAGUSD=X", period, display_ticker="SIW00 (PROXY)")
    spot["name"] = "Silver Futures (SIW00 proxy via Spot)"
    spot["reasons"] = ["Using Spot Silver as a proxy for COMEX silver futures."] + spot.get("reasons", [])
    met_rows.append(spot)

    # Also show spot explicitly (optional clarity)
    spot2 = make_signal("XAGUSD=X", period, display_ticker="XAGUSD (SPOT)")
    spot2["name"] = "Silver Spot (XAG/USD)"
    met_rows.append(spot2)

    # SLV ETF
    slv = make_signal("SLV", period)
    slv["name"] = "SLV ETF"
    met_rows.append(slv)

met_df = pd.DataFrame(met_rows)
met_df["price"] = pd.to_numeric(met_df["price"], errors="coerce")
met_df["confidence"] = pd.to_numeric(met_df["confidence"], errors="coerce").fillna(0.0)

st.dataframe(
    met_df[["name", "ticker", "signal", "confidence", "price", "status"]],
    use_container_width=True,
    hide_index=True,
)

with st.expander("Metals details"):
    met_choice = st.selectbox("Pick a metals tracker", options=met_df["name"].tolist())
    met_one = next((r for r in met_rows if r.get("name") == met_choice), None)
    if met_one:
        st.write(f"**{met_one.get('name','')}** ({met_one.get('ticker','')})")
        st.write(
            f"Signal: **{met_one['signal']}** | Confidence: **{met_one['confidence']:.2f}** | Status: **{met_one['status']}**"
        )
        st.write("Why:")
        for r in met_one.get("reasons", []):
            st.write(f"• {r}")


# ---- Details (main tickers) ----
st.subheader("Details (Main Watchlist)")
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
