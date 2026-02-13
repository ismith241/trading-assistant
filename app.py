import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from io import StringIO
from urllib.request import Request, urlopen
from datetime import date, timedelta

st.set_page_config(page_title="Trading Assistant (MVP)", layout="wide")
st.title("Trading Assistant (MVP)")
st.caption("Decision-support only. Not financial advice.")


# =========================
# Helpers
# =========================
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


def _filter_dates(df: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df
    if start is not None:
        out = out[out.index.date >= start]
    if end is not None:
        out = out[out.index.date <= end]
    return out


def _stooq_read(symbol: str, start: date | None, end: date | None, period: str | None) -> pd.DataFrame:
    """
    Cloud-safe Stooq fetch (good fallback for many equities; also for xagusd spot).
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
    df = _filter_dates(df, start, end)

    if period and (start is None and end is None):
        trims = {"1mo": 25, "3mo": 70, "6mo": 140, "1y": 260, "2y": 520, "5y": 1300}
        if period in trims:
            df = df.tail(trims[period])

    return df


def _yahoo_download(symbol: str, start: date | None, end: date | None, period: str | None) -> pd.DataFrame:
    """
    More robust Yahoo fetch:
      1) yf.download()
      2) yf.Ticker().history()
    """
    symbol = symbol.strip()

    # Method 1: download
    try:
        if start or end:
            df = yf.download(
                symbol,
                start=start,
                end=(end + timedelta(days=1)) if end else None,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        else:
            df = yf.download(
                symbol,
                period=period or "1y",
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

    # Method 2: history
    try:
        t = yf.Ticker(symbol)
        if start or end:
            h = t.history(
                start=start,
                end=(end + timedelta(days=1)) if end else None,
                interval="1d",
                auto_adjust=False,
            )
        else:
            h = t.history(period=period or "1y", interval="1d", auto_adjust=False)
        h = _normalize_df(h)
        if not h.empty and "Close" in h.columns:
            return h
    except Exception:
        pass

    return pd.DataFrame()


@st.cache_data(ttl=900)  # cache 15 minutes
def fetch_prices(ticker: str, start: date | None, end: date | None, period: str | None) -> pd.DataFrame:
    """
    Fetch daily prices with robust fallbacks.
    - Spot silver: Yahoo (XAGUSD=X) then Stooq (xagusd)
    - Others: Yahoo first, then Stooq (ticker.us) for many equities/ETFs
    """
    t = ticker.strip().upper()
    if not t:
        return pd.DataFrame()

    # Spot silver special-case (restore the "it works" behavior)
    if t in ("XAGUSD", "XAGUSD=X"):
        df = _yahoo_download("XAGUSD=X", start, end, period)
        if df.empty:
            df = _stooq_read("xagusd", start, end, period)
        return df

    # Yahoo first
    df = _yahoo_download(t, start, end, period)
    if not df.empty and "Close" in df.columns:
        return df

    # Stooq fallback for US equities/ETFs
    t_low = t.lower()
    candidates = [f"{t_low}.us"]
    if "-" in t_low:
        candidates.append(f"{t_low.replace('-', '.')}.us")  # BRK-B -> brk.b.us

    for sym in candidates:
        sdf = _stooq_read(sym, start, end, period)
        if not sdf.empty and "Close" in sdf.columns:
            return sdf

    return pd.DataFrame()


def _latest(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _slope_annualized(close: pd.Series, lookback: int = 90) -> float | None:
    close = close.dropna()
    if len(close) < lookback + 5:
        return None
    y = np.log(close.tail(lookback).astype(float).values)
    x = np.arange(len(y))
    m = np.polyfit(x, y, 1)[0]  # slope per day in log space
    return float(m * 252.0)  # annualized


def _ratio_series(a: pd.Series, b: pd.Series) -> pd.Series:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return (df["a"] / df["b"]).rename("ratio")


# =========================
# Signal engine
# =========================
def make_signal(
    ticker: str,
    start: date | None,
    end: date | None,
    period: str | None,
    display_ticker: str | None = None,
) -> dict:
    df = fetch_prices(ticker, start, end, period)
    if df.empty or "Close" not in df.columns:
        return {
            "ticker": (display_ticker or ticker).upper().strip(),
            "signal": "HOLD",
            "confidence": 0.0,
            "price": None,
            "status": "NO DATA",
            "reasons": ["No data returned (source blocked/rate-limited or symbol unsupported)."],
        }

    close = df["Close"].astype(float)
    price = float(close.iloc[-1])

    score = 0.0
    reasons: list[str] = []

    sma20 = sma(close, 20).iloc[-1] if len(close) >= 20 else np.nan
    sma50 = sma(close, 50).iloc[-1] if len(close) >= 50 else np.nan
    sma200 = sma(close, 200).iloc[-1] if len(close) >= 200 else np.nan
    rsi14 = rsi(close, 14).iloc[-1] if len(close) >= 15 else np.nan

    # Trend
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
    else:
        reasons.append("Not enough history for 20/50D averages.")

    # RSI
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
    else:
        reasons.append("RSI unavailable.")

    # Volatility
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

    sig = "BUY" if score >= 0.35 else ("SELL" if score <= -0.35 else "HOLD")
    conf = float(min(1.0, max(0.0, abs(score))))

    return {
        "ticker": (display_ticker or ticker).upper().strip(),
        "signal": sig,
        "confidence": conf,
        "price": price,
        "status": "OK",
        "reasons": reasons,
    }


# =========================
# Regime engine
# =========================
def compute_regime(
    start: date | None,
    end: date | None,
    period: str | None,
) -> tuple[str, int, pd.DataFrame, list[str], dict]:
    """
    Returns: (regime, score, table, notes, series_pack)
    """
    # Ensure enough history if not using custom dates
    r_period = "2y" if (start is None and end is None) else period

    spy = fetch_prices("SPY", start, end, r_period)
    vix = fetch_prices("^VIX", start, end, r_period)
    hyg = fetch_prices("HYG", start, end, r_period)
    ief = fetch_prices("IEF", start, end, r_period)
    rsp = fetch_prices("RSP", start, end, r_period)
    uup = fetch_prices("UUP", start, end, r_period)

    score = 0
    rows = []

    series_pack = {"SPY": spy, "VIX": vix, "HYG": hyg, "IEF": ief, "RSP": rsp, "UUP": uup}

    def add(factor: str, state: str, pts: int, detail: str, explain: str):
        nonlocal score
        score += pts
        rows.append({"Factor": factor, "State": state, "Points": pts, "Detail": detail, "Explain": explain})

    # SPY trend: above/below 200D + slope
    if spy.empty or "Close" not in spy.columns:
        add(
            "SPY Trend",
            "NO DATA",
            0,
            "—",
            "Broad market trend. Above the 200D average with a positive slope is generally risk-on.",
        )
    else:
        c = spy["Close"].astype(float)
        p = _latest(c)
        ma = _latest(sma(c, 200))
        sl = _slope_annualized(c, 90)

        pts = 0
        bits = []

        if p is not None and ma is not None:
            if p > ma:
                pts += 1
                bits.append("Above 200D")
            else:
                pts -= 1
                bits.append("Below 200D")

        if sl is not None:
            if sl > 0:
                pts += 1
                bits.append("Slope +")
            else:
                pts -= 1
                bits.append("Slope -")

        add(
            "SPY Trend",
            ", ".join(bits) if bits else "Unknown",
            pts,
            f"Slope≈{sl:.2f}" if sl is not None else "Slope NA",
            "Captures broad market health. Above 200D and rising slope tends to support staying invested.",
        )

    # VIX: below/above 50D
    if vix.empty or "Close" not in vix.columns:
        add(
            "Volatility (VIX)",
            "NO DATA",
            0,
            "—",
            "VIX is a fear gauge. Below its 50D average implies calmer conditions; rising VIX signals stress.",
        )
    else:
        vc = vix["Close"].astype(float)
        vp = _latest(vc)
        vma = _latest(sma(vc, 50))
        pts = 0
        if vp is not None and vma is not None:
            if vp < vma:
                pts += 1
                state = "Calm (VIX < 50D)"
            else:
                pts -= 1
                state = "Risk (VIX > 50D)"
        else:
            state = "Unknown"
        add("Volatility (VIX)", state, pts, f"VIX={vp:.2f}" if vp is not None else "VIX NA", "Measures market stress/uncertainty.")

    # Credit: HYG/IEF slope
    if hyg.empty or ief.empty or "Close" not in hyg.columns or "Close" not in ief.columns:
        add(
            "Credit (HYG/IEF)",
            "NO DATA",
            0,
            "—",
            "High yield vs Treasuries. If junk bonds outperform, credit conditions are improving (risk-on).",
        )
    else:
        ratio = _ratio_series(hyg["Close"].astype(float), ief["Close"].astype(float))
        sl = _slope_annualized(ratio, 90)
        pts = 0
        if sl is not None:
            if sl > 0:
                pts += 1
                state = "Improving"
            else:
                pts -= 1
                state = "Deteriorating"
        else:
            state = "Unknown"
        add("Credit (HYG/IEF)", state, pts, f"Slope≈{sl:.2f}" if sl is not None else "Slope NA", "A proxy for risk appetite in credit markets.")

    # Breadth: RSP/SPY slope
    if rsp.empty or spy.empty or "Close" not in rsp.columns or "Close" not in spy.columns:
        add(
            "Breadth (RSP/SPY)",
            "NO DATA",
            0,
            "—",
            "Equal-weight outperforming cap-weight suggests broader participation (healthier rallies).",
        )
    else:
        ratio = _ratio_series(rsp["Close"].astype(float), spy["Close"].astype(float))
        sl = _slope_annualized(ratio, 90)
        pts = 0
        if sl is not None:
            if sl > 0:
                pts += 1
                state = "Broadening"
            else:
                pts -= 1
                state = "Narrowing"
        else:
            state = "Unknown"
        add("Breadth (RSP/SPY)", state, pts, f"Slope≈{sl:.2f}" if sl is not None else "Slope NA", "Broad strength tends to be more durable than narrow rallies.")

    # Dollar: UUP slope (strength often a headwind)
    if uup.empty or "Close" not in uup.columns:
        add(
            "Dollar (UUP)",
            "NO DATA",
            0,
            "—",
            "A strengthening dollar can tighten conditions; a weakening dollar can be a tailwind for risk assets.",
        )
    else:
        uc = uup["Close"].astype(float)
        sl = _slope_annualized(uc, 90)
        pts = 0
        if sl is not None:
            if sl > 0:
                pts -= 1
                state = "Strengthening (headwind)"
            else:
                pts += 1
                state = "Weakening (tailwind)"
        else:
            state = "Unknown"
        add("Dollar (UUP)", state, pts, f"Slope≈{sl:.2f}" if sl is not None else "Slope NA", "Dollar direction often influences global liquidity.")

    regime = "RISK ON" if score >= 2 else ("RISK OFF" if score <= -2 else "NEUTRAL")

    notes = []
    if regime == "RISK ON":
        notes.append("Suggested equity exposure: 80–100% (stay invested; rotate within strength).")
    elif regime == "NEUTRAL":
        notes.append("Suggested equity exposure: 50–70% (be selective; avoid overtrading).")
    else:
        notes.append("Suggested equity exposure: 20–40% (defensive posture; prioritize drawdown control).")

    return regime, int(score), pd.DataFrame(rows), notes, series_pack


# =========================
# Sidebar controls (timeframes)
# =========================
st.sidebar.header("Time Range")
use_custom = st.sidebar.toggle("Custom date range", value=False)

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
default_period = "1y"

if use_custom:
    today = date.today()
    default_start = today - timedelta(days=365)
    start_date = st.sidebar.date_input("Start date", value=default_start)
    end_date = st.sidebar.date_input("End date", value=today)
    # guard: if user flips them
    if isinstance(start_date, list) or isinstance(end_date, list):
        # streamlit can return lists in some cases; normalize
        start_date = start_date[0] if start_date else default_start
        end_date = end_date[0] if end_date else today
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
    period = None
else:
    start_date = None
    end_date = None
    period = st.sidebar.selectbox("Preset range", PERIODS, index=PERIODS.index(default_period))


# =========================
# Main UI Controls
# =========================
left, right = st.columns([2, 1])

with left:
    tickers_text = st.text_input(
        "Tickers (comma separated)",
        value="",
        placeholder="Example: AAPL, MSFT, NVDA, SPY",
    )
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    st.caption("Leave blank if you only want the Regime + Metals panels.")

with right:
    min_conf_user = st.slider("Min confidence", 0.0, 1.0, 0.20, 0.05)
    show = st.multiselect("Show", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD", "SELL"])
    st.caption("Signals are filtered by the Market Regime rules below.")


# =========================
# Market Regime Panel
# =========================
st.subheader("Market Regime")

with st.spinner("Calculating regime..."):
    regime, rscore, rtable, rnotes, regime_series = compute_regime(start_date, end_date, period)

c1, c2, c3 = st.columns(3)
c1.metric("Regime", regime)
c2.metric("Regime Score", rscore)
c3.metric("Cache", "15 min")

# Explanations: quick access under each indicator
with st.expander("What do these regime indicators mean?"):
    for _, row in rtable.iterrows():
        st.markdown(f"**{row['Factor']}** — {row['Explain']}")

# Show scorecard
st.dataframe(rtable[["Factor", "State", "Points", "Detail"]], use_container_width=True, hide_index=True)
for n in rnotes:
    st.write(f"• {n}")

# Optional mini-charts
with st.expander("Regime charts"):
    # SPY chart with 200D
    spy_df = regime_series.get("SPY", pd.DataFrame())
    if not spy_df.empty and "Close" in spy_df.columns:
        tmp = spy_df[["Close"]].copy()
        tmp["SMA200"] = sma(tmp["Close"].astype(float), 200)
        st.markdown("**SPY (Close + 200D)**")
        st.line_chart(tmp[["Close", "SMA200"]].dropna())

    # VIX chart with 50D
    vix_df = regime_series.get("VIX", pd.DataFrame())
    if not vix_df.empty and "Close" in vix_df.columns:
        tmp = vix_df[["Close"]].copy()
        tmp["SMA50"] = sma(tmp["Close"].astype(float), 50)
        st.markdown("**VIX (Close + 50D)**")
        st.line_chart(tmp[["Close", "SMA50"]].dropna())

    # Credit ratio
    hyg_df = regime_series.get("HYG", pd.DataFrame())
    ief_df = regime_series.get("IEF", pd.DataFrame())
    if (
        isinstance(hyg_df, pd.DataFrame)
        and isinstance(ief_df, pd.DataFrame)
        and not hyg_df.empty
        and not ief_df.empty
        and "Close" in hyg_df.columns
        and "Close" in ief_df.columns
    ):
        ratio = _ratio_series(hyg_df["Close"].astype(float), ief_df["Close"].astype(float))
        if not ratio.empty:
            st.markdown("**Credit Stress (HYG/IEF)**")
            st.line_chart(ratio)

# =========================
# Regime controls Signals tab
# =========================
if regime == "RISK ON":
    regime_min_conf = min_conf_user
    buy_gate = 0.35
    msg = "Risk On: normal filtering."
elif regime == "NEUTRAL":
    regime_min_conf = min(1.0, min_conf_user + 0.05)
    buy_gate = 0.45
    msg = "Neutral: BUY signals must be stronger; confidence floor slightly raised."
else:
    regime_min_conf = min(1.0, min_conf_user + 0.10)
    buy_gate = 0.65
    msg = "Risk Off: BUY signals are heavily restricted; focus on defense / patience."

st.info(f"Regime rule applied → **{msg}**  |  Effective Min Confidence: **{regime_min_conf:.2f}**  |  BUY Gate: **{buy_gate:.2f}**")

st.divider()


# =========================
# Signals (user tickers)
# =========================
st.subheader("Signals")

if not tickers:
    st.caption("No tickers entered — signals table is empty by design.")
else:
    rows = []
    with st.spinner("Generating signals..."):
        for t in tickers[:200]:
            rows.append(make_signal(t, start_date, end_date, period))

    df = pd.DataFrame(rows)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

    # Apply regime gating for BUY signals (and user-selected signal filter)
    df["regime_allowed"] = True
    df.loc[(df["signal"] == "BUY") & (df["confidence"] < buy_gate), "regime_allowed"] = False

    df_view = df[df["signal"].isin(show)].copy()
    df_view = df_view[df_view["regime_allowed"]].copy()
    df_view = df_view[(df_view["confidence"] >= float(regime_min_conf)) | (df_view["status"] != "OK")]

    # Make it easier to read
    df_view = df_view.sort_values(["signal", "confidence"], ascending=[True, False])

    st.dataframe(
        df_view[["ticker", "signal", "confidence", "price", "status"]],
        use_container_width=True,
        hide_index=True,
    )


# =========================
# Metals (independent)
# =========================
st.subheader("Metals")
st.caption("Spot silver is used as a stable proxy for COMEX silver futures (they move very closely).")

met_rows = []
with st.spinner("Updating metals..."):
    proxy = make_signal("XAGUSD", start_date, end_date, period, display_ticker="SIW00 (PROXY)")
    proxy["name"] = "Silver Futures (SIW00 proxy via Spot)"
    proxy["reasons"] = ["Using Spot Silver (XAG/USD) as a proxy for COMEX silver futures."] + proxy.get("reasons", [])
    met_rows.append(proxy)

    spot = make_signal("XAGUSD", start_date, end_date, period, display_ticker="XAGUSD (SPOT)")
    spot["name"] = "Silver Spot (XAG/USD)"
    met_rows.append(spot)

    slv = make_signal("SLV", start_date, end_date, period)
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
        st.write(f"Signal: **{met_one['signal']}** | Confidence: **{met_one['confidence']:.2f}** | Status: **{met_one['status']}**")
        st.write("Why:")
        for r in met_one.get("reasons", []):
            st.write(f"• {r}")

# Metals graphs
with st.expander("Metals charts"):
    for sym, title in [("XAGUSD", "Spot Silver (XAG/USD)"), ("SLV", "SLV ETF")]:
        dfm = fetch_prices(sym, start_date, end_date, period)
        if not dfm.empty and "Close" in dfm.columns:
            st.markdown(f"**{title}**")
            st.line_chart(dfm["Close"].astype(float))


# =========================
# Details + graphs (signals)
# =========================
st.subheader("Details + Charts (Signals)")

if tickers:
    # Use the filtered view if it exists; otherwise use full tickers list
    try:
        options = df_view["ticker"].dropna().unique().tolist()
    except Exception:
        options = tickers

    if options:
        pick = st.selectbox("Pick a ticker", options=options)
        ddf = fetch_prices(pick, start_date, end_date, period)

        if ddf.empty or "Close" not in ddf.columns:
            st.warning("No chart data available for this ticker in the selected range.")
        else:
            close = ddf["Close"].astype(float)
            chart = pd.DataFrame({"Close": close})
            chart["SMA50"] = sma(close, 50)
            chart["SMA200"] = sma(close, 200)

            c1, c2 = st.columns([1, 2])
            with c1:
                sig = make_signal(pick, start_date, end_date, period)
                st.metric("Signal", sig["signal"])
                st.metric("Confidence", f"{sig['confidence']:.2f}")
                st.metric("Last Price", f"{float(sig['price']):.2f}" if sig["price"] is not None else "—")
            with c2:
                st.line_chart(chart.dropna())

            with st.expander("Why this signal?"):
                for r in sig.get("reasons", []):
                    st.write(f"• {r}")
else:
    st.caption("Enter tickers above to enable signal charts.")

