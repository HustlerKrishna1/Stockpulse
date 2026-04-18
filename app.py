"""StockPulse AI — Advanced agentic multi-market analysis platform.
Free stack: yfinance · Plotly · Streamlit · Groq (Llama 3.3).
"""
from __future__ import annotations
import os
import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="StockPulse AI", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

# ==========  PREMIUM STYLE (injected via JS to bypass markdown parser)  ==========
import streamlit.components.v1 as _components_style

_PREMIUM_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stText, input, textarea, button, label, p, span, div {
  font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(0,255,163,0.07) 0%, transparent 60%),
    radial-gradient(900px 500px at 110% 10%, rgba(136,132,216,0.09) 0%, transparent 60%),
    linear-gradient(180deg, #0a0e1a 0%, #06080f 100%) !important;
  color: #e6eaf2 !important;
}
h1 { font-weight: 800 !important; letter-spacing: -0.02em !important; font-size: 2.3rem !important; color: #fff !important; }
h2 { font-weight: 700 !important; letter-spacing: -0.01em !important; color: #fff !important; }
h3, h4 { font-weight: 600 !important; color: #f2f5fa !important; }
p, label, .stMarkdown, .stCaption, li { color: #c8d0dd !important; }

div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 14px 16px;
  transition: all .2s ease;
}
div[data-testid="stMetric"]:hover { border-color: rgba(0,255,163,0.35); transform: translateY(-1px); }
div[data-testid="stMetricLabel"] { font-size: .72rem !important; letter-spacing: .1em; text-transform: uppercase; color: #8892a6 !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] { color: #fff !important; font-weight: 700 !important; font-size: 1.55rem !important; font-family: 'JetBrains Mono', monospace !important; }
div[data-testid="stMetricDelta"] { font-weight: 600 !important; }

.stTabs div[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 2px; flex-wrap: wrap; }
.stTabs button[data-baseweb="tab"] {
  background: transparent !important; border-radius: 10px !important; padding: 8px 14px !important;
  color: #8892a6 !important; border: 1px solid transparent !important;
  font-weight: 500 !important; font-size: .88rem !important;
}
.stTabs button[data-baseweb="tab"]:hover { color: #fff !important; background: rgba(255,255,255,0.04) !important; }
.stTabs button[aria-selected="true"] {
  background: rgba(0,255,163,0.1) !important;
  color: #00ffa3 !important;
  border: 1px solid rgba(0,255,163,0.32) !important;
}

.stTextInput input, .stTextArea textarea {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 10px !important;
  color: #fff !important;
  font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: rgba(0,255,163,0.5) !important; box-shadow: 0 0 0 3px rgba(0,255,163,0.08) !important; }
.stSelectbox div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 10px !important;
  color: #fff !important;
}

.stButton button {
  background: rgba(0,255,163,0.1) !important;
  border: 1px solid rgba(0,255,163,0.3) !important;
  color: #00ffa3 !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  transition: all .15s ease;
}
.stButton button:hover { background: rgba(0,255,163,0.18) !important; border-color: rgba(0,255,163,0.6) !important; transform: translateY(-1px); }
.stButton button[kind="primary"] { background: #00ffa3 !important; color: #06080f !important; border-color: #00ffa3 !important; }
.stButton button[kind="primary"]:hover { background: #00e891 !important; }

.verdict-box {
  padding: 22px; border-radius: 16px;
  background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
  border: 1px solid rgba(0,255,163,0.22);
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
}
.news-card {
  padding: 14px 16px; border-radius: 12px;
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.06);
  margin-bottom: 10px;
  transition: all .15s;
}
.news-card:hover { border-color: rgba(0,255,163,0.3); background: rgba(255,255,255,0.04); transform: translateX(3px); }
.chip {
  display: inline-block; padding: 4px 12px; border-radius: 999px;
  background: rgba(0,255,163,0.1); color: #00ffa3; font-size: .72rem;
  border: 1px solid rgba(0,255,163,0.25); margin-right: 6px; font-weight: 600;
  letter-spacing: .02em;
}

div[data-testid="stChatMessage"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 12px !important;
  padding: 14px !important;
}
.stDataFrame, div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.06); }
div[data-testid="stToggle"] label { color: #c8d0dd !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }

header[data-testid="stHeader"] { background: transparent !important; }
footer, #MainMenu { visibility: hidden !important; height: 0 !important; }
div[data-testid="stDecoration"] { display: none !important; }

section[data-testid="stSidebar"] {
  background: rgba(10,14,26,0.8) !important;
  border-right: 1px solid rgba(255,255,255,0.06);
  backdrop-filter: blur(10px);
}

/* Block container spacing */
.block-container { padding-top: 1.5rem !important; padding-bottom: 4rem !important; max-width: 1600px !important; }

/* Expander */
.streamlit-expanderHeader, div[data-testid="stExpander"] summary {
  background: rgba(255,255,255,0.025) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 10px !important;
  color: #c8d0dd !important;
}

/* Alerts */
div[data-testid="stAlert"] { border-radius: 12px !important; border-width: 1px !important; }

/* Plotly charts */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
"""

# Inject CSS into parent document via JS (bypasses Streamlit's markdown parser completely)
_components_style.html(
    f"<script>(function(){{var s=parent.document.getElementById('sp-premium-css');"
    f"if(s)s.remove();"
    f"var n=parent.document.createElement('style');n.id='sp-premium-css';"
    f"n.textContent={repr(_PREMIUM_CSS)};"
    f"parent.document.head.appendChild(n);}})();</script>",
    height=0, width=0,
)

# ==========  PRESETS  ==========
PRESETS = {
    "🇺🇸 US": {"Apple": "AAPL", "Tesla": "TSLA", "NVIDIA": "NVDA", "Microsoft": "MSFT",
              "Google": "GOOGL", "Amazon": "AMZN", "Meta": "META", "S&P 500": "^GSPC"},
    "🇮🇳 India": {"Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "Infosys": "INFY.NS",
                  "HDFC Bank": "HDFCBANK.NS", "NIFTY 50": "^NSEI", "SENSEX": "^BSESN"},
    "🇦🇪 GCC": {"Emaar (DFM)": "EMAAR.AE", "DIB": "DIB.AE", "FAB (ADX)": "FAB.AE",
                "ADCB": "ADCB.AE", "Aramco": "2222.SR", "DP World": "DPW.DI"},
    "🇯🇵 Asia": {"Toyota": "7203.T", "Sony": "6758.T", "Alibaba": "9988.HK",
                 "Tencent": "0700.HK", "TSMC": "TSM", "Nikkei": "^N225"},
    "🇪🇺 Europe": {"BMW": "BMW.DE", "SAP": "SAP.DE", "LVMH": "MC.PA",
                   "HSBC": "HSBA.L", "Shell": "SHEL.L", "DAX": "^GDAXI"},
    "₿ Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD",
                 "BNB": "BNB-USD", "XRP": "XRP-USD"},
    "💱 FX/Commodity": {"EUR/USD": "EURUSD=X", "USD/INR": "INR=X", "USD/AED": "AED=X",
                        "Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F"},
}

CURRENCY_SYM = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥",
                "HKD": "HK$", "CNY": "¥", "AED": "AED ", "SAR": "SAR ", "AUD": "A$"}

# ==========  GLOBAL SEARCH  ==========
import requests

@st.cache_data(ttl=1800, show_spinner=False)
def yahoo_search(q: str, limit: int = 10):
    """Search Yahoo Finance for any company/asset. Returns list of {symbol, name, exchange, type}."""
    if not q or len(q.strip()) < 1: return []
    try:
        r = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": q, "quotesCount": limit, "newsCount": 0, "enableFuzzyQuery": True},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36"},
            timeout=5,
        )
        quotes = r.json().get("quotes", [])
        out = []
        for qu in quotes:
            sym = qu.get("symbol")
            if not sym: continue
            out.append({
                "symbol": sym,
                "name": qu.get("longname") or qu.get("shortname") or sym,
                "exchange": qu.get("exchDisp") or qu.get("exchange") or "",
                "type": qu.get("quoteType", "").upper(),
            })
        return out
    except Exception:
        return []

# ==========  HEADER / HERO  ==========
if "ticker" not in st.session_state: st.session_state.ticker = "AAPL"
if "search_results" not in st.session_state: st.session_state.search_results = []

import streamlit.components.v1 as components

# TradingView Ticker Tape (live global)
components.html("""
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
  {
  "symbols": [
    {"proName": "FOREXCOM:SPXUSD", "title": "S&P 500"},
    {"proName": "FOREXCOM:NSXUSD", "title": "Nasdaq 100"},
    {"proName": "FX_IDC:EURUSD", "title": "EUR/USD"},
    {"proName": "BITSTAMP:BTCUSD", "title": "BTC"},
    {"proName": "BITSTAMP:ETHUSD", "title": "ETH"},
    {"description": "Gold", "proName": "TVC:GOLD"},
    {"description": "Oil WTI", "proName": "TVC:USOIL"},
    {"description": "DXY", "proName": "TVC:DXY"},
    {"description": "VIX", "proName": "TVC:VIX"},
    {"description": "NIFTY", "proName": "NSE:NIFTY"},
    {"description": "Apple", "proName": "NASDAQ:AAPL"},
    {"description": "Nvidia", "proName": "NASDAQ:NVDA"},
    {"description": "Tesla", "proName": "NASDAQ:TSLA"}
  ],
  "showSymbolLogo": true,
  "colorTheme": "dark",
  "isTransparent": true,
  "displayMode": "adaptive",
  "locale": "en"
  }
  </script>
</div>
""", height=46)

hcol1, hcol2 = st.columns([3, 2])
with hcol1:
    st.markdown("<h1 style='margin-bottom:4px'>📈 StockPulse <span style='color:#00ffa3'>AI</span></h1>"
                "<p style='color:#8892a6; margin-top:0'>Institutional-grade multi-market intelligence · "
                "<span class='chip'>Agentic</span><span class='chip'>ML</span>"
                "<span class='chip'>Backtest</span><span class='chip'>Macro</span>"
                "<span class='chip'>TradingView</span><span class='chip'>Dalio AI</span></p>",
                unsafe_allow_html=True)
with hcol2:
    st.write("")
    q = st.text_input("🔍 Search any company, crypto, index — globally",
        placeholder="Type 'apple', 'reliance', 'bitcoin', 'aramco', 'nifty', 'EUR/USD'...",
        key="search_q", label_visibility="collapsed")
    if q:
        st.session_state.search_results = yahoo_search(q)

# Render search results as pills
if st.session_state.search_results:
    st.markdown("<p style='margin-top:12px; color:#8892a6; font-size:.85rem'>Matches — click to select:</p>",
                unsafe_allow_html=True)
    cols = st.columns(min(5, len(st.session_state.search_results)))
    for i, item in enumerate(st.session_state.search_results[:10]):
        with cols[i % len(cols)]:
            label = f"{item['symbol']}\n{item['name'][:22]}"
            if st.button(label, key=f"pick_{i}", help=f"{item['name']} · {item['exchange']} · {item['type']}"):
                st.session_state.ticker = item["symbol"]
                st.session_state.search_results = []
                st.rerun()
st.divider()

# ==========  CONTROL BAR  ==========
cc1, cc2, cc3, cc4, cc5 = st.columns([2, 1.2, 1.2, 1, 1])
with cc1:
    ticker_in = st.text_input("Ticker", value=st.session_state.ticker,
        help="yfinance symbol — e.g. AAPL · RELIANCE.NS · EMAAR.AE · BTC-USD · EURUSD=X").upper().strip()
    if ticker_in != st.session_state.ticker:
        st.session_state.ticker = ticker_in
ticker = st.session_state.ticker or "AAPL"

with cc2:
    period = st.selectbox("Period", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=4)
with cc3:
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"], index=5)
with cc4:
    st.write("")
    live = st.toggle("🔴 Live")
    if live:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=30_000, key="live")
        except ImportError:
            pass
with cc5:
    st.write("")
    groq_ok = bool(os.getenv("GROQ_API_KEY") or
        (hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets))
    st.markdown(f"<div style='padding-top:6px'><span class='chip'>{'🟢 AI Connected' if groq_ok else '⚪ No AI key'}</span></div>",
                unsafe_allow_html=True)

with st.expander("💼 Portfolio (comma-separated tickers)"):
    portfolio_raw = st.text_area("Portfolio tickers", value="AAPL, MSFT, NVDA, GOOGL, TSLA",
        label_visibility="collapsed", help="Up to 10 tickers for correlation, perf, risk.")
portfolio = [t.strip().upper() for t in portfolio_raw.split(",") if t.strip()][:10]

if not ticker:
    st.stop()

# ==========  INDICATORS  ==========
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = -d.clip(upper=0).rolling(n).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def macd(s: pd.Series, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m = ef - es
    sg = m.ewm(span=sig, adjust=False).mean()
    return m, sg, m - sg

def bollinger(s: pd.Series, n=20, k=2):
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return ma + k * sd, ma, ma - k * sd

def stochastic(df, n=14):
    low = df["Low"].rolling(n).min()
    high = df["High"].rolling(n).max()
    k = 100 * (df["Close"] - low) / (high - low)
    return k, k.rolling(3).mean()

def atr(df, n=14):
    h_l = df["High"] - df["Low"]
    h_c = (df["High"] - df["Close"].shift()).abs()
    l_c = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ==========  DATA  ==========
@st.cache_data(ttl=30, show_spinner=False)
def load_data(sym: str, period: str, interval: str):
    t = yf.Ticker(sym)
    hist = t.history(period=period, interval=interval, auto_adjust=True)
    info = {}
    news = []
    try: info = t.info or {}
    except Exception: pass
    try: news = (t.news or [])[:8]
    except Exception: pass
    return hist, info, news

with st.spinner(f"Loading {ticker}..."):
    df, info, news = load_data(ticker, period, interval)

if df is None or df.empty:
    st.error(f"No data for '{ticker}'. Try a different symbol.")
    st.stop()

# Enrich
df["RSI"] = rsi(df["Close"])
df["MACD"], df["Signal"], df["Hist"] = macd(df["Close"])
df["BB_U"], df["BB_M"], df["BB_L"] = bollinger(df["Close"])
df["SMA20"] = df["Close"].rolling(20).mean()
df["SMA50"] = df["Close"].rolling(50).mean()
df["SMA200"] = df["Close"].rolling(200).mean()
df["StochK"], df["StochD"] = stochastic(df)
df["ATR"] = atr(df)
df["Returns"] = df["Close"].pct_change()

# ==========  HERO METRICS  ==========
last = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else last
change = last["Close"] - prev["Close"]
pct = (change / prev["Close"]) * 100 if prev["Close"] else 0
ccy = info.get("currency", "USD")
sym = CURRENCY_SYM.get(ccy, ccy + " ")

h1, h2, h3, h4, h5, h6 = st.columns(6)
h1.metric("Price", f"{sym}{last['Close']:,.2f}", f"{pct:+.2f}%")
h2.metric("Volume", f"{int(last['Volume']):,}" if not math.isnan(last['Volume']) else "—")
h3.metric("RSI(14)", f"{last['RSI']:.1f}")
h4.metric("ATR(14)", f"{last['ATR']:.2f}")
mcap = info.get("marketCap")
h5.metric("Market Cap", f"{sym}{mcap/1e9:.2f}B" if mcap else "—")
pe = info.get("trailingPE")
h6.metric("P/E", f"{pe:.1f}" if pe else "—")

st.caption(f"📍 **{info.get('longName', ticker)}** · "
           f"{info.get('exchange', '—')} · {ccy} · "
           f"{info.get('sector', '—')} / {info.get('industry', '—')} · "
           f"Updated {df.index[-1]}")

# ==========  TABS  ==========
tabs = st.tabs(["📊 Chart", "🎯 TV Terminal", "🤖 Verdict", "🧠 HF Council",
                "💬 Chat", "🛠️ Tool Agent", "🧙 Dalio AI", "🧪 Strategy Lab",
                "🔮 ML Forecast", "⚖️ Risk", "📅 Fundamentals", "🏛️ SEC Filings",
                "👥 Insiders", "📜 Transcripts", "🌍 Macro", "💼 Portfolio",
                "🔗 Options", "📰 News", "⭐ Watchlist", "🕌 Sharia", "⚙️ Settings"])

# ---------- TAB 1: CHART ----------
with tabs[0]:
    show_bb = st.checkbox("Bollinger Bands", value=True)
    show_sma = st.checkbox("SMAs (20/50/200)", value=True)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.02,
        subplot_titles=("Price", "Volume", "RSI", "MACD"))

    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                  low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], line=dict(color="#00ffa3", width=1), name="SMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], line=dict(color="#ff6b9d", width=1), name="SMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], line=dict(color="#ffd93d", width=1), name="SMA200"), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_U"], line=dict(color="rgba(136,132,216,0.4)", dash="dot"), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_L"], line=dict(color="rgba(136,132,216,0.4)", dash="dot"), name="BB Lower", fill="tonexty", fillcolor="rgba(136,132,216,0.05)"), row=1, col=1)

    colors = ["#00ffa3" if c >= o else "#ff6b9d" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#ffd93d"), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Hist"], marker_color="#8884d8", name="MACD Hist"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#00ffa3"), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], line=dict(color="#ff6b9d"), name="Signal"), row=4, col=1)

    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.04), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("⬇️ Download CSV", df.to_csv().encode(),
                       file_name=f"{ticker}_{period}.csv", mime="text/csv")

# ---------- TAB 2: AI VERDICT ----------
def rule_verdict(df):
    r = df["RSI"].iloc[-1]
    macd_x = df["MACD"].iloc[-1] - df["Signal"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    sma200 = df["SMA200"].iloc[-1] if not math.isnan(df["SMA200"].iloc[-1]) else sma50
    close = df["Close"].iloc[-1]
    k = df["StochK"].iloc[-1]
    bb_u = df["BB_U"].iloc[-1]; bb_l = df["BB_L"].iloc[-1]
    score, reasons = 0, []
    if r < 30: score += 2; reasons.append(f"🟢 RSI {r:.1f} — oversold")
    elif r > 70: score -= 2; reasons.append(f"🔴 RSI {r:.1f} — overbought")
    else: reasons.append(f"⚪ RSI {r:.1f} — neutral")
    if macd_x > 0: score += 1; reasons.append("🟢 MACD > Signal — bullish momentum")
    else: score -= 1; reasons.append("🔴 MACD < Signal — bearish momentum")
    if close > sma50 > sma200: score += 2; reasons.append("🟢 Golden arrangement (Price > SMA50 > SMA200)")
    elif close < sma50 < sma200: score -= 2; reasons.append("🔴 Death arrangement (Price < SMA50 < SMA200)")
    if k < 20: score += 1; reasons.append(f"🟢 Stochastic {k:.1f} — oversold")
    elif k > 80: score -= 1; reasons.append(f"🔴 Stochastic {k:.1f} — overbought")
    if close < bb_l: score += 1; reasons.append("🟢 Below lower Bollinger — mean-reversion setup")
    elif close > bb_u: score -= 1; reasons.append("🔴 Above upper Bollinger — overextended")
    verdict = "STRONG BUY" if score >= 4 else ("BUY" if score >= 2 else
              ("SELL" if score <= -2 else ("STRONG SELL" if score <= -4 else "HOLD")))
    emoji = {"STRONG BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "STRONG SELL": "🔴🔴"}[verdict]
    return verdict, emoji, reasons, score

def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key and hasattr(st, "secrets"):
        try: key = st.secrets["GROQ_API_KEY"]
        except Exception: key = None
    if not key: return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except Exception:
        return None

def llm_analysis(ticker, info, df, news):
    client = get_groq_client()
    if not client: return None
    try:
        snapshot = {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "price": float(df["Close"].iloc[-1]),
            "change_pct_period": float((df["Close"].iloc[-1]/df["Close"].iloc[0]-1)*100),
            "rsi": float(df["RSI"].iloc[-1]),
            "macd_hist": float(df["Hist"].iloc[-1]),
            "sma20": float(df["SMA20"].iloc[-1]),
            "sma50": float(df["SMA50"].iloc[-1]),
            "volatility_annualized": float(df["Returns"].std() * np.sqrt(252) * 100),
            "recent_headlines": [n.get("title") for n in news[:5]],
        }
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are StockPulse AI — a decisive trader-analyst (BingX/Bybit style). NEVER say you lack real-time data — the snapshot IS real-time. NO disclaimers, NO hedging. Output in this exact format:\n\n**📊 Technical read** (3 bullets, specific numbers)\n**💼 Fundamental read** (2 bullets)\n**📰 News read** (1-2 bullets)\n**⚠️ Key risks** (2 bullets)\n\n**🎯 Trade setup**\n• Bias: BULLISH/NEUTRAL/BEARISH (confidence X%)\n• Entry zone: <price>\n• Stop loss: <price> (−X%)\n• TP1: <price> (+X%)\n• TP2: <price> (+X%)\n• Timeframe: scalp/swing/position\n\n**🏁 Final verdict:** STRONG BUY / BUY / HOLD / SELL / STRONG SELL — confidence X%"},
                {"role": "user", "content": f"Analyze: {snapshot}"},
            ],
            temperature=0.3, max_tokens=700,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"_LLM error: {e}_"

with tabs[2]:
    verdict, emoji, reasons, score = rule_verdict(df)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""<div class='verdict-box'>
            <h2 style='margin:0'>{emoji} {verdict}</h2>
            <p style='font-size:1.1em; margin:8px 0'>Score: <b>{score:+d}</b> / ±8</p>
            <p style='opacity:0.8'>Rule-engine verdict from 5 weighted signals.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("**Signals**")
        for r in reasons: st.write(r)
    with c2:
        st.markdown("### 🧠 LLM Analyst (Groq · Llama 3.3 70B)")
        ai = llm_analysis(ticker, info, df, news)
        if ai: st.markdown(ai)
        else: st.info("Set `GROQ_API_KEY` to enable AI analysis. Free at console.groq.com")

# ---------- TAB 3: RISK & FORECAST ----------
def monte_carlo(df, days=30, sims=500):
    r = df["Returns"].dropna()
    mu, sigma = r.mean(), r.std()
    last_price = df["Close"].iloc[-1]
    paths = np.zeros((sims, days))
    for i in range(sims):
        shocks = np.random.normal(mu, sigma, days)
        paths[i] = last_price * np.exp(np.cumsum(shocks))
    return paths

with tabs[9]:
    st.subheader("📐 Risk Metrics")
    r = df["Returns"].dropna()
    ann_vol = r.std() * np.sqrt(252) * 100
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() else 0
    roll_max = df["Close"].cummax()
    dd = (df["Close"] / roll_max - 1) * 100
    max_dd = dd.min()
    var95 = np.percentile(r, 5) * 100
    cvar95 = r[r <= np.percentile(r, 5)].mean() * 100

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Annual Volatility", f"{ann_vol:.2f}%")
    r2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    r3.metric("Max Drawdown", f"{max_dd:.2f}%")
    r4.metric("VaR 95%", f"{var95:.2f}%")
    r5.metric("CVaR 95%", f"{cvar95:.2f}%")

    st.subheader("🎲 Monte Carlo Forecast (30-day · 500 sims)")
    days = st.slider("Horizon (days)", 7, 90, 30)
    sims = st.slider("Simulations", 100, 2000, 500, step=100)
    paths = monte_carlo(df, days=days, sims=sims)

    mc_fig = go.Figure()
    for i in range(min(100, sims)):
        mc_fig.add_trace(go.Scatter(y=paths[i], mode="lines",
            line=dict(color="rgba(0,255,163,0.08)", width=1), showlegend=False))
    mc_fig.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=0), mode="lines",
        line=dict(color="#00ffa3", width=3), name="Median"))
    mc_fig.add_trace(go.Scatter(y=np.percentile(paths, 95, axis=0), mode="lines",
        line=dict(color="#ffd93d", width=2, dash="dash"), name="95th %"))
    mc_fig.add_trace(go.Scatter(y=np.percentile(paths, 5, axis=0), mode="lines",
        line=dict(color="#ff6b9d", width=2, dash="dash"), name="5th %"))
    mc_fig.update_layout(height=450, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Day", yaxis_title=f"Price ({ccy})")
    st.plotly_chart(mc_fig, use_container_width=True)

    end_prices = paths[:, -1]
    p1, p2, p3 = st.columns(3)
    p1.metric("Expected (median)", f"{sym}{np.median(end_prices):,.2f}")
    p2.metric("Bull case (95%)", f"{sym}{np.percentile(end_prices, 95):,.2f}")
    p3.metric("Bear case (5%)", f"{sym}{np.percentile(end_prices, 5):,.2f}")

# ---------- TAB 4: PORTFOLIO ----------
@st.cache_data(ttl=60, show_spinner=False)
def load_portfolio(tickers, period):
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
    if "Close" in data: return data["Close"]
    return data

with tabs[15]:
    st.subheader(f"💼 Portfolio ({len(portfolio)} tickers)")
    if len(portfolio) < 2:
        st.info("Add 2+ tickers in the sidebar.")
    else:
        try:
            pdata = load_portfolio(portfolio, period)
            if isinstance(pdata, pd.Series): pdata = pdata.to_frame()
            pdata = pdata.dropna(how="all")
            norm = (pdata / pdata.iloc[0]) * 100

            pfig = go.Figure()
            for col in norm.columns:
                pfig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=col))
            pfig.update_layout(height=400, template="plotly_dark",
                title="Normalized performance (start = 100)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(pfig, use_container_width=True)

            st.subheader("🔗 Correlation matrix")
            corr = pdata.pct_change().corr()
            hfig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                colorscale="RdYlGn", zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}"))
            hfig.update_layout(height=400, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(hfig, use_container_width=True)

            st.subheader("📊 Risk-Return profile")
            rets = pdata.pct_change().dropna()
            summary = pd.DataFrame({
                "Ticker": rets.columns,
                "Annual Return %": (rets.mean() * 252 * 100).values,
                "Volatility %": (rets.std() * np.sqrt(252) * 100).values,
                "Sharpe": (rets.mean() / rets.std() * np.sqrt(252)).values,
            })
            st.dataframe(summary.style.format({"Annual Return %": "{:.2f}",
                "Volatility %": "{:.2f}", "Sharpe": "{:.2f}"}), use_container_width=True)
        except Exception as e:
            st.error(f"Portfolio load failed: {e}")

# ---------- TAB 5: OPTIONS ----------
with tabs[16]:
    st.subheader("🔗 Options Chain")
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            st.info("No options available for this symbol.")
        else:
            exp = st.selectbox("Expiration", expirations[:12])
            chain = t.option_chain(exp)
            oc1, oc2 = st.columns(2)
            with oc1:
                st.markdown("**📈 Calls**")
                st.dataframe(chain.calls[["strike", "lastPrice", "bid", "ask", "volume",
                    "openInterest", "impliedVolatility"]].head(20), use_container_width=True)
            with oc2:
                st.markdown("**📉 Puts**")
                st.dataframe(chain.puts[["strike", "lastPrice", "bid", "ask", "volume",
                    "openInterest", "impliedVolatility"]].head(20), use_container_width=True)
    except Exception as e:
        st.info(f"Options not available: {e}")

# ---------- TAB 6: NEWS ----------
def news_sentiment(headlines: list[str]) -> Optional[str]:
    client = get_groq_client()
    if not client or not headlines: return None
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Classify the overall sentiment of these headlines as BULLISH / NEUTRAL / BEARISH. One word, then a 1-line reason."},
                {"role": "user", "content": "\n".join(headlines)},
            ],
            temperature=0.1, max_tokens=80,
        )
        return r.choices[0].message.content
    except Exception:
        return None

with tabs[17]:
    st.subheader(f"📰 Latest news on {ticker}")
    if not news:
        st.info("No news available.")
    else:
        heads = [n.get("title", "") for n in news if n.get("title")]
        sent = news_sentiment(heads)
        if sent:
            st.markdown(f"**🧠 AI Sentiment:** {sent}")
        for n in news:
            title = n.get("title") or (n.get("content", {}) or {}).get("title", "")
            link = n.get("link") or (n.get("content", {}) or {}).get("canonicalUrl", {}).get("url", "#")
            pub = n.get("publisher") or (n.get("content", {}) or {}).get("provider", {}).get("displayName", "")
            if title:
                st.markdown(f"<div class='news-card'><a href='{link}' target='_blank' style='color:#00ffa3; text-decoration:none'><b>{title}</b></a><br><span style='opacity:0.6'>{pub}</span></div>", unsafe_allow_html=True)

# ---------- TAB 7: AI CHAT ----------
import re as _re

def _build_snapshot(sym, period="6mo", interval="1d"):
    try:
        h, inf, nw = load_data(sym, period, interval)
        if h is None or h.empty: return None
        h = h.copy()
        h["RSI"] = rsi(h["Close"])
        h["MACD"], h["Signal"], h["Hist"] = macd(h["Close"])
        h["SMA20"] = h["Close"].rolling(20).mean()
        h["SMA50"] = h["Close"].rolling(50).mean()
        h["SMA200"] = h["Close"].rolling(200).mean()
        h["ATR"] = atr(h)
        h["Returns"] = h["Close"].pct_change()
        last = h.iloc[-1]
        return {
            "ticker": sym,
            "name": inf.get("longName", sym),
            "sector": inf.get("sector"),
            "price": float(last["Close"]),
            "change_1d_pct": float((last["Close"]/h["Close"].iloc[-2]-1)*100) if len(h) > 1 else 0,
            "change_period_pct": float((last["Close"]/h["Close"].iloc[0]-1)*100),
            "rsi": float(last["RSI"]),
            "macd_hist": float(last["Hist"]),
            "macd_cross": "bullish" if last["MACD"] > last["Signal"] else "bearish",
            "sma20": float(last["SMA20"]),
            "sma50": float(last["SMA50"]),
            "sma200": float(last["SMA200"]) if not math.isnan(last["SMA200"]) else None,
            "atr": float(last["ATR"]),
            "period_high": float(h["High"].max()),
            "period_low": float(h["Low"].min()),
            "annual_vol_pct": float(h["Returns"].std() * np.sqrt(252) * 100),
            "volume": int(last["Volume"]) if not math.isnan(last["Volume"]) else 0,
            "headlines": [n.get("title") for n in (nw or [])[:5] if n.get("title")],
        }
    except Exception:
        return None

def _extract_tickers_regex(text: str) -> list[str]:
    cands = _re.findall(r"\b(\^?[A-Z]{1,6}(?:[.-][A-Z0-9]{1,4})?(?:=X)?(?:-USD)?)\b", text.upper())
    stop = {"AI", "RSI", "MACD", "SMA", "ATR", "BUY", "SELL", "HOLD", "CEO", "USD", "EUR",
            "INR", "AED", "PM", "AM", "IV", "OI", "ETF", "IPO", "EPS", "PE", "TP", "SL", "Q1",
            "Q2", "Q3", "Q4", "YTD", "YOY", "I", "A", "THE", "IS", "IT", "AS", "OF", "TO",
            "IN", "ON", "AT", "BE", "OR", "AND", "FOR", "CAN", "YOU", "ARE", "WAS", "NOT",
            "DO", "HAS", "ITS", "GOOD", "TIME", "ENTRY", "LONG", "SHORT", "STOP", "LOSS",
            "NOW", "SOON", "BEST", "TOP", "NEXT", "MOVE", "WHAT", "WHEN", "WHERE", "WHY",
            "HOW", "WHO", "SHOULD", "COULD", "WOULD", "MARKET", "STOCK", "PRICE", "TREND"}
    return [c for c in cands if c not in stop and len(c) >= 2]

@st.cache_data(ttl=600, show_spinner=False)
def _resolve_with_llm(question: str, current_ticker: str) -> list[str]:
    """Use Groq to convert natural-language entities to yfinance tickers."""
    client = get_groq_client()
    if not client: return []
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": (
                    "Extract every company/asset/index/crypto/FX pair mentioned in the user's question "
                    "and return them as valid yfinance symbols. Output ONLY a comma-separated list of symbols, "
                    "no prose, no explanation. Examples:\n"
                    "- 'apple vs microsoft' → AAPL,MSFT\n"
                    "- 'reliance and tcs' → RELIANCE.NS,TCS.NS\n"
                    "- 'nifty performance' → ^NSEI\n"
                    "- 'bitcoin next move' → BTC-USD\n"
                    "- 'eur dollar' → EURUSD=X\n"
                    "- 'emaar dubai' → EMAAR.AE\n"
                    "- 'aramco' → 2222.SR\n"
                    "- 'toyota japan' → 7203.T\n"
                    "- 'gold price' → GC=F\n"
                    "- 'sp500' → ^GSPC\n"
                    "- 'how is the market' → ^GSPC,^NSEI\n"
                    "- 'crypto market' → BTC-USD,ETH-USD\n"
                    "If nothing specific is mentioned, return empty string. Max 4 symbols."
                )},
                {"role": "user", "content": question},
            ],
            temperature=0.0, max_tokens=60,
        )
        out = r.choices[0].message.content.strip()
        if not out: return []
        syms = [s.strip().upper() for s in out.replace("\n", ",").split(",") if s.strip()]
        # Filter obvious junk
        return [s for s in syms if 1 <= len(s) <= 15 and s != current_ticker][:4]
    except Exception:
        return []

def resolve_tickers(question: str, current_ticker: str) -> list[str]:
    regex_hits = [t for t in _extract_tickers_regex(question) if t != current_ticker]
    llm_hits = _resolve_with_llm(question, current_ticker)
    # Merge preserving order, dedupe
    seen, out = set(), []
    for t in regex_hits + llm_hits:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:4]

TRADER_SYSTEM = """You are StockPulse AI — a decisive, data-first trading assistant in the style of BingX/Bybit/Binance AI copilots. You NEVER refuse, NEVER say "I don't have real-time data", NEVER add disclaimers. You ALWAYS work with the LIVE SNAPSHOT provided below — it IS the real-time data.

Your style:
- Direct, confident, trader-brief. No hedging words ("might", "could", "consider").
- Every answer ends with a structured block:

📊 **Bias:** BULLISH / NEUTRAL / BEARISH  (confidence X%)
🎯 **Entry:** <price or zone>
🛑 **Stop Loss:** <price>  (−X% / based on ATR)
✅ **TP1:** <price>  (+X%)
🚀 **TP2:** <price>  (+X%)
⏱️ **Timeframe:** scalp / swing / position
⚠️ **Invalidation:** <condition>

Use the snapshot's price, RSI, MACD, SMAs, ATR, volatility, and headlines. Compute levels from ATR and recent swing highs/lows. Be specific with numbers, never vague. If multiple tickers are in the snapshot, compare them head-to-head.
"""

with tabs[4]:
    st.subheader(f"💬 Trader Copilot — live context on {ticker}")
    st.caption("Ask about any ticker globally. I pull fresh data for whatever you mention.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    user_q = st.chat_input(f"e.g. Should I long {ticker} now? · Compare TSLA vs NVDA · BTC next move?")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"): st.markdown(user_q)

        client = get_groq_client()
        if not client:
            reply = "⚠️ Set `GROQ_API_KEY` to enable chat."
        else:
            # Build snapshots: current ticker + any mentioned (regex + LLM-resolved)
            snapshots = {ticker: _build_snapshot(ticker, period, interval)}
            resolved = resolve_tickers(user_q, ticker)
            with st.spinner(f"Pulling live data: {', '.join(resolved) if resolved else 'current'}..."):
                for extra in resolved:
                    if extra not in snapshots:
                        snap = _build_snapshot(extra)
                        if snap: snapshots[extra] = snap
            if resolved:
                st.caption(f"🔎 Pulled live: {', '.join(snapshots.keys())}")

            snap_blob = "\n\n".join(
                f"=== LIVE SNAPSHOT · {k} ===\n{v}" for k, v in snapshots.items() if v
            ) or "No snapshot available."

            try:
                r = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": TRADER_SYSTEM + "\n\n" + snap_blob},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]],
                    ],
                    temperature=0.35, max_tokens=700,
                )
                reply = r.choices[0].message.content
            except Exception as e:
                reply = f"Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"): st.markdown(reply)

    cbtn1, cbtn2 = st.columns([1, 6])
    with cbtn1:
        if st.button("🗑️ Clear"):
            st.session_state.messages = []
            st.rerun()

# ---------- TAB 8: SHARIA COMPLIANCE ----------
HARAM_SECTORS = {"banking", "insurance", "alcohol", "tobacco", "gambling", "adult", "pork", "weapons", "defense"}
HARAM_KEYWORDS = ["bank", "insurance", "alcohol", "beer", "wine", "tobacco", "casino",
                  "gambling", "pork", "weapon", "defense", "adult"]

with tabs[19]:
    st.subheader("🕌 Sharia Compliance Screen")
    st.caption("Simplified AAOIFI-style screen. Not a fatwa — consult a scholar for certification.")
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    name = (info.get("longName") or ticker).lower()
    combined = f"{sector} {industry} {name}"

    haram_hits = [k for k in HARAM_KEYWORDS if k in combined]

    debt_ratio = None
    try:
        total_debt = info.get("totalDebt")
        market_cap = info.get("marketCap")
        if total_debt and market_cap:
            debt_ratio = (total_debt / market_cap) * 100
    except Exception:
        pass

    compliant = not haram_hits and (debt_ratio is None or debt_ratio < 33)

    cc1, cc2 = st.columns(2)
    with cc1:
        if compliant:
            st.success(f"✅ Likely Sharia-compliant · {info.get('longName', ticker)}")
        else:
            st.error(f"❌ Likely non-compliant · {info.get('longName', ticker)}")
        st.write(f"**Sector:** {info.get('sector', '—')}")
        st.write(f"**Industry:** {info.get('industry', '—')}")
        if haram_hits:
            st.write(f"**Flagged keywords:** {', '.join(haram_hits)}")
    with cc2:
        st.markdown("**Financial ratios (AAOIFI thresholds)**")
        st.write(f"Debt / Market Cap: **{debt_ratio:.1f}%**" if debt_ratio else "Debt ratio: —")
        st.caption("Rule: < 33% debt · < 33% interest-bearing securities · < 5% impure income")

# ==========  TAB 9: HEDGE FUND COUNCIL  ==========
PERSONAS = {
    "🔬 Technical Analyst": ("Be a chart-first quant. Focus ONLY on RSI, MACD, SMAs, Bollinger, volume, ATR, "
                             "trend structure, support/resistance. 3 bullets + BUY/HOLD/SELL call with conviction %."),
    "📚 Fundamental Analyst": ("Be a Buffett-style fundamental analyst. Focus ONLY on PE, margins, sector health, "
                               "moat, valuation vs peers. 3 bullets + BUY/HOLD/SELL call with conviction %."),
    "🛡️ Risk Officer": ("Be a paranoid risk manager. Focus ONLY on volatility, drawdown risk, tail risk, liquidity, "
                        "headline risk. Push back on FOMO. 3 bullets + RISK RATING low/med/high and whether to proceed."),
    "💼 Portfolio Manager": ("You synthesize the other 3 analysts + macro context. Give the FINAL institutional call: "
                             "Position size % of portfolio, entry zone, stop, TP1, TP2, time horizon, and a 1-line thesis."),
}

def run_persona(persona_name, instruction, snapshot):
    client = get_groq_client()
    if not client: return f"_Need GROQ_API_KEY_"
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are the {persona_name} on an elite hedge fund desk. {instruction} "
                                              f"Be terse, specific, and numeric. No hedging. The snapshot below IS real-time."},
                {"role": "user", "content": f"Ticker snapshot: {snapshot}"},
            ],
            temperature=0.4, max_tokens=350,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"_Error: {e}_"

with tabs[3]:
    st.subheader("🧠 Hedge Fund Council — 4-agent debate")
    st.caption("Parallel analysis from Technical · Fundamental · Risk · Portfolio Manager — then a synthesized call.")
    if st.button("🎯 Convene Council", type="primary"):
        snap = _build_snapshot(ticker, period, interval)
        if not snap:
            st.error("Snapshot unavailable.")
        else:
            cols = st.columns(2)
            persona_items = list(PERSONAS.items())
            # run in pairs, 2 columns
            with st.spinner("4 analysts thinking..."):
                outputs = {name: run_persona(name, instr, snap) for name, instr in persona_items}
            for i, (name, text) in enumerate(outputs.items()):
                with cols[i % 2]:
                    st.markdown(f"### {name}")
                    st.markdown(f"<div class='verdict-box'>{text}</div>", unsafe_allow_html=True)
                    st.write("")
    else:
        st.info("Click 'Convene Council' to run the 4-agent debate.")

# ==========  TAB 10: STRATEGY LAB  ==========
def backtest_sma(df, fast=20, slow=50):
    d = df.copy()
    d["f"] = d["Close"].rolling(fast).mean()
    d["s"] = d["Close"].rolling(slow).mean()
    d["sig"] = np.where(d["f"] > d["s"], 1, 0)
    d["ret"] = d["Close"].pct_change()
    d["stratret"] = d["sig"].shift(1) * d["ret"]
    d["eq"] = (1 + d["stratret"].fillna(0)).cumprod()
    d["bh"] = (1 + d["ret"].fillna(0)).cumprod()
    return d

def backtest_rsi(df, low=30, high=70):
    d = df.copy()
    d["rsi"] = rsi(d["Close"])
    # long when RSI crosses above low (oversold exit), flat when crosses below high (overbought entry)
    sig = np.where(d["rsi"] < low, 1, np.where(d["rsi"] > high, 0, np.nan))
    d["sig"] = pd.Series(sig, index=d.index).ffill().fillna(0)
    d["ret"] = d["Close"].pct_change()
    d["stratret"] = d["sig"].shift(1) * d["ret"]
    d["eq"] = (1 + d["stratret"].fillna(0)).cumprod()
    d["bh"] = (1 + d["ret"].fillna(0)).cumprod()
    return d

def backtest_macd(df):
    d = df.copy()
    m, s, _ = macd(d["Close"])
    d["sig"] = np.where(m > s, 1, 0)
    d["ret"] = d["Close"].pct_change()
    d["stratret"] = d["sig"].shift(1) * d["ret"]
    d["eq"] = (1 + d["stratret"].fillna(0)).cumprod()
    d["bh"] = (1 + d["ret"].fillna(0)).cumprod()
    return d

def bt_metrics(d):
    r = d["stratret"].dropna()
    total = d["eq"].iloc[-1] - 1
    bh_total = d["bh"].iloc[-1] - 1
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() else 0
    roll_max = d["eq"].cummax()
    dd = (d["eq"] / roll_max - 1).min()
    trades = int((d["sig"].diff().fillna(0) != 0).sum())
    win = (r[r > 0].count() / max(1, r[r != 0].count())) * 100
    return total * 100, bh_total * 100, sharpe, dd * 100, trades, win

with tabs[7]:
    st.subheader("🧪 Strategy Lab — Backtest your edge")
    strat = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean-Reversion", "MACD Momentum"])
    if strat == "SMA Crossover":
        cf, cs = st.columns(2)
        fast = cf.slider("Fast SMA", 5, 50, 20)
        slow = cs.slider("Slow SMA", 20, 200, 50)
        bt = backtest_sma(df, fast, slow)
    elif strat == "RSI Mean-Reversion":
        cl, ch = st.columns(2)
        low = cl.slider("Oversold", 10, 40, 30)
        high = ch.slider("Overbought", 60, 90, 70)
        bt = backtest_rsi(df, low, high)
    else:
        bt = backtest_macd(df)

    strat_ret, bh_ret, sharpe, mdd, trades, win = bt_metrics(bt)
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("Strategy Return", f"{strat_ret:+.2f}%")
    b2.metric("Buy & Hold", f"{bh_ret:+.2f}%")
    b3.metric("Sharpe", f"{sharpe:.2f}")
    b4.metric("Max DD", f"{mdd:.2f}%")
    b5.metric("Trades", f"{trades}")
    b6.metric("Win Rate", f"{win:.1f}%")

    bfig = go.Figure()
    bfig.add_trace(go.Scatter(x=bt.index, y=bt["eq"], name=f"{strat}", line=dict(color="#00ffa3", width=2)))
    bfig.add_trace(go.Scatter(x=bt.index, y=bt["bh"], name="Buy & Hold", line=dict(color="#ff6b9d", width=2, dash="dash")))
    bfig.update_layout(height=450, template="plotly_dark", title="Equity Curve (growth of $1)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(bfig, use_container_width=True)

# ==========  TAB 11: ML FORECAST  ==========
@st.cache_data(ttl=300, show_spinner=False)
def ml_forecast(sym, period):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError:
        return None
    h = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
    if len(h) < 100: return None
    h["ret"] = h["Close"].pct_change()
    h["rsi"] = rsi(h["Close"])
    m, s, hh = macd(h["Close"])
    h["macd_h"] = hh
    h["sma_r"] = h["Close"] / h["Close"].rolling(20).mean() - 1
    h["vol_r"] = h["Volume"] / h["Volume"].rolling(20).mean()
    h["mom_5"] = h["Close"].pct_change(5)
    h["mom_20"] = h["Close"].pct_change(20)
    h["vola"] = h["ret"].rolling(20).std()
    h["target"] = (h["Close"].shift(-1) > h["Close"]).astype(int)
    feats = ["ret", "rsi", "macd_h", "sma_r", "vol_r", "mom_5", "mom_20", "vola"]
    data = h[feats + ["target"]].dropna()
    if len(data) < 60: return None
    X = data[feats].iloc[:-1]
    y = data["target"].iloc[:-1]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    last_X = data[feats].iloc[[-1]]
    proba_up = float(clf.predict_proba(last_X)[0, 1])
    importances = dict(zip(feats, clf.feature_importances_))
    return {"accuracy": acc, "proba_up": proba_up, "importances": importances, "n_samples": len(data)}

with tabs[8]:
    st.subheader("🔮 ML Forecast — Next-day direction")
    st.caption("Random Forest (200 trees) trained on 8 engineered features. Hold-out accuracy shown honestly.")
    with st.spinner("Training model..."):
        result = ml_forecast(ticker, "2y" if period in ("5y", "10y", "max") else "1y")
    if result is None:
        st.warning("Not enough data or sklearn missing. Try `pip install scikit-learn`.")
    else:
        proba = result["proba_up"] * 100
        direction = "🟢 UP" if proba > 50 else "🔴 DOWN"
        m1, m2, m3 = st.columns(3)
        m1.metric("Next-day direction", direction)
        m2.metric("Probability UP", f"{proba:.1f}%")
        m3.metric("Backtest accuracy", f"{result['accuracy']*100:.1f}%")
        # Gauge
        gfig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba,
            title={"text": "Bullish probability (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#00ffa3" if proba > 50 else "#ff6b9d"},
                   "steps": [{"range": [0, 40], "color": "rgba(255,107,157,0.25)"},
                             {"range": [40, 60], "color": "rgba(255,217,61,0.25)"},
                             {"range": [60, 100], "color": "rgba(0,255,163,0.25)"}]}))
        gfig.update_layout(height=350, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
        st.plotly_chart(gfig, use_container_width=True)

        st.markdown("**Feature importance**")
        imp_df = pd.DataFrame(sorted(result["importances"].items(), key=lambda x: -x[1]),
                              columns=["Feature", "Importance"])
        ifig = go.Figure(go.Bar(x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
                                marker_color="#00ffa3"))
        ifig.update_layout(height=300, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(ifig, use_container_width=True)
        st.caption(f"Trained on {result['n_samples']} samples. Not a guarantee — markets are noisy.")

# ==========  TAB 12: FUNDAMENTALS  ==========
with tabs[10]:
    st.subheader("📅 Fundamentals, Earnings & Analyst Consensus")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("P/E (trailing)", f"{info.get('trailingPE', 0):.1f}" if info.get('trailingPE') else "—")
    f2.metric("P/E (forward)", f"{info.get('forwardPE', 0):.1f}" if info.get('forwardPE') else "—")
    f3.metric("PEG", f"{info.get('pegRatio', 0):.2f}" if info.get('pegRatio') else "—")
    f4.metric("Price / Book", f"{info.get('priceToBook', 0):.2f}" if info.get('priceToBook') else "—")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else "—")
    g2.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "—")
    g3.metric("Revenue Growth", f"{info.get('revenueGrowth', 0)*100:.2f}%" if info.get('revenueGrowth') else "—")
    g4.metric("Debt / Equity", f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else "—")

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "—")
    h2.metric("Beta", f"{info.get('beta', 0):.2f}" if info.get('beta') else "—")
    h3.metric("52W High", f"{sym}{info.get('fiftyTwoWeekHigh', 0):,.2f}" if info.get('fiftyTwoWeekHigh') else "—")
    h4.metric("52W Low", f"{sym}{info.get('fiftyTwoWeekLow', 0):,.2f}" if info.get('fiftyTwoWeekLow') else "—")

    # Analyst recommendation
    tgt_mean = info.get("targetMeanPrice")
    tgt_high = info.get("targetHighPrice")
    tgt_low = info.get("targetLowPrice")
    rec = info.get("recommendationKey", "—")
    num_an = info.get("numberOfAnalystOpinions", 0)
    if tgt_mean:
        st.subheader("🎯 Analyst Price Targets")
        upside = (tgt_mean / last["Close"] - 1) * 100
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Mean Target", f"{sym}{tgt_mean:,.2f}", f"{upside:+.1f}%")
        t2.metric("High Target", f"{sym}{tgt_high:,.2f}" if tgt_high else "—")
        t3.metric("Low Target", f"{sym}{tgt_low:,.2f}" if tgt_low else "—")
        t4.metric("Consensus", rec.upper(), f"{num_an} analysts")

    # Earnings dates
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            st.subheader("📅 Upcoming Events")
            st.dataframe(cal, use_container_width=True)
        elif isinstance(cal, dict) and cal:
            st.subheader("📅 Upcoming Events")
            st.json(cal)
    except Exception:
        pass

    # Business summary
    if info.get("longBusinessSummary"):
        with st.expander("🏢 Business Summary"):
            st.write(info["longBusinessSummary"])

# ==========  TAB 13: MACRO  ==========
SECTOR_ETFS = {
    "Tech": "XLK", "Financials": "XLF", "Energy": "XLE", "Health": "XLV",
    "Industrials": "XLI", "Discretionary": "XLY", "Staples": "XLP",
    "Utilities": "XLU", "Materials": "XLB", "Real Estate": "XLRE", "Comms": "XLC",
}
MACRO_SET = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow": "^DJI", "Nikkei": "^N225",
             "NIFTY": "^NSEI", "DAX": "^GDAXI", "FTSE": "^FTSE", "Gold": "GC=F",
             "Oil": "CL=F", "Dollar Index": "DX=F", "10Y Yield": "^TNX",
             "VIX": "^VIX", "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}

@st.cache_data(ttl=300, show_spinner=False)
def perf_grid(symbols: dict):
    rows = []
    for label, sym in symbols.items():
        try:
            h = yf.Ticker(sym).history(period="1y", interval="1d", auto_adjust=True)
            if h.empty: continue
            last_p = h["Close"].iloc[-1]
            rows.append({
                "Label": label, "Symbol": sym, "Last": last_p,
                "1D": (h["Close"].iloc[-1]/h["Close"].iloc[-2]-1)*100 if len(h) > 1 else 0,
                "1W": (h["Close"].iloc[-1]/h["Close"].iloc[-5]-1)*100 if len(h) > 5 else 0,
                "1M": (h["Close"].iloc[-1]/h["Close"].iloc[-21]-1)*100 if len(h) > 21 else 0,
                "3M": (h["Close"].iloc[-1]/h["Close"].iloc[-63]-1)*100 if len(h) > 63 else 0,
                "YTD": (h["Close"].iloc[-1]/h["Close"].iloc[0]-1)*100,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

with tabs[14]:
    st.subheader("🌍 Macro Dashboard")
    st.caption("One-glance global risk-on/risk-off regime.")
    with st.spinner("Pulling macro..."):
        macro_df = perf_grid(MACRO_SET)
    if not macro_df.empty:
        st.markdown("**🌐 Global assets**")
        styled = macro_df.style.format({
            "Last": "{:,.2f}", "1D": "{:+.2f}%", "1W": "{:+.2f}%",
            "1M": "{:+.2f}%", "3M": "{:+.2f}%", "YTD": "{:+.2f}%"
        }).background_gradient(cmap="RdYlGn", subset=["1D", "1W", "1M", "3M", "YTD"])
        st.dataframe(styled, use_container_width=True)

    with st.spinner("Pulling sectors..."):
        sec_df = perf_grid(SECTOR_ETFS)
    if not sec_df.empty:
        st.markdown("**🏭 US sector rotation (SPDR ETFs)**")
        styled2 = sec_df.style.format({
            "Last": "{:,.2f}", "1D": "{:+.2f}%", "1W": "{:+.2f}%",
            "1M": "{:+.2f}%", "3M": "{:+.2f}%", "YTD": "{:+.2f}%"
        }).background_gradient(cmap="RdYlGn", subset=["1D", "1W", "1M", "3M", "YTD"])
        st.dataframe(styled2, use_container_width=True)

        # Bar chart of sector 1M
        sfig = go.Figure(go.Bar(
            x=sec_df["Label"], y=sec_df["1M"],
            marker_color=["#00ffa3" if v > 0 else "#ff6b9d" for v in sec_df["1M"]],
            text=[f"{v:+.1f}%" for v in sec_df["1M"]], textposition="outside"))
        sfig.update_layout(title="Sector rotation — 1 month", height=400, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(sfig, use_container_width=True)

    # Risk-on / risk-off signal
    try:
        vix_row = macro_df[macro_df["Symbol"] == "^VIX"]
        if not vix_row.empty:
            vix = vix_row["Last"].iloc[0]
            regime = "🟢 Risk-ON (calm)" if vix < 16 else ("🟡 Risk-NEUTRAL" if vix < 25 else "🔴 Risk-OFF (fear)")
            st.info(f"**Volatility regime:** {regime} · VIX = {vix:.2f}")
    except Exception:
        pass

# ==========  TAB: TV TERMINAL  ==========
TV_SYMBOL_MAP = {
    "AAPL": "NASDAQ:AAPL", "MSFT": "NASDAQ:MSFT", "NVDA": "NASDAQ:NVDA",
    "TSLA": "NASDAQ:TSLA", "GOOGL": "NASDAQ:GOOGL", "AMZN": "NASDAQ:AMZN",
    "META": "NASDAQ:META", "^GSPC": "FOREXCOM:SPXUSD", "^IXIC": "NASDAQ:IXIC",
    "^NSEI": "NSE:NIFTY", "^BSESN": "BSE:SENSEX", "^N225": "TVC:NI225",
    "^GDAXI": "XETR:DAX", "^FTSE": "TVC:UKX", "^VIX": "TVC:VIX",
    "BTC-USD": "BITSTAMP:BTCUSD", "ETH-USD": "BITSTAMP:ETHUSD",
    "EURUSD=X": "FX:EURUSD", "GC=F": "TVC:GOLD", "CL=F": "TVC:USOIL",
}

def tv_sym(t):
    if t in TV_SYMBOL_MAP: return TV_SYMBOL_MAP[t]
    if t.endswith(".NS"): return f"NSE:{t[:-3]}"
    if t.endswith(".BO"): return f"BSE:{t[:-3]}"
    if t.endswith(".L"):  return f"LSE:{t[:-2]}"
    if t.endswith(".DE"): return f"XETR:{t[:-3]}"
    if t.endswith(".HK"): return f"HKEX:{t[:-3]}"
    if t.endswith(".T"):  return f"TSE:{t[:-2]}"
    if t.endswith(".AX"): return f"ASX:{t[:-3]}"
    if t.endswith(".SR"): return f"TADAWUL:{t[:-3]}"
    if t.endswith(".AE"): return f"DFM:{t[:-3]}"
    return t

def tv_widget(kind, symbol, height=500):
    tvs = tv_sym(symbol)
    if kind == "advanced_chart":
        cfg = f'{{"autosize":true,"symbol":"{tvs}","interval":"D","timezone":"Etc/UTC","theme":"dark","style":"1","locale":"en","enable_publishing":false,"allow_symbol_change":true,"studies":["RSI@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies"]}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
    elif kind == "technical_analysis":
        cfg = f'{{"interval":"1D","width":"100%","isTransparent":true,"height":"{height}","symbol":"{tvs}","showIntervalTabs":true,"locale":"en","colorTheme":"dark"}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js"
    elif kind == "symbol_info":
        cfg = f'{{"symbol":"{tvs}","width":"100%","locale":"en","colorTheme":"dark","isTransparent":true}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js"
    elif kind == "fundamental":
        cfg = f'{{"colorTheme":"dark","isTransparent":true,"displayMode":"adaptive","width":"100%","height":"{height}","symbol":"{tvs}","locale":"en"}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-financials.js"
    elif kind == "mini_chart":
        cfg = f'{{"symbol":"{tvs}","width":"100%","height":"{height}","locale":"en","dateRange":"12M","colorTheme":"dark","isTransparent":true,"autosize":true}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js"
    elif kind == "market_overview":
        cfg = f'{{"colorTheme":"dark","dateRange":"12M","showChart":true,"locale":"en","isTransparent":true,"showSymbolLogo":true,"width":"100%","height":"{height}","tabs":[{{"title":"Indices","symbols":[{{"s":"FOREXCOM:SPXUSD","d":"S&P"}},{{"s":"NASDAQ:IXIC","d":"Nasdaq"}},{{"s":"NSE:NIFTY","d":"NIFTY"}},{{"s":"TVC:NI225","d":"Nikkei"}}]}},{{"title":"Crypto","symbols":[{{"s":"BITSTAMP:BTCUSD"}},{{"s":"BITSTAMP:ETHUSD"}}]}}]}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js"
    elif kind == "stock_heatmap":
        cfg = f'{{"dataSource":"SPX500","grouping":"sector","blockSize":"market_cap_basic","blockColor":"change","locale":"en","colorTheme":"dark","hasTopBar":true,"isDataSetEnabled":true,"isZoomEnabled":true,"hasSymbolTooltip":true,"width":"100%","height":"{height}"}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js"
    elif kind == "crypto_heatmap":
        cfg = f'{{"dataSource":"Crypto","blockSize":"market_cap_calc","blockColor":"24h_close_change|5","locale":"en","colorTheme":"dark","hasTopBar":true,"isDataSetEnabled":true,"isZoomEnabled":true,"hasSymbolTooltip":true,"width":"100%","height":"{height}"}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-crypto-coins-heatmap.js"
    elif kind == "forex_cross":
        cfg = f'{{"width":"100%","height":"{height}","currencies":["EUR","USD","JPY","GBP","INR","AED","CHF","AUD","CAD","CNY"],"isTransparent":true,"colorTheme":"dark","locale":"en"}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-forex-cross-rates.js"
    elif kind == "economic_calendar":
        cfg = f'{{"colorTheme":"dark","isTransparent":true,"width":"100%","height":"{height}","locale":"en","importanceFilter":"0,1","countryFilter":"us,eu,in,ae,jp,gb,cn"}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-events.js"
    elif kind == "top_stories":
        cfg = f'{{"feedMode":"symbol","symbol":"{tvs}","isTransparent":true,"displayMode":"regular","width":"100%","height":"{height}","colorTheme":"dark","locale":"en"}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-timeline.js"
    elif kind == "screener":
        cfg = f'{{"width":"100%","height":"{height}","defaultColumn":"overview","defaultScreen":"most_capitalized","market":"america","showToolbar":true,"colorTheme":"dark","locale":"en","isTransparent":true}}'
        url = "https://s3.tradingview.com/external-embedding/embed-widget-screener.js"
    else: return
    html = f'<div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div><script type="text/javascript" src="{url}" async>{cfg}</script></div>'
    components.html(html, height=height + 30)

with tabs[1]:
    st.subheader("🎯 TradingView Terminal")
    st.caption(f"Pro widgets — live on **{ticker}** ({tv_sym(ticker)})")
    tv_tabs = st.tabs(["Advanced Chart", "Technical Analysis", "Symbol Info",
                       "Financials", "Top Stories", "Market Overview",
                       "Stock Heatmap", "Crypto Heatmap", "FX Rates",
                       "Economic Calendar", "Screener"])
    with tv_tabs[0]: tv_widget("advanced_chart", ticker, 620)
    with tv_tabs[1]:
        c1, c2 = st.columns(2)
        with c1: tv_widget("technical_analysis", ticker, 450)
        with c2: tv_widget("mini_chart", ticker, 450)
    with tv_tabs[2]: tv_widget("symbol_info", ticker, 200)
    with tv_tabs[3]: tv_widget("fundamental", ticker, 500)
    with tv_tabs[4]: tv_widget("top_stories", ticker, 500)
    with tv_tabs[5]: tv_widget("market_overview", ticker, 500)
    with tv_tabs[6]: tv_widget("stock_heatmap", ticker, 600)
    with tv_tabs[7]: tv_widget("crypto_heatmap", ticker, 600)
    with tv_tabs[8]: tv_widget("forex_cross", ticker, 500)
    with tv_tabs[9]: tv_widget("economic_calendar", ticker, 500)
    with tv_tabs[10]: tv_widget("screener", ticker, 550)

# ==========  TAB: TOOL-CALLING AGENT  ==========
def tool_get_price(symbol):
    h = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=True)
    if h.empty: return {"error": f"No data for {symbol}"}
    last = h.iloc[-1]; prev = h.iloc[-2] if len(h) > 1 else last
    return {"symbol": symbol, "price": float(last["Close"]),
            "change_pct": float((last["Close"]/prev["Close"]-1)*100) if prev["Close"] else 0}

def tool_get_snapshot(symbol): return _build_snapshot(symbol) or {"error": "no data"}
def tool_compare(symbols): return {s: tool_get_price(s) for s in symbols[:5]}
def tool_news(symbol):
    try:
        n = (yf.Ticker(symbol).news or [])[:5]
        return [{"title": x.get("title"), "publisher": x.get("publisher")} for x in n if x.get("title")]
    except Exception as e: return {"error": str(e)}
def tool_backtest(symbol, strategy="sma"):
    h = yf.Ticker(symbol).history(period="2y", interval="1d", auto_adjust=True)
    if h.empty: return {"error": "no data"}
    h["RSI"] = rsi(h["Close"])
    bt = backtest_sma(h, 20, 50) if strategy == "sma" else (
         backtest_rsi(h, 30, 70) if strategy == "rsi" else backtest_macd(h))
    s, bh, sh, mdd, tr, win = bt_metrics(bt)
    return {"strategy": strategy, "return_pct": round(s, 2), "bh_pct": round(bh, 2),
            "sharpe": round(sh, 2), "max_dd_pct": round(mdd, 2), "trades": tr, "win_rate_pct": round(win, 1)}

TOOL_SPECS = [
    {"type": "function", "function": {"name": "get_price", "description": "Live price/change for a ticker.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {"name": "get_snapshot", "description": "Full technical snapshot.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {"name": "compare", "description": "Compare multiple tickers.",
        "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"}}}, "required": ["symbols"]}}},
    {"type": "function", "function": {"name": "news", "description": "Latest headlines.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {"name": "backtest", "description": "Backtest SMA/RSI/MACD.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "strategy": {"type": "string", "enum": ["sma","rsi","macd"]}}, "required": ["symbol", "strategy"]}}},
]
TOOL_FNS = {"get_price": tool_get_price, "get_snapshot": tool_get_snapshot,
            "compare": lambda symbols: tool_compare(symbols), "news": tool_news,
            "backtest": lambda symbol, strategy="sma": tool_backtest(symbol, strategy)}

with tabs[5]:
    st.subheader("🛠️ Tool-Calling Agent")
    st.caption("Multi-step reasoning. It calls tools, gets live data, and synthesizes.")
    if "agent_log" not in st.session_state: st.session_state.agent_log = []
    user_task = st.text_input("Task",
        placeholder="e.g. Compare AAPL, NVDA, RELIANCE.NS — backtest RSI on the best — recommend buy", key="agent_input")
    if st.button("🚀 Run agent", type="primary", key="run_agent_btn") and user_task:
        client = get_groq_client()
        if not client: st.error("Need GROQ_API_KEY")
        else:
            import json as _json
            msgs = [
                {"role": "system", "content": "You are StockPulse Agent. Use tools to fetch LIVE data before answering. Plan multiple tool calls. End with a trader-style verdict including Entry/SL/TP. No disclaimers."},
                {"role": "user", "content": user_task}]
            trace = []; final = "(no response)"
            for _ in range(6):
                try:
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile", messages=msgs,
                        tools=TOOL_SPECS, tool_choice="auto", temperature=0.3, max_tokens=800)
                    m = resp.choices[0].message
                    if m.tool_calls:
                        msgs.append({"role": "assistant", "content": m.content or "",
                            "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                for tc in m.tool_calls]})
                        for tc in m.tool_calls:
                            fn = tc.function.name
                            try: args = _json.loads(tc.function.arguments or "{}")
                            except: args = {}
                            try: result = TOOL_FNS[fn](**args)
                            except Exception as e: result = {"error": str(e)}
                            trace.append({"tool": fn, "args": args, "result": result})
                            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": _json.dumps(result)[:2500]})
                    else:
                        final = m.content or "(empty)"; break
                except Exception as e:
                    final = f"Error: {e}"; break
            st.session_state.agent_log.append({"task": user_task, "trace": trace, "answer": final})
    for entry in reversed(st.session_state.agent_log[-3:]):
        with st.expander(f"📝 {entry['task'][:80]}", expanded=True):
            st.markdown(f"<div class='verdict-box'>{entry['answer']}</div>", unsafe_allow_html=True)
            if entry["trace"]:
                with st.expander("🔍 Tool trace"):
                    for t in entry["trace"]:
                        st.code(f"{t['tool']}({t['args']}) → {str(t['result'])[:400]}")

# ==========  TAB: DALIO AI + INVESTING.COM  ==========
with tabs[6]:
    st.subheader("🧙 Dalio AI + Investing Intelligence")
    st.caption("Ray Dalio's Digital Ray + Investing.com widgets + SeekingAlpha + Benzinga")
    dtabs = st.tabs(["🧙 Digital Ray", "📊 Investing.com", "📰 Seeking Alpha", "🔔 Benzinga"])
    with dtabs[0]:
        st.markdown("**Ask Ray Dalio's public AI about macro cycles, debt, geopolitics, principles.**")
        components.iframe("https://www.digitalray.ai/", height=720, scrolling=True)
    with dtabs[1]:
        st.markdown(f"**Investing.com technicals for {ticker}**")
        components.iframe(f"https://sslwidget.investing.com/widget/technicalStudies?width=100%25&theme=darkTheme&pair={ticker}", height=500, scrolling=True)
        st.markdown("**Economic Calendar**")
        components.iframe("https://sslecal2.investing.com/?columns=exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous&importance=2,3&features=datepicker,timezone&countries=5,32,37,72&calType=week&timeZone=8&lang=1", height=500, scrolling=True)
    with dtabs[2]:
        components.iframe(f"https://seekingalpha.com/symbol/{ticker.split('.')[0]}", height=720, scrolling=True)
    with dtabs[3]:
        components.iframe(f"https://www.benzinga.com/quote/{ticker.split('.')[0]}", height=720, scrolling=True)

# ==========  TAB: SEC FILINGS  ==========
@st.cache_data(ttl=3600, show_spinner=False)
def sec_filings(symbol):
    try:
        hdr = {"User-Agent": "StockPulseAI research@stockpulse.ai"}
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=hdr, timeout=8)
        tks = r.json(); cik = None
        for _, row in tks.items():
            if row["ticker"].upper() == symbol.split(".")[0].upper():
                cik = str(row["cik_str"]).zfill(10); break
        if not cik: return None
        r2 = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=hdr, timeout=8)
        data = r2.json()
        recent = data.get("filings", {}).get("recent", {})
        out = []
        for i in range(min(30, len(recent.get("form", [])))):
            out.append({"form": recent["form"][i], "date": recent["filingDate"][i],
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{recent['accessionNumber'][i].replace('-','')}/{recent['primaryDocument'][i]}"})
        return {"name": data.get("name"), "cik": cik, "filings": out}
    except Exception as e: return {"error": str(e)}

with tabs[11]:
    st.subheader("🏛️ SEC EDGAR Filings")
    st.caption("Official filings from SEC. US tickers only.")
    data = sec_filings(ticker)
    if not data or (isinstance(data, dict) and "error" in data):
        st.info("No SEC data (non-US or lookup failed).")
    else:
        st.markdown(f"**{data['name']}** · CIK {data['cik']}")
        df_fil = pd.DataFrame(data["filings"])
        key = df_fil[df_fil["form"].isin(["10-K", "10-Q", "8-K", "DEF 14A", "S-1", "13F-HR"])].head(20)
        st.markdown("**📌 Key filings**")
        for _, row in key.iterrows():
            st.markdown(f"<div class='news-card'><a href='{row['url']}' target='_blank' style='color:#00ffa3; text-decoration:none'><b>{row['form']}</b> · {row['date']}</a></div>", unsafe_allow_html=True)
        with st.expander("All filings"):
            st.dataframe(df_fil, use_container_width=True)
        latest_10k = df_fil[df_fil["form"] == "10-K"].head(1)
        if not latest_10k.empty and st.button("🧠 AI-summarize latest 10-K"):
            client = get_groq_client()
            if client:
                try:
                    url = latest_10k.iloc[0]["url"]
                    r = requests.get(url, headers={"User-Agent": "StockPulseAI"}, timeout=15)
                    text = _re.sub(r"<[^>]+>", " ", r.text)[:12000]
                    with st.spinner("Summarizing..."):
                        s = client.chat.completions.create(model="llama-3.3-70b-versatile",
                            messages=[
                                {"role":"system","content":"Summarize this 10-K: (1) Business, (2) 3 biggest risks, (3) financial highlights, (4) outlook."},
                                {"role":"user","content":text}],
                            temperature=0.2, max_tokens=700)
                        st.markdown(f"<div class='verdict-box'>{s.choices[0].message.content}</div>", unsafe_allow_html=True)
                except Exception as e: st.error(f"Failed: {e}")

# ==========  TAB: INSIDERS  ==========
with tabs[12]:
    st.subheader("👥 Insider Transactions & Institutional Holders")
    try:
        t = yf.Ticker(ticker)
        mh = getattr(t, "major_holders", None)
        ih = getattr(t, "institutional_holders", None)
        ip = getattr(t, "insider_purchases", None)
        it = getattr(t, "insider_transactions", None)
        if isinstance(mh, pd.DataFrame) and not mh.empty:
            st.markdown("**🏦 Ownership breakdown**"); st.dataframe(mh, use_container_width=True)
        if isinstance(ih, pd.DataFrame) and not ih.empty:
            st.markdown("**🏛️ Top institutional holders**"); st.dataframe(ih.head(15), use_container_width=True)
        if isinstance(ip, pd.DataFrame) and not ip.empty:
            st.markdown("**💰 Insider purchases (net)**"); st.dataframe(ip, use_container_width=True)
        if isinstance(it, pd.DataFrame) and not it.empty:
            st.markdown("**📋 Recent insider transactions**"); st.dataframe(it.head(25), use_container_width=True)
        if not any([isinstance(x, pd.DataFrame) and not x.empty for x in [mh, ih, ip, it]]):
            st.info("No insider data available for this ticker.")
    except Exception as e: st.info(f"Unavailable: {e}")

# ==========  TAB: TRANSCRIPTS / EARNINGS  ==========
with tabs[13]:
    st.subheader("📜 Earnings & Analyst Brief")
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            st.markdown("**📅 Earnings history & next date**")
            st.dataframe(ed.head(12), use_container_width=True)
    except Exception: pass
    if st.button("🎤 AI-generate earnings Q&A brief"):
        client = get_groq_client()
        if client:
            snap = _build_snapshot(ticker)
            with st.spinner("Preparing brief..."):
                rr = client.chat.completions.create(model="llama-3.3-70b-versatile",
                    messages=[
                        {"role":"system","content":"Sellside analyst prepping for earnings. Output: (1) Top 5 mgmt questions, (2) KPIs to watch, (3) Base/bull/bear with price targets, (4) Post-earnings trade setup."},
                        {"role":"user","content":f"Ticker: {ticker}. Snapshot: {snap}"}],
                    temperature=0.3, max_tokens=900)
                st.markdown(f"<div class='verdict-box'>{rr.choices[0].message.content}</div>", unsafe_allow_html=True)

# ==========  TAB: WATCHLIST  ==========
import sqlite3 as _sq
WATCHLIST_DB = os.path.join(os.path.dirname(__file__), "watchlist.db")
def wl_conn():
    c = _sq.connect(WATCHLIST_DB)
    c.execute("CREATE TABLE IF NOT EXISTS watch (symbol TEXT PRIMARY KEY, note TEXT, rsi_below REAL, rsi_above REAL, price_above REAL, price_below REAL, added_at TEXT)")
    return c
def wl_list():
    with wl_conn() as c: return pd.read_sql_query("SELECT * FROM watch ORDER BY added_at DESC", c)
def wl_add(sym, note="", rb=None, ra=None, pa=None, pb=None):
    with wl_conn() as c:
        c.execute("INSERT OR REPLACE INTO watch VALUES (?,?,?,?,?,?,?)",
            (sym, note, rb, ra, pa, pb, datetime.now().isoformat()))
def wl_remove(sym):
    with wl_conn() as c: c.execute("DELETE FROM watch WHERE symbol=?", (sym,))

with tabs[18]:
    st.subheader("⭐ Watchlist & Alerts")
    st.caption("Persistent — stored in SQLite. Triggers evaluated on every refresh.")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        wl_sym = st.text_input("Symbol", value=ticker, key="wl_sym_in").upper().strip()
        wl_note = st.text_input("Note", key="wl_note_in")
    with c2:
        wl_rb = st.number_input("Alert if RSI <", value=0.0, step=1.0, key="wl_rb_in")
        wl_ra = st.number_input("Alert if RSI >", value=0.0, step=1.0, key="wl_ra_in")
    with c3:
        wl_pa = st.number_input("Price >", value=0.0, step=1.0, key="wl_pa_in")
        wl_pb = st.number_input("Price <", value=0.0, step=1.0, key="wl_pb_in")
    ab1, ab2 = st.columns(2)
    with ab1:
        if st.button("➕ Add/update", type="primary", key="wl_add_btn"):
            wl_add(wl_sym, wl_note, wl_rb or None, wl_ra or None, wl_pa or None, wl_pb or None)
            st.success(f"Added {wl_sym}")
    with ab2:
        if st.button("🗑️ Remove", key="wl_rem_btn"):
            wl_remove(wl_sym); st.warning(f"Removed {wl_sym}")
    wl_df = wl_list()
    if wl_df.empty:
        st.info("Empty watchlist.")
    else:
        rows = []
        for _, r in wl_df.iterrows():
            snap = _build_snapshot(r["symbol"])
            if not snap: continue
            alerts = []
            if r["rsi_below"] and snap["rsi"] < r["rsi_below"]: alerts.append(f"RSI {snap['rsi']:.1f} < {r['rsi_below']}")
            if r["rsi_above"] and snap["rsi"] > r["rsi_above"]: alerts.append(f"RSI {snap['rsi']:.1f} > {r['rsi_above']}")
            if r["price_above"] and snap["price"] > r["price_above"]: alerts.append(f"Price > {r['price_above']}")
            if r["price_below"] and snap["price"] < r["price_below"]: alerts.append(f"Price < {r['price_below']}")
            rows.append({"Symbol": r["symbol"], "Price": round(snap["price"], 2),
                "Change 1D %": round(snap["change_1d_pct"], 2), "RSI": round(snap["rsi"], 1),
                "🔔 Alerts": " · ".join(alerts) if alerts else "—", "Note": r["note"] or ""})
        if rows:
            dfwl = pd.DataFrame(rows)
            st.dataframe(dfwl, use_container_width=True)
            trig = [r for r in rows if r["🔔 Alerts"] != "—"]
            if trig: st.warning(f"🔔 {len(trig)} alert(s) firing!")

# ==========  TAB: SETTINGS  ==========
with tabs[20]:
    st.subheader("⚙️ Settings")
    st.caption("Configure features, theme, AI parameters, and integrations.")
    if "settings" not in st.session_state: st.session_state.settings = {}
    s = st.session_state.settings
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**🎨 Appearance**")
        s["theme"] = st.radio("Theme", ["dark", "midnight", "terminal"], horizontal=True, index=0)
        s["compact"] = st.toggle("Compact mode", value=False)
    with s2:
        st.markdown("**🤖 AI**")
        s["ai_temp"] = st.slider("LLM creativity", 0.0, 1.0, 0.3, 0.05)
        s["ai_model"] = st.selectbox("Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    st.markdown("**🧩 Feature toggles**")
    features = ["Chart","TV Terminal","Verdict","HF Council","Chat","Tool Agent",
                "Dalio AI","Strategy Lab","ML Forecast","Risk","Fundamentals",
                "SEC","Insiders","Transcripts","Macro","Portfolio","Options","News","Watchlist","Sharia"]
    cols = st.columns(4)
    for i, f in enumerate(features):
        with cols[i % 4]:
            s[f"show_{f}"] = st.toggle(f, value=True, key=f"tog_{f}")
    st.markdown("**🔗 Integrations**")
    i1, i2 = st.columns(2)
    with i1:
        s["discord"] = st.text_input("Discord webhook (alerts)", value=s.get("discord",""), type="password")
    with i2:
        s["telegram"] = st.text_input("Telegram bot token", value=s.get("telegram",""), type="password")
    st.success("Saved in session.")

st.divider()
st.caption("Built with Streamlit · yfinance · Plotly · Groq Llama 3.3 · TradingView · Digital Ray · SEC EDGAR · scikit-learn · "
           f"[@hustlerkrishna1](https://github.com/hustlerkrishna1) · "
           f"{datetime.now().strftime('%H:%M:%S')} · Not financial advice.")
