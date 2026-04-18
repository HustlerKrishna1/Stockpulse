<div align="center">

# 📈 StockPulse AI

### Institutional-grade agentic equity research terminal — built in a single file.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036)](https://groq.com)
[![TradingView](https://img.shields.io/badge/TradingView-Widgets-131722?logo=tradingview&logoColor=white)](https://tradingview.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Live](https://img.shields.io/badge/demo-streamlit_cloud-brightgreen)]()

**21 synchronized research tabs · 6 live data sources · 5-tool LLM agent · multi-agent hedge-fund council · ML + Monte Carlo · global coverage**

</div>

---

## 🎯 What it is

**StockPulse AI** is a single-page agentic research terminal that ingests live market data, SEC filings, insider flows, news and transcripts, passes them through a deterministic signal engine **and** a Groq-hosted Llama 3.3 70B agent with **real function-calling**, and returns a BingX/Bybit-style decisive verdict — Entry / Stop-Loss / Targets included.

It works for **any ticker on any exchange** — NYSE, NASDAQ, NSE, BSE, LSE, TSE, HKEX, Tadawul, DFM, ADX, XETRA, Euronext, ASX — resolved from natural language (`"aramco"` → `2222.SR`, `"nifty"` → `^NSEI`).

> **One `streamlit run app.py` and you get a Bloomberg-lite that costs $0/month.**

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                            USER  (Browser @ 8501)                              │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                    │  natural-language query / ticker click
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                       STREAMLIT UI  (21 tabs · Plotly · TradingView)           │
│   Chart │ TV Terminal │ Verdict │ HF Council │ Chat │ Tool Agent │ Dalio AI   │
│   Strategy Lab │ ML Forecast │ Risk │ Fundamentals │ SEC │ Insiders │ …       │
└──────┬─────────────────────────────┬──────────────────────────┬────────────────┘
       │                             │                          │
       ▼                             ▼                          ▼
┌──────────────┐       ┌──────────────────────────┐   ┌──────────────────────┐
│  TICKER      │       │   DETERMINISTIC SIGNAL    │   │   AGENTIC LAYER      │
│  RESOLVER    │       │         ENGINE            │   │   (Groq Llama 3.3)   │
│              │       │                           │   │                      │
│ · regex       │       │ · RSI(14) · MACD(12,26,9)│   │ ┌──────────────────┐ │
│ · Yahoo srch  │──┐    │ · SMA20/50/200            │   │ │ Tool-calling     │ │
│ · LLM fallback│  │    │ · Weighted score          │   │ │   get_price      │ │
└──────────────┘  │    │ · BUY / HOLD / SELL       │   │ │   get_snapshot   │ │
                  │    └──────────────┬───────────┘   │ │   compare        │ │
                  │                   │               │ │   news           │ │
                  │                   ▼               │ │   backtest       │ │
                  │    ┌──────────────────────────┐  │ └──────────────────┘ │
                  └───▶│      DATA FABRIC          │◀─│                      │
                       │                           │  │ ┌──────────────────┐ │
                       │ · yfinance  (OHLCV)       │  │ │ HF Council       │ │
                       │ · Yahoo Search API        │  │ │   Technical      │ │
                       │ · SEC EDGAR (CIK+filings) │  │ │   Fundamental    │ │
                       │ · Insider transactions    │  │ │   Risk Officer   │ │
                       │ · Options chain           │  │ │   Portfolio PM   │ │
                       │ · News RSS                │  │ └──────────────────┘ │
                       └──────────────┬───────────┘  └──────────┬───────────┘
                                      │                          │
                                      ▼                          ▼
                       ┌──────────────────────────────────────────────────────┐
                       │          ANALYTICS & PERSISTENCE LAYER               │
                       │                                                      │
                       │  scikit-learn RF   │   Monte Carlo   │   Backtester │
                       │  (8-feature ML)    │   (10k paths)   │   (SMA/RSI/  │
                       │                    │                 │    MACD)     │
                       │                                                      │
                       │  SQLite watchlist  │  5-min TTL cache │  Sharia filt│
                       └──────────────────────────────────────────────────────┘
```

### Why this shape?

| Layer | Principle |
|---|---|
| **UI** | Single-file Streamlit, zero build step, deployable in one click |
| **Signal engine** | 100% deterministic — works without any LLM key |
| **Agentic layer** | LLM *decides which tool to call*; tools return grounded numbers — no hallucinated prices |
| **Data fabric** | Only free, public APIs — no vendor lock-in, no credit card |
| **Persistence** | SQLite for watchlist; 5-minute TTL cache for market data |

---

## 🔄 Data Flow

### 1 · Natural-language query → structured action

```
 "should i buy aramco before earnings?"
            │
            ▼
 ┌───────────────────────┐
 │  resolve_tickers()    │   ① regex hits → none
 │                       │   ② Yahoo search → 2222.SR
 │                       │   ③ LLM fallback if needed
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │  _build_snapshot()    │   OHLCV · RSI · MACD · SMAs · vol
 │   (cached 5 min)      │   earnings date · P/E · market cap
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │  Groq tool-calling    │   LLM sees snapshot, may call:
 │  loop (max 4 hops)    │     news(2222.SR)
 │                       │     backtest(2222.SR, "sma")
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │  Decisive verdict     │   Entry: 28.40
 │  (BingX/Bybit style)  │   SL:    27.10   (-4.6%)
 │                       │   TP1:   30.20  TP2: 31.80
 └───────────────────────┘
```

### 2 · Chart/tab render path

```
 user picks ticker
        │
        ▼
 yfinance.download(period, interval)   ←── @st.cache_data(ttl=300)
        │
        ▼
 pandas → indicators (RSI, MACD, SMAs)
        │
        ├─────▶ Plotly candlestick + overlays
        ├─────▶ TradingView widget (symbol-mapped)
        ├─────▶ Signal engine → Verdict tab
        ├─────▶ sklearn RF → ML Forecast tab
        ├─────▶ Monte Carlo → Risk tab
        └─────▶ Backtester → Strategy Lab tab
```

### 3 · SEC / Insider / Fundamentals pipeline

```
 symbol ──▶ sec.gov/files/company_tickers.json  ──▶ CIK
                                                    │
                                                    ▼
                          data.sec.gov/submissions/CIK{cik}.json
                                                    │
                                ┌───────────────────┼───────────────────┐
                                ▼                   ▼                   ▼
                           Filings (10-K)      Insider txns        Earnings cal
                                │                   │                   │
                                └───────────────────┴───────────────────┘
                                                    │
                                                    ▼
                                          Fundamentals tab
```

---

## ✨ Feature Matrix

<table>
<tr><th>Module</th><th>Tabs</th><th>Powered by</th></tr>
<tr><td><b>Market intelligence</b></td><td>Chart · TV Terminal · News · Watchlist</td><td>yfinance · TradingView · Yahoo RSS · SQLite</td></tr>
<tr><td><b>Decisive AI</b></td><td>Verdict · Chat · Tool Agent · Dalio AI</td><td>Groq Llama 3.3 70B · function-calling</td></tr>
<tr><td><b>Multi-agent research</b></td><td>HF Council</td><td>4-persona debate (Technical / Fundamental / Risk / PM)</td></tr>
<tr><td><b>Quant</b></td><td>Strategy Lab · ML Forecast · Risk</td><td>scikit-learn RF · Monte Carlo · SMA/RSI/MACD backtests</td></tr>
<tr><td><b>Corporate intel</b></td><td>Fundamentals · SEC Filings · Insiders · Transcripts</td><td>SEC EDGAR · yfinance actions</td></tr>
<tr><td><b>Macro</b></td><td>Macro · Options · Portfolio</td><td>TradingView econ calendar · options chain · mini-optimizer</td></tr>
<tr><td><b>Compliance</b></td><td>Sharia</td><td>AAOIFI debt-ratio + keyword filter</td></tr>
</table>

---

## 📋 Requirements

### Runtime

| What | Version | Why |
|---|---|---|
| Python | **3.10+** | f-strings, walrus, type hints |
| OS | Windows / macOS / Linux | Tested on Win11 & Ubuntu 22 |
| RAM | 1 GB min · 2 GB recommended | Plotly + sklearn |
| Network | Outbound HTTPS | yfinance, SEC, Groq, Yahoo |

### Python packages (see [`requirements.txt`](requirements.txt))

```txt
streamlit>=1.30          # UI runtime
yfinance>=0.2.40         # global market data
plotly>=5.20             # charts
pandas>=2.0              # data wrangling
numpy>=1.24              # math
groq>=0.11               # LLM + tool calling
streamlit-autorefresh>=1.0
scikit-learn>=1.3        # ML forecast
```

### Secrets

Create `.streamlit/secrets.toml` (gitignored):

```toml
GROQ_API_KEY = "gsk_your_key_here"   # free at console.groq.com
```

> The deterministic signal engine, charts, SEC data, backtester and ML forecast **all work without a key**. Only the AI chat/verdict layers need Groq.

---

## 🚀 Quick start

```bash
git clone https://github.com/HustlerKrishna1/Stockpulse.git
cd Stockpulse

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# optional — for AI features
mkdir -p .streamlit && echo 'GROQ_API_KEY = "your_key"' > .streamlit/secrets.toml

streamlit run app.py
```

Open **http://localhost:8501** and start typing a company name.

---

## 🧭 What each tab does (and how)

| # | Tab | Inputs | Processing | Output |
|---|---|---|---|---|
| 1 | **Chart** | symbol, period, interval | yfinance → pandas → Plotly candlestick + SMA20/50 | Interactive OHLC chart |
| 2 | **TV Terminal** | symbol | Yahoo→TradingView symbol map | Advanced chart + technicals + financials widgets |
| 3 | **Verdict** | snapshot | Weighted RSI/MACD/SMA score + LLM narrative | BUY/HOLD/SELL + Entry/SL/TP |
| 4 | **HF Council** | snapshot, question | 4 personas debate in sequence | Consensus memo |
| 5 | **Chat** | free text | Ticker resolver → snapshot → LLM | Decisive answer, no disclaimers |
| 6 | **Tool Agent** | free text | Groq function-calling loop (4 hops) | Answer with tool call trace |
| 7 | **Dalio AI** | — | Iframe embed | Ray Dalio chatbot |
| 8 | **Strategy Lab** | symbol, strategy, params | Vectorized backtest | Equity curve + Sharpe/DD/winrate |
| 9 | **ML Forecast** | symbol | 8-feature RF on 2y history | Next-day up/down probability + feature importance |
| 10 | **Risk** | symbol, days, sims | Monte Carlo GBM | Fan chart + percentile table |
| 11 | **Fundamentals** | symbol | yfinance.info | P/E, margins, growth, ratios |
| 12 | **SEC Filings** | symbol | EDGAR CIK lookup | Latest 10-K/10-Q/8-K links |
| 13 | **Insiders** | symbol | yfinance insider_transactions | Last 70 txns, net flow |
| 14 | **Transcripts** | symbol | yfinance earnings_dates | Earnings calendar |
| 15 | **Macro** | — | TradingView widgets | Heatmaps, econ calendar |
| 16 | **Portfolio** | tickers, weights | Equal-risk simple optimizer | Allocation chart |
| 17 | **Options** | symbol, expiry | yfinance.option_chain | Calls/puts with IV |
| 18 | **News** | symbol | Yahoo Finance RSS | Headlines with sentiment tag |
| 19 | **Watchlist** | symbol, alerts | SQLite CRUD | Persistent list + triggers |
| 20 | **Sharia** | symbol | Debt ratio + keyword filter | Compliant / Non-compliant verdict |
| 21 | **Settings** | — | st.session_state | Theme, temp, feature toggles |

---

## 🧠 How the agentic verdict really works

The LLM **never hallucinates a price**. It sees a JSON snapshot and must either:

1. Respond with analysis, **or**
2. Call one of these tools (OpenAI function-calling spec):

```python
TOOL_SPECS = [
    get_price(symbol),                 # live last-trade
    get_snapshot(symbol, period),      # OHLCV + indicators
    compare(symbols: List[str]),       # multi-asset ranking
    news(symbol, limit),               # RSS headlines
    backtest(symbol, strategy, params) # run a vectorized backtest
]
```

Loop runs up to 4 hops, then the final message must contain an **Entry / SL / TP** block.

```
┌─ turn 1: user asks "compare aramco vs adnoc"
├─ turn 2: LLM calls compare(["2222.SR","ADNOCDIST.AE"])
├─ turn 3: tool returns ranked snapshot
└─ turn 4: LLM writes verdict with numbers
```

---

## 🧪 Live smoke test

Ships with verified paths:

- ✅ Yahoo search: `"apple"` → 5 results
- ✅ yfinance AAPL: **$270.23** (cached 300 s)
- ✅ SEC CIK lookup: AAPL → **0000320193**
- ✅ Groq tool-calling: invokes `get_price(AAPL)` correctly
- ✅ Insider txns pulled: **70 rows**
- ✅ Earnings dates: **25** · Options expiries: **22** · News items: **10**

---

## 🌐 Deploy in 60 seconds

**Streamlit Community Cloud**:

1. Push to this repo
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app** → pick this repo
3. In **Secrets**, paste `GROQ_API_KEY = "..."`
4. Deploy

Works on free tier. No Dockerfile needed.

---

## 🛣️ Roadmap

- [ ] Walk-forward backtester with purged k-fold
- [ ] Vector memory (Chroma) for persistent user context
- [ ] Command palette (⌘K) for global navigation
- [ ] PDF/HTML one-click report export
- [ ] Discord / Telegram webhook alerts
- [ ] Real-time tape via websocket (when a free provider shows up)

---

## 📁 Project structure

```
Stockpulse/
├── app.py                       # everything — 1600 LoC, 21 tabs
├── requirements.txt             # 8 pinned deps
├── .streamlit/
│   └── secrets.toml             # GROQ_API_KEY (gitignored)
├── .gitignore
└── README.md                    # you are here
```

Single-file is a feature, not a bug — easier to audit, easier to deploy, easier to fork.

---

## ⚠️ Disclaimer

Educational project. **Not financial advice.** The "decisive trader" voice is a UX choice — always do your own research, size positions responsibly, and never risk capital you can't lose.

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Built with ❤️ by [@HustlerKrishna1](https://github.com/HustlerKrishna1)**

*If this project impressed you, a ⭐ goes a long way.*

</div>
