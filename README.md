# QuantEdge — Systematic Investing Platform

A production-grade Dash/Plotly dashboard backed by PostgreSQL.  
All market data comes from FMP Premium. No CSV files. No 20-minute first loads.

---

## Architecture

```
quantedge/
├── config.py                   ← All settings (reads from env vars)
│
├── db/
│   ├── connection.py           ← SQLAlchemy engine + Session
│   └── models.py               ← All ORM table definitions
│
├── ingestion/
│   ├── fmp_client.py           ← ALL FMP API calls (single source of truth)
│   ├── prices.py               ← Nightly OHLCV + FX ingestion → DB
│   ├── universe.py             ← Stock universe sync (replaces stocks_all_pages.csv)
│   └── scheduler.py            ← APScheduler: nightly jobs + 15-min alert checker
│
├── analytics/
│   ├── technicals.py           ← Reads prices from DB; same signals as before
│   ├── fundamental.py          ← DB-cached FMP fundamentals (24h staleness)
│   └── portfolio.py            ← FX + prices from DB; same P&L logic
│
├── modules/
│   ├── stop_loss.py            ← (copy from original) ATR/Trailing/Fixed backtester
│   └── backtester.py           ← (copy from original) DCA strategy backtester
│
├── app.py                      ← Main Dash application
├── setup.py                    ← ONE-TIME bootstrap script
├── gunicorn_config.py          ← Production server config
├── Procfile                    ← Heroku / Railway / Render deployment
├── requirements.txt
├── .env.example                ← Copy → .env and fill in your values
└── transactions.csv            ← Your trade history (unchanged)
```

---

## Quick Start

### 1. Prerequisites

```bash
# PostgreSQL running locally
createdb quantedge
createuser quantedge
psql -c "ALTER USER quantedge WITH PASSWORD 'quantedge';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE quantedge TO quantedge;"

# Python dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set FMP_API_KEY and DATABASE_URL at minimum
export $(cat .env | xargs)
```

### 3. Bootstrap the database (one-time, ~16 minutes)

```bash
python setup.py
```

This will:
- Create all DB tables
- Sync the FMP stock universe (~7,000+ tickers)
- Ingest 10 years of price history for your portfolio + screener universe
- Cache fundamentals for all screener tickers
- Sync the next 90 days of earnings events

### 4. Run

```bash
# Development
python app.py

# Production
gunicorn --config gunicorn_config.py app:server
```

Open **http://127.0.0.1:8050**

---

## Tabs

### 📊 Portfolio Overview
- 7 KPI cards: Total P/L, Day P/L, Wallet Value, Total Return, CAGR, Sharpe, Sortino
- Cumulative P/L chart + sector allocation donut
- Open positions table with unrealised P/L, weights, scores
- Stop-loss recommendations (ATR / Trailing / Fixed %)

### 🔍 Screener
- Full factor table: Technical Score, Fundamental Score, YTD, Momentum, RSI, P/E, DCF Upside
- Click any row → stock detail view (chart + metrics)
- Filter by sector

### 🧪 Strategy Lab
- Signal comparison (Technical Score, Momentum, RSI) per ticker
- Visual backtest output

### 🔍 Watchlist
- Add tickers by symbol
- Table of all watchlist items with live scores
- Click any row → full stock detail view

### 🔔 Alerts
- Create: price above / price below / stop-loss alerts
- Optional email notification (Gmail App Password)
- Active alerts table + triggered alerts (last 30 days)
- Auto-checks every 15 minutes in background

### 📅 Earnings Calendar
- Next 90 days of earnings events
- Portfolio holdings highlighted
- EPS estimate vs actual, revenue estimate vs actual
- One-click refresh

---

## Nightly Data Refresh

The scheduler runs automatically inside `app.py` (no cron setup needed):

| Job | Schedule | What it does |
|-----|----------|-------------|
| Price refresh | 22:00 UTC daily | Incremental OHLCV update for all tickers |
| Earnings sync | 22:30 UTC daily | Next 90 days of earnings events |
| Universe sync | Sunday 23:00 UTC | Re-sync FMP stock list |
| Alert checker | Every 15 minutes | Check thresholds, fire emails |

Configure timing in `.env`:
```
NIGHTLY_CRON_HOUR=22
NIGHTLY_CRON_MINUTE=0
```

---

## Stock Universe

`stocks_all_pages.csv` is gone. The `company_info` table is the source of truth.

Control screener size in `.env`:
```bash
SCREENER_UNIVERSE=500          # top 500 by market cap (recommended to start)
SCREENER_UNIVERSE=all          # all ~7,000 FMP tickers (setup takes ~40 min)
SCREENER_UNIVERSE=AAPL,MSFT,NVDA,TSLA   # exact list
```

Your portfolio tickers are always included regardless of this setting.

---

## Manual Data Operations

```bash
# Force-refresh prices for specific tickers
python -m ingestion.prices AAPL MSFT NVDA

# Re-sync full universe
python -m ingestion.universe

# Run the scheduler standalone (useful for debugging)
python -m ingestion.scheduler
```

---

## Adding Your Existing Modules

Copy your original stop_loss and backtester modules:

```bash
cp /path/to/original/modules/stop_loss.py  modules/
cp /path/to/original/modules/backtester.py modules/
```

They require no changes — `app.py` imports them with a graceful fallback if missing.

---

## Production Deployment (VPS / Railway / Render)

```bash
# Set environment variables in your platform's dashboard, then:
gunicorn --config gunicorn_config.py app:server

# Or with Procfile (Heroku / Railway):
# Platform reads Procfile automatically
```

**PostgreSQL**: use the managed database your platform provides.  
Set `DATABASE_URL` to the connection string they give you.

---

## Data Flow

```
FMP API
  │
  ├── ingestion/prices.py     → prices table       (nightly, incremental)
  ├── ingestion/prices.py     → fx_rates table      (nightly, incremental)
  ├── ingestion/universe.py   → company_info table  (weekly)
  ├── analytics/fundamental.py→ fundamentals table  (24h cache)
  └── ingestion/scheduler.py  → earnings_calendar   (nightly)
        │
        └── app.py reads from DB (instant)
              ├── analytics/technicals.py   computes signals in-memory
              ├── analytics/portfolio.py    computes P/L in-memory
              └── Dash renders charts/tables
```

---

## FAQ

**Q: The app is empty / shows "no data"**  
A: Run `python setup.py` first. The DB must be populated before the app can render.

**Q: I get a DB connection error**  
A: Check `DATABASE_URL` in your `.env`. Make sure PostgreSQL is running: `pg_isready`

**Q: FX rates are missing / showing 1.0**  
A: During setup, FX pairs are derived from your `transactions.csv` currencies.  
If your transactions are all in USD, make sure `currency` column contains `USD`.

**Q: Alerts aren't sending emails**  
A: Check `SMTP_USER` and `SMTP_PASS` in `.env`. Use a Gmail App Password  
(not your account password): https://support.google.com/accounts/answer/185833

**Q: Can I still use the old screener with stocks_all_pages.csv?**  
A: No — it's replaced by the `company_info` DB table.  
Run `python -m ingestion.universe` to repopulate it from FMP.
