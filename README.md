# QuantEdge

> A production-grade quantitative equity analytics dashboard built in Python/Dash, backed by PostgreSQL and powered by the FMP Premium data feed.

QuantEdge combines portfolio tracking, multi-factor screening (technical + fundamental + ML), market-regime detection, backtesting, and stop-loss optimization in a single self-hosted web application.

---

## Highlights

- **Portfolio tracking** with multi-currency P&L, FX-adjusted holdings, sector allocation, and per-position stop-loss recommendations.
- **Multi-factor screener** combining technical scores, fundamental quality scores, and an ML ranking model (LightGBM, 10 features, 3-month horizon, q10/q90 confidence band).
- **Market regime detector** based on a Gaussian Mixture Model fit on SPY features — classifies the current state as BULL STRONG / BULL CALM / UNCERTAIN / STRESS and surfaces it as a banner above the screener.
- **Backtesting lab** with a DCA strategy, live weight sliders, an integrated weight optimiser, and a buy-and-hold benchmark.
- **Stop-loss comparison** across ATR-, trailing-percentage-, and fixed-percentage-based exits.
- **Production-ready deployment** with gunicorn, a `Procfile` for Railway/Render/Heroku, and a centralised `.env` configuration.

---

## Architecture

```
quantedge/
├── app.py                       # Dash entry point: layouts, callbacks, design tokens
├── config.py                    # Centralised settings, reads from environment / .env
├── setup.py                     # One-time database bootstrap
├── gunicorn_config.py           # Production server configuration
├── Procfile                     # Railway / Render / Heroku deploy descriptor
│
├── db/
│   ├── connection.py            # SQLAlchemy engine + Session context manager
│   └── models.py                # ORM models for every table
│
├── ingestion/
│   ├── fmp_client.py            # All outbound FMP API calls (single source of truth)
│   ├── prices.py                # Nightly OHLCV + FX ingestion → Postgres
│   ├── universe.py              # Stock universe sync (weekly)
│   └── scheduler.py             # APScheduler: nightly jobs + 15-min alert checker
│
├── analytics/
│   ├── technicals.py            # Momentum, MACD, volatility, drawdown features
│   ├── fundamental.py           # FMP-cached fundamentals (ROIC, FCF, growth, valuation)
│   ├── portfolio.py             # P&L computation, FX conversion, holdings table
│   └── multibagger.py           # Multi-bagger pattern scanner
│
└── modules/
    ├── backtester.py            # DCA strategy backtester (CAGR, Sharpe, drawdown)
    ├── stop_loss.py             # ATR / trailing / fixed stop-loss comparison
    ├── ml_scorer.py             # LightGBM ranking model with quantile uncertainty
    └── regime_detector.py       # GMM-based 4-state market regime classifier
```

### Tech stack

| Layer | Technology |
|---|---|
| Frontend | Dash · Plotly · Bootstrap (via `dash-bootstrap-components`) |
| Backend | Python 3.10+ · Flask (under Dash) · gunicorn in production |
| Database | PostgreSQL via SQLAlchemy ORM |
| Data feed | FMP Premium `/stable/` API (August 2025+) |
| Machine learning | LightGBM · scikit-learn (GaussianMixture, StandardScaler) |
| Scheduling | APScheduler with a SQLAlchemy job store |

---

## Quick start

### 1. Prerequisites

```bash
# PostgreSQL running locally
createdb quantedge
createuser quantedge
psql -c "ALTER USER quantedge WITH PASSWORD 'replace_with_strong_password';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE quantedge TO quantedge;"

# Python dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set at minimum FMP_API_KEY and DATABASE_URL
```

### 3. Bootstrap the database (one-time, ~16 minutes for top 500)

```bash
python setup.py
```

This will:

- Create every database table.
- Sync the FMP stock universe (up to ~48,000 tickers, or filtered by market cap).
- Ingest 10 years of price history for your portfolio plus the screener universe.
- Cache fundamentals for all screener tickers.
- Sync the next 90 days of earnings events.

### 4. Run

```bash
# Development
python app.py

# Production
gunicorn --config gunicorn_config.py app:server
```

Open the UI at **http://127.0.0.1:8050** (or the URL assigned by your hosting platform).

---

## Tabs

### Portfolio Overview
- Seven KPI cards: Total P/L, Day P/L, Wallet Value, Total Return, CAGR, Sharpe, Sortino.
- Cumulative P/L chart and sector allocation donut.
- Open positions table with unrealised P/L, weights, and composite scores.
- Stop-loss recommendations (ATR / trailing / fixed-percentage) per position.

### Screener
- **Market regime banner** at the top: current state, recommendation, stay probability, and expected duration.
- Full factor table: technical score, fundamental score, **ML score**, **ML decile**, **ML confidence band**, YTD, momentum, RSI, P/E, DCF upside, and other quality metrics.
- Click any row to drill into a detail view (chart + metrics).
- Sector filter.

### Strategy Lab
- Side-by-side signal comparison (technical score, momentum, RSI) per ticker.
- Interactive backtester with live weight sliders and a one-click weight optimiser.
- Equity curve and drawdown chart against a buy-and-hold SPY benchmark.

### Watchlist
- Add tickers by symbol.
- Real-time scores for every watchlist entry.

### Alerts
- Price-above, price-below, and stop-loss alert rules.
- Optional email notification (Gmail app password).
- Background checker runs every 15 minutes.

### Earnings Calendar
- Upcoming earnings events for the next 90 days.
- Portfolio holdings highlighted.
- EPS estimate vs actual, revenue estimate vs actual.

---

## Nightly data refresh

The embedded `APScheduler` runs four jobs automatically:

| Job | Schedule (UTC) | What it does |
|---|---|---|
| Price refresh | 22:00 daily | Incremental OHLCV update for every ticker in the DB |
| Earnings sync | 22:30 daily | Refresh next 90 days of earnings events |
| Universe sync | Sun 23:00 | Re-sync the FMP stock list and company profiles |
| Alert checker | every 15 min | Evaluate active alerts, dispatch email notifications |

Configure timing via `.env`:

```env
NIGHTLY_CRON_HOUR=22
NIGHTLY_CRON_MINUTE=0
```

> **Production note:** Free tiers on Railway/Render put the web service to sleep when idle, which kills the embedded scheduler. For reliable nightly ingestion in production, run the ingestion modules as platform-native cron jobs instead of relying on the embedded `APScheduler`. See [Production deployment](#production-deployment) below.

---

## Stock universe

The `company_info` table is the single source of truth for available tickers and metadata. Control the screener size via `.env`:

```env
SCREENER_UNIVERSE=500                       # Top 500 by market cap (recommended)
SCREENER_UNIVERSE=all                       # Every actively-traded ticker (~48,000)
SCREENER_UNIVERSE=AAPL,MSFT,NVDA,TSLA       # Explicit list
```

Portfolio tickers and SPY are always included regardless of this setting.

---

## Manual operations

```bash
# Refresh prices for specific tickers
python -m ingestion.prices AAPL MSFT NVDA

# Re-sync the full stock universe from FMP
python -m ingestion.universe

# Run the embedded scheduler standalone (useful for debugging)
python -m ingestion.scheduler
```

All three respect the `LOG_LEVEL` environment variable (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

---

## Logging

The application uses Python's standard `logging` module. The root configuration is bootstrapped once in `app.py`:

```text
2026-05-14 12:34:56 | INFO    | quantedge.app          | Loading market data...
2026-05-14 12:34:58 | INFO    | modules.ml_scorer      | ML Scorer: scored 487 tickers
2026-05-14 12:34:59 | INFO    | modules.regime_detector| Regime: BULL CALM ★★★★ (stay 92%, duration ~120d)
```

Override the level at runtime:

```bash
LOG_LEVEL=DEBUG python app.py
```

In production (Railway / Render / Heroku) the platform captures stdout/stderr automatically and surfaces logs via its dashboard.

---

## Production deployment

QuantEdge ships with a `Procfile` that works out of the box on Railway, Render, and Heroku.

1. Push the repo to GitHub.
2. Create a new project on your platform of choice and link it to the GitHub repo.
3. Add a managed PostgreSQL service; the platform will inject `DATABASE_URL` automatically.
4. Set the remaining environment variables in the platform's dashboard:
   - `FMP_API_KEY`
   - `SMTP_USER`, `SMTP_PASS`, `ALERT_FROM` (optional, for email alerts)
   - `DEBUG=False`
   - `LOG_LEVEL=INFO`
5. Run `python setup.py` once via the platform's shell to bootstrap the database.

### Reliable nightly ingestion in production

The embedded `APScheduler` lives inside the web process and stops running when the web service sleeps. For production-grade ingestion, set up separate cron services that run:

```
python -m ingestion.prices       # daily at 22:00 UTC
python -m ingestion.universe     # weekly Sunday 23:00 UTC
python -m ingestion.scheduler --earnings-only   # daily at 22:30 UTC (future flag)
```

Both Railway and Render expose cron jobs as a separate service type — see their respective docs.

---

## Data flow

```
FMP API
   │
   ├── ingestion/prices.py     → prices table         (nightly, incremental)
   ├── ingestion/prices.py     → fx_rates table       (nightly, incremental)
   ├── ingestion/universe.py   → company_info table   (weekly)
   ├── analytics/fundamental.py→ fundamentals table   (24h cache)
   └── ingestion/scheduler.py  → earnings_calendar    (nightly)
         │
         └── app.py reads from DB (instant on cold start)
               ├── analytics/technicals.py    derives signals in-memory
               ├── analytics/portfolio.py     computes P/L in-memory
               ├── modules/ml_scorer.py       trains and scores in-memory
               ├── modules/regime_detector.py fits GMM in-memory
               └── Dash renders the UI
```

---

## FAQ

**The app shows no data on first run**
Run `python setup.py` first. The database must be populated before any tab will render meaningful data.

**`psycopg2.OperationalError: could not connect to server`**
Check `DATABASE_URL` in your `.env` and confirm PostgreSQL is running locally:
```bash
pg_isready
```

**FX rates appear as 1.0 for every position**
During setup, FX pairs are derived from the `currency` column in `transactions.csv`. If every transaction is in USD, set `currency=USD` explicitly so the EUR→USD pair is fetched.

**Email alerts are not delivered**
Set `SMTP_USER` and `SMTP_PASS` in `.env`. For Gmail, generate an [App Password](https://support.google.com/accounts/answer/185833) instead of using your account password.

**ML scores look noisy / unreliable**
The LightGBM model is trained against a 3-month forward return, so live scores predict a horizon that is not yet observable. Treat the q10–q90 band as an indication of model uncertainty and re-evaluate scores roughly monthly.

**Regime banner is missing**
The regime detector requires at least 500 SPY observations. Ensure SPY is in your `prices` table — `python -m ingestion.prices SPY` will backfill it if needed.

---

## Development

This project follows a feature-branch + Pull Request workflow with `main` protected against direct pushes. Commit messages use the [Conventional Commits](https://www.conventionalcommits.org/) prefixes (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, etc.).

```bash
git checkout -b feature/short-name
# ... edit, test ...
git add -p
git commit -m "feat: ..."
git push -u origin feature/short-name
# Open a Pull Request on GitHub
```

---

## Configuration reference

All settings live in `.env` (locally) or in the platform's environment-variable panel (in production). See `.env.example` for the full template.

| Variable | Default | Description |
|---|---|---|
| `FMP_API_KEY` | _required_ | FMP Premium API key |
| `DATABASE_URL` | `postgresql://quantedge:quantedge@localhost:5432/quantedge` | PostgreSQL connection string |
| `YEARS_BACK` | `10` | Price history depth ingested by `setup.py` |
| `SCREENER_UNIVERSE` | `500` | `"all"`, an integer (top-N by market cap), or a comma-separated ticker list |
| `NIGHTLY_CRON_HOUR` | `22` | Hour (UTC) for the embedded nightly job |
| `NIGHTLY_CRON_MINUTE` | `0` | Minute (UTC) for the embedded nightly job |
| `SMTP_HOST` | `smtp.gmail.com` | SMTP server for email alerts |
| `SMTP_PORT` | `587` | SMTP port |
| `SMTP_USER` | _empty_ | SMTP username |
| `SMTP_PASS` | _empty_ | SMTP password (Gmail App Password recommended) |
| `ALERT_FROM` | _equals `SMTP_USER`_ | "From" address used on outgoing alert emails |
| `DEBUG` | `True` | Dash debug mode toggle |
| `PORT` | `8050` | Web server port |
| `HOST` | `127.0.0.1` | Web server bind host (`0.0.0.0` in production) |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## License

This project is released for personal and educational use. The FMP API key, transaction data, and any deployed instance remain your sole responsibility.
