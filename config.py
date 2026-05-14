"""
config.py — Centralised configuration for QuantEdge.
Loads .env automatically via python-dotenv (pip install python-dotenv).
"""
import os

# Load .env file automatically — no more manual PowerShell exports needed
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass  # python-dotenv not installed — fall back to environment variables

# ── FMP API ──────────────────────────────────────────────────────────────────
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE    = "https://financialmodelingprep.com/stable"
FMP_TIMEOUT = 20
FMP_RETRIES = 2

# ── PostgreSQL ───────────────────────────────────────────────────────────────
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://quantedge:quantedge@localhost:5432/quantedge",
)

# ── Email alerts ─────────────────────────────────────────────────────────────
SMTP_HOST  = os.environ.get("SMTP_HOST",  "smtp.gmail.com")
SMTP_PORT  = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER  = os.environ.get("SMTP_USER",  "")
SMTP_PASS  = os.environ.get("SMTP_PASS",  "")
ALERT_FROM = os.environ.get("ALERT_FROM", SMTP_USER)

# ── Data settings ─────────────────────────────────────────────────────────────
YEARS_BACK    = int(os.environ.get("YEARS_BACK", "10"))

_raw = os.environ.get("SCREENER_UNIVERSE", "500")
if _raw.lower() == "all":
    SCREENER_UNIVERSE = "all"
elif "," in _raw:
    SCREENER_UNIVERSE = [t.strip().upper() for t in _raw.split(",")]
else:
    try:
        SCREENER_UNIVERSE = int(_raw)
    except ValueError:
        SCREENER_UNIVERSE = 500

# ── Nightly scheduler (UTC) ───────────────────────────────────────────────────
NIGHTLY_CRON_HOUR   = int(os.environ.get("NIGHTLY_CRON_HOUR",   "22"))
NIGHTLY_CRON_MINUTE = int(os.environ.get("NIGHTLY_CRON_MINUTE",  "0"))

# ── App server ────────────────────────────────────────────────────────────────
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
PORT  = int(os.environ.get("PORT",  "8050"))
HOST  = os.environ.get("HOST", "127.0.0.1")
