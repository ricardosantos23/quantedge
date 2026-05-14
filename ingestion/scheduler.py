"""
ingestion/scheduler.py — Background job scheduler + alert checker.

Jobs
----
  22:00 UTC daily    → nightly price + FX refresh
  22:30 UTC daily    → earnings calendar (next 90 days)
  23:00 UTC Sunday   → universe sync (weekly)
  every 15 minutes   → check active alerts, fire emails

The scheduler uses a SQLAlchemy job store so:
  - Jobs survive restarts
  - Only ONE Gunicorn worker runs each job (no duplicates)

Usage
-----
  Embedded (recommended):
      from ingestion.scheduler import start_scheduler
      scheduler = start_scheduler()   # call once in app.py

  Standalone (alternative):
      python -m ingestion.scheduler
"""
import smtplib
import time
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

import config
from db.connection import Session, engine
from db.models import Alert, EarningsCalendar, create_all_tables
from ingestion.fmp_client import fetch_earnings_calendar
from ingestion.prices import run_price_ingestion, run_fx_ingestion
from ingestion.universe import run_universe_sync


# ════════════════════════════════════════════════════════════════════
#  EMAIL
# ════════════════════════════════════════════════════════════════════

def send_alert_email(to: str, subject: str, body: str) -> bool:
    """
    Send a plain-text alert email.
    Returns True on success, False on failure.
    Silently skips if SMTP is not configured.
    """
    if not config.SMTP_USER or not config.SMTP_PASS:
        print(f"  [alert] SMTP not configured — skipping email: {subject}")
        return False
    try:
        msg            = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = config.ALERT_FROM
        msg["To"]      = to
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.login(config.SMTP_USER, config.SMTP_PASS)
            s.sendmail(config.ALERT_FROM, [to], msg.as_string())
        print(f"  [alert] ✓ email sent to {to}: {subject}")
        return True
    except Exception as e:
        print(f"  [alert] ✗ email failed: {e}")
        return False


# ════════════════════════════════════════════════════════════════════
#  ALERT CHECKER
# ════════════════════════════════════════════════════════════════════

def check_and_fire_alerts() -> int:
    """
    Compare latest prices against active alert thresholds.
    Marks triggered alerts, sends email notifications.
    Returns the number of alerts fired.
    """
    fired_count = 0
    with Session() as s:
        alerts = (
            s.query(Alert)
            .filter(Alert.is_active == True, Alert.triggered == False)  # noqa: E712
            .all()
        )
        if not alerts:
            return 0

        # Batch-load latest price for all alerted tickers in one query
        tickers = list({a.ticker for a in alerts})
        price_rows = s.execute(
            text("""
                SELECT DISTINCT ON (ticker)
                    ticker, adj_close
                FROM prices
                WHERE ticker = ANY(:t)
                ORDER BY ticker, date DESC
            """),
            {"t": tickers},
        ).fetchall()
        price_map = {row[0]: float(row[1]) for row in price_rows if row[1]}

        now = datetime.utcnow()
        for alert in alerts:
            price = price_map.get(alert.ticker)
            if price is None:
                continue

            triggered = (
                (alert.alert_type == "price_above" and price >= alert.threshold)
                or (alert.alert_type in ("price_below", "stop_loss") and price <= alert.threshold)
            )

            if not triggered:
                continue

            alert.triggered    = True
            alert.triggered_at = now
            alert.is_active    = False
            fired_count       += 1

            if alert.email:
                direction = "≥" if alert.alert_type == "price_above" else "≤"
                subject   = f"[QuantEdge] {alert.ticker} {alert.alert_type} triggered"
                body      = (
                    f"Your alert has been triggered.\n\n"
                    f"Ticker:        {alert.ticker}\n"
                    f"Alert Type:    {alert.alert_type}\n"
                    f"Condition:     Price {direction} {alert.threshold:.2f}\n"
                    f"Current Price: {price:.2f}\n"
                    f"Triggered at:  {now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"— QuantEdge"
                )
                send_alert_email(alert.email, subject, body)

        s.commit()

    if fired_count:
        print(f"  [alerts] {fired_count} alert(s) fired")
    return fired_count


# ════════════════════════════════════════════════════════════════════
#  EARNINGS SYNC
# ════════════════════════════════════════════════════════════════════

def run_earnings_sync(days_ahead: int = 90) -> int:
    create_all_tables(engine)
    start = date.today().strftime("%Y-%m-%d")
    end   = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    rows = fetch_earnings_calendar(start, end)
    if not rows:
        print("  [earnings] no data returned")
        return 0

    seen = set()
    unique_rows = []
    for row in rows:
        key = (row["ticker"], row["earnings_date"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    rows = unique_rows
    print(f"  [earnings] {len(rows)} eventos únicos após deduplicação")

    BATCH = 500
    total_inserted = 0
    with Session() as s:
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            stmt = insert(EarningsCalendar).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "earnings_date"],
                set_={
                    "eps_estimate":     stmt.excluded.eps_estimate,
                    "eps_actual":       stmt.excluded.eps_actual,
                    "revenue_estimate": stmt.excluded.revenue_estimate,
                    "revenue_actual":   stmt.excluded.revenue_actual,
                    "updated_at":       datetime.utcnow(),
                },
            )
            s.execute(stmt)
            total_inserted += len(batch)
        s.commit()

    print(f"  [earnings] {total_inserted} events synced ({start} → {end})")
    return total_inserted


# ════════════════════════════════════════════════════════════════════
#  NIGHTLY PRICE REFRESH
# ════════════════════════════════════════════════════════════════════

def run_nightly_prices() -> None:
    """Refresh all tickers and FX pairs currently in the DB."""
    with Session() as s:
        tickers = [r[0] for r in s.execute(
            text("SELECT DISTINCT ticker FROM prices")
        ).fetchall()]
        pairs = [r[0] for r in s.execute(
            text("SELECT DISTINCT pair FROM fx_rates")
        ).fetchall()]

    if not tickers:
        print("  [nightly] no tickers in DB — run setup.py first")
        return

    run_price_ingestion(tickers)
    if pairs:
        run_fx_ingestion(pairs)


# ════════════════════════════════════════════════════════════════════
#  SCHEDULER
# ════════════════════════════════════════════════════════════════════

def start_scheduler():
    """
    Start APScheduler in background thread.
    Uses SQLAlchemy job store → only ONE worker fires each job
    even with multiple Gunicorn workers.

    Returns the scheduler instance (or None if APScheduler is not installed).
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
        from apscheduler.executors.pool import ThreadPoolExecutor as APThreadPool
    except ImportError:
        print("  [scheduler] APScheduler not installed — scheduler disabled.")
        print("  Install with: pip install apscheduler")
        return None

    create_all_tables(engine)

    jobstores  = {"default": SQLAlchemyJobStore(url=config.DB_URL)}
    executors  = {"default": APThreadPool(max_workers=2)}
    job_defaults = {"coalesce": True, "max_instances": 1, "misfire_grace_time": 600}

    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone="UTC",
    )

    # Nightly prices + FX — 22:00 UTC
    scheduler.add_job(
        run_nightly_prices, "cron",
        hour=config.NIGHTLY_CRON_HOUR,
        minute=config.NIGHTLY_CRON_MINUTE,
        id="nightly_prices",
        replace_existing=True,
    )
    # Earnings calendar — 22:30 UTC daily
    scheduler.add_job(
        run_earnings_sync, "cron",
        hour=config.NIGHTLY_CRON_HOUR, minute=30,
        id="nightly_earnings",
        replace_existing=True,
    )
    # Universe sync — Sunday 23:00 UTC (weekly is enough)
    scheduler.add_job(
        run_universe_sync, "cron",
        day_of_week="sun", hour=23, minute=0,
        id="weekly_universe",
        replace_existing=True,
    )
    # Alert checker — every 15 minutes
    scheduler.add_job(
        check_and_fire_alerts, "interval",
        minutes=15,
        id="alert_checker",
        replace_existing=True,
    )

    scheduler.start()
    print(
        "  [scheduler] ✓ started — "
        f"nightly prices @ {config.NIGHTLY_CRON_HOUR:02d}:{config.NIGHTLY_CRON_MINUTE:02d} UTC, "
        "alerts every 15 min"
    )
    return scheduler


# ════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import signal

    print("Starting scheduler (Ctrl+C to stop)...")
    sched = start_scheduler()

    if sched:
        def _shutdown(sig, frame):
            print("\nShutting down scheduler...")
            sched.shutdown()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while True:
            time.sleep(60)
