"""
gunicorn_config.py — Production server configuration.

Usage
-----
    gunicorn --config gunicorn_config.py app:server

Notes
-----
- workers=2 is intentional: APScheduler uses a SQLAlchemy job store
  so only 1 worker fires each scheduled job regardless of worker count.
  More than 4 workers wastes RAM on a typical 8GB VPS without benefit.
- Sync workers (default) are fine for Dash — it's not an async app.
- --timeout 120 gives the first data load time to finish.
"""
import multiprocessing
import os

# Bind
bind    = f"0.0.0.0:{os.environ.get('PORT', '8050')}"

# Workers: 2-4 is sweet spot for a dashboard app
workers = int(os.environ.get("WEB_CONCURRENCY", min(2, multiprocessing.cpu_count())))

# Timeouts
timeout      = 120   # seconds; covers slow DB queries on cold start
keepalive    = 5
graceful_timeout = 30

# Logging
loglevel    = os.environ.get("LOG_LEVEL", "info")
accesslog   = "-"   # stdout
errorlog    = "-"   # stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(D)sµs'

# Restart workers after N requests to avoid memory leaks
max_requests          = 1000
max_requests_jitter   = 50
