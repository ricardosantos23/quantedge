# Railway's default builder (Railpack — note /mise/installs/ in the
# crash traceback) does not ship libgomp.so.1, the OpenMP runtime that
# LightGBM and scikit-learn link against. nixpacks.toml was ignored
# because the active builder is Railpack, not Nixpacks. A Dockerfile is
# honoured by Railway above any auto-detected builder and gives us
# explicit, reproducible control of the system libraries.

FROM python:3.13-slim

# OpenMP runtime required by lightgbm / scikit-learn at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across code-only
# changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY . .

# gunicorn_config.py binds 0.0.0.0:$PORT (Railway injects $PORT) and
# reads WEB_CONCURRENCY for the worker count. Keep the start command
# identical to the Procfile so local and cloud behave the same.
CMD ["gunicorn", "--config", "gunicorn_config.py", "app:server"]
