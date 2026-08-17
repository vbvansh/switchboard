# Switchboard container image.
#
# Two stages. The builder installs dependencies into a virtualenv; the runtime
# stage copies only that venv and the application. Build tools, caches and
# compilers never reach the final image - it stays small, and a smaller image
# has less in it that can be vulnerable.

# --- Stage 1: dependencies --------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied alone, before the source, so Docker reuses the cached dependency layer
# whenever only application code has changed. Rebuilds go from minutes to
# seconds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Switchboard" \
      org.opencontainers.image.description="Self-hostable AI model router with an auditable savings ledger." \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as a non-root user. If the process is ever compromised, the attacker
# lands as an unprivileged account rather than as root inside the container.
RUN groupadd --system switchboard \
    && useradd --system --gid switchboard --home-dir /app switchboard

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=switchboard:switchboard switchboard/ ./switchboard/
COPY --chown=switchboard:switchboard migrations/ ./migrations/
COPY --chown=switchboard:switchboard docker/entrypoint.sh ./docker/entrypoint.sh
COPY --chown=switchboard:switchboard alembic.ini providers.yaml pyproject.toml \
     LICENSE NOTICE README.md ./

# The default SQLite database lives here. Mount a volume over it to keep the
# ledger across container replacements - without one, every restart loses all
# users, budgets and spending history.
RUN mkdir -p /app/data \
    && chown switchboard:switchboard /app/data \
    && chmod +x /app/docker/entrypoint.sh

USER switchboard

# 127.0.0.1 is the right default on a laptop and useless in a container -
# nothing outside could reach it. Overridden here, not in config.py, so the
# safe default still applies when running locally.
ENV SWITCHBOARD_HOST=0.0.0.0 \
    SWITCHBOARD_PORT=8000 \
    SWITCHBOARD_DATABASE_URL=sqlite:////app/data/switchboard.db

EXPOSE 8000
VOLUME ["/app/data"]

# Liveness only - deliberately does not check providers. See /health/live.
# Uses urllib rather than curl, which is not installed in the slim image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status==200 else 1)"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
