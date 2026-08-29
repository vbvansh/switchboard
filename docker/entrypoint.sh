#!/bin/sh
# Container entrypoint: bring the schema up to date, then serve.
#
# Running migrations on start is the right default for a self-hosted single
# instance - it means `docker compose up` just works, with no second command to
# forget. It is NOT right when several replicas start at once: they would race
# to migrate the same database. Set SWITCHBOARD_AUTO_MIGRATE=false in that
# case and run `switchboard db upgrade` as a separate job first.
set -eu

if [ "${SWITCHBOARD_AUTO_MIGRATE:-true}" = "true" ]; then
    echo "Switchboard: applying database migrations..."
    python -m switchboard db upgrade
else
    echo "Switchboard: auto-migrate disabled; checking schema..."
    python -m switchboard db status
fi

# Every hosting platform - Render, Railway, Fly, Heroku, Cloud Run - assigns a
# port at runtime and passes it as PORT. A container that ignores it binds to
# the wrong port and the platform reports a startup timeout with no explanation,
# because from the outside it looks like the process simply never came up.
#
# Precedence: the platform's PORT, then an explicit SWITCHBOARD_PORT, then 8000.
BIND_PORT="${PORT:-${SWITCHBOARD_PORT:-8000}}"
BIND_HOST="${SWITCHBOARD_HOST:-0.0.0.0}"

echo "Switchboard: binding ${BIND_HOST}:${BIND_PORT}"

# exec replaces this shell with the server, so the server becomes PID 1 and
# receives SIGTERM directly. Without exec, the shell would hold PID 1, swallow
# the signal, and Docker would kill the container after the grace period -
# cutting off in-flight requests instead of letting them finish.
exec python -m switchboard serve --host "$BIND_HOST" --port "$BIND_PORT" "$@"
