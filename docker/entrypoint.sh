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

# exec replaces this shell with the server, so the server becomes PID 1 and
# receives SIGTERM directly. Without exec, the shell would hold PID 1, swallow
# the signal, and Docker would kill the container after the grace period -
# cutting off in-flight requests instead of letting them finish.
exec python -m switchboard serve "$@"
