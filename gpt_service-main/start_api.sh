#!/usr/bin/env sh
#export PYTHONPATH=$(pwd)
#DEFAULT_PORT=5000
#PORT=${1:-$DEFAULT_PORT}
## python api/main.py runserver -p "${PORT}"
## uwsgi --http :"${PORT}" --module api.main:app --processes 1 --threads 12 --buffer-size=32768 –-timeout=900 --harakiri=900 --socket-timeout=900
#uwsgi --http-socket 0.0.0.0:"$PORT" --ini uwsgi.ini

set -e

export APP_MODULE=api.main:app

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
LOG_LEVEL=${LOG_LEVEL:-info}

# If there's a prestart.sh script in the /app directory or other path specified, run it before starting
PRE_START_PATH=${PRE_START_PATH:-/app/prestart.sh}
if [ -f "$PRE_START_PATH" ]; then
  echo "Running script $PRE_START_PATH"
  ."$PRE_START_PATH"
fi

exec uvicorn --reload --reload-dir ${PWD} --host $HOST --port $PORT --log-level $LOG_LEVEL "$APP_MODULE" --access-log