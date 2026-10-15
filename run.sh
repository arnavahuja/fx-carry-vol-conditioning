#!/usr/bin/env bash
# Launch the analysis app.
set -euo pipefail
PY="${PYTHON:-/opt/anaconda3/envs/backbone/bin/python}"
exec "$PY" -m streamlit run "$(dirname "$0")/app.py" "$@"
