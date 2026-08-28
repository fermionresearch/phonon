#!/bin/zsh
set -euo pipefail
HERE="${0:A:h}"
exec "$HERE/.venv/bin/python" "$HERE/stt.py" "$@"
