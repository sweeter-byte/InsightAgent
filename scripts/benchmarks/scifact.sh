#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SCIFACT_PYTHON="${SCIFACT_PYTHON:-python}"
COMMAND="${1:-}"

if [[ -z "$COMMAND" ]]; then
  echo "Usage: $0 {prepare|index|smoke|full|report} [options]" >&2
  exit 2
fi

case "$COMMAND" in
  prepare|index|smoke|full|report) ;;
  *)
    echo "Unknown SciFact command: $COMMAND" >&2
    exit 2
    ;;
esac

shift
cd "$REPO_ROOT"
echo "[scifact.sh] repo=$REPO_ROOT command=$COMMAND python=$SCIFACT_PYTHON" >&2
exec "$SCIFACT_PYTHON" -m evals.beir_scifact "$COMMAND" "$@"
