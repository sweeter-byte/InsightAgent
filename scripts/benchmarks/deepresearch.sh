#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMMAND="${1:-}"

if [[ -z "$COMMAND" ]]; then
  echo "Usage: scripts/benchmarks/deepresearch.sh <prepare|doctor|smoke|pilot|full|export|score|report> [options]" >&2
  exit 2
fi

shift
cd "$REPO_ROOT"
DEEPRESEARCH_PYTHON="${DEEPRESEARCH_PYTHON:-$(command -v python)}"
exec "$DEEPRESEARCH_PYTHON" -m evals.deepresearch_bench "$COMMAND" "$@"
