#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSIGHT_AGENT_URL="${INSIGHT_AGENT_URL:-http://127.0.0.1:8000}"
INTERRUPTED_RUN_ID="${INTERRUPTED_RUN_ID:-}"
DEMO_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVAL_ROOT="${EVAL_ROOT:-/tmp/insight-agent-chapter18-${DEMO_STAMP}}"

json_field() {
  local field="$1"
  python -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$field"
}

cd "$PROJECT_ROOT"

echo "[operations] liveness"
curl --fail --silent --show-error "$INSIGHT_AGENT_URL/health"
echo

echo "[operations] readiness"
curl --fail --silent --show-error "$INSIGHT_AGENT_URL/ready"
echo

echo "[analyze] import fixed Markdown material"
curl --fail --silent --show-error \
  -F "file=@examples/chapter18_demo_material.md;type=text/markdown" \
  "$INSIGHT_AGENT_URL/v1/materials"
echo

echo "[direct] unified query; expected intent=direct and no run_id"
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"query":"Direct answer only: what is 2 + 2?"}' \
  "$INSIGHT_AGENT_URL/v1/query"
echo

echo "[analyze] unified query over imported local knowledge"
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"query":"Analyze the imported Chapter 18 demo notes and report the exact acceptance marker."}' \
  "$INSIGHT_AGENT_URL/v1/query"
echo

echo "[research] submit an explicit asynchronous research run"
RESEARCH_RESPONSE="$(curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"query":"Compare direct answering, local document analysis, and multi-step research. Produce a sourced report."}' \
  "$INSIGHT_AGENT_URL/v1/research/runs")"
echo "$RESEARCH_RESPONSE"
RUN_ID="$(printf '%s' "$RESEARCH_RESPONSE" | json_field run_id)"

echo "[research] initial status for $RUN_ID"
curl --fail --silent --show-error \
  "$INSIGHT_AGENT_URL/v1/research/runs/$RUN_ID"
echo

echo "[research] SSE events; stream ends at a terminal event"
curl -N --fail --silent --show-error \
  "$INSIGHT_AGENT_URL/v1/research/runs/$RUN_ID/events"
echo

echo "[research] final status and Research Report"
curl --fail --silent --show-error \
  "$INSIGHT_AGENT_URL/v1/research/runs/$RUN_ID"
echo

if [[ -n "$INTERRUPTED_RUN_ID" ]]; then
  echo "[resume] resume explicitly supplied interrupted/timed-out run"
  curl --fail --silent --show-error -X POST \
    "$INSIGHT_AGENT_URL/v1/research/runs/$INTERRUPTED_RUN_ID/resume"
  echo
  curl -N --fail --silent --show-error \
    "$INSIGHT_AGENT_URL/v1/research/runs/$INTERRUPTED_RUN_ID/events"
  echo
else
  echo "[resume] skipped: set INTERRUPTED_RUN_ID to an interrupted or timed-out run"
fi

echo "[evaluation] deterministic offline fixture run"
python -m evals \
  --dataset eval_data/research_eval_smoke_v1.jsonl \
  --fixture evals/fixtures/smoke_v1.json \
  --mode offline \
  --output "$EVAL_ROOT/runs" \
  --run-id chapter18-candidate

echo "[evaluation] explicit mock-only baseline registration"
python -m evals \
  --register-baseline "$EVAL_ROOT/runs/chapter18-candidate" \
  --baseline-output "$EVAL_ROOT/baseline.json" \
  --baseline-kind mock_only \
  --confirm-baseline

echo "[evaluation] regression comparison"
python -m evals \
  --baseline "$EVAL_ROOT/baseline.json" \
  --candidate "$EVAL_ROOT/runs/chapter18-candidate" \
  --policy evals/policies/default_v1.json \
  --report-output "$EVAL_ROOT/reports"

echo "Demo artifacts: $EVAL_ROOT"
