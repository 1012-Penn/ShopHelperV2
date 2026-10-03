#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVER_URL="http://${HOST}:${PORT}"
server_log="$(mktemp)"

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -f "$server_log"
}
trap cleanup EXIT

python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT" >"$server_log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 30); do
  if curl -fsS "${SERVER_URL}/openapi.json" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  cat "$server_log" >&2
  exit 1
fi

stream_response="$(curl -fsS -N "${SERVER_URL}/api/v1/chat/stream" \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"smoke-1","message":"这是合成烟囱测试，请回复测试。","history":[]}' )"
printf '%s\n' "$stream_response"
grep -q 'event: token' <<<"$stream_response"
grep -q 'event: done' <<<"$stream_response"

after_sale_response="$(curl -fsS "${SERVER_URL}/api/v1/after-sale/extract" \
  -H 'Content-Type: application/json' \
  -d '{"text":"测试订单TEST-123的商品损坏，我想换货。"}' )"
printf '%s\n' "$after_sale_response"
for field in order_id request_type expected_solution; do
  grep -q "\"${field}\"" <<<"$after_sale_response"
done

echo "smoke test passed"
