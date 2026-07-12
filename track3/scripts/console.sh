#!/usr/bin/env bash
# Manage the self-hosted TerraceGate console + its Cloudflare Tunnel.
#   ./scripts/console.sh up      # start the console + connector (amd.alvnvnc.site)
#   ./scripts/console.sh down     # stop both
#   ./scripts/console.sh status   # show local + public health
#
# The console runs next to your local Ollama (gemma3:12b + qwen2.5:3b on the AMD
# GPU) and is plan/gate-only — no execute path, so it is safe to expose.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${TG_HOST:-amd.alvnvnc.site}"
PORT="${TG_PORT:-8787}"
VLM_PROXY_TOKEN="${VLM_PROXY_TOKEN:-terraceroute-track2-v1}"
LOG="${TMPDIR:-/tmp}/terracegate-console"; mkdir -p "$LOG"

up() {
  pgrep -f "agent.webdemo --port $PORT" >/dev/null || {
    VLM_PROXY_TOKEN="$VLM_PROXY_TOKEN" \
      nohup python3 -m agent.webdemo --port "$PORT" > "$LOG/web.log" 2>&1 &
    echo "started console on :$PORT"; }
  pgrep -f "cloudflared tunnel run" >/dev/null || {
    [ -f .amd_token ] || { echo "missing .amd_token (run scripts/publish_amd.py first)"; exit 1; }
    # token via env so it never shows up in 'ps'
    TUNNEL_TOKEN="$(cat .amd_token)" nohup cloudflared tunnel run > "$LOG/cloudflared.log" 2>&1 &
    echo "started connector -> https://$HOST"; }
  sleep 6; status
}

down() {
  pkill -9 -f "cloudflared tunnel run" 2>/dev/null && echo "stopped connector" || true
  pkill -9 -f "agent.webdemo --port $PORT" 2>/dev/null && echo "stopped console" || true
  sleep 2  # let the pattern-matched processes actually exit before any restart
}

status() {
  echo "console  : $(pgrep -f "agent.webdemo --port $PORT" >/dev/null && echo up || echo down)"
  echo "connector: $(pgrep -f "cloudflared tunnel run" >/dev/null && echo up || echo down) ($(grep -c "Registered tunnel connection" "$LOG/cloudflared.log" 2>/dev/null || echo 0) edge conns)"
  echo "local    : HTTP $(curl -s -m5 -o /dev/null -w '%{http_code}' "localhost:$PORT/healthz" || echo down)"
  echo "public   : HTTP $(curl -s -m10 -o /dev/null -w '%{http_code}' "https://$HOST/healthz" || echo down)  ($HOST)"
}

case "${1:-status}" in
  up) up;; down) down;; restart) down; sleep 1; up;; status) status;;
  *) echo "usage: $0 {up|down|restart|status}"; exit 1;;
esac
