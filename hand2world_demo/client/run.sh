#!/usr/bin/env bash
# Closed-loop AR demo client launcher (run on the Mac with the iPhone connected).
#
# Usage:
#   bash hand2world_demo/client/run.sh --server ws://gpu-host:8501
#
# The Hand2WorldCam SDK opens its own WebSocket on :8765 for the phone — no
# explicit start needed; it boots inside the client process.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

exec python -u -m hand2world_demo.client.client "$@"
