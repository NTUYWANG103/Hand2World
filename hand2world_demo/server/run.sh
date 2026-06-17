#!/usr/bin/env bash
# Closed-loop AR demo server launcher.
#
# Usage:
#   bash hand2world_demo/server/run.sh [args passed to hand2world_demo.server.server]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Single-GPU layout: cuda:0 runs Wan + VAE/TAE + WiLoR + xray render. They serialize
# through the engine's CUDA pool so splitting buys nothing.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# nvdiffrast / WiLoR want EGL-headless rendering on GPU.
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec python -u -m hand2world_demo.server.server "$@"
