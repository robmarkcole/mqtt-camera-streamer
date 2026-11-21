#!/bin/sh
# Use the same environment for preparation, model caching, benchmarks and systemd.
set -eu
project_root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/var/lib/mqtt-camera/venv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/var/lib/mqtt-camera/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/var/cache/mqtt-camera/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/var/cache/mqtt-camera}"
cd "$project_root"
exec "${UV_BIN:-/usr/local/bin/uv}" "$@"
