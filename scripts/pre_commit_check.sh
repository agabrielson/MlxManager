#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m py_compile mlx_manager.py mlx_manager_bootstrap.py
python3 scripts/project_health_snapshot.py --static-only --check >/dev/null

echo "pre-commit health checks passed"
