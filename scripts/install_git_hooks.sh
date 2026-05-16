#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

git config core.hooksPath .githooks
chmod +x .githooks/pre-commit scripts/pre_commit_check.sh scripts/project_health_snapshot.py

echo "Installed repo-local git hooks from .githooks"
