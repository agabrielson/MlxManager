#!/usr/bin/env python3
"""Generate a project-health snapshot for MLX Manager.

The default mode is intentionally safe for local development: it performs
static checks plus read-only runtime probes. Use ``--static-only`` for
pre-commit where the GUI/gateway may not be running.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "docs" / "project_health_report.md"
SETTINGS_PATH = REPO_ROOT / ".mlx_manager.json"


def _run(cmd: list[str], *, timeout: float = 8.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return 127, "", str(exc)


def _git_lines(args: list[str]) -> list[str]:
    code, out, _err = _run(["git", *args], timeout=8)
    if code != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]


def tracked_files() -> list[Path]:
    files = [REPO_ROOT / item for item in _git_lines(["ls-files"])]
    # Include this health machinery before it has been committed for the first
    # time, while avoiding unrelated local scratch files.
    health_files = [
        ".githooks/pre-commit",
        "docs/project_health_report.md",
        "scripts/install_git_hooks.sh",
        "scripts/pre_commit_check.sh",
        "scripts/project_health_snapshot.py",
    ]
    seen = {path.resolve() for path in files}
    for rel in health_files:
        path = REPO_ROOT / rel
        if path.exists() and path.resolve() not in seen:
            files.append(path)
            seen.add(path.resolve())
    return files


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _line_count(path: Path) -> int:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".icns"}:
        return 0
    text = _read_text(path)
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _category(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    suffix = path.suffix.lower()
    if rel.startswith(".githooks/"):
        return "Git hooks"
    if rel.startswith("scripts/"):
        return "Health automation"
    if rel.startswith("docs/") or suffix == ".md":
        return "Documentation"
    if rel.startswith("assets/"):
        return "Assets"
    if suffix == ".py":
        return "Python product code"
    if suffix in {".sh", ".bash"}:
        return "Shell/setup scripts"
    return "Repo metadata/config"


class _ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.cyclomatic = 1
        self.cognitive = 0
        self._nesting = 0

    def _branch(self, node: ast.AST, *, extra_paths: int = 1) -> None:
        self.cyclomatic += extra_paths
        self.cognitive += 1 + self._nesting
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    def visit_If(self, node: ast.If) -> None:  # noqa: N802 - AST visitor API
        self._branch(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._branch(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
        self._branch(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self._branch(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:  # noqa: N802
        self._branch(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        self._branch(node)

    def visit_Assert(self, node: ast.Assert) -> None:  # noqa: N802
        self.cyclomatic += 1
        self.cognitive += 1 + self._nesting
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:  # noqa: N802
        extra = max(0, len(node.values) - 1)
        self.cyclomatic += extra
        self.cognitive += extra
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:  # noqa: N802
        self._branch(node, extra_paths=max(1, len(node.cases)))


def _function_complexity(path: Path) -> list[dict[str, Any]]:
    text = _read_text(path)
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [
            {
                "file": path.relative_to(REPO_ROOT).as_posix(),
                "function": "<syntax-error>",
                "line": exc.lineno or 1,
                "cyclomatic": 0,
                "cognitive": 0,
                "length": 0,
                "error": str(exc),
            }
        ]
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        visitor = _ComplexityVisitor()
        visitor.visit(node)
        end_line = getattr(node, "end_lineno", node.lineno)
        rows.append(
            {
                "file": path.relative_to(REPO_ROOT).as_posix(),
                "function": node.name,
                "line": node.lineno,
                "cyclomatic": visitor.cyclomatic,
                "cognitive": visitor.cognitive,
                "length": max(1, end_line - node.lineno + 1),
                "error": "",
            }
        )
    return rows


def _non_python_heuristic(path: Path) -> dict[str, Any]:
    rel = path.relative_to(REPO_ROOT).as_posix()
    text = _read_text(path)
    suffix = path.suffix.lower()
    if suffix == ".md":
        return {
            "file": rel,
            "heuristic": "markdown_structure",
            "score": len(re.findall(r"^#{1,6}\s+", text, flags=re.M))
            + len(re.findall(r"```mermaid", text))
            + len(re.findall(r"^\|", text, flags=re.M)),
        }
    if suffix in {".sh", ".bash"}:
        return {
            "file": rel,
            "heuristic": "shell_control_tokens",
            "score": len(re.findall(r"\b(if|then|fi|for|while|case|esac|trap|exit)\b", text)),
        }
    return {"file": rel, "heuristic": "none", "score": 0}


def _asset_summary(files: list[Path]) -> dict[str, Any]:
    asset_files = [p for p in files if p.relative_to(REPO_ROOT).as_posix().startswith("assets/")]
    total_bytes = 0
    for path in asset_files:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
    return {"files": len(asset_files), "bytes": total_bytes}


def _settings_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(SETTINGS_PATH),
        "exists": SETTINGS_PATH.exists(),
        "mode": None,
        "healthy_permissions": False,
        "parse_ok": False,
        "hf_token_present": False,
        "gateway_token_present": False,
        "auth_enabled": None,
        "host_mode": None,
        "scheme": None,
        "port": None,
        "idle_enabled": None,
        "idle_minutes": None,
        "sleep_mode": None,
        "selected_model_present": False,
        "last_running_model_present": False,
    }
    if not SETTINGS_PATH.exists():
        return summary
    try:
        mode = SETTINGS_PATH.stat().st_mode & 0o777
        summary["mode"] = oct(mode)
        summary["healthy_permissions"] = mode == 0o600
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        summary["error"] = str(exc)
        return summary
    summary["parse_ok"] = isinstance(data, dict)
    if not isinstance(data, dict):
        return summary
    summary.update(
        {
            "hf_token_present": bool(str(data.get("hf_token") or "").strip()),
            "gateway_token_present": bool(str(data.get("auth_token") or "").strip()),
            "auth_enabled": bool(data.get("auth_enabled")),
            "host_mode": data.get("host_mode"),
            "scheme": data.get("gateway_scheme"),
            "port": data.get("port"),
            "idle_enabled": bool(data.get("idle_enabled")),
            "idle_minutes": data.get("idle_minutes"),
            "sleep_mode": data.get("sleep_mode"),
            "selected_model_present": bool(str(data.get("selected_model") or "").strip()),
            "last_running_model_present": bool(str(data.get("last_running_model") or "").strip()),
        }
    )
    return summary


def _hf_cache_dir() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser()
    return Path("~/.cache/huggingface/hub").expanduser()


def _model_cache_summary() -> dict[str, Any]:
    cache = _hf_cache_dir()
    models: list[str] = []
    bytes_total = 0
    if cache.exists():
        for path in cache.iterdir():
            if not path.name.startswith("models--"):
                continue
            model_id = path.name.replace("models--", "", 1).replace("--", "/")
            if model_id.startswith("mlx-community/"):
                models.append(model_id)
                try:
                    bytes_total += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                except OSError:
                    pass
    return {
        "cache_dir": str(cache),
        "mlx_community_models": len(models),
        "approx_bytes": bytes_total,
        "sample": sorted(models)[:8],
    }


def _process_summary() -> dict[str, Any]:
    code, out, _err = _run(["pgrep", "-fl", "mlx_manager.py"], timeout=4)
    manager = [line for line in out.splitlines() if "mlx_manager.py" in line] if code == 0 else []
    code, out, _err = _run(["pgrep", "-fl", "mlx_lm|mlx-lm"], timeout=4)
    mlx_servers = [line for line in out.splitlines() if "mlx" in line.lower()] if code == 0 else []
    return {
        "manager_processes": len(manager),
        "mlx_server_processes": len(mlx_servers),
        "manager_running": bool(manager),
        "mlx_server_running": bool(mlx_servers),
    }


def _http_json(url: str, *, token: str = "", verify_ssl: bool = False, timeout: float = 4.0) -> dict[str, Any]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    context = None
    if url.startswith("https://") and not verify_ssl:
        context = ssl._create_unverified_context()
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as resp:
            body = resp.read(256_000)
            elapsed_ms = round((time.time() - started) * 1000, 1)
            payload = json.loads(body.decode("utf-8", errors="replace") or "{}")
            return {"ok": True, "status": resp.status, "elapsed_ms": elapsed_ms, "json": payload}
    except urllib.error.HTTPError as exc:
        body = exc.read(256_000)
        elapsed_ms = round((time.time() - started) * 1000, 1)
        try:
            payload = json.loads(body.decode("utf-8", errors="replace") or "{}")
        except json.JSONDecodeError:
            payload = {"raw": body.decode("utf-8", errors="replace")[:500]}
        return {"ok": False, "status": exc.code, "elapsed_ms": elapsed_ms, "json": payload}
    except Exception as exc:
        return {"ok": False, "status": None, "elapsed_ms": round((time.time() - started) * 1000, 1), "error": str(exc)}


def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def _gateway_summary(settings: dict[str, Any], static_only: bool) -> dict[str, Any]:
    if static_only:
        return {"checked": False, "reason": "static_only"}
    port = settings.get("port") or 9000
    scheme = str(settings.get("scheme") or "http").lower()
    token = ""
    verify_ssl = False
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            token = str(data.get("auth_token") or "")
            verify_ssl = bool(data.get("client_verify_ssl"))
        except Exception:
            pass
    base_url = f"{scheme}://127.0.0.1:{port}"
    result = {
        "checked": True,
        "base_url": base_url,
        "port_open": _port_open("127.0.0.1", int(port)),
        "health": None,
        "ready": None,
        "status": None,
        "models": None,
    }
    result["health"] = _http_json(f"{base_url}/health", verify_ssl=verify_ssl)
    result["ready"] = _http_json(f"{base_url}/ready", verify_ssl=verify_ssl)
    result["status"] = _http_json(f"{base_url}/status", token=token, verify_ssl=verify_ssl)
    result["models"] = _http_json(f"{base_url}/v1/models", token=token, verify_ssl=verify_ssl)
    return result


def _compile_check() -> dict[str, Any]:
    python = REPO_ROOT / "mlx-env" / "bin" / "python"
    interpreter = str(python) if python.exists() else sys.executable
    code, out, err = _run([interpreter, "-m", "py_compile", "mlx_manager.py", "mlx_manager_bootstrap.py"], timeout=12)
    return {"ok": code == 0, "interpreter": interpreter, "stdout": out, "stderr": err}


def build_snapshot(*, static_only: bool = False) -> dict[str, Any]:
    files = tracked_files()
    loc_by_category: dict[str, int] = {}
    for path in files:
        loc_by_category[_category(path)] = loc_by_category.get(_category(path), 0) + _line_count(path)

    py_files = [p for p in files if p.suffix == ".py"]
    functions: list[dict[str, Any]] = []
    for path in py_files:
        functions.extend(_function_complexity(path))
    functions_sorted = sorted(
        [row for row in functions if not row.get("error")],
        key=lambda r: (r["cyclomatic"], r["cognitive"], r["length"]),
        reverse=True,
    )

    file_rows: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        rows = [row for row in functions if row["file"] == rel and not row.get("error")]
        heuristic = _non_python_heuristic(path)
        file_rows.append(
            {
                "file": rel,
                "loc": _line_count(path),
                "function_count": len(rows),
                "max_cyclomatic": max([row["cyclomatic"] for row in rows], default=0),
                "max_cognitive": max([row["cognitive"] for row in rows], default=0),
                "max_function_length": max([row["length"] for row in rows], default=0),
                "heuristic": heuristic["score"],
                "category": _category(path),
            }
        )
    hotspot_rows = sorted(
        file_rows,
        key=lambda row: (
            row["max_cyclomatic"] * 2
            + row["max_cognitive"] * 1.5
            + row["max_function_length"] / 10
            + row["loc"] / 250
            + row["heuristic"] / 25
        ),
        reverse=True,
    )

    git_status = _git_lines(["status", "--short"])
    settings = _settings_summary()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch": (_git_lines(["branch", "--show-current"]) or ["unknown"])[0],
            "commit": (_git_lines(["rev-parse", "--short", "HEAD"]) or ["unknown"])[0],
            "dirty_entries": git_status,
            "dirty_count": len(git_status),
        },
        "loc": {
            "total": sum(loc_by_category.values()),
            "by_category": dict(sorted(loc_by_category.items())),
            "tracked_files": len(files),
        },
        "complexity": {
            "python_files": len(py_files),
            "functions": len([row for row in functions if not row.get("error")]),
            "functions_cyclomatic_gt_10": sum(1 for row in functions if row.get("cyclomatic", 0) > 10),
            "functions_cognitive_gt_15": sum(1 for row in functions if row.get("cognitive", 0) > 15),
            "functions_longer_than_75": sum(1 for row in functions if row.get("length", 0) > 75),
            "files_longer_than_750": sum(1 for row in file_rows if row["loc"] > 750),
            "top_functions": functions_sorted[:10],
            "top_hotspots": hotspot_rows[:10],
        },
        "runtime": {
            "settings": settings,
            "processes": _process_summary() if not static_only else {"checked": False, "reason": "static_only"},
            "gateway": _gateway_summary(settings, static_only),
            "model_cache": _model_cache_summary(),
            "compile_check": _compile_check(),
            "asset_summary": _asset_summary(files),
        },
    }


def _fmt_bytes(num: int) -> str:
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1000 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000
    return f"{num} B"


def _display_path(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    try:
        path = Path(text)
        if path.is_absolute():
            return path.relative_to(REPO_ROOT).as_posix()
    except Exception:
        pass
    return text


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def render_markdown(snapshot: dict[str, Any]) -> str:
    runtime = snapshot["runtime"]
    settings = runtime["settings"]
    gateway = runtime["gateway"]
    compile_check = runtime["compile_check"]
    processes = runtime["processes"]
    cache = runtime["model_cache"]
    complexity = snapshot["complexity"]

    gateway_state = "not checked"
    gateway_status = "Not checked"
    if gateway.get("checked"):
        health = gateway.get("health") or {}
        ready = gateway.get("ready") or {}
        ready_payload = ready.get("json") if isinstance(ready.get("json"), dict) else {}
        model_state = ready_payload.get("state") or (health.get("json") or {}).get("state")
        gateway_state = f"health={health.get('status') or 'error'}, ready={ready.get('status') or 'error'}, state={model_state or 'unknown'}"
        gateway_status = "Green" if health.get("status") == 200 and ready.get("status") in {200, 503} else "Review"

    rows = [
        ["Compile check", "Green" if compile_check["ok"] else "Review", _display_path(compile_check["interpreter"])],
        ["Gateway status", gateway_status, f"{gateway.get('base_url', gateway.get('reason', ''))}; {gateway_state}"],
        [
            "Settings persistence",
            "Green" if settings["parse_ok"] and settings["healthy_permissions"] else "Review",
            f"mode={settings.get('mode')}; hf_token={'present' if settings.get('hf_token_present') else 'missing'}",
        ],
        [
            "Process hygiene",
            "Review" if processes.get("manager_processes", 0) > 1 else "Green",
            f"manager={processes.get('manager_processes', 'n/a')}; mlx_server={processes.get('mlx_server_processes', 'n/a')}",
        ],
        [
            "Model inventory",
            "Green" if cache["mlx_community_models"] else "Review",
            f"{cache['mlx_community_models']} cached mlx-community model(s), approx {_fmt_bytes(cache['approx_bytes'])}",
        ],
    ]

    health_metrics = [
        ["Startup/wake reliability", "Startup success rate, p50/p95 startup time, wake-on-request success, wake latency, stuck-warming count.", "Core user experience and client reliability."],
        ["Sleep/battery behavior", "Idle timeout accuracy, real-question/probe split, CPU and memory while running/light sleep/deep sleep/stopped.", "Keeps local inference useful on a laptop."],
        ["Gateway contract", "`/health`, `/ready`, `/status`, `/v1/models`, `/v1/chat/completions`, streaming/non-streaming behavior.", "Keeps LLM-OS, opencode, scripts, and IDE tools interoperable."],
        ["Auth/TLS matrix", "HTTP/HTTPS, auth on/off, localhost/all-hosts, SSL verification on/off.", "Most connection bugs hide at configuration edges."],
        ["Config persistence", "Selected model, last-running model, HF token availability, gateway token, host mode, sleep mode, file mode `0600`.", "Prevents silent config drift and forgotten credentials."],
        ["Process hygiene", "Orphaned `mlx_lm` servers, port collisions, close latency, forced-kill count.", "Prevents battery drain and stuck shutdowns."],
        ["UI responsiveness", "Long work on UI thread, button latency, close latency, log visibility.", "The manager must stay usable while models warm or fail."],
        ["Client sync", "LLM-OS sync success/failure, base URL, protocol, auth flag, token-present status, model name.", "Avoids “connected” status with broken requests."],
        ["Token telemetry", "Real chat request count, prompt/completion/total tokens, exact vs estimated source.", "Explains use patterns and helps tune battery/performance tradeoffs."],
        ["Model inventory", "Cached model count, disk use, selected model exists, HF browse/download success.", "Prevents selected-model startup surprises."],
        ["Security hygiene", "No secrets in logs/git, masked diagnostics, auth-on when all-hosts, cert/key presence.", "Important when the gateway is reachable beyond localhost."],
    ]
    cs_best_practices = [
        ["Separate mechanism from policy", "Keep generic gateway/process/settings helpers reusable; keep MLX-specific policy in small call sites.", "Makes the manager easier to adapt for opencode or other local clients."],
        ["Keep UI thread thin", "Never perform network calls, model startup, shutdown waits, downloads, or gateway blocking work on the Qt UI thread.", "Protects responsiveness during warmup, sleep, TLS failures, and model download."],
        ["Use explicit state transitions", "Represent running, warming, sleeping, deep sleep, stopped, and error as deliberate states with logs and API status.", "Prevents clients from guessing whether to retry, wake, or fail."],
        ["Make side effects bounded", "Every subprocess, thread, socket, and timer should have a clear owner and bounded shutdown path.", "Avoids orphaned `mlx_lm` servers, stuck ports, and battery drain."],
        ["Treat configuration as data", "Persist settings through one adapter, mask secrets in diagnostics, and report token fields only as present/missing.", "Prevents the class of config drift that made the HF token appear forgotten."],
        ["Test contracts, not only functions", "For gateway work, verify HTTP status, JSON envelope, auth behavior, retry semantics, and state fields.", "Clients depend on the contract more than the internal implementation."],
        ["Prefer small deterministic checks in hooks", "Pre-commit should compile, regenerate/check health, and avoid live model dependencies.", "Keeps commits fast while still catching broken startup syntax and stale reports."],
        ["Use hotspots to guide refactors", "Refactor files/functions that are complex, changing, and operationally critical before cosmetic cleanup.", "Targets the next bug cluster instead of chasing raw LOC."],
    ]
    cs_thresholds = [
        ["Cyclomatic complexity", "`> 10` review; `> 20` refactor candidate", "Split branch-heavy request handling, gateway state, and settings normalization."],
        ["Cognitive complexity", "`> 15` review; `> 25` refactor candidate", "Flatten nested state logic and move decision tables into helpers."],
        ["Function length", "`> 75` review; `> 150` refactor candidate", "Extract one responsibility: parsing, status building, forwarding, or UI rendering."],
        ["File length", "`> 750` review", "Consider splitting by lifecycle, gateway, settings, telemetry, or UI widgets."],
        ["Thread/process ownership", "One owner per worker/process/socket", "Name threads, use bridge signals for GUI updates, and keep shutdown bounded."],
        ["Runtime contract coverage", "Every endpoint and mode has a smoke check", "Cover HTTP/HTTPS, auth on/off, sleeping/warming/running, streaming/non-streaming."],
        ["Secret exposure", "Zero raw tokens in docs, logs, or commits", "Only report present/missing and run staged diff secret scans before push."],
    ]

    loc_rows = [[k, f"{v:,}"] for k, v in snapshot["loc"]["by_category"].items()]
    hotspot_rows = [
        [
            idx,
            row["file"],
            row["loc"],
            row["max_cyclomatic"],
            row["max_cognitive"],
            row["max_function_length"],
            row["heuristic"],
        ]
        for idx, row in enumerate(complexity["top_hotspots"][:8], start=1)
    ]
    function_rows = [
        [idx, row["file"], row["function"], row["line"], row["cyclomatic"], row["cognitive"], row["length"]]
        for idx, row in enumerate(complexity["top_functions"][:8], start=1)
    ]

    return "\n".join(
        [
            "# MLX Manager Project Health Report",
            "",
            f"Last generated: {snapshot['generated_at']}",
            "",
            "This is the living health report for MLX Manager. It tracks whether the project is safe to change, pleasant to operate, and reliable as a local OpenAI-compatible gateway for MLX models.",
            "",
            "The report deliberately combines classic computer-science/code metrics with operator metrics that matter for this specific code region: GUI responsiveness, model lifecycle, sleep/wake, gateway compatibility, token accounting, settings persistence, and battery/process hygiene.",
            "",
            "## Current Snapshot",
            "",
            _table(["Area", "Status", "Evidence"], rows),
            "",
            "## MLX-Specific Health Scorecard",
            "",
            _table(["Area", "Track", "Why"], health_metrics),
            "",
            "## Computer Engineering Metrics",
            "",
            "### LOC Baseline",
            "",
            _table(["Area", "Lines"], loc_rows + [["Total reported LOC", f"{snapshot['loc']['total']:,}"]]),
            "",
            "### Complexity Snapshot",
            "",
            _table(
                ["Metric", "Current Value"],
                [
                    ["Python files analyzed", complexity["python_files"]],
                    ["Python functions analyzed", complexity["functions"]],
                    ["Functions with cyclomatic complexity `> 10`", complexity["functions_cyclomatic_gt_10"]],
                    ["Functions with cognitive complexity `> 15`", complexity["functions_cognitive_gt_15"]],
                    ["Functions longer than 75 lines", complexity["functions_longer_than_75"]],
                    ["Files longer than 750 lines", complexity["files_longer_than_750"]],
                ],
            ),
            "",
            "Complexity should be read as a refactoring signal, not a grade. For this project, high-complexity code is most risky when it also controls gateway routing, subprocess lifecycle, sleep/wake, auth/TLS, or persistent settings.",
            "",
            "### CS Best Practices For This Code Region",
            "",
            _table(["Practice", "Guidance", "Why It Matters Here"], cs_best_practices),
            "",
            "### Suggested Engineering Thresholds",
            "",
            _table(["Metric", "Threshold", "Preferred Response"], cs_thresholds),
            "",
            "### Top Hotspots",
            "",
            _table(["Rank", "File", "LOC", "Max Cyclomatic", "Max Cognitive", "Max Function Length", "Structural Heuristic"], hotspot_rows),
            "",
            "### Top Python Functions By Cyclomatic Complexity",
            "",
            _table(["Rank", "File", "Function", "Line", "Cyclomatic", "Cognitive", "Length"], function_rows),
            "",
            "## Runtime And Operator Metrics",
            "",
            _table(
                ["Metric", "Current Value"],
                [
                    ["Settings file", _display_path(settings["path"])],
                    ["Settings permissions", f"{settings.get('mode')} ({'ok' if settings.get('healthy_permissions') else 'review'})"],
                    ["HF token", "present" if settings.get("hf_token_present") else "missing"],
                    ["Gateway auth", "enabled" if settings.get("auth_enabled") else "disabled"],
                    ["Gateway mode", f"{settings.get('scheme')} / {settings.get('host_mode')} / port {settings.get('port')}"],
                    ["Idle policy", f"{settings.get('sleep_mode')} after {settings.get('idle_minutes')} min ({'enabled' if settings.get('idle_enabled') else 'disabled'})"],
                    ["Manager process count", processes.get("manager_processes", "n/a")],
                    ["MLX server process count", processes.get("mlx_server_processes", "n/a")],
                    ["Cached mlx-community models", cache["mlx_community_models"]],
                    ["Cached model disk footprint", _fmt_bytes(cache["approx_bytes"])],
                    ["Assets tracked", f"{runtime['asset_summary']['files']} files, {_fmt_bytes(runtime['asset_summary']['bytes'])}"],
                ],
            ),
            "",
            "## Pre-Commit Policy",
            "",
            "Pre-commit should stay fast and deterministic. The tracked hook runs syntax compilation plus static health snapshot generation. Full runtime checks should be run manually before a release or after changing gateway/sleep/auth behavior.",
            "",
            "Recommended commands:",
            "",
            "```bash",
            "scripts/pre_commit_check.sh",
            "scripts/project_health_snapshot.py --static-only --check",
            "scripts/project_health_snapshot.py --write",
            "```",
            "",
            "Install the optional local hook with:",
            "",
            "```bash",
            "scripts/install_git_hooks.sh",
            "```",
            "",
            "## Open Health Gaps",
            "",
            "- Add automated startup/wake latency sampling once a stable small default model is selected.",
            "- Add a repeatable auth/TLS matrix test for HTTP, HTTPS, auth-on, auth-off, localhost, and all-hosts modes.",
            "- Add a sleep/wake soak test that proves probes do not reset idle timers but real chat requests do.",
            "- Add a close/shutdown regression that verifies the GUI exits without orphaning `mlx_lm` servers.",
            "- Add client-sync smoke tests against LLM-OS and future opencode-style clients.",
            "",
            "## Interpretation",
            "",
            "The most important risk in this repo is not raw LOC. It is the state boundary between GUI, gateway, subprocess, settings, and external clients. Health work should prioritize anything that makes that boundary more observable and repeatable.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write docs/project_health_report.md")
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    parser.add_argument("--static-only", action="store_true", help="skip live gateway/process probes")
    parser.add_argument("--check", action="store_true", help="fail if snapshot cannot be generated or compile check fails")
    args = parser.parse_args()

    snapshot = build_snapshot(static_only=args.static_only)
    if args.write:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(render_markdown(snapshot), encoding="utf-8")
    if args.json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    elif not args.write:
        print(render_markdown(snapshot))

    if args.check and not snapshot["runtime"]["compile_check"]["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
