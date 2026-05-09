#!/usr/bin/env python3
"""
mlx_manager.py — GUI for managing the mlx-lm model server.

Run with:  /Users/agabriel/Documents/cs/mlx-llm/mlx-env/bin/python mlx_manager.py

One model runs at a time on a single port (default 9000).
Switching kills the current server before starting the next.
"""

import json
import os
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parent
REPO_PYTHON = REPO_ROOT / "mlx-env" / "bin" / "python"
BOOTSTRAP_PATH = REPO_ROOT / "mlx_manager_bootstrap.py"


def _run_bootstrap_and_relaunch(reason: str) -> None:
    if not BOOTSTRAP_PATH.exists():
        return
    if os.environ.get("MLX_MANAGER_SKIP_BOOTSTRAP") == "1":
        return
    os.environ["MLX_MANAGER_SKIP_BOOTSTRAP"] = "1"
    launcher = str(Path(sys.executable).resolve())
    os.execv(launcher, [launcher, str(BOOTSTRAP_PATH), "--launch", str(Path(__file__).resolve()), "--reason", reason])


if not REPO_PYTHON.exists():
    _run_bootstrap_and_relaunch("missing mlx-env runtime")

if os.environ.get("MLX_MANAGER_SKIP_REEXEC") != "1" and REPO_PYTHON.exists():
    try:
        current_python = Path(sys.executable).resolve()
        desired_python = REPO_PYTHON.resolve()
        if current_python != desired_python:
            os.environ["MLX_MANAGER_SKIP_REEXEC"] = "1"
            os.execv(str(desired_python), [str(desired_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    except Exception:
        pass

try:
    from huggingface_hub import HfApi, snapshot_download
except Exception:
    HfApi = None
    snapshot_download = None

try:
    from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
    from PyQt6.QtGui import QColor, QFont, QIcon, QPalette, QTextCursor
    from PyQt6.QtWidgets import (
        QApplication, QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
        QLineEdit, QMessageBox, QPushButton, QRadioButton, QTextEdit, QVBoxLayout, QWidget,
    )
except Exception:
    _run_bootstrap_and_relaunch("missing PyQt6 dependency")
    raise

# ── Colours (matches llm-os dark theme) ───────────────────────────────────────

BG     = "#0d1117"
SURF   = "#161b26"
SURF2  = "#1e2435"
BORDER = "#252d3d"
ACC    = "#7c6af7"
ACC2   = "#4ecdc4"
TEXT   = "#dde1ec"
MUTED  = "#6b7280"
OK     = "#22c55e"
WARN   = "#f59e0b"
ERR    = "#ef4444"

DEFAULT_PORT = 9000
DEFAULT_HOST = "0.0.0.0"
LOCALHOST_HOST = "127.0.0.1"
INTERNAL_HOST = "127.0.0.1"
INTERNAL_PORT_OFFSET = 1
DEFAULT_IDLE_MINUTES = 15
DEFAULT_IDLE_ENABLED = True
DEFAULT_SLEEP_MODE = "light"
DEFAULT_AUTH_ENABLED = True
GATEWAY_POLL_INTERVAL_S = 120.0
GATEWAY_STATS_WINDOW_S = 300.0
IDLE_WATCHDOG_INTERVAL_S = 30.0
STARTUP_TIMEOUT_S = 90
STARTUP_REQUEST_TIMEOUT_S = 25
INTERNAL_FORWARD_TIMEOUT_S = 45
INTERNAL_READY_TIMEOUT_S = 20
GENERATION_PROBE_TIMEOUT_S = 12
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_MAX_TOKENS_PER_REQUEST = 4096
DEFAULT_GATEWAY_SCHEME = "http"
AVAILABLE_MODEL_LIMIT = 200
HF_ORG = "mlx-community"
LLM_OS_API_BASE = os.environ.get("LLM_OS_API_BASE", "http://localhost:8080")
DEFAULT_GATEWAY_TOKEN = "sk-mytoken"
DEFAULT_LLM_OS_LOCAL_BASE_URL = os.environ.get("LLM_OS_LOCAL_BASE_URL")
ICON_PATH = Path(__file__).parent / "assets" / "mlx_gui_icon.png"
SETTINGS_PATH = REPO_ROOT / ".mlx_manager.json"
CERTS_DIR = REPO_ROOT / "certs"
DEFAULT_GATEWAY_CERT_PATH = CERTS_DIR / "mlx-manager-gateway.crt"
DEFAULT_GATEWAY_KEY_PATH = CERTS_DIR / "mlx-manager-gateway.key"

# ── Model discovery ───────────────────────────────────────────────────────────

def _hf_cache_dir() -> Path:
    if hf := os.environ.get("HF_HOME"):
        return Path(hf) / "hub"
    if hf := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(hf)
    return Path("~/.cache/huggingface/hub").expanduser()


def scan_cached_models() -> List[Dict]:
    """Return [{id, path, size_gb}] for cached mlx-community models."""
    cache = _hf_cache_dir()
    results = []
    if not cache.exists():
        return results
    for d in sorted(cache.iterdir()):
        name = d.name
        if not name.startswith("models--"):
            continue
        model_id = name.replace("models--", "", 1).replace("--", "/")
        try:
            snap = next((d / "snapshots").iterdir())
            size_b = sum(f.stat().st_size for f in snap.rglob("*") if f.is_file())
            size_gb = size_b / 1e9
        except Exception:
            size_gb = 0.0
        results.append({"id": model_id, "path": str(d), "size_gb": size_gb})
    return results


# ── mlx_lm binary ─────────────────────────────────────────────────────────────

def find_mlx_binary() -> Optional[str]:
    if REPO_PYTHON.exists():
        return f"{REPO_PYTHON} -m mlx_lm"
    candidates = [
        Path(__file__).parent / "mlx-env" / "bin" / "mlx_lm",
        Path(sys.prefix) / "bin" / "mlx_lm",
        Path("~/venvs/mlx/bin/mlx_lm").expanduser(),
        Path("~/.venvs/mlx/bin/mlx_lm").expanduser(),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    try:
        import mlx_lm  # noqa: F401
        return f"{sys.executable} -m mlx_lm"
    except Exception:
        pass
    try:
        r = subprocess.run(["which", "mlx_lm"], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def build_mlx_server_cmd(model: str, host: str, port: int) -> List[str]:
    if REPO_PYTHON.exists():
        return [str(REPO_PYTHON), "-m", "mlx_lm", "server", "--model", model, "--host", host, "--port", str(port), "--log-level", "INFO"]
    binary = Path(__file__).parent / "mlx-env" / "bin" / "mlx_lm"
    if binary.exists():
        return [str(binary), "server", "--model", model, "--host", host, "--port", str(port), "--log-level", "INFO"]
    return [sys.executable, "-m", "mlx_lm", "server", "--model", model, "--host", host, "--port", str(port), "--log-level", "INFO"]


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        settings = json.loads(SETTINGS_PATH.read_text()) or {}
        if "host_mode_user_set" not in settings:
            settings["host_mode"] = "all"
        if "auth_enabled" not in settings:
            settings["auth_enabled"] = DEFAULT_AUTH_ENABLED
        if "auth_token" not in settings:
            settings["auth_token"] = DEFAULT_GATEWAY_TOKEN
        if "idle_enabled" not in settings:
            settings["idle_enabled"] = DEFAULT_IDLE_ENABLED
        if "idle_minutes" not in settings:
            settings["idle_minutes"] = DEFAULT_IDLE_MINUTES
        if "sleep_mode" not in settings:
            settings["sleep_mode"] = DEFAULT_SLEEP_MODE
        if "request_timeout_seconds" not in settings:
            settings["request_timeout_seconds"] = DEFAULT_REQUEST_TIMEOUT_SECONDS
        if "max_tokens_per_request" not in settings:
            settings["max_tokens_per_request"] = DEFAULT_MAX_TOKENS_PER_REQUEST
        if "gateway_scheme" not in settings:
            settings["gateway_scheme"] = DEFAULT_GATEWAY_SCHEME
        if "gateway_cert_file" not in settings:
            settings["gateway_cert_file"] = str(DEFAULT_GATEWAY_CERT_PATH)
        if "gateway_key_file" not in settings:
            settings["gateway_key_file"] = str(DEFAULT_GATEWAY_KEY_PATH)
        if "client_verify_ssl" not in settings:
            settings["client_verify_ssl"] = False
        return settings
    except Exception:
        return {}


def save_settings(settings: dict):
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2, sort_keys=True))
        os.chmod(SETTINGS_PATH, 0o600)
    except Exception:
        pass


def _path_from_setting(value: Optional[Any], default: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return default
    return Path(raw).expanduser()


def ensure_gateway_tls_files(cert_file: Any, key_file: Any) -> tuple[Path, Path]:
    """Ensure a usable local TLS certificate/key pair exists for the gateway.

    The default path is generated as a local self-signed development cert. If a
    custom cert/key path is supplied, both files must already exist so we do not
    accidentally overwrite operator-managed material.
    """
    cert_path = Path(cert_file).expanduser()
    key_path = Path(key_file).expanduser()
    using_default_paths = (
        cert_path.resolve() == DEFAULT_GATEWAY_CERT_PATH.resolve()
        and key_path.resolve() == DEFAULT_GATEWAY_KEY_PATH.resolve()
    )
    if cert_path.exists() and key_path.exists():
        return cert_path, key_path
    if not using_default_paths:
        missing = [str(p) for p in (cert_path, key_path) if not p.exists()]
        raise FileNotFoundError("Missing TLS file(s): " + ", ".join(missing))

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False) as cfg:
        cfg.write(
            "\n".join([
                "[req]",
                "default_bits = 2048",
                "prompt = no",
                "distinguished_name = dn",
                "x509_extensions = v3_req",
                "",
                "[dn]",
                "CN = localhost",
                "",
                "[v3_req]",
                "subjectAltName = @alt_names",
                "",
                "[alt_names]",
                "DNS.1 = localhost",
                "DNS.2 = host.docker.internal",
                "IP.1 = 127.0.0.1",
            ])
        )
        cfg_path = cfg.name
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-nodes", "-newkey", "rsa:2048",
                "-sha256", "-days", "3650",
                "-keyout", str(key_path),
                "-out", str(cert_path),
                "-config", cfg_path,
                "-extensions", "v3_req",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        os.chmod(key_path, 0o600)
        os.chmod(cert_path, 0o644)
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass
    return cert_path, key_path


def create_gateway_ssl_context(cert_file: Any, key_file: Any) -> ssl.SSLContext:
    cert_path, key_path = ensure_gateway_tls_files(cert_file, key_file)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    return context


def startup_diagnostics(mlx_binary: Optional[str]) -> List[str]:
    lines = [
        f"Repo: {REPO_ROOT}",
        f"Runtime: {'ok' if REPO_PYTHON.exists() else 'missing'} — {REPO_PYTHON}",
        f"Bootstrap helper: {'present' if BOOTSTRAP_PATH.exists() else 'missing'} — {BOOTSTRAP_PATH}",
        f"mlx launcher: {mlx_binary or 'missing'}",
        f"huggingface_hub: {'available' if HfApi is not None and snapshot_download is not None else 'missing'}",
        f"Settings: {SETTINGS_PATH}",
    ]
    return lines


# ── Server helpers ────────────────────────────────────────────────────────────

def get_server_models(port: int, host: str = "localhost", auth_token: str = "") -> list[str]:
    """Returns the model IDs advertised by the server, or an empty list on failure."""
    try:
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        req = urllib.request.Request(f"http://{host}:{port}/v1/models", headers=headers)
        with urllib.request.urlopen(req, timeout=2) as r:
            data = json.loads(r.read())
            models = data.get("data", [])
            return [m.get("id") for m in models if m.get("id")]
    except Exception:
        return []


def check_server(port: int, host: str = "localhost", auth_token: str = "", preferred_model: str = "") -> Optional[str]:
    """Returns the preferred advertised model ID when available, else the first advertised model."""
    models = get_server_models(port, host=host, auth_token=auth_token)
    if not models:
        return None
    if preferred_model and preferred_model in models:
        return preferred_model
    return models[0]


def check_generation_ready(port: int, host: str = "localhost", auth_token: str = "", preferred_model: str = "") -> Optional[str]:
    """Returns the preferred advertised model ID only after a tiny chat completion succeeds."""
    model_id = check_server(port, host=host, auth_token=auth_token, preferred_model=preferred_model)
    if not model_id:
        return None
    try:
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        payload = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with exactly: READY"}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            f"http://{host}:{port}/v1/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=GENERATION_PROBE_TIMEOUT_S) as r:
            data = json.loads(r.read())
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return model_id if content else None
    except Exception:
        return None


def kill_all_servers(log_fn=None):
    """SIGTERM all mlx_lm server processes system-wide."""
    r = subprocess.run(["pgrep", "-f", "mlx_lm server"], capture_output=True, text=True)
    pids = r.stdout.strip().split()
    if pids:
        subprocess.run(["kill", "-TERM"] + pids, capture_output=True)
        if log_fn:
            log_fn(f"  Sent TERM to {len(pids)} process(es): {', '.join(pids)}")
        time.sleep(1.5)


def sync_llm_os_local_model(
    model_id: str,
    base_url: str,
    require_auth: bool = False,
    auth_token: str = "",
    verify_ssl: bool = True,
    ca_bundle: str = "",
    log_fn=None,
) -> bool:
    """Update llm-os live config so it points at the MLX gateway."""
    try:
        with urllib.request.urlopen(f"{LLM_OS_API_BASE}/api/config/llm", timeout=5) as resp:
            cfg = json.loads(resp.read().decode())
    except Exception as exc:
        if log_fn:
            log_fn(f"  llm-os sync skipped: could not read config ({exc})")
        return False

    local = cfg.get("local") or {}
    roles = cfg.get("roles") or {}
    local["enabled"] = True
    local["backend"] = "mlx_lm"
    local["base_url"] = base_url
    local["scheme"] = urlsplit(base_url).scheme or DEFAULT_GATEWAY_SCHEME
    local["verify_ssl"] = bool(verify_ssl)
    local["ca_bundle"] = ca_bundle
    local["model"] = model_id
    local["auth_enabled"] = bool(require_auth and auth_token)
    local["api_key"] = auth_token if (require_auth and auth_token) else ""
    local.setdefault("timeout", 180)
    roles["primary_local_model"] = model_id
    cfg["local"] = local
    cfg["roles"] = roles

    payload = json.dumps(cfg).encode()
    req = urllib.request.Request(
        f"{LLM_OS_API_BASE}/api/config/llm",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            json.loads(resp.read().decode())
        if log_fn:
            mode = "secured" if (require_auth and auth_token) else "open"
            log_fn(f"  llm-os synced: {model_id} via {base_url} ({mode} gateway)")
        return True
    except Exception as exc:
        if log_fn:
            log_fn(f"  llm-os sync failed: {exc}")
        return False


# ── Signal bridge (worker threads → Qt main thread) ───────────────────────────

class _Bridge(QObject):
    log_line   = pyqtSignal(str)
    status_set = pyqtSignal(str, str)   # state, label
    hf_models_loaded = pyqtSignal(object)
    cached_models_loaded = pyqtSignal(object)
    setup_refresh = pyqtSignal()
    activity_pulse = pyqtSignal()
    idle_timer_stop = pyqtSignal()
    idle_timeout_request = pyqtSignal()
    usage_refresh = pyqtSignal()
    close_finished = pyqtSignal()


# ── Main window ───────────────────────────────────────────────────────────────

class MLXManagerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MLX-LM Model Manager")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1180, 720)
        self.setMinimumSize(960, 600)

        self._bridge       = _Bridge()
        self._bridge.log_line.connect(self._append_log)
        self._bridge.status_set.connect(self._apply_status)
        self._bridge.hf_models_loaded.connect(self._set_available_models)
        self._bridge.cached_models_loaded.connect(self._set_cached_models)
        self._bridge.setup_refresh.connect(self._refresh_setup_status)
        self._bridge.activity_pulse.connect(self._apply_activity_pulse)
        self._bridge.idle_timer_stop.connect(self._stop_idle_timer)
        self._bridge.idle_timeout_request.connect(self._on_idle_timeout)
        self._bridge.usage_refresh.connect(self._refresh_usage_status)
        self._bridge.close_finished.connect(self._finish_close)

        self._settings     = load_settings()
        self.mlx_binary    = find_mlx_binary()
        self.models        = scan_cached_models()
        self.server_proc   = None
        self.gateway_server = None
        self.gateway_thread = None
        self._gateway_port = None
        self._gateway_bound_host = None
        self._gateway_bound_scheme = None
        self._gateway_tls_self_signed = False
        self._status       = "unknown"
        self._running_id   = None
        self._current_model = None
        self._sleeping     = False
        self._generation_ready = False
        self._deep_sleeping = False
        self._status_changed_at = ""
        self._started_at = time.time()
        self._last_activity = time.time()
        self._last_real_question_at = 0.0
        self._last_real_question = ""
        self._token_stats = {
            "real_questions": 0,
            "chat_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "exact_tokens": 0,
            "last_total_tokens": 0,
            "last_token_source": "none",
        }
        self._last_synced_model = None
        self._last_synced_state = None
        self._last_synced_signature = None
        self._startup_event = threading.Event()
        self._startup_event.set()
        self._startup_thread = None
        self._startup_error = None
        self._startup_model = None
        self._startup_started_at = 0.0
        self._shutdown_requested = False
        self._close_in_progress = False
        self._close_finalizing = False
        self._idle_watchdog_thread = None
        self._expected_exit_pid = None
        self._expected_exit_reason = ""
        self._gateway_request_times = deque()
        self._gateway_lock = threading.RLock()
        self._server_lock  = threading.RLock()
        self._stats_lock = threading.RLock()
        self._radio_group  = QButtonGroup(self)
        self._radio_map    = {}   # model_id → QRadioButton
        self._dot_map      = {}   # model_id → QLabel
        self._available_models = []
        self._preferred_cached_selection = None
        self._hf_api = HfApi() if HfApi is not None else None

        self._build_ui()
        self._setup_timers()
        self._start_idle_watchdog()
        self._ensure_gateway_running()
        QTimer.singleShot(250, self._on_refresh_available_models)

        self._append_log(f"mlx_lm binary: {self.mlx_binary or '⚠ NOT FOUND'}")
        self._append_log(f"HF cache: {_hf_cache_dir()}")
        self._append_log(f"Found {len(self.models)} cached model(s)")

        # Pre-select running model, then last running model, then saved selection
        if self.models:
            running = check_server(self._port(), auth_token=self._auth_token())
            last_running = self._settings.get("last_running_model")
            saved_model = self._settings.get("selected_model")
            pre = running or last_running or saved_model or self.models[0]["id"]
            if pre in self._radio_map:
                self._radio_map[pre].setChecked(True)
            if last_running:
                self._current_model = last_running

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setStyleSheet(f"background-color: {BG}; color: {TEXT};")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        hdr = QWidget()
        hdr.setStyleSheet(f"background-color: {SURF};")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(20, 12, 20, 12)

        title = QLabel("🧠  MLX-LM Model Manager")
        title.setFont(QFont("Helvetica", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT}; background: transparent;")

        status_wrap = QWidget()
        status_wrap.setStyleSheet("background: transparent;")
        status_layout = QVBoxLayout(status_wrap)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(2)

        self._status_lbl = QLabel("◌ Checking")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._status_lbl.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
        self._status_lbl.setMinimumWidth(220)

        self._status_meta_lbl = QLabel("")
        self._status_meta_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._status_meta_lbl.setFont(QFont("Menlo", 9))
        self._status_meta_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")

        self._header_tokens_lbl = QLabel("Tokens: 0")
        self._header_tokens_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._header_tokens_lbl.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
        self._header_tokens_lbl.setStyleSheet(f"color: {ACC2}; background: transparent;")

        status_layout.addWidget(self._status_lbl, alignment=Qt.AlignmentFlag.AlignRight)
        status_layout.addWidget(self._status_meta_lbl, alignment=Qt.AlignmentFlag.AlignRight)
        status_layout.addWidget(self._header_tokens_lbl, alignment=Qt.AlignmentFlag.AlignRight)

        hdr_layout.addWidget(title)
        hdr_layout.addStretch()
        hdr_layout.addWidget(status_wrap)
        root_layout.addWidget(hdr)
        root_layout.addWidget(self._divider())

        body = QWidget()
        body.setStyleSheet(f"background-color: {BG};")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(20, 16, 20, 16)
        body_layout.setSpacing(16)

        log_container = QWidget()
        log_container.setMinimumWidth(400)
        log_container.setStyleSheet(f"background: {BG};")
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(6)

        log_lbl = QLabel("LOG")
        log_lbl.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
        log_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        log_layout.addWidget(log_lbl)

        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setFont(QFont("Menlo", 10))
        self._log_box.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._log_box.setStyleSheet(
            f"background: {SURF}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 8px;"
        )
        log_layout.addWidget(self._log_box, stretch=1)
        body_layout.addWidget(log_container, stretch=2)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        lbl = QLabel("CACHED MODELS")
        lbl.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        left_layout.addWidget(lbl)

        self._list_frame = QFrame()
        self._list_frame.setStyleSheet(
            f"background-color: {SURF2}; border-radius: 6px; padding: 8px;"
        )
        self._list_inner = QVBoxLayout(self._list_frame)
        self._list_inner.setContentsMargins(10, 8, 10, 8)
        self._list_inner.setSpacing(4)
        left_layout.addWidget(self._list_frame, stretch=1)
        body_layout.addWidget(left, stretch=1)

        ctrl = QWidget()
        ctrl.setFixedWidth(320)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(6)

        ctrl_hdr = QLabel("CONTROLS")
        ctrl_hdr.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
        ctrl_hdr.setStyleSheet(f"color: {MUTED}; background: transparent;")
        ctrl_layout.addWidget(ctrl_hdr)

        host_lbl = QLabel("Gateway Host")
        host_lbl.setFont(QFont("Helvetica", 10))
        host_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        ctrl_layout.addWidget(host_lbl)

        self._host_mode_combo = QComboBox()
        self._host_mode_combo.setEditable(False)
        self._host_mode_combo.setFont(QFont("Menlo", 10))
        self._host_mode_combo.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 4px 8px;"
        )
        self._host_mode_combo.addItem("All Interfaces", "all")
        self._host_mode_combo.addItem("Localhost Only", "localhost")
        host_mode = self._settings.get("host_mode", "all")
        idx = self._host_mode_combo.findData(host_mode)
        self._host_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._host_mode_combo.currentIndexChanged.connect(lambda _idx: self._on_gateway_config_changed(restart_gateway=True))
        ctrl_layout.addWidget(self._host_mode_combo)

        port_lbl = QLabel("Gateway Port")
        port_lbl.setFont(QFont("Helvetica", 10))
        port_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        ctrl_layout.addWidget(port_lbl)

        self._port_entry = QLineEdit(str(self._settings.get("port", DEFAULT_PORT)))
        self._port_entry.setFont(QFont("Menlo", 12))
        self._port_entry.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 4px 8px;"
        )
        self._port_entry.editingFinished.connect(lambda: self._on_gateway_config_changed(restart_gateway=True))
        ctrl_layout.addWidget(self._port_entry)

        protocol_lbl = QLabel("Gateway Protocol")
        protocol_lbl.setFont(QFont("Helvetica", 10))
        protocol_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        ctrl_layout.addWidget(protocol_lbl)

        self._gateway_scheme_combo = QComboBox()
        self._gateway_scheme_combo.setEditable(False)
        self._gateway_scheme_combo.setFont(QFont("Menlo", 10))
        self._gateway_scheme_combo.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 4px 8px;"
        )
        self._gateway_scheme_combo.addItem("HTTP", "http")
        self._gateway_scheme_combo.addItem("HTTPS / TLS", "https")
        scheme = self._settings.get("gateway_scheme", DEFAULT_GATEWAY_SCHEME)
        idx = self._gateway_scheme_combo.findData(scheme)
        self._gateway_scheme_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._gateway_scheme_combo.currentIndexChanged.connect(lambda _idx: self._on_gateway_config_changed(restart_gateway=True))
        ctrl_layout.addWidget(self._gateway_scheme_combo)

        self._auth_toggle = QCheckBox("Require gateway token")
        self._auth_toggle.setChecked(bool(self._settings.get("auth_enabled", False)))
        self._auth_toggle.setStyleSheet(f"color: {TEXT};")
        self._auth_toggle.stateChanged.connect(lambda _state: self._on_gateway_config_changed())
        ctrl_layout.addWidget(self._auth_toggle)

        self._auth_token_entry = QLineEdit(self._settings.get("auth_token", DEFAULT_GATEWAY_TOKEN))
        self._auth_token_entry.setPlaceholderText("Bearer token")
        self._auth_token_entry.setFont(QFont("Menlo", 11))
        self._auth_token_entry.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 4px 8px;"
        )
        self._auth_token_entry.editingFinished.connect(lambda: self._on_gateway_config_changed())
        ctrl_layout.addWidget(self._auth_token_entry)

        self._client_verify_ssl_toggle = QCheckBox("Clients verify TLS cert")
        self._client_verify_ssl_toggle.setChecked(bool(self._settings.get("client_verify_ssl", False)))
        self._client_verify_ssl_toggle.setStyleSheet(f"color: {TEXT};")
        self._client_verify_ssl_toggle.stateChanged.connect(lambda _state: self._on_gateway_config_changed())
        ctrl_layout.addWidget(self._client_verify_ssl_toggle)

        self._gateway_cert_entry = QLineEdit(str(self._settings.get("gateway_cert_file", DEFAULT_GATEWAY_CERT_PATH)))
        self._gateway_cert_entry.setPlaceholderText("TLS certificate path")
        self._gateway_cert_entry.setFont(QFont("Menlo", 9))
        self._gateway_cert_entry.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 4px 8px;"
        )
        self._gateway_cert_entry.editingFinished.connect(lambda: self._on_gateway_config_changed(restart_gateway=True))
        ctrl_layout.addWidget(self._gateway_cert_entry)

        self._gateway_key_entry = QLineEdit(str(self._settings.get("gateway_key_file", DEFAULT_GATEWAY_KEY_PATH)))
        self._gateway_key_entry.setPlaceholderText("TLS private key path")
        self._gateway_key_entry.setFont(QFont("Menlo", 9))
        self._gateway_key_entry.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 4px 8px;"
        )
        self._gateway_key_entry.editingFinished.connect(lambda: self._on_gateway_config_changed(restart_gateway=True))
        ctrl_layout.addWidget(self._gateway_key_entry)

        self._idle_toggle = QCheckBox("Sleep when idle")
        self._idle_toggle.setChecked(bool(self._settings.get("idle_enabled", DEFAULT_IDLE_ENABLED)))
        self._idle_toggle.setStyleSheet(f"color: {TEXT};")
        self._idle_toggle.stateChanged.connect(lambda _state: self._on_idle_config_changed())
        ctrl_layout.addWidget(self._idle_toggle)

        idle_row = QHBoxLayout()
        idle_lbl = QLabel("Idle min")
        idle_lbl.setFont(QFont("Helvetica", 10))
        idle_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        idle_row.addWidget(idle_lbl)
        self._idle_minutes_entry = QLineEdit(str(self._settings.get("idle_minutes", DEFAULT_IDLE_MINUTES)))
        self._idle_minutes_entry.setFont(QFont("Menlo", 11))
        self._idle_minutes_entry.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 4px 8px;"
        )
        self._idle_minutes_entry.editingFinished.connect(self._on_idle_config_changed)
        idle_row.addWidget(self._idle_minutes_entry)
        ctrl_layout.addLayout(idle_row)

        sleep_mode_row = QHBoxLayout()
        sleep_mode_lbl = QLabel("Sleep Mode")
        sleep_mode_lbl.setFont(QFont("Helvetica", 10))
        sleep_mode_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        sleep_mode_row.addWidget(sleep_mode_lbl)
        self._sleep_mode_combo = QComboBox()
        self._sleep_mode_combo.setEditable(False)
        self._sleep_mode_combo.setFont(QFont("Menlo", 10))
        self._sleep_mode_combo.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 4px 8px;"
        )
        self._sleep_mode_combo.addItem("Light (wake on request)", "light")
        self._sleep_mode_combo.addItem("Deep (manual wake)", "deep")
        sleep_mode = self._settings.get("sleep_mode", DEFAULT_SLEEP_MODE)
        idx = self._sleep_mode_combo.findData(sleep_mode)
        self._sleep_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._sleep_mode_combo.currentIndexChanged.connect(lambda _idx: self._on_idle_config_changed())
        sleep_mode_row.addWidget(self._sleep_mode_combo)
        ctrl_layout.addLayout(sleep_mode_row)

        ctrl_layout.addWidget(self._divider())

        usage_lbl = QLabel("IDLE / TOKEN USE")
        usage_lbl.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
        usage_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        ctrl_layout.addWidget(usage_lbl)

        self._usage_status_frame = QFrame()
        self._usage_status_frame.setStyleSheet(
            f"background: {SURF2}; border: 1px solid {BORDER};"
            " border-radius: 6px; padding: 6px;"
        )
        usage_layout = QVBoxLayout(self._usage_status_frame)
        usage_layout.setContentsMargins(8, 6, 8, 6)
        usage_layout.setSpacing(4)

        self._idle_state_lbl = QLabel("")
        self._idle_state_lbl.setFont(QFont("Helvetica", 10, QFont.Weight.Bold))
        self._idle_state_lbl.setWordWrap(True)
        self._idle_state_lbl.setStyleSheet(f"color: {TEXT}; background: transparent;")
        usage_layout.addWidget(self._idle_state_lbl)

        self._usage_tokens_lbl = QLabel("")
        self._usage_tokens_lbl.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
        self._usage_tokens_lbl.setWordWrap(True)
        self._usage_tokens_lbl.setStyleSheet(f"color: {ACC2}; background: transparent;")
        usage_layout.addWidget(self._usage_tokens_lbl)

        self._last_question_lbl = QLabel("")
        self._last_question_lbl.setFont(QFont("Menlo", 9))
        self._last_question_lbl.setWordWrap(True)
        self._last_question_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        usage_layout.addWidget(self._last_question_lbl)

        ctrl_layout.addWidget(self._usage_status_frame)

        ctrl_layout.addWidget(self._divider())

        hub_lbl = QLabel("HUGGING FACE")
        hub_lbl.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
        hub_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        ctrl_layout.addWidget(hub_lbl)

        self._hf_token_entry = QLineEdit(self._settings.get("hf_token", self._default_hf_token()))
        self._hf_token_entry.setPlaceholderText("HF token (optional, uses env if blank)")
        self._hf_token_entry.setEchoMode(QLineEdit.EchoMode.Password)
        self._hf_token_entry.setFont(QFont("Menlo", 11))
        self._hf_token_entry.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 4px 8px;"
        )
        ctrl_layout.addWidget(self._hf_token_entry)

        self._hf_model_combo = QComboBox()
        self._hf_model_combo.setEditable(False)
        self._hf_model_combo.setFont(QFont("Menlo", 10))
        self._hf_model_combo.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 4px 8px;"
        )
        self._hf_model_combo.addItem("Loading mlx-community models…", "")
        ctrl_layout.addWidget(self._hf_model_combo)

        self._hf_refresh_btn = self._make_btn("⇣  Refresh MLX Models", self._on_refresh_available_models, bg=SURF2, fg=ACC2)
        ctrl_layout.addWidget(self._hf_refresh_btn)

        self._hf_download_btn = self._make_btn("⬇  Download Selected", self._on_download_selected_model, bg=SURF2, fg=OK)
        ctrl_layout.addWidget(self._hf_download_btn)

        ctrl_layout.addWidget(self._divider())

        setup_lbl = QLabel("SETUP STATUS")
        setup_lbl.setFont(QFont("Helvetica", 9, QFont.Weight.Bold))
        setup_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
        ctrl_layout.addWidget(setup_lbl)

        self._setup_status_box = QTextEdit()
        self._setup_status_box.setReadOnly(True)
        self._setup_status_box.setFont(QFont("Menlo", 9))
        self._setup_status_box.setStyleSheet(
            f"background: {SURF2}; color: {TEXT}; border: none;"
            f" border-radius: 4px; padding: 6px;"
        )
        self._setup_status_box.setFixedHeight(100)
        ctrl_layout.addWidget(self._setup_status_box)

        self._repair_btn = self._make_btn("🛠  Repair Runtime", self._on_repair_runtime, bg=SURF2, fg=ACC2)
        ctrl_layout.addWidget(self._repair_btn)

        ctrl_layout.addWidget(self._divider())

        self._start_btn = self._make_btn("▶  Load & Start", self._on_start, bg=ACC, fg="#ffffff")
        ctrl_layout.addWidget(self._start_btn)

        self._stop_btn = self._make_btn("■  Stop All", self._on_stop, bg=SURF2, fg=ERR)
        ctrl_layout.addWidget(self._stop_btn)

        self._refresh_btn = self._make_btn("↺  Refresh Cache", self._on_refresh, bg=SURF2, fg=ACC2)
        ctrl_layout.addWidget(self._refresh_btn)
        ctrl_layout.addStretch()

        if self._hf_api is None or snapshot_download is None:
            self._hf_model_combo.clear()
            self._hf_model_combo.addItem("huggingface_hub not available", "")
            self._hf_model_combo.setEnabled(False)
            self._hf_refresh_btn.setEnabled(False)
            self._hf_download_btn.setEnabled(False)

        body_layout.addWidget(ctrl)
        root_layout.addWidget(body, stretch=1)
        self._set_cached_models(self.models)
        self._refresh_setup_status()
        self._refresh_usage_status()

    def _make_btn(self, text, cmd, bg=SURF2, fg=TEXT):
        btn = QPushButton(text)
        btn.setFont(QFont("Helvetica", 11))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: none;
                border-radius: 5px;
                padding: 8px 12px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {BORDER};
            }}
            QPushButton:pressed {{
                background-color: {SURF};
            }}
        """)
        btn.clicked.connect(cmd)
        return btn

    def _divider(self, margin=0):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background: {BORDER}; border: none; max-height: 1px;")
        if margin:
            line.setContentsMargins(margin, 0, margin, 0)
        return line

    # ── Actions ────────────────────────────────────────────────────────────────

    def _on_start(self):
        mid = self._selected_model()
        if not mid:
            QMessageBox.critical(self, "No model selected", "Select a model from the list.")
            return
        if not self.mlx_binary:
            QMessageBox.critical(
                self, "mlx_lm not found",
                "Cannot locate a working mlx_lm launcher.\n\n"
                "Expected either an mlx_lm executable or a working python -m mlx_lm installation in mlx-env."
            )
            return
        if self._auth_enabled() and not self._auth_token():
            QMessageBox.critical(self, "Token required", "Enter a bearer token or disable gateway auth.")
            return
        if not self._ensure_gateway_running():
            return

        self._current_model = mid
        self._persist_settings()
        self._sleeping = False
        self._append_log(f"\n{'─'*50}")
        self._append_log(f"▶ Starting: {mid}  (gateway {self._gateway_scheme()} port {self._port()} → internal {self._internal_port()})")

        def _worker():
            self._ensure_gateway_running()
            self._ensure_server_running(reason="manual start")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_stop(self):
        self._apply_status("stopping", "Stopping…")
        self._append_log("\n■ Stopping active mlx_lm model…")

        def _worker():
            self._stop_model_server(reason="manual stop", sleeping=False)
            selected = self._selected_model() or self._current_model or self._running_id
            if selected:
                sync_llm_os_local_model(
                    selected,
                    self._gateway_client_base_url(),
                    require_auth=self._auth_enabled(),
                    auth_token=self._auth_token(),
                    verify_ssl=self._client_verify_ssl(),
                    log_fn=self._bridge.log_line.emit,
                )
            self._bridge.log_line.emit("  Gateway remains ready for wake-on-request.")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_refresh(self):
        self._set_cached_models(scan_cached_models())
        self._persist_settings()
        self._append_log(f"↺ Refreshed: {len(self.models)} model(s) in cache")
        self._ensure_gateway_running()

    def _on_gateway_config_changed(self, restart_gateway: bool = False):
        """Persist gateway settings and push the effective connection to llm-os."""
        self._persist_settings()
        self._last_synced_signature = None
        if restart_gateway:
            self._shutdown_gateway()
            self._ensure_gateway_running()
        self._sync_current_llm_os_state(force=True, reason="gateway config change")
        self._refresh_usage_status()

    def _on_refresh_available_models(self):
        if self._hf_api is None:
            self._append_log("  Hugging Face browser unavailable: huggingface_hub is not installed")
            return
        token = self._hf_token()
        self._persist_settings()
        self._hf_model_combo.clear()
        self._hf_model_combo.addItem("Loading mlx-community models…", "")
        self._append_log(f"⇣ Loading available models from {HF_ORG}…")

        def _worker():
            try:
                models = []
                for info in self._hf_api.list_models(author=HF_ORG, limit=AVAILABLE_MODEL_LIMIT, sort="downloads", direction=-1, token=token or None):
                    model_id = getattr(info, "id", "")
                    if model_id.startswith(f"{HF_ORG}/"):
                        models.append(model_id)
                self._bridge.hf_models_loaded.emit(sorted(set(models), key=lambda mid: (mid.split('/')[-1].lower(), mid.lower())))
                self._bridge.log_line.emit(f"  Loaded {len(models)} available models from {HF_ORG}")
            except Exception as exc:
                self._bridge.log_line.emit(f"  Hugging Face browse failed: {exc}")
                self._bridge.hf_models_loaded.emit([])

        threading.Thread(target=_worker, daemon=True).start()

    def _on_download_selected_model(self):
        if snapshot_download is None:
            QMessageBox.critical(self, "Missing dependency", "huggingface_hub is not available in mlx-env.")
            return
        model_id = self._hf_model_combo.currentData() or self._hf_model_combo.currentText().strip()
        if not model_id:
            QMessageBox.critical(self, "No model selected", "Choose a model from the Hugging Face list first.")
            return
        token = self._hf_token()
        self._preferred_cached_selection = model_id
        self._persist_settings()
        self._append_log(f"⬇ Downloading {model_id}…")

        def _worker():
            try:
                local_path = snapshot_download(repo_id=model_id, token=token or None, repo_type="model")
                self._bridge.log_line.emit(f"  Download complete: {model_id}")
                self._bridge.log_line.emit(f"  Snapshot: {local_path}")
                self._bridge.cached_models_loaded.emit(scan_cached_models())
            except Exception as exc:
                self._bridge.log_line.emit(f"  Download failed for {model_id}: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_repair_runtime(self):
        if not BOOTSTRAP_PATH.exists():
            QMessageBox.critical(self, "Bootstrap missing", f"Could not find bootstrap helper:\n{BOOTSTRAP_PATH}")
            return
        self._append_log("🛠 Repairing mlx_manager runtime…")

        def _worker():
            cmd = [sys.executable, str(BOOTSTRAP_PATH), "--ensure-only", "--reason", "manual repair"]
            self._bridge.log_line.emit(f"  cmd: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._bridge.log_line.emit(line)
            finally:
                rc = proc.wait()
                self._bridge.log_line.emit(f"  Runtime repair exited with code {rc}")
                self.mlx_binary = find_mlx_binary()
                self._bridge.log_line.emit(f"  mlx_lm binary: {self.mlx_binary or '⚠ NOT FOUND'}")
                self._bridge.setup_refresh.emit()

        threading.Thread(target=_worker, daemon=True).start()

    # ── Timers ─────────────────────────────────────────────────────────────────

    def _setup_timers(self):
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._on_idle_timeout)

    def _start_idle_watchdog(self):
        if self._idle_watchdog_thread and self._idle_watchdog_thread.is_alive():
            return

        def _watchdog():
            while not self._shutdown_requested:
                time.sleep(IDLE_WATCHDOG_INTERVAL_S)
                if self._shutdown_requested:
                    break
                try:
                    enabled, idle_seconds, _sleep_mode = self._idle_config_snapshot()
                    if not enabled or self._sleeping or self._deep_sleeping:
                        continue
                    if not self._server_process_active():
                        continue
                    reference = self._idle_reference_time()
                    if reference and (time.time() - reference) >= idle_seconds:
                        self._bridge.idle_timeout_request.emit()
                except Exception as exc:
                    if self._shutdown_requested:
                        break
                    try:
                        self._bridge.log_line.emit(f"  Idle watchdog warning: {exc}")
                    except RuntimeError:
                        break

        self._idle_watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        self._idle_watchdog_thread.start()

    def _on_idle_config_changed(self):
        self._persist_settings()
        self._arm_idle_timer()
        self._refresh_usage_status()

    def _apply_activity_pulse(self):
        self._last_activity = time.time()
        self._arm_idle_timer()
        self._refresh_usage_status()

    def _stop_idle_timer(self):
        if hasattr(self, "_idle_timer"):
            self._idle_timer.stop()
        self._refresh_usage_status()

    def _mark_expected_process_exit(self, proc, reason: str):
        if proc is None:
            return
        try:
            self._expected_exit_pid = proc.pid
        except Exception:
            self._expected_exit_pid = None
        self._expected_exit_reason = reason or ""

    def _clear_expected_process_exit(self, proc):
        try:
            pid = proc.pid if proc is not None else None
        except Exception:
            pid = None
        if pid is not None and pid == self._expected_exit_pid:
            self._expected_exit_pid = None
            self._expected_exit_reason = ""

    def _is_expected_process_exit(self, proc) -> bool:
        try:
            pid = proc.pid if proc is not None else None
        except Exception:
            pid = None
        return pid is not None and pid == self._expected_exit_pid

    def _should_filter_process_log_line(self, proc, line: str) -> bool:
        if not self._is_expected_process_exit(proc):
            return False
        lowered = (line or "").lower()
        return "resource_tracker:" in lowered or "warnings.warn('resource_tracker:" in lowered

    def _arm_idle_timer(self):
        if not hasattr(self, "_idle_timer"):
            return
        enabled, idle_seconds, _sleep_mode = self._idle_config_snapshot()
        if enabled and not self._sleeping and (self._generation_ready or bool(self.server_proc and self.server_proc.poll() is None)):
            self._idle_timer.start(max(1000, int(idle_seconds * 1000)))
        else:
            self._idle_timer.stop()

    def _on_idle_timeout(self):
        current = self._running_id or self._current_model or self._selected_model()
        enabled, _idle_seconds, sleep_mode = self._idle_config_snapshot()
        if not current or not enabled:
            return
        idle_minutes = max(1.0, _idle_seconds / 60.0)
        if sleep_mode == "deep":
            self._append_log(f"⏸ No real question for {idle_minutes:g} min; entering deep sleep to save battery.")
            self._enter_deep_sleep(reason="idle timeout")
            return
        self._append_log(f"⏸ No real question for {idle_minutes:g} min; pausing model to save battery.")
        self._stop_model_server(reason="idle timeout", sleeping=True)

    def _refresh_dots(self, running_id: Optional[str]):
        selected = self._selected_model() or self._current_model
        for mid, dot in self._dot_map.items():
            if mid == running_id:
                dot.setText("● running")
                dot.setStyleSheet(f"color: {OK}; background: transparent;")
            elif self._sleeping and mid == selected:
                dot.setText("◐ sleeping")
                dot.setStyleSheet(f"color: {WARN}; background: transparent;")
            else:
                dot.setText("")


    def _set_available_models(self, model_ids):
        self._available_models = list(model_ids or [])
        self._hf_model_combo.clear()
        if not self._available_models:
            self._hf_model_combo.addItem("No mlx-community models found", "")
            return
        for model_id in self._available_models:
            self._hf_model_combo.addItem(model_id, model_id)

    def _set_cached_models(self, models):
        self.models = list(models or [])
        preferred = self._preferred_cached_selection or self._settings.get("last_running_model") or self._settings.get("selected_model") or self._selected_model() or self._current_model or self._running_id
        self._preferred_cached_selection = None
        self._radio_group = QButtonGroup(self)
        self._radio_map = {}
        self._dot_map = {}
        while self._list_inner.count():
            item = self._list_inner.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not self.models:
            err_lbl = QLabel("No mlx-community models found\nin HuggingFace cache.")
            err_lbl.setStyleSheet(f"color: {ERR}; background: transparent;")
            self._list_inner.addWidget(err_lbl)
            self._list_inner.addStretch()
            return
        for m in self.models:
            mid = m["id"]
            size = f"  {m['size_gb']:.1f} GB" if m.get("size_gb", 0) > 0 else ""
            row = QWidget()
            row.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 4, 0, 4)
            row_layout.setSpacing(6)
            rb = QRadioButton()
            rb.setStyleSheet(f"""
                QRadioButton::indicator {{ width: 14px; height: 14px; }}
                QRadioButton::indicator:checked {{ background-color: {ACC}; border-radius: 7px; border: 2px solid {ACC}; }}
                QRadioButton::indicator:unchecked {{ border: 2px solid {MUTED}; border-radius: 7px; background: transparent; }}
                background: transparent;
            """)
            self._radio_group.addButton(rb)
            self._radio_map[mid] = rb
            row_layout.addWidget(rb)
            name_lbl = QLabel(mid.split("/")[-1])
            name_lbl.setToolTip(mid)
            name_lbl.setFont(QFont("Menlo", 12))
            name_lbl.setStyleSheet(f"color: {TEXT}; background: transparent;")
            name_lbl.mousePressEvent = lambda e, r=rb: r.setChecked(True)
            row_layout.addWidget(name_lbl)
            size_lbl = QLabel(size)
            size_lbl.setFont(QFont("Menlo", 10))
            size_lbl.setStyleSheet(f"color: {MUTED}; background: transparent;")
            row_layout.addWidget(size_lbl)
            row_layout.addStretch()
            dot = QLabel("")
            dot.setFont(QFont("Menlo", 10))
            dot.setStyleSheet(f"color: {MUTED}; background: transparent;")
            self._dot_map[mid] = dot
            row_layout.addWidget(dot)
            self._list_inner.addWidget(row)
            if preferred and preferred == mid:
                rb.setChecked(True)
        if preferred and preferred in self._radio_map:
            self._radio_map[preferred].setChecked(True)
        elif self.models and self.models[0]["id"] in self._radio_map:
            self._radio_map[self.models[0]["id"]].setChecked(True)
        self._list_inner.addStretch()
        self._refresh_dots(self._running_id)
        self._persist_settings()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _default_hf_token(self) -> str:
        return (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or ""
        )

    def _hf_token(self) -> str:
        token = self._hf_token_entry.text().strip() if hasattr(self, "_hf_token_entry") else ""
        return token or self._default_hf_token()

    def _persist_settings(self):
        auth_enabled = self._auth_enabled()
        raw_auth_token = self._auth_token_entry.text().strip() if hasattr(self, "_auth_token_entry") else ""
        auth_token = raw_auth_token or (DEFAULT_GATEWAY_TOKEN if auth_enabled else "")
        settings = {
            "host_mode": self._gateway_host_mode(),
            "host_mode_user_set": True,
            "port": self._port(),
            "gateway_scheme": self._gateway_scheme(),
            "gateway_cert_file": str(self._gateway_cert_file()),
            "gateway_key_file": str(self._gateway_key_file()),
            "client_verify_ssl": self._client_verify_ssl(),
            "auth_enabled": auth_enabled,
            "auth_token": auth_token,
            "idle_enabled": self._idle_enabled(),
            "idle_minutes": self._idle_minutes(),
            "sleep_mode": self._sleep_mode(),
            "request_timeout_seconds": self._request_timeout_seconds(),
            "max_tokens_per_request": self._max_tokens_per_request(),
            "hf_token": self._hf_token_entry.text().strip() if hasattr(self, "_hf_token_entry") else "",
            "selected_model": self._selected_model() or self._current_model or self._settings.get("selected_model", ""),
            "last_running_model": self._running_id or self._settings.get("last_running_model", ""),
        }
        self._settings = settings
        save_settings(settings)

    def _selected_model(self) -> Optional[str]:
        for mid, rb in self._radio_map.items():
            if rb.isChecked():
                return mid
        return None

    def _gateway_host_mode(self) -> str:
        return (self._host_mode_combo.currentData() or "all") if hasattr(self, "_host_mode_combo") else "all"

    def _gateway_bind_host(self) -> str:
        return LOCALHOST_HOST if self._gateway_host_mode() == "localhost" else DEFAULT_HOST

    def _gateway_scheme(self) -> str:
        value = (self._gateway_scheme_combo.currentData() if hasattr(self, "_gateway_scheme_combo") else None)
        scheme = str(value or self._settings.get("gateway_scheme", DEFAULT_GATEWAY_SCHEME) or DEFAULT_GATEWAY_SCHEME).lower()
        return "https" if scheme == "https" else "http"

    def _gateway_https_enabled(self) -> bool:
        return self._gateway_scheme() == "https"

    def _gateway_cert_file(self) -> Path:
        value = self._gateway_cert_entry.text().strip() if hasattr(self, "_gateway_cert_entry") else ""
        return _path_from_setting(value or self._settings.get("gateway_cert_file"), DEFAULT_GATEWAY_CERT_PATH)

    def _gateway_key_file(self) -> Path:
        value = self._gateway_key_entry.text().strip() if hasattr(self, "_gateway_key_entry") else ""
        return _path_from_setting(value or self._settings.get("gateway_key_file"), DEFAULT_GATEWAY_KEY_PATH)

    def _client_verify_ssl(self) -> bool:
        if hasattr(self, "_client_verify_ssl_toggle"):
            return bool(self._client_verify_ssl_toggle.isChecked())
        return bool(self._settings.get("client_verify_ssl", False))

    def _gateway_client_base_url(self) -> str:
        if DEFAULT_LLM_OS_LOCAL_BASE_URL:
            return DEFAULT_LLM_OS_LOCAL_BASE_URL.rstrip("/")
        return f"{self._gateway_scheme()}://host.docker.internal:{self._port()}"

    def _port(self) -> int:
        try:
            return int(self._port_entry.text())
        except ValueError:
            return DEFAULT_PORT

    def _internal_port(self) -> int:
        return self._port() + INTERNAL_PORT_OFFSET

    def _request_timeout_seconds(self) -> int:
        raw = self._settings.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
        try:
            return max(5, int(float(raw)))
        except Exception:
            return DEFAULT_REQUEST_TIMEOUT_SECONDS

    def _max_tokens_per_request(self) -> int:
        raw = self._settings.get("max_tokens_per_request", DEFAULT_MAX_TOKENS_PER_REQUEST)
        try:
            return max(1, int(float(raw)))
        except Exception:
            return DEFAULT_MAX_TOKENS_PER_REQUEST

    def _auth_enabled(self) -> bool:
        return bool(self._auth_toggle.isChecked())

    def _auth_token(self) -> str:
        token = self._auth_token_entry.text().strip()
        return token or DEFAULT_GATEWAY_TOKEN

    def _idle_enabled(self) -> bool:
        return bool(self._idle_toggle.isChecked())

    def _idle_minutes(self) -> float:
        try:
            return max(1.0, float(self._idle_minutes_entry.text()))
        except ValueError:
            return float(DEFAULT_IDLE_MINUTES)

    def _idle_seconds(self) -> float:
        return self._idle_minutes() * 60.0

    def _sleep_mode(self) -> str:
        return (self._sleep_mode_combo.currentData() or DEFAULT_SLEEP_MODE) if hasattr(self, "_sleep_mode_combo") else DEFAULT_SLEEP_MODE

    def _idle_config_snapshot(self) -> tuple[bool, float, str]:
        """Return the effective idle policy.

        The saved settings file is treated as the source of truth because the
        long-running GUI can be older than the on-disk settings after repair or
        restart work. UI changes are persisted immediately, so this still honors
        user changes while preventing stale in-memory controls from disabling
        battery-saving sleep.
        """
        settings = load_settings()
        if settings:
            enabled = bool(settings.get("idle_enabled", DEFAULT_IDLE_ENABLED))
            try:
                minutes = max(1.0, float(settings.get("idle_minutes", DEFAULT_IDLE_MINUTES)))
            except Exception:
                minutes = float(DEFAULT_IDLE_MINUTES)
            sleep_mode = str(settings.get("sleep_mode", DEFAULT_SLEEP_MODE) or DEFAULT_SLEEP_MODE)
            if sleep_mode not in {"light", "deep"}:
                sleep_mode = DEFAULT_SLEEP_MODE
            return enabled, minutes * 60.0, sleep_mode
        return self._idle_enabled(), self._idle_seconds(), self._sleep_mode()

    def _idle_reference_time(self) -> float:
        stats = self._token_stats_snapshot()
        return max(float(stats.get("last_real_question_at") or 0.0), float(self._last_activity or 0.0))

    def _format_clock(self, ts: float) -> str:
        if not ts:
            return "never"
        return time.strftime("%I:%M %p", time.localtime(ts)).lstrip("0")

    def _server_process_active(self) -> bool:
        return bool(self.server_proc and self.server_proc.poll() is None)

    def _decode_json_body(self, body: bytes) -> dict:
        try:
            return json.loads((body or b"{}").decode("utf-8", errors="replace")) or {}
        except Exception:
            return {}

    def _content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("input_text") or item.get("content") or ""
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(p for p in parts if p)
        if isinstance(content, dict):
            text = content.get("text") or content.get("input_text") or content.get("content") or ""
            return text if isinstance(text, str) else ""
        return ""

    def _payload_user_texts(self, payload: dict) -> List[str]:
        texts = []
        for message in payload.get("messages") or []:
            if not isinstance(message, dict):
                continue
            if (message.get("role") or "").lower() != "user":
                continue
            text = self._content_text(message.get("content")).strip()
            if text:
                texts.append(text)
        return texts

    def _is_probe_payload(self, payload: dict, user_texts: List[str]) -> bool:
        if len(user_texts) != 1:
            return False
        text = " ".join(user_texts[0].lower().split())
        max_tokens = payload.get("max_tokens")
        try:
            max_tokens = int(max_tokens) if max_tokens is not None else None
        except Exception:
            max_tokens = None
        probe_texts = {
            "ping",
            "health",
            "health check",
            "readiness check",
            "ready",
            "reply with exactly: ready",
        }
        looks_like_ready_probe = text in probe_texts or (
            text.startswith("reply with exactly") and "ready" in text
        )
        return looks_like_ready_probe and (max_tokens is None or max_tokens <= 16)

    def _is_real_chat_question(self, payload: dict) -> bool:
        user_texts = self._payload_user_texts(payload)
        if not user_texts:
            return False
        return not self._is_probe_payload(payload, user_texts)

    def _question_summary(self, payload: dict) -> str:
        text = " ".join(self._payload_user_texts(payload)).strip()
        if not text:
            return ""
        text = " ".join(text.split())
        return text if len(text) <= 84 else f"{text[:81]}..."

    def _record_real_question(self, payload: dict):
        summary = self._question_summary(payload)
        now = time.time()
        with self._stats_lock:
            self._last_real_question_at = now
            self._last_real_question = summary
            self._token_stats["real_questions"] += 1
        self._bridge.usage_refresh.emit()

    def _estimate_tokens_from_text(self, text: str) -> int:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return 0
        return max(1, int((len(cleaned) + 3) / 4))

    def _estimate_prompt_tokens(self, payload: dict) -> int:
        texts = []
        for message in payload.get("messages") or []:
            if isinstance(message, dict):
                texts.append(self._content_text(message.get("content")))
        return self._estimate_tokens_from_text("\n".join(texts))

    def _extract_completion_text(self, response_payload: dict) -> str:
        parts = []
        for choice in response_payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            if isinstance(message, dict):
                parts.append(self._content_text(message.get("content")))
            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                parts.append(self._content_text(delta.get("content")))
        return "\n".join(p for p in parts if p)

    def _record_token_usage(self, request_payload: dict, response_body: bytes, content_type: str = ""):
        response_payload = {}
        if (content_type or "").startswith("application/json"):
            response_payload = self._decode_json_body(response_body)

        usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
        prompt_tokens = completion_tokens = total_tokens = 0
        source = "estimated"

        if isinstance(usage, dict):
            try:
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
                source = "usage"
            except Exception:
                prompt_tokens = completion_tokens = total_tokens = 0

        if total_tokens <= 0:
            prompt_tokens = self._estimate_prompt_tokens(request_payload)
            completion_tokens = self._estimate_tokens_from_text(self._extract_completion_text(response_payload))
            if completion_tokens <= 0:
                completion_tokens = self._estimate_tokens_from_text((response_body or b"").decode("utf-8", errors="replace"))
            total_tokens = prompt_tokens + completion_tokens

        with self._stats_lock:
            self._token_stats["chat_requests"] += 1
            self._token_stats["prompt_tokens"] += prompt_tokens
            self._token_stats["completion_tokens"] += completion_tokens
            self._token_stats["total_tokens"] += total_tokens
            self._token_stats["last_total_tokens"] = total_tokens
            self._token_stats["last_token_source"] = source
            if source == "usage":
                self._token_stats["exact_tokens"] += total_tokens
            else:
                self._token_stats["estimated_tokens"] += total_tokens
        self._bridge.usage_refresh.emit()

    def _token_stats_snapshot(self) -> dict:
        with self._stats_lock:
            stats = dict(self._token_stats)
            stats["last_real_question_at"] = self._last_real_question_at
            stats["last_real_question"] = self._last_real_question
            return stats

    def _refresh_usage_status(self):
        if not hasattr(self, "_idle_state_lbl"):
            return
        stats = self._token_stats_snapshot()
        idle_enabled, idle_seconds, sleep_mode = self._idle_config_snapshot()
        idle_minutes = max(1.0, idle_seconds / 60.0)
        if self._sleeping or self._deep_sleeping:
            if self._deep_sleeping:
                idle_text = f"Deep sleep active. Use Load & Start to wake. Idle policy: {idle_minutes:g} min."
            else:
                idle_text = f"Sleeping now. The next real question wakes the last model. Idle policy: {idle_minutes:g} min."
            idle_color = WARN
        elif idle_enabled:
            idle_text = f"Idle sleep ON: pauses after {idle_minutes:g} min without a real user question."
            idle_color = OK
        else:
            idle_text = "Idle sleep OFF: model will stay loaded until stopped."
            idle_color = WARN

        self._idle_state_lbl.setText(idle_text)
        self._idle_state_lbl.setStyleSheet(f"color: {idle_color}; background: transparent;")

        total = stats.get("total_tokens", 0)
        prompt = stats.get("prompt_tokens", 0)
        completion = stats.get("completion_tokens", 0)
        questions = stats.get("real_questions", 0)
        source = stats.get("last_token_source", "none")
        last_total = stats.get("last_total_tokens", 0)
        if hasattr(self, "_header_tokens_lbl"):
            self._header_tokens_lbl.setText(
                f"Tokens: {total:,} total • last {last_total:,}"
            )
        self._usage_tokens_lbl.setText(
            f"Tokens used: {total:,} total\n"
            f"Prompt: {prompt:,}  •  Completion: {completion:,}\n"
            f"Questions: {questions:,}  •  Last: {last_total:,} [{source}]"
        )

        last_at = stats.get("last_real_question_at", 0.0)
        last_q = stats.get("last_real_question") or "none"
        self._last_question_lbl.setText(f"Last real question: {self._format_clock(last_at)} • {last_q}")

    def _mark_activity(self):
        self._bridge.activity_pulse.emit()

    def _record_gateway_request(self, kind: str):
        now = time.time()
        self._gateway_request_times.append((now, kind))
        cutoff = now - GATEWAY_STATS_WINDOW_S
        while self._gateway_request_times and self._gateway_request_times[0][0] < cutoff:
            self._gateway_request_times.popleft()

    def _gateway_request_summary(self) -> str:
        now = time.time()
        cutoff = now - GATEWAY_STATS_WINDOW_S
        while self._gateway_request_times and self._gateway_request_times[0][0] < cutoff:
            self._gateway_request_times.popleft()
        if not self._gateway_request_times:
            return "0 requests in last 5m"
        model_hits = sum(1 for _, kind in self._gateway_request_times if kind == 'models')
        chat_hits = sum(1 for _, kind in self._gateway_request_times if kind == 'chat')
        question_hits = sum(1 for _, kind in self._gateway_request_times if kind == 'question')
        return f"{len(self._gateway_request_times)} requests in last 5m ({model_hits} models, {chat_hits} chat, {question_hits} real questions)"

    def _gateway_error_payload(self, message: str, error_type: str, code: str, state: Optional[str] = None) -> dict:
        payload = {
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            },
            "message": message,
        }
        if state:
            payload["state"] = state
            payload["error"]["state"] = state
        return payload

    def _gateway_status_payload(self) -> dict:
        models_payload = self._gateway_models_payload()
        state = models_payload.get("state") or "unknown"
        models = models_payload.get("data") or []
        ready_models = [m for m in models if m.get("ready")]
        model_id = ready_models[0].get("id") if ready_models else (models[0].get("id") if models else "")
        stats = self._token_stats_snapshot()
        last_real = float(stats.get("last_real_question_at") or 0.0)
        return {
            "service": "mlx-manager",
            "status": "ready" if state == "running" else "not_ready",
            "ready": state == "running",
            "state": state,
            "model": model_id,
            "models": models,
            "gateway": {
                "host": self._gateway_bound_host or self._gateway_bind_host(),
                "port": self._gateway_port or self._port(),
                "scheme": self._gateway_scheme(),
                "auth_enabled": self._auth_enabled(),
                "base_url": self._gateway_client_base_url(),
                "tls": {
                    "enabled": self._gateway_https_enabled(),
                    "cert_file": str(self._gateway_cert_file()) if self._gateway_https_enabled() else "",
                    "self_signed": bool(self._gateway_tls_self_signed),
                    "client_verify_ssl": self._client_verify_ssl(),
                },
            },
            "internal": {
                "host": INTERNAL_HOST,
                "port": self._internal_port(),
                "process_active": self._server_process_active(),
                "generation_ready": self._generation_ready,
            },
            "idle": models_payload.get("idle") or {},
            "usage": models_payload.get("usage") or {},
            "uptime_seconds": max(0.0, time.time() - self._started_at),
            "last_real_question_ago_seconds": (time.time() - last_real) if last_real else None,
            "request_timeout_seconds": self._request_timeout_seconds(),
        }

    def _sync_llm_os_if_needed(self, model_id: Optional[str], state: str, log_fn=None, force: bool = False):
        if not model_id:
            return False
        signature = (
            model_id,
            state,
            self._gateway_client_base_url(),
            bool(self._auth_enabled()),
            self._auth_token() if self._auth_enabled() else "",
            bool(self._client_verify_ssl()),
        )
        if (
            not force
            and self._last_synced_signature == signature
            and self._last_synced_model == model_id
            and self._last_synced_state == state
        ):
            return False
        ok = sync_llm_os_local_model(
            model_id,
            self._gateway_client_base_url(),
            require_auth=self._auth_enabled(),
            auth_token=self._auth_token(),
            verify_ssl=self._client_verify_ssl(),
            log_fn=log_fn or self._append_log,
        )
        if ok:
            self._last_synced_model = model_id
            self._last_synced_state = state
            self._last_synced_signature = signature
        return ok

    def _sync_current_llm_os_state(self, *, force: bool = False, reason: str = "state sync"):
        model_id = self._running_id or self._current_model or self._selected_model() or self._settings.get("last_running_model", "")
        if not model_id:
            return False
        if self._generation_ready and self._server_process_active():
            state = "running"
        elif self._sleeping or self._deep_sleeping:
            state = "sleeping"
        elif self._startup_inflight():
            state = "warming"
        else:
            idle_enabled, _idle_seconds, _sleep_mode = self._idle_config_snapshot()
            state = "sleeping" if idle_enabled and not self._server_process_active() else "stopped"

        def _worker():
            self._sync_llm_os_if_needed(
                model_id,
                state,
                log_fn=self._bridge.log_line.emit,
                force=force,
            )

        threading.Thread(target=_worker, daemon=True, name=f"llm-os-sync-{reason}").start()
        return True

    def _startup_inflight(self) -> bool:
        return bool(self._startup_thread and self._startup_thread.is_alive())

    def _gateway_models_payload(self) -> dict:
        proc_running = bool(self.server_proc and self.server_proc.poll() is None)
        startup_inflight = self._startup_inflight()
        ready_model = self._running_id if (self._generation_ready and proc_running) else None
        model_id = ready_model or self._current_model or self._selected_model() or self._settings.get("last_running_model", "")
        idle_enabled, idle_seconds, sleep_mode = self._idle_config_snapshot()
        sleep_available = bool(model_id and idle_enabled and not proc_running and not startup_inflight)
        data = []
        if model_id:
            data.append({
                "id": model_id,
                "object": "model",
                "owned_by": "mlx-community",
                "state": "running" if ready_model else ("sleeping" if (self._sleeping or sleep_available) else "warming"),
                "ready": bool(ready_model),
            })
        if ready_model:
            state = "running"
        elif self._sleeping or sleep_available:
            state = "sleeping"
        elif proc_running or startup_inflight or model_id:
            state = "warming"
        else:
            state = "stopped"
        stats = self._token_stats_snapshot()
        return {
            "object": "list",
            "data": data,
            "state": state,
            "idle": {
                "enabled": idle_enabled,
                "minutes": max(1.0, idle_seconds / 60.0),
                "sleep_mode": sleep_mode,
                "last_real_question_at": stats.get("last_real_question_at", 0.0),
            },
            "usage": {
                "real_questions": stats.get("real_questions", 0),
                "chat_requests": stats.get("chat_requests", 0),
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "completion_tokens": stats.get("completion_tokens", 0),
                "total_tokens": stats.get("total_tokens", 0),
                "estimated_tokens": stats.get("estimated_tokens", 0),
                "exact_tokens": stats.get("exact_tokens", 0),
            },
        }

    def _shutdown_gateway(self):
        server = None
        host = None
        port = None
        with self._gateway_lock:
            if self.gateway_server:
                server = self.gateway_server
                host = self._gateway_bound_host
                port = self._gateway_port
                self.gateway_server = None
                self.gateway_thread = None
                self._gateway_port = None
                self._gateway_bound_host = None
                self._gateway_bound_scheme = None
                self._gateway_tls_self_signed = False
        self._shutdown_gateway_instance(server, host, port)

    def _shutdown_gateway_instance(self, server, host: Optional[str], port: Optional[int]):
        if not server:
            return
        self._wake_gateway_for_shutdown(host, port)
        try:
            server.shutdown()
            server.server_close()
        except Exception as exc:
            self._bridge.log_line.emit(f"  Gateway shutdown warning: {exc}")

    def _wake_gateway_for_shutdown(self, host: Optional[str], port: Optional[int]):
        if not host or not port:
            return

        def _poke():
            time.sleep(0.05)
            connect_host = LOCALHOST_HOST if host in {"0.0.0.0", "::", ""} else host
            try:
                with socket.create_connection((connect_host, int(port)), timeout=0.75):
                    pass
            except Exception:
                pass

        threading.Thread(target=_poke, daemon=True, name="gateway-shutdown-wake").start()

    def _enter_deep_sleep(self, reason: str = "idle timeout"):
        self._stop_model_server(reason=reason, sleeping=False)
        self._shutdown_gateway()
        self._deep_sleeping = True
        label = (self._current_model or self._selected_model() or "Sleeping").split("/")[-1]
        self._bridge.status_set.emit("sleeping", f"{label} (deep)")
        self._bridge.log_line.emit("  Deep sleep active: gateway stopped; use Load & Start to wake.")

    def _ensure_gateway_running(self) -> bool:
        desired_port = self._port()
        bind_host = self._gateway_bind_host()
        scheme = self._gateway_scheme()
        with self._gateway_lock:
            if (
                self.gateway_server
                and self._gateway_port == desired_port
                and self._gateway_bound_host == bind_host
                and self._gateway_bound_scheme == scheme
            ):
                return True
            if self.gateway_server:
                self._append_log(f"↺ Restarting gateway on {scheme}://{bind_host}:{desired_port}")
                old_server = self.gateway_server
                old_host = self._gateway_bound_host
                old_port = self._gateway_port
                self.gateway_server = None
                self.gateway_thread = None
                self._gateway_port = None
                self._gateway_bound_host = None
                self._gateway_bound_scheme = None
                self._gateway_tls_self_signed = False
                self._shutdown_gateway_instance(old_server, old_host, old_port)

            manager = self

            class _GatewayServer(ThreadingHTTPServer):
                allow_reuse_address = True
                daemon_threads = True
                block_on_close = False

            class _Handler(BaseHTTPRequestHandler):
                def log_message(self, fmt, *args):
                    return

                def _authorized(self):
                    if not manager._auth_enabled():
                        return True
                    expected = manager._auth_token()
                    return bool(expected) and self.headers.get("Authorization") == f"Bearer {expected}"

                def _send_json(self, status: int, payload: dict, extra_headers: Optional[dict] = None):
                    body = json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    for key, value in (extra_headers or {}).items():
                        self.send_header(str(key), str(value))
                    self.end_headers()
                    self.wfile.write(body)

                def _send_error(self, status: int, message: str, error_type: str, code: str, state: Optional[str] = None, retry_after: Optional[int] = None):
                    headers = {}
                    if retry_after is not None:
                        headers["Retry-After"] = str(max(1, int(retry_after)))
                    self._send_json(status, manager._gateway_error_payload(message, error_type, code, state), headers)

                def do_GET(self):
                    path = urlsplit(self.path).path
                    if path == "/health":
                        self._send_json(200, {
                            "status": "ok",
                            "service": "mlx-manager",
                            "state": manager._gateway_models_payload().get("state"),
                            "uptime_seconds": max(0.0, time.time() - manager._started_at),
                        })
                        return
                    if path == "/ready":
                        status_payload = manager._gateway_status_payload()
                        status = 200 if status_payload.get("ready") else 503
                        headers = {} if status == 200 else {"Retry-After": "5"}
                        self._send_json(status, status_payload, headers)
                        return
                    if path in {"/status", "/v1/models"}:
                        if not self._authorized():
                            self._send_error(401, "Unauthorized", "authentication_error", "unauthorized")
                            return
                        if path == "/status":
                            manager._record_gateway_request('status')
                            self._send_json(200, manager._gateway_status_payload())
                            return
                        manager._record_gateway_request('models')
                        self._send_json(200, manager._gateway_models_payload())
                        return
                    self._send_error(404, "Not found", "invalid_request_error", "not_found")

                def do_POST(self):
                    path = urlsplit(self.path).path
                    if path != "/v1/chat/completions":
                        self._send_error(404, "Not found", "invalid_request_error", "not_found")
                        return
                    if not self._authorized():
                        self._send_error(401, "Unauthorized", "authentication_error", "unauthorized")
                        return
                    length = int(self.headers.get("Content-Length", "0") or 0)
                    body = self.rfile.read(length)
                    request_payload = manager._decode_json_body(body)
                    real_question = manager._is_real_chat_question(request_payload)
                    manager._record_gateway_request('question' if real_question else 'chat')
                    if real_question:
                        manager._record_real_question(request_payload)
                        manager._mark_activity()
                    elif not manager._server_process_active():
                        self._send_json(
                            202,
                            manager._gateway_error_payload(
                                "No real user question detected; model remains asleep/stopped.",
                                "model_sleeping",
                                "probe_ignored",
                                manager._gateway_models_payload().get("state"),
                            ),
                            {"Retry-After": "5"},
                        )
                        return
                    if not manager._ensure_server_running(
                        reason="incoming request",
                        fast_request_wake=True,
                        count_as_activity=real_question,
                    ):
                        self._send_error(
                            503,
                            "Model is warming or waking; retry shortly.",
                            "model_warming",
                            "service_unavailable",
                            manager._gateway_models_payload().get("state"),
                            retry_after=5,
                        )
                        return
                    if request_payload.get("stream") is True:
                        manager._forward_stream_to_internal(self, path, body, request_payload, real_question)
                        return
                    status, response_body, content_type = manager._forward_to_internal(path, body)
                    if real_question and 200 <= status < 300:
                        manager._record_token_usage(request_payload, response_body, content_type)
                    self.send_response(status)
                    self.send_header("Content-Type", content_type or "application/json")
                    self.send_header("Content-Length", str(len(response_body)))
                    self.end_headers()
                    self.wfile.write(response_body)

            try:
                self.gateway_server = _GatewayServer((bind_host, desired_port), _Handler)
                if scheme == "https":
                    cert_path, key_path = ensure_gateway_tls_files(self._gateway_cert_file(), self._gateway_key_file())
                    context = create_gateway_ssl_context(cert_path, key_path)
                    self.gateway_server.socket = context.wrap_socket(self.gateway_server.socket, server_side=True)
                    self._gateway_tls_self_signed = (
                        cert_path.resolve() == DEFAULT_GATEWAY_CERT_PATH.resolve()
                        and key_path.resolve() == DEFAULT_GATEWAY_KEY_PATH.resolve()
                    )
                else:
                    self._gateway_tls_self_signed = False
                self.gateway_thread = threading.Thread(
                    target=lambda: self.gateway_server.serve_forever(poll_interval=GATEWAY_POLL_INTERVAL_S),
                    daemon=True,
                )
                self.gateway_thread.start()
                self._gateway_port = desired_port
                self._gateway_bound_host = bind_host
                self._gateway_bound_scheme = scheme
                self._append_log(f"↔ Gateway ready on {scheme}://{bind_host}:{desired_port} (internal mlx-lm on {INTERNAL_HOST}:{self._internal_port()})")
                if self._gateway_host_mode() == "localhost":
                    self._append_log(f"  NOTE: localhost mode keeps the gateway private; llm-os is synced via {self._gateway_client_base_url()} for container access.")
                if scheme == "https":
                    verify = "enabled" if self._client_verify_ssl() else "disabled for self-signed/local testing"
                    self._append_log(f"  Gateway TLS: enabled (client certificate verification {verify})")
                if self._auth_enabled():
                    self._append_log("  Gateway auth: enabled")
                else:
                    self._append_log("  Gateway auth: disabled")
                self._append_log(f"  Gateway idle mode: quiet poll every {int(GATEWAY_POLL_INTERVAL_S)}s")
                return True
            except Exception as exc:
                self._append_log(f"  ERROR: gateway failed to start ({exc})")
                self._apply_status("error", "Gateway error")
                return False

    def _start_server_worker(self, mid: str, reason: str):
        try:
            if self._shutdown_requested:
                self._startup_error = "startup cancelled by shutdown"
                return
            preferred = self._current_model or self._selected_model() or self._settings.get("last_running_model", "")
            running = check_server(self._internal_port(), host=INTERNAL_HOST, preferred_model=preferred)
            if not running:
                self._bridge.log_line.emit(f"  Starting model for {reason}: {mid}")
                self._launch_server_process(mid)
            else:
                self._bridge.log_line.emit(f"  Waiting for generation readiness: {running}")

            deadline = time.time() + STARTUP_TIMEOUT_S
            while time.time() < deadline:
                if self._shutdown_requested:
                    self._startup_error = "startup cancelled by shutdown"
                    return
                if self.server_proc and self.server_proc.poll() is not None:
                    self._startup_error = f"server exited before ready (code {self.server_proc.returncode})"
                    break
                preferred = self._current_model or self._selected_model() or self._settings.get("last_running_model", "")
                running = check_server(self._internal_port(), host=INTERNAL_HOST, preferred_model=preferred)
                if running:
                    self._bridge.status_set.emit("starting", f"Loading {running.split('/')[-1]}…")
                    if check_generation_ready(self._internal_port(), host=INTERNAL_HOST, preferred_model=mid):
                        self._generation_ready = True
                        self._running_id = running
                        self._current_model = running
                        self._sleeping = False
                        self._deep_sleeping = False
                        self._mark_activity()
                        elapsed = max(0.0, time.time() - self._startup_started_at)
                        self._bridge.log_line.emit(f"  Model ready in {elapsed:.1f}s: {running}")
                        self._bridge.status_set.emit("running", running.split("/")[-1])
                        self._sync_llm_os_if_needed(running, "running", log_fn=self._bridge.log_line.emit)
                        return
                time.sleep(0.5)

            if not self._startup_error:
                self._startup_error = "model start timed out"
            self._bridge.log_line.emit(f"  ERROR: {self._startup_error}")
            self._bridge.status_set.emit("error", "Start timeout")
        finally:
            self._startup_event.set()

    def _begin_startup(self, mid: str, reason: str):
        with self._server_lock:
            if self._startup_inflight() and self._startup_model == mid:
                return self._startup_event
            self._startup_event = threading.Event()
            self._startup_error = None
            self._startup_model = mid
            self._startup_started_at = time.time()
            self._startup_thread = threading.Thread(target=self._start_server_worker, args=(mid, reason), daemon=True)
            self._startup_thread.start()
            return self._startup_event

    def _forward_stream_to_internal(self, handler, path: str, body: bytes, request_payload: dict, real_question: bool):
        req = urllib.request.Request(
            f"http://{INTERNAL_HOST}:{self._internal_port()}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started_at = time.time()
        buffered = bytearray()
        try:
            with urllib.request.urlopen(req, timeout=self._request_timeout_seconds()) as resp:
                self._generation_ready = True
                content_type = resp.headers.get_content_type() or "text/event-stream"
                handler.send_response(resp.status)
                handler.send_header("Content-Type", content_type)
                handler.send_header("Cache-Control", "no-cache")
                handler.send_header("Connection", "close")
                handler.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    if len(buffered) < 2_000_000:
                        buffered.extend(chunk)
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                if real_question and 200 <= resp.status < 300:
                    self._record_token_usage(request_payload, bytes(buffered), content_type)
                elapsed = time.time() - started_at
                if elapsed >= 2.0:
                    self._bridge.log_line.emit(f"  Streamed request completed in {elapsed:.1f}s")
        except urllib.error.HTTPError as exc:
            raw_body = exc.read()
            text = raw_body.decode(errors="replace")
            try:
                payload = json.loads(text or "{}")
                if not isinstance(payload, dict):
                    raise ValueError("non-object error")
            except Exception:
                payload = self._gateway_error_payload(
                    text[:300] or "MLX stream request failed.",
                    "api_error",
                    "upstream_error",
                    self._gateway_models_payload().get("state"),
                )
            handler._send_json(exc.code, payload)
        except Exception as exc:
            elapsed = time.time() - started_at
            state = self._gateway_models_payload().get("state")
            self._bridge.log_line.emit(f"  Stream request failed after {elapsed:.1f}s (state={state}): {exc}")
            handler._send_json(
                504,
                self._gateway_error_payload(str(exc), "timeout_error", "gateway_timeout", state),
                {"Retry-After": "5"},
            )

    def _forward_to_internal(self, path: str, body: bytes):
        req = urllib.request.Request(
            f"http://{INTERNAL_HOST}:{self._internal_port()}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started_at = time.time()
        timeout_s = self._request_timeout_seconds()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    self._generation_ready = True
                    if self._running_id or self._current_model:
                        self._bridge.status_set.emit("running", (self._running_id or self._current_model).split("/")[-1])
                        self._sync_llm_os_if_needed(self._running_id or self._current_model, "running", log_fn=self._bridge.log_line.emit)
                    elapsed = time.time() - started_at
                    if elapsed >= 2.0:
                        self._bridge.log_line.emit(f"  Forwarded request completed in {elapsed:.1f}s")
                    return resp.status, resp.read(), resp.headers.get_content_type()
            except urllib.error.HTTPError as exc:
                body = exc.read()
                text = body.decode(errors="replace")
                if attempt < 2 and self._is_retryable_warm_response(exc.code, text):
                    self._bridge.log_line.emit("  Internal server is still warming; retrying forwarded request…")
                    time.sleep(2.0)
                    continue
                return exc.code, body, exc.headers.get_content_type()
            except Exception as exc:
                elapsed = time.time() - started_at
                state = self._gateway_models_payload().get("state")
                self._bridge.log_line.emit(f"  Forward request failed after {elapsed:.1f}s (state={state}): {exc}")
                payload = json.dumps(self._gateway_error_payload(str(exc), "timeout_error", "gateway_timeout", state)).encode()
                return 504, payload, "application/json"

        payload = json.dumps(self._gateway_error_payload(
            "Model is warming or waking; retry shortly.",
            "model_warming",
            "service_unavailable",
            self._gateway_models_payload().get("state"),
        )).encode()
        return 503, payload, "application/json"

    @staticmethod
    def _is_retryable_warm_response(status_code: int, body: str) -> bool:
        if status_code not in {202, 503, 504}:
            return False
        try:
            payload = json.loads(body or "{}")
        except Exception:
            payload = {}
        state = str(payload.get("state") or "").lower()
        message = str(payload.get("message") or payload.get("error") or body or "").lower()
        return state in {"warming", "starting", "sleeping", "running"} or "warming" in message or "waking" in message

    def _ensure_server_running(self, reason: str = "manual start", fast_request_wake: bool = False, count_as_activity: bool = True) -> bool:
        with self._server_lock:
            if self._shutdown_requested:
                return False
            preferred = self._current_model or self._selected_model() or self._settings.get("last_running_model", "")
            running = check_server(self._internal_port(), host=INTERNAL_HOST, preferred_model=preferred)
            if running and self._generation_ready:
                self._running_id = running
                self._current_model = running
                self._sleeping = False
                self._deep_sleeping = False
                if count_as_activity:
                    self._mark_activity()
                self._sync_llm_os_if_needed(running, "running", log_fn=self._bridge.log_line.emit)
                return True

            mid = self._selected_model() or self._current_model or self._running_id
            if not mid:
                self._bridge.log_line.emit("  ERROR: no model selected for startup")
                return False
            if not self.mlx_binary:
                self._bridge.log_line.emit("  ERROR: mlx_lm binary not found")
                return False

            self._current_model = mid
            self._sleeping = False
            self._deep_sleeping = False
            self._generation_ready = False
            self._bridge.status_set.emit("starting", f"Loading {mid.split('/')[-1]}…")
            if running and fast_request_wake:
                self._running_id = running
                if count_as_activity:
                    self._mark_activity()
                self._bridge.status_set.emit("starting", f"Waking {running.split('/')[-1]}…")
                return True

        event = self._begin_startup(mid, reason)
        wait_timeout = STARTUP_REQUEST_TIMEOUT_S if fast_request_wake else STARTUP_TIMEOUT_S
        finished = event.wait(wait_timeout)

        if self._shutdown_requested:
            return False

        preferred = self._current_model or self._selected_model() or self._settings.get("last_running_model", "")
        running = check_server(self._internal_port(), host=INTERNAL_HOST, preferred_model=preferred)
        if running and (self._generation_ready or fast_request_wake):
            self._running_id = running
            self._current_model = running
            self._sleeping = False
            self._deep_sleeping = False
            if count_as_activity:
                self._mark_activity()
            if self._generation_ready:
                self._bridge.status_set.emit("running", running.split("/")[-1])
                self._sync_llm_os_if_needed(running, "running", log_fn=self._bridge.log_line.emit)
            else:
                self._bridge.status_set.emit("starting", f"Waking {running.split('/')[-1]}…")
            return True

        if not finished:
            state = self._gateway_models_payload().get("state")
            self._bridge.log_line.emit(f"  Startup wait exceeded {wait_timeout}s for {mid.split('/')[-1]} (state={state})")
        return False

    def _launch_server_process(self, mid: str):
        if self._shutdown_requested:
            self._bridge.log_line.emit("  Startup skipped because shutdown is in progress.")
            return
        kill_all_servers(lambda m: self._bridge.log_line.emit(m))
        if self.server_proc and self.server_proc.poll() is None:
            try:
                self._mark_expected_process_exit(self.server_proc, "server restart")
                self.server_proc.terminate()
                self.server_proc.wait(timeout=5)
            except Exception:
                pass

        cmd = build_mlx_server_cmd(mid, INTERNAL_HOST, self._internal_port())
        self._generation_ready = False
        self._bridge.log_line.emit(f"  cmd: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        self.server_proc = proc

        def _stream_logs(active_proc):
            try:
                for line in active_proc.stdout:
                    line = line.rstrip()
                    if line:
                        if self._should_filter_process_log_line(active_proc, line):
                            continue
                        self._bridge.log_line.emit(line)
            finally:
                rc = active_proc.wait()
                expected = self._is_expected_process_exit(active_proc)
                if not expected:
                    self._bridge.log_line.emit(f"  server exited (code {rc})")
                if self.server_proc is active_proc:
                    self.server_proc = None
                    self._running_id = None
                    self._generation_ready = False
                    self._bridge.idle_timer_stop.emit()
                    if not self._sleeping:
                        self._bridge.status_set.emit("stopped", "")
                self._clear_expected_process_exit(active_proc)

        threading.Thread(target=_stream_logs, args=(proc,), daemon=True).start()

    def _stop_model_server(
        self,
        reason: str = "manual stop",
        sleeping: bool = False,
        wait_timeout: float = 5.0,
        lock_timeout: Optional[float] = None,
    ):
        if lock_timeout is None:
            acquired = self._server_lock.acquire()
        else:
            acquired = self._server_lock.acquire(timeout=max(0.0, float(lock_timeout)))
        if not acquired:
            self._bridge.log_line.emit("  Server lock busy during shutdown; sending TERM to mlx_lm servers.")
            kill_all_servers(lambda m: self._bridge.log_line.emit(m))
            self.server_proc = None
            self._running_id = None
            self._generation_ready = False
            self._bridge.idle_timer_stop.emit()
            self._sleeping = False
            self._bridge.status_set.emit("stopped", "")
            return
        try:
            if self.server_proc and self.server_proc.poll() is None:
                try:
                    self._mark_expected_process_exit(self.server_proc, reason)
                    self.server_proc.terminate()
                    self.server_proc.wait(timeout=max(0.1, float(wait_timeout)))
                except Exception:
                    try:
                        self.server_proc.kill()
                        self.server_proc.wait(timeout=2)
                    except Exception:
                        pass
            self.server_proc = None
            self._running_id = None
            self._generation_ready = False
            self._bridge.idle_timer_stop.emit()
            self._sleeping = sleeping and bool(self._current_model or self._selected_model())
            self._bridge.log_line.emit(f"  Model stopped ({reason}).")
            if self._sleeping:
                label = (self._current_model or self._selected_model() or "Sleeping").split("/")[-1]
                self._bridge.status_set.emit("sleeping", label)
                self._sync_llm_os_if_needed(self._current_model or self._selected_model(), "sleeping", log_fn=self._bridge.log_line.emit)
                self._bridge.log_line.emit(f"  Gateway activity while paused: {self._gateway_request_summary()}")
            else:
                self._bridge.status_set.emit("stopped", "")
        finally:
            self._server_lock.release()

    def _apply_status(self, state: str, label: str = ""):
        self._status = state
        self._status_changed_at = time.strftime("%I:%M %p").lstrip("0")
        styles = {
            "running":  (OK,   "●", "Running",     "#133222"),
            "starting": (WARN, "◌", "Loading",     "#3a2b14"),
            "stopping": (WARN, "◌", "Stopping",    "#3a2b14"),
            "sleeping": (WARN, "◐", "Sleeping",    "#3a2b14"),
            "stopped":  (MUTED, "○", "Stopped",    SURF2),
            "error":    (ERR,  "✕", "Gateway Error", "#341a1c"),
            "unknown":  (MUTED, "?", "Checking",   SURF2),
        }
        col, sym, title, bg = styles.get(state, (MUTED, "?", "Checking", SURF2))
        cleaned_label = (label or "").strip()
        if state == "starting" and cleaned_label.lower().startswith("waking "):
            title = "Waking"
        elif state == "sleeping" and (self._deep_sleeping or "(deep)" in cleaned_label.lower()):
            title = "Deep Sleep"
            cleaned_label = cleaned_label.replace("(deep)", "").strip()
        pill_text = f"{sym} {title}"
        self._status_lbl.setText(pill_text)
        self._status_lbl.setStyleSheet(
            f"color: {col}; background: {bg}; border: 1px solid {col};"
            " border-radius: 12px; padding: 4px 10px;"
        )
        meta_parts = []
        if cleaned_label:
            meta_parts.append(cleaned_label)
        if self._status_changed_at:
            meta_parts.append(self._status_changed_at)
        meta_text = " • ".join(meta_parts)
        self._status_meta_lbl.setText(meta_text)
        self._status_meta_lbl.setVisible(bool(meta_text))
        self._refresh_usage_status()

    def _append_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log_box.append(f"[{ts}] {msg}")
        self._log_box.moveCursor(QTextCursor.MoveOperation.End)

    def _begin_close(self):
        self._close_in_progress = True
        self._shutdown_requested = True
        self.setEnabled(False)
        self._apply_status("stopping", "Closing…")
        self._append_log("■ Closing MLX Manager; stopping gateway and model in the background…")

        def _worker():
            try:
                self._persist_settings()
                self._shutdown_gateway()
                self._stop_model_server(
                    reason="window close",
                    sleeping=False,
                    wait_timeout=3.0,
                    lock_timeout=2.0,
                )
            finally:
                self._bridge.close_finished.emit()

        threading.Thread(target=_worker, daemon=True, name="mlx-manager-close").start()

    def _finish_close(self):
        self._close_finalizing = True
        self.close()

    def _refresh_setup_status(self):
        if not hasattr(self, "_setup_status_box"):
            return
        self._setup_status_box.setPlainText("\n".join(startup_diagnostics(self.mlx_binary)))

    def closeEvent(self, event):
        if self._close_finalizing:
            super().closeEvent(event)
            return
        if not self._close_in_progress:
            self._begin_close()
        event.ignore()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Global dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base,            QColor(SURF))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(SURF2))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(SURF))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Text,            QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button,          QColor(SURF2))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(ACC))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    win = MLXManagerWindow()
    win.show()
    sys.exit(app.exec())
