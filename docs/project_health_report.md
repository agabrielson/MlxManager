# MLX Manager Project Health Report

Last generated: 2026-05-16T00:19:12.902973+00:00

This is the living health report for MLX Manager. It tracks whether the project is safe to change, pleasant to operate, and reliable as a local OpenAI-compatible gateway for MLX models.

The report deliberately combines classic computer-science/code metrics with operator metrics that matter for this specific code region: GUI responsiveness, model lifecycle, sleep/wake, gateway compatibility, token accounting, settings persistence, and battery/process hygiene.

## Current Snapshot

| Area | Status | Evidence |
| --- | --- | --- |
| Compile check | Green | /Users/agabriel/Documents/cs/mlx-llm/mlx-env/bin/python |
| Gateway status | Green | http://127.0.0.1:9000; health=200, ready=503, state=sleeping |
| Settings persistence | Green | mode=0o600; hf_token=present |
| Process hygiene | Green | manager=1; mlx_server=0 |
| Model inventory | Green | 5 cached mlx-community model(s), approx 39.9 GB |

## MLX-Specific Health Scorecard

| Area | Track | Why |
| --- | --- | --- |
| Startup/wake reliability | Startup success rate, p50/p95 startup time, wake-on-request success, wake latency, stuck-warming count. | Core user experience and client reliability. |
| Sleep/battery behavior | Idle timeout accuracy, real-question/probe split, CPU and memory while running/light sleep/deep sleep/stopped. | Keeps local inference useful on a laptop. |
| Gateway contract | `/health`, `/ready`, `/status`, `/v1/models`, `/v1/chat/completions`, streaming/non-streaming behavior. | Keeps LLM-OS, opencode, scripts, and IDE tools interoperable. |
| Auth/TLS matrix | HTTP/HTTPS, auth on/off, localhost/all-hosts, SSL verification on/off. | Most connection bugs hide at configuration edges. |
| Config persistence | Selected model, last-running model, HF token availability, gateway token, host mode, sleep mode, file mode `0600`. | Prevents silent config drift and forgotten credentials. |
| Process hygiene | Orphaned `mlx_lm` servers, port collisions, close latency, forced-kill count. | Prevents battery drain and stuck shutdowns. |
| UI responsiveness | Long work on UI thread, button latency, close latency, log visibility. | The manager must stay usable while models warm or fail. |
| Client sync | LLM-OS sync success/failure, base URL, protocol, auth flag, token-present status, model name. | Avoids “connected” status with broken requests. |
| Token telemetry | Real chat request count, prompt/completion/total tokens, exact vs estimated source. | Explains use patterns and helps tune battery/performance tradeoffs. |
| Model inventory | Cached model count, disk use, selected model exists, HF browse/download success. | Prevents selected-model startup surprises. |
| Security hygiene | No secrets in logs/git, masked diagnostics, auth-on when all-hosts, cert/key presence. | Important when the gateway is reachable beyond localhost. |

## Computer Engineering Metrics

### LOC Baseline

| Area | Lines |
| --- | --- |
| Assets | 0 |
| Documentation | 1,178 |
| Git hooks | 4 |
| Health automation | 730 |
| Python product code | 2,615 |
| Repo metadata/config | 7 |
| Total reported LOC | 4,534 |

### Complexity Snapshot

| Metric | Current Value |
| --- | --- |
| Python files analyzed | 3 |
| Python functions analyzed | 165 |
| Functions with cyclomatic complexity `> 10` | 19 |
| Functions with cognitive complexity `> 15` | 14 |
| Functions longer than 75 lines | 5 |
| Files longer than 750 lines | 1 |

Complexity should be read as a refactoring signal, not a grade. For this project, high-complexity code is most risky when it also controls gateway routing, subprocess lifecycle, sleep/wake, auth/TLS, or persistent settings.

### CS Best Practices For This Code Region

| Practice | Guidance | Why It Matters Here |
| --- | --- | --- |
| Separate mechanism from policy | Keep generic gateway/process/settings helpers reusable; keep MLX-specific policy in small call sites. | Makes the manager easier to adapt for opencode or other local clients. |
| Keep UI thread thin | Never perform network calls, model startup, shutdown waits, downloads, or gateway blocking work on the Qt UI thread. | Protects responsiveness during warmup, sleep, TLS failures, and model download. |
| Use explicit state transitions | Represent running, warming, sleeping, deep sleep, stopped, and error as deliberate states with logs and API status. | Prevents clients from guessing whether to retry, wake, or fail. |
| Make side effects bounded | Every subprocess, thread, socket, and timer should have a clear owner and bounded shutdown path. | Avoids orphaned `mlx_lm` servers, stuck ports, and battery drain. |
| Treat configuration as data | Persist settings through one adapter, mask secrets in diagnostics, and report token fields only as present/missing. | Prevents the class of config drift that made the HF token appear forgotten. |
| Test contracts, not only functions | For gateway work, verify HTTP status, JSON envelope, auth behavior, retry semantics, and state fields. | Clients depend on the contract more than the internal implementation. |
| Prefer small deterministic checks in hooks | Pre-commit should compile, regenerate/check health, and avoid live model dependencies. | Keeps commits fast while still catching broken startup syntax and stale reports. |
| Use hotspots to guide refactors | Refactor files/functions that are complex, changing, and operationally critical before cosmetic cleanup. | Targets the next bug cluster instead of chasing raw LOC. |

### Suggested Engineering Thresholds

| Metric | Threshold | Preferred Response |
| --- | --- | --- |
| Cyclomatic complexity | `> 10` review; `> 20` refactor candidate | Split branch-heavy request handling, gateway state, and settings normalization. |
| Cognitive complexity | `> 15` review; `> 25` refactor candidate | Flatten nested state logic and move decision tables into helpers. |
| Function length | `> 75` review; `> 150` refactor candidate | Extract one responsibility: parsing, status building, forwarding, or UI rendering. |
| File length | `> 750` review | Consider splitting by lifecycle, gateway, settings, telemetry, or UI widgets. |
| Thread/process ownership | One owner per worker/process/socket | Name threads, use bridge signals for GUI updates, and keep shutdown bounded. |
| Runtime contract coverage | Every endpoint and mode has a smoke check | Cover HTTP/HTTPS, auth on/off, sleeping/warming/running, streaming/non-streaming. |
| Secret exposure | Zero raw tokens in docs, logs, or commits | Only report present/missing and run staged diff secret scans before push. |

### Top Hotspots

| Rank | File | LOC | Max Cyclomatic | Max Cognitive | Max Function Length | Structural Heuristic |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | mlx_manager.py | 2513 | 36 | 41 | 358 | 0 |
| 2 | scripts/project_health_snapshot.py | 712 | 22 | 23 | 197 | 0 |
| 3 | mlx_manager_bootstrap.py | 102 | 7 | 6 | 35 | 0 |
| 4 | README.md | 563 | 0 | 0 | 0 | 50 |
| 5 | docs/project_health_report.md | 132 | 0 | 0 | 0 | 82 |
| 6 | docs/c4_architecture.md | 483 | 0 | 0 | 0 | 29 |
| 7 | scripts/install_git_hooks.sh | 9 | 0 | 0 | 0 | 0 |
| 8 | scripts/pre_commit_check.sh | 9 | 0 | 0 | 0 | 0 |

### Top Python Functions By Cyclomatic Complexity

| Rank | File | Function | Line | Cyclomatic | Cognitive | Length |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | mlx_manager.py | _ensure_gateway_running | 1900 | 36 | 41 | 177 |
| 2 | mlx_manager.py | _ensure_server_running | 2247 | 24 | 27 | 63 |
| 3 | scripts/project_health_snapshot.py | render_markdown | 490 | 22 | 23 | 197 |
| 4 | mlx_manager.py | _gateway_models_payload | 1804 | 20 | 25 | 45 |
| 5 | mlx_manager.py | _set_cached_models | 1256 | 18 | 21 | 62 |
| 6 | mlx_manager.py | load_settings | 219 | 18 | 17 | 34 |
| 7 | mlx_manager.py | _content_text | 1480 | 15 | 25 | 17 |
| 8 | mlx_manager.py | _start_idle_watchdog | 1137 | 14 | 22 | 28 |

## Runtime And Operator Metrics

| Metric | Current Value |
| --- | --- |
| Settings file | /Users/agabriel/Documents/cs/mlx-llm/.mlx_manager.json |
| Settings permissions | 0o600 (ok) |
| HF token | present |
| Gateway auth | enabled |
| Gateway mode | http / all / port 9000 |
| Idle policy | light after 15.0 min (enabled) |
| Manager process count | 1 |
| MLX server process count | 0 |
| Cached mlx-community models | 5 |
| Cached model disk footprint | 39.9 GB |
| Assets tracked | 11 files, 1.3 MB |

## Pre-Commit Policy

Pre-commit should stay fast and deterministic. The tracked hook runs syntax compilation plus static health snapshot generation. Full runtime checks should be run manually before a release or after changing gateway/sleep/auth behavior.

Recommended commands:

```bash
scripts/pre_commit_check.sh
scripts/project_health_snapshot.py --static-only --check
scripts/project_health_snapshot.py --write
```

Install the optional local hook with:

```bash
scripts/install_git_hooks.sh
```

## Open Health Gaps

- Add automated startup/wake latency sampling once a stable small default model is selected.
- Add a repeatable auth/TLS matrix test for HTTP, HTTPS, auth-on, auth-off, localhost, and all-hosts modes.
- Add a sleep/wake soak test that proves probes do not reset idle timers but real chat requests do.
- Add a close/shutdown regression that verifies the GUI exits without orphaning `mlx_lm` servers.
- Add client-sync smoke tests against LLM-OS and future opencode-style clients.

## Interpretation

The most important risk in this repo is not raw LOC. It is the state boundary between GUI, gateway, subprocess, settings, and external clients. Health work should prioritize anything that makes that boundary more observable and repeatable.
