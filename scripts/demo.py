#!/usr/bin/env python3
"""
scripts/demo.py — Oceanographic MLOps classroom demonstration orchestrator.

Subcommands:
    start    Run preflight, export bundles if needed, start Docker Compose services,
             wait for health, then print service URLs.
    status   Show current container health and service status.
    verify   Submit a Gulf of Mannar prediction and verify all three services.
    stop     Stop services cleanly without deleting host data or the registry.

Usage:
    python scripts/demo.py start
    python scripts/demo.py start --skip-preflight
    python scripts/demo.py status
    python scripts/demo.py verify
    python scripts/demo.py stop

Exit codes:
    0  Success
    1  Failure (details printed to stderr)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = sys.executable

# ── Service configuration ──────────────────────────────────────────────────────
SERVICES = {
    "mlflow": {
        "container": "coralsense-mlflow",
        "url": "http://localhost:5000/health",
        "label": "MLflow UI",
        "public_url": "http://localhost:5000",
    },
    "api": {
        "container": "coralsense-api",
        "url": "http://localhost:8000/health",
        "label": "FastAPI",
        "public_url": "http://localhost:8000",
    },
    "dashboard": {
        "container": "coralsense-dashboard",
        "url": "http://localhost:8501/_stcore/health",
        "label": "Streamlit dashboard",
        "public_url": "http://localhost:8501",
    },
}

# Gulf of Mannar healthy reef observation used for the demo prediction
_DEMO_OBSERVATION = {
    "region": "Gulf of Mannar",
    "depth_m": 5.0,
    "water_temperature_c": 27.5,
    "ph": 8.1,
    "salinity_ppt": 35.0,
    "dissolved_oxygen_mg_l": 7.0,
    "turbidity_ntu": 2.0,
    "light_intensity": 800.0,
    "current_speed_m_s": 0.2,
    "sonar_backscatter": -15.0,
    "rugosity_index": 3.5,
    "hard_substrate_percentage": 60.0,
    "acoustic_complexity_index": 0.7,
    "coral_cover_percentage": 45.0,
    "bleaching_percentage": 5.0,
    "disease_percentage": 2.0,
}

HEALTH_POLL_SECONDS = 5
HEALTH_MAX_ATTEMPTS = 36  # up to 3 minutes per service


# ── Utilities ──────────────────────────────────────────────────────────────────
def _run(
    cmd: list[str], *, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=str(_ROOT),
        check=check,
    )


def _http_get(url: str, timeout: int = 8) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, str(exc)
    except Exception as exc:
        return 0, str(exc)


def _http_post_json(url: str, payload: dict, timeout: int = 15) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except Exception as exc:
        return 0, str(exc)


def _poll_health(service_name: str, url: str) -> bool:
    print(f"  Waiting for {service_name}...", end="", flush=True)
    for attempt in range(1, HEALTH_MAX_ATTEMPTS + 1):
        code, _ = _http_get(url)
        if code == 200:
            print(f" healthy ({attempt * HEALTH_POLL_SECONDS}s)")
            return True
        print(".", end="", flush=True)
        time.sleep(HEALTH_POLL_SECONDS)
    print(f" TIMEOUT after {HEALTH_MAX_ATTEMPTS * HEALTH_POLL_SECONDS}s")
    return False


def _containers_running() -> dict[str, str]:
    """Return {container_name: status} for all project containers."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    containers: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            name, status = parts
            if any(c["container"] == name for c in SERVICES.values()):
                containers[name] = status
    return containers


# ── Subcommands ────────────────────────────────────────────────────────────────
def cmd_start(skip_preflight: bool = False) -> int:
    print("\nOceanographic MLOps — Demo Start")
    print("=" * 50)

    # 1. Preflight
    if not skip_preflight:
        print("\n[1/5] Running preflight checks...")
        result = subprocess.run(
            [_PYTHON, "scripts/preflight.py"],
            cwd=str(_ROOT),
            text=True,
        )
        if result.returncode != 0:
            print("  Preflight reported blocking failures. Fix them before starting demo.")
            print("  To start anyway:  python scripts/demo.py start --skip-preflight")
            return 1
        print("  Preflight passed.")
    else:
        print("\n[1/5] Preflight skipped (--skip-preflight).")

    # 2. Export / verify bundles
    print("\n[2/5] Checking deployment bundles...")
    bundle_manifest = _ROOT / "deploy" / "bundles" / "manifest.json"
    if not bundle_manifest.exists():
        print("  Bundles not found — exporting champion models...")
        _run([_PYTHON, "scripts/export_champions.py"])
    result = subprocess.run(
        [_PYTHON, "scripts/verify_deployment_bundle.py"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    if result.returncode != 0:
        print("  Bundle verification failed:")
        print(result.stdout[-500:])
        return 1
    print("  Deployment bundles verified.")

    # 3. Validate Docker Compose file
    print("\n[3/5] Validating Docker Compose configuration...")
    result = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    if result.returncode != 0:
        print(f"  docker-compose.yml validation failed: {result.stderr[:300]}")
        return 1
    print("  docker-compose.yml is valid.")

    # 4. Start services
    print("\n[4/5] Starting Docker services (this may take 2-3 minutes first run)...")
    _run(["docker", "compose", "up", "--build", "-d"])
    print("  Services started in background.")

    # 5. Health polling
    print("\n[5/5] Waiting for services to become healthy...")
    all_healthy = True
    for _svc_name, svc_cfg in SERVICES.items():
        ok = _poll_health(svc_cfg["label"], svc_cfg["url"])
        if not ok:
            print(f"  ERROR: {svc_cfg['label']} did not become healthy.")
            all_healthy = False

    print()
    if all_healthy:
        print("All services are healthy. Demo is ready.")
        print()
        print("  Service URLs:")
        for svc_cfg in SERVICES.values():
            print(f"    {svc_cfg['label']:<24} {svc_cfg['public_url']}")
        print()
        print("  Quick verification:  python scripts/demo.py verify")
        print("  Stop demo:           python scripts/demo.py stop")
        return 0
    else:
        print("Some services failed to start. Check logs:")
        print("  docker compose logs -f")
        return 1


def cmd_status() -> int:
    print("\nOceanographic MLOps — Demo Status")
    print("=" * 50)

    # Container state
    print("\nContainer state:")
    containers = _containers_running()
    if containers:
        for name, status in containers.items():
            print(f"  {name:<30} {status}")
    else:
        print("  No project containers found.")

    print("\nService health:")
    for _svc_name, svc_cfg in SERVICES.items():
        code, body = _http_get(svc_cfg["url"])
        status = "healthy" if code == 200 else f"unreachable (HTTP {code})"
        print(f"  {svc_cfg['label']:<24} {status}")

    # Champion metadata
    print("\nChampion model metadata:")
    code, body = _http_get("http://localhost:8000/model-info")
    if code == 200:
        try:
            info = json.loads(body)
            for task, meta in info.get("models", {}).items():
                algo = meta.get("algorithm", "?")
                ver = meta.get("version", "?")
                print(f"  {task:<14} v{ver}  {algo}")
        except (json.JSONDecodeError, KeyError):
            print("  (could not parse model-info response)")
    else:
        print("  FastAPI not available — start demo first: python scripts/demo.py start")

    return 0


def cmd_verify() -> int:
    print("\nOceanographic MLOps — Demo Verify")
    print("=" * 50)
    all_ok = True

    # 1. FastAPI health
    code, _ = _http_get("http://localhost:8000/health")
    ok = code == 200
    print(f"  FastAPI /health       {'OK' if ok else 'FAIL (HTTP ' + str(code) + ')'}")
    all_ok = all_ok and ok

    # 2. FastAPI model-info
    code, body = _http_get("http://localhost:8000/model-info")
    ok = code == 200
    print(f"  FastAPI /model-info   {'OK' if ok else 'FAIL (HTTP ' + str(code) + ')'}")
    all_ok = all_ok and ok

    # 3. Gulf of Mannar prediction
    code, resp = _http_post_json("http://localhost:8000/predict/both", _DEMO_OBSERVATION)
    if code == 200 and isinstance(resp, dict):
        health = resp.get("health", {})
        restoration = resp.get("restoration", {})
        h_class = health.get("predicted_class", "?")
        h_conf = health.get("confidence", 0)
        r_class = restoration.get("predicted_class", "?")
        r_conf = restoration.get("confidence", 0)

        # Verify probability sums to ~1
        h_probs = sum(health.get("probabilities", {}).values())
        r_probs = sum(restoration.get("probabilities", {}).values())
        prob_ok = abs(h_probs - 1.0) < 0.01 and abs(r_probs - 1.0) < 0.01

        print("  Gulf of Mannar prediction:")
        print(f"    reef_health          {h_class} (confidence={h_conf:.3f})")
        print(f"    restoration          {r_class} (confidence={r_conf:.3f})")
        print(
            f"    probability sums     health={h_probs:.4f}  restoration={r_probs:.4f}  "
            f"{'OK' if prob_ok else 'WARN'}"
        )

        h_version = health.get("model_version", "?")
        h_alias = health.get("model_alias", "?")
        r_version = restoration.get("model_version", "?")
        r_alias = restoration.get("model_alias", "?")
        print(f"    health model         v{h_version} alias={h_alias}")
        print(f"    restoration model    v{r_version} alias={r_alias}")
        all_ok = all_ok and prob_ok
    else:
        print(f"  Prediction FAIL (HTTP {code}): {str(resp)[:200]}")
        all_ok = False

    # 4. Dashboard health
    code, _ = _http_get("http://localhost:8501/_stcore/health")
    ok = code == 200
    print(f"  Streamlit health      {'OK' if ok else 'FAIL (HTTP ' + str(code) + ')'}")
    all_ok = all_ok and ok

    # 5. MLflow health
    code, _ = _http_get("http://localhost:5000/health")
    ok = code == 200
    print(f"  MLflow health         {'OK' if ok else 'FAIL (HTTP ' + str(code) + ')'}")
    all_ok = all_ok and ok

    print()
    print(f"  Result: {'VERIFICATION PASSED' if all_ok else 'VERIFICATION FAILED'}")
    return 0 if all_ok else 1


def cmd_stop() -> int:
    print("\nOceanographic MLOps — Demo Stop")
    print("=" * 50)
    print("\nStopping Docker services (data and registry are preserved)...")

    result = subprocess.run(
        ["docker", "compose", "down", "--timeout", "30"],
        cwd=str(_ROOT),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  Warning: docker compose down returned {result.returncode}")
        print(result.stderr[:300])
    else:
        print("  Services stopped.")

    # Confirm no project containers remain
    containers = _containers_running()
    remaining = {n: s for n, s in containers.items() if "Exited" not in s and "Up" in s}
    if remaining:
        print(f"  WARNING: containers still running: {list(remaining.keys())}")
        return 1

    print("  No project containers running.")
    print("  Canonical registry is intact — champion aliases unchanged.")
    return 0


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Oceanographic MLOps demonstration orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start the demo stack")
    p_start.add_argument(
        "--skip-preflight", action="store_true", help="Skip preflight checks (not recommended)"
    )

    sub.add_parser("status", help="Show current service status")
    sub.add_parser("verify", help="Verify all services and submit a test prediction")
    sub.add_parser("stop", help="Stop demo services cleanly")

    args = parser.parse_args()

    if args.command == "start":
        return cmd_start(skip_preflight=args.skip_preflight)
    elif args.command == "status":
        return cmd_status()
    elif args.command == "verify":
        return cmd_verify()
    elif args.command == "stop":
        return cmd_stop()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
