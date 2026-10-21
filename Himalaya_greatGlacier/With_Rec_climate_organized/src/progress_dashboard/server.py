"""Serve a read-only local progress dashboard for a registered run directory."""

from __future__ import annotations

import argparse
import csv
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"_error": "invalid-json", "path": str(path)}


def tail(path: Path, count: int = 24) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-count:]
    except OSError:
        return []


def csv_stats(path: Path, key: str | None = None) -> dict:
    if not path.exists():
        return {"rows": 0, "counts": {}}
    counts: dict[str, int] = {}
    rows = 0
    try:
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            for row in reader:
                rows += 1
                if key and row.get(key):
                    counts[row[key]] = counts.get(row[key], 0) + 1
    except OSError:
        return {"rows": 0, "counts": {}, "error": "read-failed"}
    return {"rows": rows, "counts": counts}


def status_payload(run_dir: Path) -> dict:
    live = read_json(run_dir / "live_status.json")
    progress = read_json(run_dir / "progress.json")
    summary = read_json(run_dir / "run_summary.json")
    attribution = read_json(run_dir / "attribution_summary.json")
    diagnostic = read_json(run_dir / "diagnostic_summary.json")
    routing = read_json(run_dir / "routing_summary.json")
    stage3_dir = run_dir.parent / "stage3_initial_state_pilot_v0"
    stage3_attempts = [
        path
        for pattern in ("operator-smoke-v*", "turnover-pilot-v*", "climate-replication-v*")
        for path in stage3_dir.glob(pattern)
    ]
    stage3_wave_dir = (
        max(
            stage3_attempts,
            key=lambda path: (
                (path / "progress.json").stat().st_mtime
                if (path / "progress.json").exists()
                else path.stat().st_mtime
            ),
        )
        if stage3_attempts
        else stage3_dir / "operator-smoke-v0"
    )
    stage3_progress = read_json(stage3_wave_dir / "progress.json")
    stage3_summary = read_json(stage3_wave_dir / "run_summary.json")
    failure_stats = csv_stats(run_dir / "failures.csv", "error_type")
    system_stats = csv_stats(run_dir / "system_attribution.csv", "attribution_class")
    now = datetime.now(timezone.utc)
    updated_at = live.get("updated_at_utc") or progress.get("updated_at_utc")
    stale_seconds = None
    if updated_at:
        try:
            stale_seconds = max(
                0.0,
                (now - datetime.fromisoformat(updated_at)).total_seconds(),
            )
        except ValueError:
            stale_seconds = None
    completed = progress.get(
        "completed_task_count", progress.get("completed_system_count", 0)
    )
    total = progress.get(
        "total_task_count", progress.get("total_system_count", 0)
    )
    state = live.get(
        "state", "complete" if total and completed >= total else "running"
    )
    if (
        not live
        and stale_seconds is not None
        and stale_seconds > 300
        and state != "complete"
    ):
        state = "stale"
    return {
        "run_dir": str(run_dir),
        "state": state,
        "progress": progress,
        "run_summary": summary,
        "attribution_summary": attribution,
        "diagnostic_summary": diagnostic,
        "routing_summary": routing,
        "stage3_progress": stage3_progress,
        "stage3_summary": stage3_summary,
        "live_status": live,
        "failure_stats": failure_stats,
        "system_stats": system_stats,
        "log_tail": tail(stage3_dir / "stage3_pilot.log")
        or live.get("events", [])
        or tail(run_dir / "attribution.log")
        or tail(run_dir / "turnover_pilot.log"),
        "stale_seconds": stale_seconds,
        "server_time_utc": now.isoformat(),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    run_dir: Path

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/status":
            body = json.dumps(
                status_payload(self.run_dir), ensure_ascii=True
            ).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
            return
        if path in ("/", "/index.html"):
            body = (SCRIPT_DIR / "dashboard.html").read_bytes()
            self._send(body, "text/html; charset=utf-8")
            return
        self._send(b"Not found", "text/plain; charset=utf-8", 404)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    DashboardHandler.run_dir = run_dir
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"dashboard http://{args.host}:{args.port}/ run_dir={run_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
