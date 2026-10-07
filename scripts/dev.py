"""Run the local LightGBM v2 API and frontend together: python scripts/dev.py."""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from nexus.bundle import BundleLoader
    from nexus.config import Settings
    from training.export_release_bundle import export_bundle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=3000)
    args = parser.parse_args()
    npm = shutil.which("npm")
    if not npm or not (ROOT / "web/node_modules/next").is_dir():
        raise SystemExit("Install frontend dependencies first: cd web && npm install")
    for port in (args.backend_port, args.frontend_port):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise SystemExit(f"Port {port} is unavailable: {exc}") from exc

    version = "v2.0.0"
    bundles = ROOT / "artifacts/bundles"
    if not (bundles / version / "manifest.json").exists():
        export_bundle(ROOT / "experiments/lightgbm_validated_v2", bundles, version)
    BundleLoader(bundles, version).load()
    env = dict(os.environ)
    env.update(
        NEXUS_BUNDLE_VERSION=version,
        NEXUS_BUNDLES_DIR=str(bundles),
        NEXUS_DATABASE_PATH=os.environ.get(
            "NEXUS_DATABASE_PATH", str(ROOT / "artifacts/nexus-v2.sqlite3")
        ),
        NEXUS_BACKEND_URL=f"http://127.0.0.1:{args.backend_port}",
        NEXUS_NEXT_DIST_DIR=f".next/local-{args.frontend_port}",
        OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"),
        NEXUS_API_TOKEN=os.environ.get("NEXUS_API_TOKEN") or secrets.token_urlsafe(32),
    )
    Settings(api_token=env["NEXUS_API_TOKEN"])
    children = []
    try:
        backend = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "nexus.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.backend_port),
            ],
            cwd=ROOT,
            env=env,
        )
        children.append(backend)
        for _ in range(120):
            if backend.poll() is not None:
                raise RuntimeError("Backend exited before becoming ready")
            try:
                with urllib.request.urlopen(f"{env['NEXUS_BACKEND_URL']}/health/ready", timeout=1):
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5)
        else:
            raise RuntimeError("Backend did not become model-ready within 60 seconds")
        children.append(
            subprocess.Popen(
                [
                    npm,
                    "run",
                    "dev",
                    "--",
                    "--hostname",
                    "127.0.0.1",
                    "--port",
                    str(args.frontend_port),
                ],
                cwd=ROOT / "web",
                env=env,
                start_new_session=True,
            )
        )
        print(
            f"LightGBM v2 ready. Open http://localhost:{args.frontend_port}/traffic. "
            "Ctrl+C stops both services.",
            flush=True,
        )
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        raise RuntimeError("A service exited; stopping the local stack")
    except KeyboardInterrupt:
        pass
    finally:
        import signal

        for child in reversed(children):
            if child is not backend:
                # npm starts a child Next.js server; stop the entire process group.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
