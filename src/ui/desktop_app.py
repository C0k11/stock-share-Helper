import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


API_URL = "http://127.0.0.1:8000"
UI_URL = "http://127.0.0.1:8501"


def _http_ok(url: str, timeout_s: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=float(timeout_s)) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 400
    except (urllib.error.URLError, ValueError):
        return False


def _wait_http(url: str, timeout_s: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < float(timeout_s):
        if _http_ok(url, timeout_s=1.0):
            return True
        time.sleep(0.3)
    return False


def _start_uvicorn() -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.Popen(cmd, cwd=str(_repo_root()), env=env)


def _start_streamlit() -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/ui/streamlit_app.py",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    return subprocess.Popen(cmd, cwd=str(_repo_root()), env=env)


def _repo_root():
    # file = <repo>/src/ui/desktop_app.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    api_proc: subprocess.Popen | None = None
    ui_proc: subprocess.Popen | None = None

    # If already running (maybe started in another terminal), reuse.
    if not _http_ok(f"{API_URL}/", timeout_s=0.5):
        api_proc = _start_uvicorn()
        if not _wait_http(f"{API_URL}/", timeout_s=30.0):
            raise SystemExit("FastAPI did not start on http://127.0.0.1:8000")

    if not _http_ok(UI_URL, timeout_s=0.5):
        ui_proc = _start_streamlit()
        if not _wait_http(UI_URL, timeout_s=60.0):
            raise SystemExit("Streamlit did not start on http://127.0.0.1:8501")

    try:
        import webview  # pywebview
    except Exception as e:
        raise SystemExit(f"pywebview import failed: {e}. Install with: python -m pip install pywebview")

    window = webview.create_window("AI Trading Terminal", UI_URL, width=1440, height=900)

    def _cleanup() -> None:
        for p in [ui_proc, api_proc]:
            if p is None:
                continue
            try:
                p.terminate()
            except Exception:
                pass

    try:
        window.events.closed += lambda: _cleanup()
    except Exception:
        pass

    try:
        webview.start(debug=False)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
