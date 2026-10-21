"""Dashboard page behaviour in a real headless browser (Edge/Chrome via DevTools protocol).

Skipped when no Chromium-family browser is installed (e.g. on the Pi)."""
import json
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from conftest import BAD_TAIL

from ionwatch import live

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            shutil.which("chromium") or "", shutil.which("chromium-browser") or "",
            shutil.which("google-chrome") or "", shutil.which("microsoft-edge") or ""]
BROWSER = next((b for b in BROWSERS if b and Path(b).exists()), None)
pytestmark = pytest.mark.skipif(BROWSER is None, reason="no Chromium-family browser for page tests")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def serve(folder):
    import uvicorn

    from ionwatch.dashboard.server import create_app
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(folder), host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}/"


class Page:
    def __init__(self, tmp_path):
        import websocket
        self.port = free_port()
        self.proc = subprocess.Popen([BROWSER, "--headless=new", "--disable-gpu", f"--remote-debugging-port={self.port}",
                                      f"--user-data-dir={tmp_path / 'profile'}", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json"))
                ws_url = next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.1)
        self.ws = websocket.create_connection(ws_url, timeout=30, suppress_origin=True)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})

    def js(self, expr):
        return self.call("Runtime.evaluate", expression=expr, returnByValue=True)["result"].get("value")

    def wait_for(self, expr, timeout=20):
        t = time.time()
        while time.time() - t < timeout:
            if self.js(expr):
                return True
            time.sleep(0.3)
        return False

    def close(self):
        self.ws.close()
        self.proc.terminate()
        self.proc.wait(timeout=10)


@pytest.fixture(scope="module")
def replay_folder(tmp_path_factory, cfg):
    out = tmp_path_factory.mktemp("page_replay")
    live.run_replay([str(BAD_TAIL)], 0, out, cfg, "haldwani-01")
    return out


@pytest.fixture
def page(tmp_path):
    p = Page(tmp_path)
    yield p
    p.close()


def test_page_with_records(replay_folder, page):
    server, url = serve(replay_folder)
    try:
        page.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=1, mobile=True)
        page.call("Page.navigate", url=url)
        assert page.wait_for("document.getElementById('event-rows').children.length > 0")
        # NavIC: the latest minutes don't list NavIC, earlier ones did (visible, not tracked)
        navic = page.js("document.getElementById('navic-status').textContent")
        assert navic.startswith("visible, not tracked (last seen ") and navic.endswith(" IST)")
        # edge_of_gap rows hidden by default, shown with the toggle
        text = "[...document.querySelectorAll('#event-rows tr')].map(r => r.textContent).join('|')"
        assert "edge of gap" not in page.js(text)
        assert "hidden" in page.js("document.getElementById('ignored-count').textContent")
        page.js("document.getElementById('show-ignored').click()")
        assert "edge of gap" in page.js(text)
        # lock-loss-only events: max fade shows an em dash
        cells = page.js("[...document.querySelectorAll('#event-rows tr td:nth-child(6)')].map(c => c.textContent)")
        assert "—" in cells and "–" not in cells
        # phone width: no horizontal page scroll
        assert page.js("document.documentElement.scrollWidth") <= 390
        assert page.js("document.getElementById('empty').hidden") is True
    finally:
        server.should_exit = True


def test_page_empty_state(tmp_path, page):
    folder = tmp_path / "empty"
    folder.mkdir()
    server, url = serve(folder)
    try:
        page.call("Page.navigate", url=url)
        assert page.wait_for("document.getElementById('refreshed').textContent.startsWith('updated')")
        assert page.js("document.getElementById('empty').hidden") is False
        assert page.js("document.getElementById('t-update').textContent") == "no records yet"
        assert page.js("document.getElementById('t-recv').textContent") == "waiting for data"
    finally:
        server.should_exit = True
