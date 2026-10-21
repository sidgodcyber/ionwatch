"""Pi deployment files: shell syntax, LF endings, and the systemd unit runs live mode with restart."""
import shutil
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


@pytest.mark.parametrize("name", ["setup_pi.sh", "demo.sh", "ionwatch-live.service"])
def test_lf_line_endings(name):
    assert b"\r\n" not in (DEPLOY / name).read_bytes()


def _working_bash():
    bash = shutil.which("bash")
    try:   # on Windows this may be the WSL launcher without a distribution
        return bash if bash and subprocess.run([bash, "-c", "true"], capture_output=True, timeout=20).returncode == 0 \
            else None
    except (OSError, subprocess.TimeoutExpired):
        return None


@pytest.mark.parametrize("name", ["setup_pi.sh", "demo.sh"])
def test_shell_script_syntax(name):
    bash = _working_bash()
    if bash is None:
        pytest.skip("no working bash")
    subprocess.run([bash, "-n"], input=(DEPLOY / name).read_bytes(), check=True)


def test_service_unit():
    unit = (DEPLOY / "ionwatch-live.service").read_text()
    assert "Restart=on-failure" in unit and "StartLimitIntervalSec=0" in unit
    exec_line = next(l for l in unit.splitlines() if l.startswith("ExecStart="))
    for part in ("-m ionwatch.cli live", "--l86 /dev/serial0", "--baud 9600", "--node-id haldwani-01",
                 "/data/live"):
        assert part in exec_line
    assert "SupplementaryGroups=dialout" in unit


def test_checklist_covers_wiring_and_acceptance():
    text = (DEPLOY / "PI_CHECKLIST.md").read_text(encoding="utf-8")
    for needle in ("GPIO15 / RXD", "GPIO14 / TXD", "3.3 V", "cat /dev/serial0", "vcgencmd get_throttled",
                   "0x0", "first fix", "acceptance test", "unplug", "validate",
                   "then 5, then 10", "-k pty", "on-board patch antenna"):
        assert needle.lower() in text.lower(), needle
