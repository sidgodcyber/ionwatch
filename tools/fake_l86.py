"""Fake Quectel L86: replays an NMEA fixture as a serial stream, re-stamped to a running clock,
and obeys the PMTK commands IonWatch sends (baud, output rate, sentence selection).

    python tools/fake_l86.py [--fixture tests/fixtures/l86_sim_10min.nmea] [--transport auto|pty|tcp]
                             [--speed 1] [--start 2026-10-22T14:00:00Z] [--duration-s 600]
                             [--baud 9600] [--max-rate 10]
                             [--inject-fade PRN:START:DUR:DEPTH] [--inject-dropout START:DUR]
                             [--inject-garbage T[,T...]] [--inject-disconnect START:DUR]

The first stdout line is ``PORT <path-or-url>``; give that to ``ionwatch live --l86``:
  * pty (Linux/macOS, the default there): a pseudo-terminal such as /dev/pts/5;
  * tcp (Windows, or --transport tcp): ``socket://127.0.0.1:<port>``, which pyserial opens like a
    serial port. Windows has no pty; this is the laptop substitute.

Clock: one fixture cycle (RMC ... GSV) is one simulated second. ``--speed`` simulated seconds run
per wall second (``--rate`` is an old alias), so ``--speed 40`` pairs with
``ionwatch live --clock-scale 40``. START/DUR/T values are simulated seconds from --start.

L86 behaviour:
  PMTK251,<baud>  switch baud (acked at the old baud). The serial line carries at most baud/10
                  bytes per second: at 9600 baud a full 1 Hz cycle (~0.6 kB) fits, 5-10 Hz does not
                  and epochs are cut off, exactly as on the real link.
  PMTK220,<ms>    fix interval -> output rate 1000/ms Hz, capped at --max-rate (a module that
                  cannot go faster).
  PMTK314,...     sentence selection (GLL,RMC,VTG,GGA,GSA,GSV,...).
  Settings are lost on --inject-disconnect (power loss: back to --baud and 1 Hz).

Injections:
  --inject-fade        lower the SNR of one PRN by DEPTH dB (fixture PRN numbering, e.g. 12 or 65)
  --inject-dropout     send nothing for DUR seconds (receiver silent, port stays open)
  --inject-garbage     at T send binary garbage, a line cut in half and a bad-checksum sentence
  --inject-disconnect  tcp: drop the client and refuse connections for DUR seconds (unplug)
"""
from __future__ import annotations

import argparse
import os
import select
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "l86_sim_10min.nmea"
PMTK314_FIELDS = ["GLL", "RMC", "VTG", "GGA", "GSA", "GSV"]


def checksum(body: str) -> str:
    x = 0
    for ch in body:
        x ^= ord(ch)
    return f"{x:02X}"


def sentence(body: str) -> str:
    return f"${body}*{checksum(body)}"


def load_cycles(path: Path) -> list[list[str]]:
    cycles, cur = [], []
    for line in path.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line.startswith("$"):
            continue
        if line[3:6] == "RMC" and cur:
            cycles.append(cur)
            cur = []
        cur.append(line)
    if cur:
        cycles.append(cur)
    return cycles


def restamp(cycle: list[str], t: datetime, fade: tuple | None, k: int, enabled: set) -> list[str]:
    out = []
    for s in cycle:
        f = s[1:s.rindex("*")].split(",")
        kind = f[0][2:]
        if kind not in enabled:
            continue
        if kind in ("RMC", "GGA"):
            f[1] = f"{t:%H%M%S}.{t.microsecond // 1000:03d}"
            if kind == "RMC" and len(f) > 9:
                f[9] = f"{t:%d%m%y}"
        elif kind == "GSV" and fade and fade[1] <= k < fade[1] + fade[2]:
            for i in range(4, len(f) - 3, 4):
                if f[i].strip().isdigit() and int(f[i]) == fade[0] and f[i + 3]:
                    f[i + 3] = f"{max(int(f[i + 3]) - fade[3], 0):02d}"
        out.append(sentence(",".join(f)))
    return out


def garbage(t: datetime) -> list[bytes]:
    hms = f"{t:%H%M%S}.000"
    body = f"GPGGA,{hms},2915.336,N,07932.220,E,1,08,1.0,500.0,M,-40.0,M,,"
    return [b"\x00\xff\xfe\x13 noise \x81\x82\r\n",
            ("$GPGSV,3,1,11,02,62,040,4").encode(),           # cut in half, no newline
            f"${body}*00\r\n".encode()]                        # bad checksum


class Transport:
    path = ""

    def send(self, data: bytes) -> None: ...
    def connected(self) -> bool: ...
    def poll(self) -> bytes: ...
    def disconnect(self) -> None: ...
    def close(self) -> None: ...


class PtyTransport(Transport):
    def __init__(self):
        import tty
        self.master, slave = os.openpty()
        tty.setraw(slave)
        self.path = os.ttyname(slave)
        self._slave = slave
        os.set_blocking(self.master, False)

    def send(self, data):
        try:
            os.write(self.master, data)
        except BlockingIOError:      # nobody reading: drop, like a UART would
            pass

    def connected(self):
        return True

    def poll(self) -> bytes:
        got = b""
        try:
            while select.select([self.master], [], [], 0)[0]:
                got += os.read(self.master, 4096)
        except OSError:
            pass
        return got

    def disconnect(self):
        pass

    def close(self):
        os.close(self.master)
        os.close(self._slave)


class TcpTransport(Transport):
    def __init__(self, port: int):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", port))
        self.srv.listen(1)
        self.srv.setblocking(False)
        self.path = f"socket://127.0.0.1:{self.srv.getsockname()[1]}"
        self.cli = None
        self.refuse = False

    def poll(self) -> bytes:
        if self.cli is None:
            try:
                c, _ = self.srv.accept()
            except (BlockingIOError, OSError):
                return b""
            if self.refuse:
                c.close()
                return b""
            c.setblocking(False)
            self.cli = c
        got = b""
        try:
            while True:
                b = self.cli.recv(4096)
                if not b:                # reader closed the port (e.g. reopening at a new baud)
                    self.cli.close()
                    self.cli = None
                    break
                got += b
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:
            self.cli = None
        return got

    def connected(self):
        return self.cli is not None

    def send(self, data):
        if self.cli is None:
            return
        try:
            self.cli.setblocking(True)
            self.cli.sendall(data)
            self.cli.setblocking(False)
        except OSError:
            self.cli = None

    def disconnect(self):
        if self.cli is not None:
            try:
                self.cli.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.cli.close()
            self.cli = None

    def close(self):
        self.disconnect()
        self.srv.close()


class Module:
    """L86 state driven by PMTK commands."""

    def __init__(self, baud: int, max_rate: float):
        self.default_baud, self.max_rate = baud, max_rate
        self.reset()
        self._buf = b""

    def reset(self):
        self.baud, self.rate = self.default_baud, 1.0
        self.enabled = {"RMC", "GGA", "GSA", "GSV", "VTG", "GLL"}

    def command(self, data: bytes) -> list[bytes]:
        """Handle incoming bytes; return acks to send."""
        acks = []
        self._buf += data
        *lines, self._buf = self._buf.split(b"\n")
        for raw in lines:
            line = raw.decode("ascii", "replace").strip()
            if not line.startswith("$PMTK") or "*" not in line:
                continue
            body = line[1:line.rindex("*")]
            if checksum(body) != line[line.rindex("*") + 1:].upper()[:2]:
                continue
            f = body.split(",")
            cmd = f[0][4:]
            flag = "3"
            if cmd == "251" and len(f) > 1 and f[1].isdigit():
                acks.append(sentence(f"PMTK001,{cmd},{flag}").encode() + b"\r\n")   # acked at the old baud
                self.baud = int(f[1])
                print(f"SIM baud {self.baud}", flush=True)
                continue
            if cmd == "220" and len(f) > 1 and f[1].isdigit():
                self.rate = min(1000 / max(int(f[1]), 100), self.max_rate)
                print(f"SIM rate {self.rate:g} Hz", flush=True)
            elif cmd == "314":
                vals = f[1:]
                self.enabled = {n for n, v in zip(PMTK314_FIELDS, vals) if v.strip() not in ("", "0")}
                print(f"SIM sentences {sorted(self.enabled)}", flush=True)
            else:
                flag = "1"
            acks.append(sentence(f"PMTK001,{cmd},{flag}").encode() + b"\r\n")
        return acks


def span(s: str | None) -> tuple[int, int] | None:
    if not s:
        return None
    a, b = s.split(":")
    return int(a), int(b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--transport", choices=["auto", "pty", "tcp"], default="auto")
    ap.add_argument("--tcp-port", type=int, default=0, help="0 = any free port")
    ap.add_argument("--speed", "--rate", dest="speed", type=float, default=1.0,
                    help="simulated seconds per wall second")
    ap.add_argument("--start", default=None, help="simulated start time, ISO UTC (default: now)")
    ap.add_argument("--duration-s", type=int, default=None, help="simulated seconds to send (default: fixture)")
    ap.add_argument("--baud", type=int, default=9600, help="power-on baud of the module")
    ap.add_argument("--max-rate", type=float, default=10.0, help="highest output rate the module accepts")
    ap.add_argument("--wait-client-s", type=float, default=30.0, help="tcp: wait this long for the first client")
    ap.add_argument("--inject-fade", default=None, help="PRN:START:DUR:DEPTH")
    ap.add_argument("--inject-dropout", default=None, help="START:DUR")
    ap.add_argument("--inject-garbage", default="", help="T[,T...]")
    ap.add_argument("--inject-disconnect", default=None, help="START:DUR (tcp)")
    a = ap.parse_args(argv)

    cycles = load_cycles(Path(a.fixture))
    total = a.duration_s or len(cycles)
    start = (datetime.fromisoformat(a.start.replace("Z", "+00:00")) if a.start
             else datetime.now(timezone.utc).replace(microsecond=0))
    fade = tuple(int(x) for x in a.inject_fade.split(":")) if a.inject_fade else None
    dropout, disconnect = span(a.inject_dropout), span(a.inject_disconnect)
    garbage_at = {int(x) for x in a.inject_garbage.split(",") if x.strip()}
    mod = Module(a.baud, a.max_rate)

    kind = a.transport
    if kind == "auto":
        kind = "pty" if hasattr(os, "openpty") and sys.platform != "win32" else "tcp"
    tr = PtyTransport() if kind == "pty" else TcpTransport(a.tcp_port)
    print(f"PORT {tr.path}", flush=True)

    def service():
        for ack in mod.command(tr.poll()):
            tr.send(ack)

    if kind == "tcp":                      # start the clock when the reader is there
        t_end = time.monotonic() + a.wait_client_s
        while not tr.connected() and time.monotonic() < t_end:
            service()
            time.sleep(0.01)
    wall0 = time.monotonic()
    try:
        for k in range(total):
            if disconnect and k == disconnect[0]:
                tr.refuse = kind == "tcp"
                tr.disconnect()
                mod.reset()                                  # power lost: settings gone
                print(f"SIM {k}s disconnect (settings reset)", flush=True)
            if disconnect and k == disconnect[0] + disconnect[1]:
                tr.refuse = False
                print(f"SIM {k}s reconnect allowed", flush=True)
            silent = dropout and dropout[0] <= k < dropout[0] + dropout[1]
            budget = mod.baud / 10                          # bytes this second (8N1)
            n = max(1, int(round(mod.rate)))
            for j in range(n):
                target = wall0 + (k + j / n) / a.speed
                while (rem := target - time.monotonic()) > 0:
                    service()
                    time.sleep(min(rem, 0.005))
                service()
                if silent:
                    continue
                t = start + timedelta(seconds=k + j / n)
                if j == 0 and k in garbage_at:
                    for g in garbage(t):
                        tr.send(g)
                data = ("\r\n".join(restamp(cycles[k % len(cycles)], t, fade, k, mod.enabled)) + "\r\n").encode()
                if budget <= 0:
                    continue
                if len(data) > budget:                      # the line is full: the epoch is cut off
                    data = data[:int(budget)]
                budget -= len(data)
                tr.send(data)
    except KeyboardInterrupt:
        pass
    finally:
        print("SIM done", flush=True)
        time.sleep(0.2)
        tr.close()


if __name__ == "__main__":
    main()
