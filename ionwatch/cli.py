"""IonWatch command-line entry point.

    python -m ionwatch.cli run --phone FILE [--phone FILE ...] [--l86 FILE] --out DIR [--config FILE]
    python -m ionwatch.cli inspect --phone FILE [--l86 FILE] [--out DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from ionwatch import pipeline
from ionwatch.io import gnsslogger
from ionwatch.proc.common import load_config


@click.group()
def cli():
    """IonWatch offline processing pipeline."""


@cli.command()
@click.option("--phone", "phone", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="GnssLogger .txt file (repeatable; split sessions are merged).")
@click.option("--l86", type=click.Path(exists=True, dir_okay=False), help="L86 NMEA log (optional).")
@click.option("--out", required=True, type=click.Path(file_okay=False),
              help="Output folder; its name becomes session_id.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--node-id", default=None, help="Node id written into minutes.jsonl (default: config node_id).")
@click.option("--session", "session_path", default=None, type=click.Path(exists=True, dir_okay=False),
              help="session.json (default: found next to the inputs, e.g. data/live/session.json).")
@click.option("--phone2", "phone2", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="Second phone's GnssLogger log at the same spot (may be L1-only).")
@click.option("--trim-start", default=0.0, type=float, help="Drop this many minutes of setup at the start of the data.")
@click.option("--trim-end", default=0.0, type=float, help="Drop this many minutes at the end of the data.")
@click.option("--no-plots", is_flag=True, help="Skip PNG plots.")
def run(phone, l86, out, config_path, node_id, session_path, phone2, trim_start, trim_end, no_plots):
    """Process a session: indices, events, plots, summary and minutes.jsonl."""
    if not phone and not l86:
        raise click.UsageError("give --phone and/or --l86")
    cfg = load_config(config_path)
    res = pipeline.run(list(phone), l86, out, cfg, plots=not no_plots, node_id=node_id, session=session_path,
                       trim_start_min=trim_start, trim_end_min=trim_end, phone2=list(phone2) or None)
    click.echo(f"session {res['session_id']}: receivers {', '.join(res['receivers'])}")
    for name, df in res["tables"].items():
        click.echo(f"  {name:<20} {len(df):>6} rows")
    for stage, note in res["notes"].items():
        click.echo(f"  note [{stage}]: {note}")
    click.echo(f"minutes.jsonl: {res['minutes']} records")
    click.echo(f"summary: {Path(out) / 'summary.md'}")


@cli.group()
def session():
    """Session notes (antenna, phone, free text). Times always come from the data."""


@session.command("note")
@click.option("--antenna", required=True, type=click.Choice(["external", "onboard"]))
@click.option("--phone", "phone_name", default="", help='e.g. "iQOO vivo I2221"')
@click.option("--phone2", "phone2_name", default="", help='second phone, e.g. "OnePlus CPH2569"')
@click.option("--notes", default="", help="Free text: site, sky view, anything unusual.")
@click.option("--dir", "folder", default="data/live", show_default=True, type=click.Path(file_okay=False),
              help="Where to write session.json (the live output folder).")
def session_note(antenna, phone_name, phone2_name, notes, folder):
    """Write session.json with the antenna, phone(s) and notes (no timing fields)."""
    from ionwatch import session as sessmod
    path = sessmod.write_note(folder, antenna, phone_name, notes, phone2=phone2_name)
    click.echo(f"wrote {path}")


@cli.command()
@click.option("--l86", "port", required=True,
              help="Serial port (/dev/serial0, COM3) or pyserial URL (socket://127.0.0.1:5000).")
@click.option("--baud", default=9600, show_default=True, type=int)
@click.option("--node-id", default=None, help="Default: config node_id.")
@click.option("--out", default="data/live", show_default=True, type=click.Path(file_okay=False))
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--clock-scale", default=1.0, show_default=True, type=float,
              help="Simulated seconds per wall second between data (fast clock for the simulator).")
@click.option("--l86-rate", default="1", show_default=True, type=click.Choice(["1", "5", "10"]),
              help="L86 output rate. Above 1 Hz: 115200 baud, RMC/GGA/GSA/GSV only, verified over 30 s, "
                   "falls back to 1 Hz at --baud if the measured rate is too low.")
@click.option("--max-minutes", default=None, type=int, help="Stop after this many minutes have their final record.")
@click.option("--duration-s", default=None, type=float, help="Stop after this many wall seconds.")
def live(port, baud, node_id, out, config_path, clock_scale, l86_rate, max_minutes, duration_s):
    """Read the L86 continuously; publish a provisional and a final record per minute to
    minutes.jsonl and refresh latest.json."""
    from ionwatch import live as livemod
    cfg = load_config(config_path)
    n = livemod.run_live(port, baud, out, cfg, node_id or cfg["node_id"], clock_scale=clock_scale,
                         max_minutes=max_minutes, duration_s=duration_s, l86_rate=int(l86_rate))
    click.echo(f"{n} records written to {Path(out) / 'minutes.jsonl'}")


@cli.command()
@click.option("--phone", "phone", multiple=True, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--speed", default=60.0, show_default=True, type=float,
              help="Replay speed x real time (1, 10, 60; 0 = as fast as possible).")
@click.option("--out", required=True, type=click.Path(file_okay=False))
@click.option("--node-id", default=None, help="Default: config node_id.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--max-minutes", default=None, type=int)
def replay(phone, speed, out, node_id, config_path, max_minutes):
    """Push a recorded GnssLogger file through the live path (demo fallback)."""
    from ionwatch import live as livemod
    cfg = load_config(config_path)
    n = livemod.run_replay(list(phone), speed, out, cfg, node_id or cfg["node_id"], max_minutes=max_minutes)
    click.echo(f"{n} records written to {Path(out) / 'minutes.jsonl'}")


@cli.command()
@click.option("--phone", "phone", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="GnssLogger .txt file (repeatable).")
@click.option("--l86", type=click.Path(exists=True, dir_okay=False), help="L86 NMEA log.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", type=click.Path(file_okay=False), help="Also write *_quality.json here.")
def inspect(phone, l86, config_path, out):
    """Print data-quality reports only (no indices)."""
    cfg = load_config(config_path)
    if not phone and not l86:
        raise click.UsageError("give --phone and/or --l86")
    if phone:
        log = gnsslogger.read_phone(phone, cfg)
        join = gnsslogger.attach_elevation(log, cfg)
        q = gnsslogger.phone_quality(log, join, cfg)
        click.echo(gnsslogger.format_quality(q))
        if out:
            pipeline.write_json(q, Path(out) / "phone_quality.json")
    if l86:
        from ionwatch.io import nmea_l86
        lg = nmea_l86.read_l86(l86, cfg)
        if phone:
            click.echo("")
        click.echo(nmea_l86.format_quality(lg.quality))
        if out:
            pipeline.write_json(lg.quality, Path(out) / "l86_quality.json")


@cli.command()
@click.option("--data", "data_dir", default="data/live", show_default=True, type=click.Path(file_okay=False),
              help="A live, replay or batch output folder (with minutes.jsonl / latest.json).")
@click.option("--host", default="0.0.0.0", show_default=True, help="0.0.0.0 = reachable from the local Wi-Fi.")
@click.option("--port", default=8080, show_default=True, type=int)
def dashboard(data_dir, host, port):
    """Serve the web dashboard for a live / replay / batch folder."""
    from ionwatch.dashboard import server
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    click.echo(f"IonWatch dashboard for {data_dir} on http://{host}:{port}  (Ctrl-C to stop)")
    server.serve(data_dir, host=host, port=port)


@cli.group()
def bench():
    """IGS benchmark: fetch reference data for the session's night, then compare."""


def _bench_dates(date, processed, cfg):
    import pandas as pd
    if date:
        return [pd.Timestamp(date, tz="UTC")]
    if processed:
        from ionwatch.bench.run import session_info
        return session_info(Path(processed), cfg)["dates"]
    live_file = Path("data/live/minutes.jsonl")
    if live_file.exists():
        from ionwatch import schema
        fin = schema.finals(schema.read_jsonl(live_file)) or schema.read_jsonl(live_file)
        if fin:
            return [pd.Timestamp(fin[-1]["minute_utc"]).normalize()]
    raise click.UsageError("give --date YYYY-MM-DD or --processed <folder> (the date comes from the session data)")


@bench.command("fetch")
@click.option("--out", default="data/bench", show_default=True, type=click.Path(file_okay=False))
@click.option("--date", default=None, help="UTC date YYYY-MM-DD (default: the session's data span).")
@click.option("--processed", default=None, type=click.Path(exists=True, file_okay=False),
              help="Processed session folder whose data span sets the date(s).")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False))
def bench_fetch(out, date, processed, config_path):
    """Download IGS RINEX (LCK3/LCK4, IISC/HYDE backups), BRDM, GIM, DCB/OSB, Kp and F10.7; validate all."""
    import logging

    from ionwatch.bench import fetch as fetchmod
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(config_path)
    bad = False
    for d in _bench_dates(date, processed, cfg):
        m = fetchmod.fetch(d, Path(out))
        click.echo(f"\n{m['date_utc']} (DOY {m['doy']}) -> {m['folder']}")
        for name, r in m["items"].items():
            click.echo(f"  {name:<12} {'OK ' if r.get('ok') else '-- '} {r.get('file') or r.get('status')}")
        for c in m["cross_checks"]:
            v = "OK" if c.get("ok") else "CHECK" if c.get("ok") is False else "n/a"
            click.echo(f"  check: {c['check']}: {v}")
        if m["missing_required"]:
            click.echo(f"  MISSING required: {m['missing_required']}")
            bad = True
    if bad:
        raise SystemExit(1)


@bench.command("run")
@click.option("--bench", "bench_dir", default="data/bench", show_default=True, type=click.Path(exists=True, file_okay=False))
@click.option("--processed", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False))
def bench_run(bench_dir, processed, config_path):
    """IGS ROTI over the night window, GIM VTEC at Haldwani, Kp, our events vs ROTI -> bench_summary.md."""
    from ionwatch.bench import run as runmod
    cfg = load_config(config_path)
    r = runmod.run(Path(bench_dir), Path(processed), cfg)
    for s in r["stations"]:
        click.echo(f"  {s['station']}: night max ROTI {s['roti_max']} TECU/min, {s['windows_above']} windows above "
                   f"{cfg['bench']['roti_irregular']}")
    click.echo(f"bench summary: {Path(r['folder']) / 'bench_summary.md'}")


@cli.command()
@click.option("--data", "data_dir", default="data/live", show_default=True, type=click.Path(file_okay=False))
@click.option("--every", default=60.0, show_default=True, type=float, help="Seconds between lines.")
@click.option("--count", default=None, type=int, help="Stop after this many lines (default: run until Ctrl-C).")
def status(data_dir, every, count):
    """Print a health line for a live folder every 60 s: time, satellites, C/N0, L86 rate, age, warnings."""
    from ionwatch import status as statusmod
    try:
        statusmod.watch(data_dir, every_s=every, count=count, echo=click.echo)
    except KeyboardInterrupt:
        pass


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def validate(path):
    """Check a minutes.jsonl: every record matches the schema; per receiver, every minute from
    the first to the last has exactly one final record (only the newest minutes may still be
    provisional-only, waiting for their final) and no minute is missing."""
    import pandas as pd

    from ionwatch import schema
    recs = schema.read_jsonl(Path(path))
    bad = 0
    for i, r in enumerate(recs, 1):
        try:
            schema.validate(r)
        except Exception as e:
            bad += 1
            click.echo(f"record {i} ({r.get('minute_utc')}): INVALID {str(e).splitlines()[0]}")
    ok = bad == 0
    for rcv in sorted({r.get("receiver") for r in recs}):
        rr = [r for r in recs if r.get("receiver") == rcv]
        fin = pd.Series([r["minute_utc"] for r in rr if r.get("status") == "final"])
        allm = pd.to_datetime(sorted({r["minute_utc"] for r in rr}))
        expected = pd.date_range(allm.min(), allm.max(), freq="1min")
        missing = len(expected) - len(allm)
        dup_final = int(fin.duplicated().sum())
        finals = pd.to_datetime(sorted(set(fin)))
        last_final = finals.max() if len(finals) else None
        no_final = [m for m in allm if m not in set(finals)]
        tail = [m for m in no_final if last_final is None or m > last_final]
        holes = len(no_final) - len(tail)
        no_data = sum("no_data" in r["warnings"] for r in rr if r.get("status") == "final")
        click.echo(f"{rcv}: {len(allm)} minutes {allm.min():%Y-%m-%d %H:%M} -> {allm.max():%H:%M} UTC, "
                   f"{len(rr)} records ({len(fin)} final), missing minutes {missing}, duplicate finals {dup_final}, "
                   f"minutes without final {holes} (+{len(tail)} newest still pending), no_data minutes {no_data}")
        ok &= missing == 0 and dup_final == 0 and holes == 0
    click.echo(f"{len(recs)} records, {bad} invalid -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    cli()


if __name__ == "__main__":
    main()
