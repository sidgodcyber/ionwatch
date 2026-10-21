/* IonWatch dashboard client: polls the API every 10 s and redraws without a page reload. */
(() => {
  "use strict";
  const REFRESH_MS = 10000;
  const SYS_ORDER = ["GPS", "GLO", "GAL", "BDS", "QZS", "NAVIC", "SBAS"];
  const SYS_NAME = { GPS: "GPS", GLO: "GLONASS", GAL: "Galileo", BDS: "BeiDou", QZS: "QZSS", NAVIC: "NavIC", SBAS: "SBAS" };
  const SYS_LETTER = { GPS: "G", GLO: "R", GAL: "E", BDS: "C", QZS: "J", NAVIC: "I", SBAS: "S" };
  const L5_BANDS = new Set(["L5", "E5a", "B2a"]);
  const CLASS_LABEL = { iono_likely: "iono likely", iono_possible: "iono possible", local: "local cause",
                        low_elev: "low elevation", edge_of_gap: "edge of gap (ignored)",
                        local_or_unknown: "local or unknown (outside night)" };
  const WARN_TEXT = {
    no_data: "receiver produced no data", receiver_partial: "receiver active for part of the minute only",
    low_cn0: "mean C/N0 below 25 dB-Hz (indoors / obstructed?)", local_cause: "local cause (many satellites faded or AGC jump)",
    serial_disconnected: "serial port lost", serial_reconnected: "serial port reopened", no_serial_data: "no bytes from the L86",
    bad_checksum: "NMEA lines with bad checksums", garbage_lines: "non-NMEA bytes on the line", partial_line: "line cut by a reconnect",
    no_fix: "L86 has no fix (RMC status V)", clock_jump: "host clock disagreed with data time",
    l86_rate_measured: "L86 rate measured", l86_rate_fallback: "L86 rate fell back to 1 Hz",
    final_capped: "final published before an event ended (30 min cap)", processing_error: "processing error (see log)",
  };

  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const istFmt = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false });
  const istFull = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", day: "2-digit", month: "short",
                                                    hour: "2-digit", minute: "2-digit", hour12: false });
  const ist = (ms) => istFmt.format(new Date(ms));

  const state = { records: new Map(), hours: 3, newest: null, status: null, events: [], warnings: [],
                  satColor: new Map(), satSlots: {}, charts: {} };

  // ------------------------------------------------------------------ data
  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }

  function mergeRecords(list) {
    for (const r of list) {
      const k = `${r.receiver}|${r.minute_utc}`;
      const old = state.records.get(k);
      if (!old || (r.rev || 1) >= (old.rev || 1)) state.records.set(k, r);
    }
    const ms = [...state.records.values()].map((r) => r.minute_utc).sort();
    state.newest = ms.length ? ms[ms.length - 1] : null;
  }

  async function refresh() {
    try {
      const status = await getJSON("/api/status");
      if (status.latest && state.newest && status.latest.minute_utc < state.newest) state.records.clear();  // new replay
      if (!status.latest) state.records.clear();
      let url = `/api/minutes?hours=${state.hours}`;
      if (state.newest) {
        const since = new Date(Date.parse(state.newest) - 30 * 60000).toISOString().slice(0, 16) + ":00Z";
        url += `&since=${encodeURIComponent(since)}`;
      }
      const [mins, ev, warn] = await Promise.all([getJSON(url), getJSON(`/api/events?hours=${state.hours}&limit=60`),
                                                  getJSON("/api/warnings?hours=1")]);
      mergeRecords(mins.records);
      prune();
      Object.assign(state, { status, events: ev.events, warnings: warn.warnings });
      render();
      $("refreshed").textContent = `updated ${new Date().toLocaleTimeString("en-GB")}`;
    } catch (e) {
      $("refreshed").textContent = `refresh failed: ${e.message}`;
    }
  }

  function prune() {
    if (!state.newest) return;
    const lo = Date.parse(state.newest) - state.hours * 3600e3;
    for (const [k, r] of state.records) if (Date.parse(r.minute_utc) < lo) state.records.delete(k);
  }

  function primaryReceiver() {
    const l = state.status && state.status.latest;
    return l ? l.receiver : (state.status && state.status.receivers[0]) || "phone";
  }

  function recordsFor(receiver) {
    return [...state.records.values()].filter((r) => r.receiver === receiver).sort((a, b) => a.minute_utc < b.minute_utc ? -1 : 1);
  }

  // ------------------------------------------------------------------ header
  function renderHeader(st) {
    $("node").textContent = st.node_id || "–";
    const mode = (st.mode || "").toLowerCase();
    const badge = $("mode");
    badge.textContent = mode ? mode.toUpperCase() : "–";
    badge.className = `badge ${mode}`;
    $("source").textContent = st.source ? (Array.isArray(st.source) ? st.source.join(", ") : st.source) +
      (st.speed ? ` · ×${st.speed}` : "") : "";
    const l = st.latest;
    $("t-update").textContent = l ? `${istFull.format(new Date(Date.parse(l.minute_utc)))} IST` : "no records yet";
    const age = st.updated_age_s;
    const ageEl = $("t-age");
    if (age == null) { ageEl.textContent = "–"; ageEl.className = "sub"; }
    else {
      const txt = age < 90 ? `${Math.round(age)} s ago` : age < 5400 ? `${Math.round(age / 60)} min ago` : `${(age / 3600).toFixed(1)} h ago`;
      ageEl.textContent = `${l ? l.status : ""} · written ${txt}`;
      ageEl.className = st.stale ? "sub age-warn" : "sub";
      if (st.stale) ageEl.textContent += " (over 3 min)";
    }
    const rs = st.receiver_status || "–";
    const cls = rs === "ok" ? "ok" : ["disconnected", "stale"].includes(rs) ? "bad" : "warn";
    $("t-recv").innerHTML = `<span class="state ${cls}">${rs}</span>`;
    $("t-recv-sub").textContent = l ? `${l.receiver === "l86" ? "L86" : "phone"} · ${st.final_records}/${st.records} records final` : "–";
    $("t-sats").textContent = l ? l.sats_tracked : "–";
    $("t-rate").textContent = l && l.l86_rate_hz != null ? `${l.l86_rate_hz} Hz` : "–";
    $("t-rate-sub").textContent = l && l.receiver === "l86" ? "epochs per second (measured)" : "phone source: not applicable";
  }

  // ------------------------------------------------------------------ sky plot
  // C/N0 colour domain follows the data (phones sit at 12-30 dB-Hz, an outdoor L86 at 30-50);
  // the legend under the plot prints the actual end values.
  let rampDomain = [15, 50];
  function rampColor(cn0) {
    const stops = css("--ramp").split(",").map((s) => s.trim());
    if (cn0 == null) return css("--other");
    const [lo, hi] = rampDomain;
    const t = Math.min(1, Math.max(0, (cn0 - lo) / (hi - lo)));
    return stops[Math.round(t * (stops.length - 1))];
  }

  function renderRamp() {
    const stops = css("--ramp").split(",").map((s) => s.trim());
    $("ramp-bar").style.background = `linear-gradient(90deg, ${stops.join(",")})`;
    $("ramp-lo").textContent = rampDomain[0];
    $("ramp-hi").textContent = `${rampDomain[1]} dB-Hz`;
  }

  let skyPoints = [];
  function renderSky(latest) {
    const cv = $("sky");
    const size = Math.min(cv.parentElement.clientWidth, 340);
    const dpr = window.devicePixelRatio || 1;
    cv.width = size * dpr; cv.height = size * dpr; cv.style.height = `${size}px`;
    const g = cv.getContext("2d");
    g.scale(dpr, dpr);
    const c = size / 2, R = size / 2 - 22;
    const rOf = (el) => R * (90 - el) / 90;
    g.clearRect(0, 0, size, size);
    g.strokeStyle = css("--grid"); g.lineWidth = 1;
    for (const el of [0, 30, 60]) { g.beginPath(); g.arc(c, c, rOf(el), 0, 2 * Math.PI); g.stroke(); }
    g.beginPath(); g.moveTo(c - R, c); g.lineTo(c + R, c); g.moveTo(c, c - R); g.lineTo(c, c + R); g.stroke();
    g.fillStyle = css("--muted"); g.font = "11px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
    [["N", 0], ["E", 90], ["S", 180], ["W", 270]].forEach(([t, az]) => {
      const a = (az - 90) * Math.PI / 180; g.fillText(t, c + (R + 12) * Math.cos(a), c + (R + 12) * Math.sin(a));
    });
    const diag = -Math.PI / 4 - 0.25;      // ring labels on a diagonal, away from N and the cross
    for (const el of [30, 60]) g.fillText(`${el}°`, c + (rOf(el) + 8) * Math.cos(diag), c + (rOf(el) + 8) * Math.sin(diag));

    skyPoints = [];
    const sats = new Map();
    for (const s of (latest ? latest.satellites : [])) {
      if (s.elev == null || s.az == null || s.cn0_mean == null) continue;
      const k = `${s.sys}${s.svid}`;
      const prev = sats.get(k);
      if (!prev || s.cn0_mean > prev.cn0_mean) sats.set(k, { ...s, bands: [...(prev ? prev.bands : []), s.band] });
      else prev.bands.push(s.band);
    }
    const vals = [...sats.values()].map((s) => s.cn0_mean);
    if (vals.length) {
      const lo = Math.floor(Math.min(...vals) / 5) * 5, hi = Math.ceil(Math.max(...vals) / 5) * 5;
      rampDomain = [lo, Math.max(hi, lo + 15)];
    }
    renderRamp();
    const pts = [...sats.values()].map((s) => {
      const a = (s.az - 90) * Math.PI / 180, r = rOf(Math.max(0, s.elev));
      return { x: c + r * Math.cos(a), y: c + r * Math.sin(a), s };
    });
    for (const p of pts) {          // marks first, labels after, so no label hides under a mark
      g.beginPath(); g.arc(p.x, p.y, 7, 0, 2 * Math.PI);
      g.fillStyle = rampColor(p.s.cn0_mean); g.fill();
      g.lineWidth = 2; g.strokeStyle = css("--surface-1"); g.stroke();
      skyPoints.push(p);
    }
    g.font = "10px system-ui, sans-serif"; g.fillStyle = css("--text-secondary"); g.textBaseline = "middle";
    const boxes = pts.map((p) => [p.x - 8, p.y - 8, p.x + 8, p.y + 8]);
    const hit = (b) => boxes.some((o) => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1]);
    for (const p of pts) {          // try right, left, above, below; skip the label if all collide
      const t = `${SYS_LETTER[p.s.sys] || "?"}${String(p.s.svid).padStart(2, "0")}`;
      const w = g.measureText(t).width, h = 11;
      const spots = [[p.x + 10, p.y, "left"], [p.x - 10, p.y, "right"], [p.x, p.y - 13, "center"], [p.x, p.y + 13, "center"]];
      for (const [x, y, align] of spots) {
        const x0 = align === "left" ? x : align === "right" ? x - w : x - w / 2;
        const b = [x0, y - h / 2, x0 + w, y + h / 2];
        if (!hit(b)) { g.textAlign = align; g.fillText(t, x, y); boxes.push(b); break; }
      }
    }
    $("sky-empty").hidden = sats.size > 0;
  }

  // The newest minute may be a receiver silence (duty cycling, unplug): show the newest minute
  // that had tracked satellites instead, and say which minute that is.
  function renderSkyFrom(recs) {
    const latest = recs[recs.length - 1];
    const has = (r) => r && r.satellites.some((s) => s.cn0_mean != null && s.elev != null);
    const src = has(latest) ? latest : [...recs].reverse().find(has);
    renderSky(src);
    const note = $("sky-note");
    if (src && src !== latest) {
      note.hidden = false;
      note.textContent = `No tracked satellites in the latest minute; showing ${istFull.format(new Date(Date.parse(src.minute_utc)))} IST.`;
    } else note.hidden = true;
    $("sky-sub").textContent = src ? `${ist(Date.parse(src.minute_utc))} IST, tracked satellites` : "latest minute, tracked satellites";
  }

  function skyHover(ev) {
    const cv = $("sky"), tip = $("sky-tip");
    const rect = cv.getBoundingClientRect();
    const px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - rect.left;
    const py = (ev.touches ? ev.touches[0].clientY : ev.clientY) - rect.top;
    let best = null, bd = 14 * 14;
    for (const p of skyPoints) { const d = (p.x - px) ** 2 + (p.y - py) ** 2; if (d < bd) { bd = d; best = p; } }
    if (!best) { tip.hidden = true; return; }
    const s = best.s;
    tip.textContent = `${SYS_NAME[s.sys] || s.sys} ${s.svid} (${s.bands.join("+")}) · C/N0 ${s.cn0_mean} dB-Hz · el ${s.elev}° az ${s.az}°`;
    tip.hidden = false;
    tip.style.left = `${Math.min(best.x + 10, rect.width - tip.offsetWidth - 4)}px`;
    tip.style.top = `${best.y - 32}px`;
  }

  // ------------------------------------------------------------------ NavIC
  function renderNavic(recs) {
    const phone = [...recs].reverse();
    const latest = phone[0];
    const st = $("navic-status"), rows = $("navic-rows"), note = $("navic-note");
    rows.innerHTML = "";
    if (!latest) { st.textContent = "no records yet"; note.textContent = ""; return; }
    const labels = { tracked: ["ok", "tracked"], visible_not_tracked: ["warn", "visible, not tracked"], no_data: ["warn", "no data"] };
    let src = latest;
    if (!latest.navic_sats.length) src = phone.find((r) => r.navic_sats.length) || latest;
    // Elevations from an earlier minute: report that minute's state, not "no data".
    const shown = src !== latest ? src.navic_status : latest.navic_status;
    const [cls, base] = labels[shown] || ["warn", shown];
    const when = src !== latest ? ` (last seen ${ist(Date.parse(src.minute_utc))} IST)` : "";
    st.innerHTML = `<span class="state ${cls}"></span>`;
    st.firstChild.textContent = base + when;
    for (const n of src.navic_sats) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>I${String(n.svid).padStart(2, "0")}</td><td class="num">${n.elev ?? "–"}°</td>` +
        `<td class="num">${n.az ?? "–"}°</td><td class="num">${n.cn0_mean ?? "not tracked"}</td>`;
      rows.appendChild(tr);
    }
    if (latest.receiver === "l86") note.textContent = "The L86-M33 (MT3333) does not track NavIC; NavIC comes from the phone.";
    else if (src !== latest) note.textContent = `Not listed in the latest minute; elevations from ${istFull.format(new Date(Date.parse(src.minute_utc)))} IST.`;
    else if (latest.navic_status === "visible_not_tracked") note.textContent = "Listed in the almanac (Status rows) but C/N0 = 0: no NavIC health metrics.";
    else note.textContent = "";
  }

  // ------------------------------------------------------------------ timelines
  function seriesColor(sys, svid, byConstellation) {
    if (byConstellation) return css(`--series-${SYS_ORDER.indexOf(sys) + 1}`) || css("--other");
    const k = `${sys}|${svid}`;
    if (!state.satColor.has(k)) {   // stable slot per satellite within its constellation, never re-assigned
      state.satSlots[sys] = (state.satSlots[sys] || 0) + 1;
      state.satColor.set(k, state.satSlots[sys] - 1);
    }
    const slot = state.satColor.get(k);
    return slot < 8 ? css(`--series-${slot + 1}`) : css("--other");
  }

  function buildSeries(recs, filter, field) {
    const byConst = filter === "ALL";
    const minutes = recs.map((r) => Date.parse(r.minute_utc) + 30000);
    const keys = new Map();
    recs.forEach((r, i) => {
      for (const s of r.satellites) {
        if (!byConst && s.sys !== filter) continue;
        const k = `${s.sys} ${s.svid} ${s.band}`;
        if (!keys.has(k)) keys.set(k, { sys: s.sys, svid: s.svid, band: s.band, vals: new Array(recs.length).fill(null), flags: new Array(recs.length).fill(null) });
        const e = keys.get(k);
        e.vals[i] = field(s);
        e.flags[i] = s;
      }
    });
    const ordered = [...keys.values()].sort((a, b) =>
      SYS_ORDER.indexOf(a.sys) - SYS_ORDER.indexOf(b.sys) || a.svid - b.svid || a.band.localeCompare(b.band));
    return { minutes, series: ordered, byConst };
  }

  function axisOpts(yTitle, yMin, yMax) {
    const grid = css("--grid"), ink = css("--muted");
    return {
      x: { type: "linear", grid: { color: grid }, ticks: { color: ink, maxTicksLimit: 8, callback: (v) => ist(v) },
           title: { display: true, text: "Time (IST)", color: ink } },
      y: { min: yMin, max: yMax, grid: { color: grid }, ticks: { color: ink }, title: { display: true, text: yTitle, color: ink } },
    };
  }

  function legendOpts(byConst) {
    return {
      display: true, position: "bottom",
      labels: { color: css("--text-secondary"), boxWidth: 14, boxHeight: 2, usePointStyle: false,
        filter: (item, data) => {
          const ds = data.datasets[item.datasetIndex];
          return !ds.hideFromLegend;
        } },
    };
  }

  function tooltipOpts(unit) {
    return {
      mode: "nearest", intersect: false,
      callbacks: {
        title: (items) => items.length ? `${ist(items[0].parsed.x)} IST` : "",
        label: (it) => {
          const ds = it.dataset;
          if (ds.isRef) return `${ds.label}`;
          const v = it.parsed.y;
          const extra = ds.flagAt ? ds.flagAt(it.dataIndex) : "";
          return `${ds.label}: ${v == null ? "–" : v.toFixed(unit === "s4" ? 2 : 1)}${unit === "s4" ? "" : " dB-Hz"}${extra}`;
        },
      },
    };
  }

  function upsertChart(id, config) {
    const old = state.charts[id];
    if (old) { old.data = config.data; old.options = config.options; old.update("none"); return old; }
    state.charts[id] = new Chart($(id), config);
    return state.charts[id];
  }

  function labelOf(e, byConst) {
    return `${SYS_LETTER[e.sys] || "?"}${String(e.svid).padStart(2, "0")} ${e.band}`;
  }

  function renderCn0(recs) {
    const filter = $("filter").value;
    const { minutes, series, byConst } = buildSeries(recs, filter, (s) => s.cn0_mean);
    const datasets = [];
    const seenConst = new Set();
    for (const e of series) {
      const color = seriesColor(e.sys, e.svid, byConst);
      const other = !byConst && color === css("--other");
      datasets.push({
        label: byConst ? (SYS_NAME[e.sys] || e.sys) : labelOf(e), data: minutes.map((x, i) => ({ x, y: e.vals[i] })),
        borderColor: color, backgroundColor: color, borderWidth: byConst ? 1.25 : 2, pointRadius: 0, pointHitRadius: 6,
        spanGaps: false, borderDash: L5_BANDS.has(e.band) ? [5, 3] : [],
        hideFromLegend: byConst ? seenConst.has(e.sys) : other,
        satLabel: labelOf(e),
      });
      seenConst.add(e.sys);
    }
    if (!byConst && datasets.some((d) => d.hideFromLegend))
      datasets.push({ label: "other satellites", data: [], borderColor: css("--other"), backgroundColor: css("--other"), isRef: true });
    const ch = upsertChart("cn0", {
      type: "line", data: { datasets },
      options: { animation: false, responsive: true, maintainAspectRatio: false, parsing: false, normalized: true,
        scales: axisOpts("C/N0 (dB-Hz)", undefined, undefined),
        plugins: { legend: legendOpts(byConst), tooltip: { ...tooltipOpts("cn0"),
          callbacks: { ...tooltipOpts("cn0").callbacks,
            label: (it) => `${it.dataset.satLabel || it.dataset.label}: ${it.parsed.y == null ? "–" : it.parsed.y.toFixed(1)} dB-Hz` } } } },
    });
    $("cn0-empty").hidden = series.length > 0;
    return ch;
  }

  const refLines = {
    id: "refLines",
    afterDatasetsDraw(chart, _args, opts) {
      const { ctx, chartArea: a, scales: { y } } = chart;
      if (!opts || !opts.lines) return;
      ctx.save();
      for (const [v, text] of opts.lines) {
        const py = y.getPixelForValue(v);
        if (py < a.top || py > a.bottom) continue;
        ctx.strokeStyle = css("--text-secondary"); ctx.setLineDash([5, 4]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(a.left, py); ctx.lineTo(a.right, py); ctx.stroke();
        ctx.setLineDash([]); ctx.fillStyle = css("--text-secondary"); ctx.font = "11px system-ui, sans-serif";
        ctx.textAlign = "right"; ctx.fillText(text, a.right - 4, py - 4);
      }
      ctx.restore();
    },
  };

  function renderS4(recs, receiver) {
    const filter = $("filter").value;
    const { minutes, series, byConst } = buildSeries(recs, filter, (s) => (s.s4_flag === "valid" ? s.s4 : null));
    $("s4-title").textContent = receiver === "phone"
      ? "S4 per satellite — indicative (phone, 1 Hz C/N0)" : "S4 per satellite (L86 C/N0)";
    // Noise floor: 2 x median s4_noise across satellites, per minute.
    const floor = recs.map((r, i) => {
      const v = r.satellites.map((s) => s.s4_noise).filter((x) => x != null).sort((a, b) => a - b);
      return { x: minutes[i], y: v.length ? Math.min(1, 2 * v[Math.floor(v.length / 2)]) : null };
    });
    const datasets = [{ label: "below 2× noise floor", data: floor, fill: "origin", borderWidth: 0, pointRadius: 0,
      backgroundColor: css("--shade"), borderColor: "transparent", spanGaps: false, isRef: true, order: 10 }];
    const seenConst = new Set();
    let ymax = 0.8;
    for (const e of series) {
      const color = seriesColor(e.sys, e.svid, byConst);
      e.vals.forEach((v) => { if (v != null) ymax = Math.max(ymax, v * 1.1); });
      datasets.push({
        label: byConst ? (SYS_NAME[e.sys] || e.sys) : labelOf(e), data: minutes.map((x, i) => ({ x, y: e.vals[i] })),
        borderColor: color, backgroundColor: color, borderWidth: 1.5, pointRadius: 3, pointHoverRadius: 5,
        spanGaps: false, borderDash: L5_BANDS.has(e.band) ? [5, 3] : [], satLabel: labelOf(e),
        hideFromLegend: byConst ? seenConst.has(e.sys) : color === css("--other"),
      });
      seenConst.add(e.sys);
    }
    upsertChart("s4", {
      type: "line", data: { datasets }, plugins: [refLines],
      options: { animation: false, responsive: true, maintainAspectRatio: false, parsing: false,
        scales: axisOpts(receiver === "phone" ? "S4 (indicative)" : "S4", 0, Math.min(1.2, Math.ceil(ymax * 10) / 10)),
        plugins: { refLines: { lines: [[0.3, "0.3 moderate"], [0.6, "0.6 strong"]] }, legend: legendOpts(byConst),
          tooltip: { ...tooltipOpts("s4"), filter: (it) => !it.dataset.isRef,
            callbacks: { ...tooltipOpts("s4").callbacks,
              label: (it) => `${it.dataset.satLabel}: S4 ${it.parsed.y == null ? "–" : it.parsed.y.toFixed(2)}` } } } },
    });
  }

  // ------------------------------------------------------------------ events, warnings
  function renderEvents(all) {
    const body = $("event-rows");
    body.innerHTML = "";
    const showIgnored = $("show-ignored").checked;
    const ignored = all.filter((e) => e.class === "edge_of_gap").length;
    $("ignored-count").textContent = ignored ? `(${ignored} hidden)` : "";
    if (showIgnored) $("ignored-count").textContent = ignored ? `(${ignored})` : "";
    const events = showIgnored ? all : all.filter((e) => e.class !== "edge_of_gap");
    for (const e of events) {
      const tr = document.createElement("tr");
      const span = e.minutes > 1 ? ` (${e.minutes} min)` : "";
      const who = e.receiver && e.receiver !== "phone" ? ` (${e.receiver})` : "";
      const two = e.both_phones == null ? "—" : e.both_phones ? "both" : "one only";
      tr.innerHTML = `<td class="num when" title="${e.start_ist}">${e.start_ist.slice(11, 16)}${span}</td><td>${SYS_NAME[e.sys] || e.sys} ${e.svid}${who}</td>` +
        `<td>${e.band}</td><td><span class="cls ${e.class}">${CLASS_LABEL[e.class] || e.class}</span></td>` +
        `<td class="reason"></td><td class="num">${e.max_fade_db != null ? e.max_fade_db.toFixed(1) + " dB" : "—"}</td>` +
        `<td>${two}</td>` +
        `<td class="${e.status === "provisional" ? "prov" : ""}">${e.status}</td>`;
      tr.querySelector(".reason").textContent = e.reason || "";
      body.appendChild(tr);
    }
    $("events-empty").hidden = events.length > 0;
  }

  function renderWarnings(list) {
    const ul = $("warn-list");
    ul.innerHTML = "";
    for (const w of list) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="code"></span> <span class="muted small"></span><br><span class="small"></span>`;
      li.children[0].textContent = WARN_TEXT[w.code] || w.code;
      li.children[1].textContent = `${w.minutes} min · last ${w.last_ist.slice(11, 16)} IST`;
      li.children[3].textContent = w.example !== w.code ? w.example : "";
      ul.appendChild(li);
    }
    $("warn-empty").hidden = list.length > 0;
  }

  function updateFilter(recs) {
    const sel = $("filter");
    const have = new Set(recs.flatMap((r) => r.satellites.map((s) => s.sys)));
    for (const sys of SYS_ORDER) {
      if (have.has(sys) && !sel.querySelector(`option[value="${sys}"]`)) {
        const o = document.createElement("option"); o.value = sys; o.textContent = `${SYS_NAME[sys]} (colour = satellite)`;
        sel.appendChild(o);
      }
    }
  }

  // ------------------------------------------------------------------ main render
  function render() {
    const st = state.status;
    if (!st) return;
    renderHeader(st);
    const receiver = primaryReceiver();
    const recs = recordsFor(receiver);
    const empty = recs.length === 0;
    $("empty").hidden = !empty;
    updateFilter(recs);
    renderSkyFrom(recs);
    renderNavic(recs);
    renderCn0(recs);
    renderS4(recs, receiver);
    renderEvents(state.events);
    renderWarnings(state.warnings);
  }

  $("filter").addEventListener("change", render);
  $("show-ignored").addEventListener("change", () => renderEvents(state.events));
  $("hours").addEventListener("change", (e) => { state.hours = +e.target.value; state.records.clear(); state.newest = null; refresh(); });
  $("sky").addEventListener("mousemove", skyHover);
  $("sky").addEventListener("touchstart", skyHover, { passive: true });
  $("sky").addEventListener("mouseleave", () => { $("sky-tip").hidden = true; });
  window.addEventListener("resize", () => renderSkyFrom(recordsFor(primaryReceiver())));

  renderRamp();
  refresh();
  setInterval(refresh, REFRESH_MS);
})();
