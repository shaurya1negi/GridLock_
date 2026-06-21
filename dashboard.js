const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmt = n => Number(n).toLocaleString();
const CRUMBS = { dashboard: "Overview", upload: "Process video", violations: "Violations", evidence: "Evidence", analytics: "Analytics" };
const VCOLORS = { car: "#2B7BD4", motorcycle: "#1D9E75", bus: "#E08A1E", truck: "#7A5BD0", bicycle: "#C8442E", person: "#8A99AB" };

let STATE = { runId: null, data: null };

function showPage(p) {
  $$(".page").forEach(x => x.classList.remove("show"));
  $("#page-" + p)?.classList.add("show");
  $$(".nav a").forEach(a => a.classList.toggle("active", a.dataset.page === p));
  $("#crumb").textContent = CRUMBS[p] || "Overview";
  history.replaceState({}, "", "/" + p);
  render(p);
}
$$(".nav a").forEach(a => a.addEventListener("click", e => { e.preventDefault(); showPage(a.dataset.page); }));
const currentPage = () => $$(".page.show")[0]?.id.replace("page-", "") || document.body.dataset.active || "dashboard";

async function loadRuns() {
  try {
    const runs = await fetch("/api/runs").then(r => r.json());
    const done = runs.filter(r => r.status === "done");
    const sel = $("#runSelect");
    if (done.length) {
      sel.innerHTML = done.map(r => `<option value="${r.run_id}">${r.video} · ${r.run_id.slice(0, 15)}</option>`).join("");
      sel.onchange = () => selectRun(sel.value);
      if (!STATE.runId || STATE.runId === "demo") await selectRun(done[0].run_id);
    }
    return runs;
  } catch { return []; }
}
async function selectRun(id) {
  if (!id || id === "undefined") return;
  STATE.runId = id;
  const [summary, violations, analytics, evidence] = await Promise.all([
    fetch(`/api/runs/${id}/summary`).then(r => r.json()),
    fetch(`/api/runs/${id}/violations`).then(r => r.json()),
    fetch(`/api/runs/${id}/analytics`).then(r => r.json()),
    fetch(`/api/runs/${id}/evidence`).then(r => r.json()),
  ]);
  STATE.data = { summary, violations, analytics, evidence };
  // re-render whatever page is showing, with the real run's data
  render(currentPage());
}
async function loadDemo() {
  const d = await fetch("/api/demo").then(r => r.json());
  STATE.runId = "demo"; STATE.data = d;
  $("#runSelect").innerHTML = `<option>demo run · sample data</option>`;
  render(currentPage());
}

function render(p) {
  if (!STATE.data) return;
  const { summary, violations, analytics, evidence } = STATE.data;
  if (p === "dashboard") renderDashboard(summary, violations, analytics);
  if (p === "violations") renderViolations(violations);
  if (p === "evidence") renderEvidence(evidence);
  if (p === "analytics") renderAnalytics(analytics, violations);
}

const IC = {
  obj: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="8" rx="2"/><path d="M5 11l2-5h10l2 5" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  det: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2" stroke-linecap="round"/></svg>`,
  alert: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/></svg>`,
  speed: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 14l4-4M5 19a9 9 0 1 1 14 0" stroke-linecap="round"/></svg>`,
};
function statCard({ label, value, unit, sub, icon, alert }) {
  return `<div class="card stat ${alert ? 'alert' : ''}"><div class="ic">${icon}</div><div class="label">${label}</div><div class="value">${value}${unit ? ` <small>${unit}</small>` : ""}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

function renderDashboard(s, v, a) {
  $("#statCards").innerHTML = [
    statCard({ label: "Unique vehicles", value: fmt(s.unique_vehicles || 0), sub: "real tracks (ghosts filtered)", icon: IC.obj }),
    statCard({ label: "Total detections", value: fmt(s.total_detections || 0), sub: `over ${fmt(s.frames || 0)} frames`, icon: IC.det }),
    statCard({ label: "Violations", value: (s.n_parking || 0) + (s.n_wrong_side || 0), sub: `${s.n_parking || 0} parking · ${s.n_wrong_side || 0} wrong-side`, icon: IC.alert, alert: true }),
    statCard({ label: "Avg speed", value: a.avg_speed_px_sec || 0, unit: "px/s", sub: "moving vehicles", icon: IC.speed }),
  ].join("");
  drawBars("#ovDensity", (a.density_over_time || []).map(d => d.count));
  drawDonut(s.class_breakdown || {});
  drawArch();
}
function drawBars(sel, vals) {
  const max = Math.max(...vals, 1);
  $(sel).innerHTML = vals.map(v => `<div class="bar" style="height:${Math.max(3, v / max * 100)}%" title="${v}"></div>`).join("");
}
function drawDonut(bd) {
  const e = Object.entries(bd), total = e.reduce((s, [, v]) => s + v, 0) || 1;
  let acc = 0;
  const segs = e.map(([k, v]) => { const len = v / total * 100; const s = `<circle cx="21" cy="21" r="15.915" fill="transparent" stroke="${VCOLORS[k] || '#8A99AB'}" stroke-width="6" stroke-dasharray="${len} ${100 - len}" stroke-dashoffset="${25 - acc}"/>`; acc += len; return s; }).join("");
  $("#ovDonut").innerHTML = `<circle cx="21" cy="21" r="15.915" fill="transparent" stroke="#EEF3F9" stroke-width="6"/>${segs}<text x="21" y="20" text-anchor="middle" font-size="7" font-weight="700" fill="#0E1B2A">${total}</text><text x="21" y="27" text-anchor="middle" font-size="3.4" fill="#8A99AB">VEHICLES</text>`;
  $("#ovDonutLegend").innerHTML = e.map(([k, v]) => `<div class="row"><i style="background:${VCOLORS[k] || '#8A99AB'}"></i>${k}<b>${v}</b></div>`).join("") || `<div style="color:var(--mist);font-size:13px">No data</div>`;
}
function drawArch() {
  const box = (x, w, t, s, fill, stroke, ink) => `<rect x="${x}" y="40" width="${w}" height="56" rx="8" fill="${fill}" stroke="${stroke}"/><text x="${x + w / 2}" y="64" text-anchor="middle" font-size="12" font-weight="600" fill="${ink}">${t}</text><text x="${x + w / 2}" y="81" text-anchor="middle" font-size="10" fill="${ink}" opacity=".75">${s}</text>`;
  const arr = x => `<line x1="${x}" y1="68" x2="${x + 24}" y2="68" stroke="#9DBEE8" stroke-width="1.6"/>`;
  $("#archDiagram").innerHTML = `<svg viewBox="0 0 920 130" style="width:100%;height:auto;margin-top:4px">
    ${box(10, 120, "Detect + track", "YOLO11m · BoT-SORT", "#E6F1FB", "#85B7EB", "#042C53")}${arr(130)}
    ${box(160, 110, "Trajectory CSV", "data contract", "#E1F5EE", "#5DCAA5", "#04342C")}${arr(270)}
    ${box(300, 120, "Parked extract", "stop segments", "#E6F1FB", "#85B7EB", "#042C53")}${arr(424)}
    ${box(450, 130, "Violation engines", "parking · wrong-side", "#EEEDFE", "#AFA9EC", "#26215C")}${arr(584)}
    ${box(610, 120, "Evidence", "best-shot patches", "#FBEAF0", "#ED93B1", "#4B1528")}
  </svg>`;
}

const VBADGE = { "Illegal parking": "parking", "Wrong-side driving": "wrong" };
function renderViolations(v) {
  const np = v.filter(x => x.type === "Illegal parking").length, nw = v.filter(x => x.type === "Wrong-side driving").length;
  $("#violStats").innerHTML = [
    statCard({ label: "Illegal parking", value: np, sub: "5-point consensus", icon: IC.alert, alert: np > 0 }),
    statCard({ label: "Wrong-side driving", value: nw, sub: "vector alignment", icon: IC.alert, alert: nw > 0 }),
  ].join("");
  if (!v.length) { $("#violTable").innerHTML = emptyState("No violations in this run."); const b=$("#dlCsvBtn"); if(b) b.style.display="none"; return; }
  const b = $("#dlCsvBtn");
  if (b) { b.style.display = ""; b.onclick = () => downloadViolationsCsv(v); }
  $("#violTable").innerHTML = `<table><thead><tr><th>Track</th><th>Type</th><th>Vehicle</th><th>Time</th><th>Confidence</th><th>Detail</th></tr></thead><tbody>${v.map(x => `<tr>
    <td><span class="id-chip">#${x.tracker_id}</span></td>
    <td><span class="badge ${VBADGE[x.type] || 'parking'}">${x.type}</span></td>
    <td><span class="badge cls">${x.vehicle_class}</span></td>
    <td class="mono">${Number(x.timestamp_sec).toFixed(1)}s</td>
    <td><span class="conf-bar"><i style="width:${Math.round(x.confidence * 100)}%"></i></span> <span class="mono">${Number(x.confidence).toFixed(2)}</span></td>
    <td style="color:var(--slate);font-size:12.5px">${x.detail}</td></tr>`).join("")}</tbody></table>`;
}

// Build a CSV from the currently displayed violations and trigger a download.
function downloadViolationsCsv(rows) {
  const cols = ["tracker_id", "type", "vehicle_class", "timestamp_sec", "confidence", "detail"];
  const esc = s => `"${String(s ?? "").replace(/"/g, '""')}"`;
  const lines = [cols.join(",")];
  rows.forEach(r => lines.push(cols.map(c => esc(r[c])).join(",")));
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = (STATE.runId && STATE.runId !== "demo") ? STATE.runId : "demo";
  a.href = url; a.download = `violations_${stamp}.csv`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function renderEvidence(ev) {
  if (!ev || !ev.length) { $("#evidenceGrid").innerHTML = emptyState("No evidence patches yet. They appear after a run with violations."); return; }
  $("#evidenceGrid").innerHTML = ev.map(e => {
    const fields = parseCitation(e.citation || "");
    const vtype = fields["VIOLATION TYPE"] || "Violation";
    const rows = Object.entries(fields)
      .filter(([k]) => k !== "VIOLATION TYPE" && !/^=/.test(k))
      .map(([k, v]) => `<div class="cite-row"><span class="cite-k">${titleCase(k)}</span><span class="cite-v">${v}</span></div>`)
      .join("");
    return `<div class="evidence-card">
      <img src="${e.image}" alt="${e.filename}">
      <div class="cite-head"><span class="badge ${/parking/i.test(vtype) ? 'parking' : 'wrong'}">${titleCase(vtype)}</span></div>
      <div class="cite-body">${rows}</div>
    </div>`;
  }).join("");
}
// turn the raw citation .txt into {KEY: value} pairs, skipping divider lines
function parseCitation(txt) {
  const out = {};
  txt.split("\n").forEach(line => {
    const t = line.trim();
    if (!t || /^[=\-]+$/.test(t) || /CITATION TICKET/i.test(t)) return;
    const isBullet = t.startsWith("•");
    const idx = t.indexOf(":");
    if (idx === -1) return;
    let k = t.slice(0, idx).replace(/^•\s*/, "").trim();
    let v = t.slice(idx + 1).trim();
    out[(isBullet ? "  " : "") + k] = v;
  });
  return out;
}
function titleCase(s) {
  return String(s).toLowerCase().replace(/\b\w/g, c => c.toUpperCase()).replace(/\bId\b/, "ID");
}

function renderAnalytics(a, v) {
  $("#anStats").innerHTML = [
    statCard({ label: "Peak concurrent", value: a.peak_concurrent || 0, sub: "vehicles in frame", icon: IC.obj }),
    statCard({ label: "Avg speed", value: a.avg_speed_px_sec || 0, unit: "px/s", sub: "moving vehicles", icon: IC.speed }),
    statCard({ label: "Total violations", value: v.length, sub: "this run", icon: IC.alert, alert: v.length > 0 }),
  ].join("");
  drawBars("#anDensity", (a.density_over_time || []).map(d => d.count));
  const counts = {}; v.forEach(x => counts[x.type] = (counts[x.type] || 0) + 1);
  const max = Math.max(...Object.values(counts), 1);
  const C = { "Wrong-side driving": "#C8442E", "Illegal parking": "#E08A1E" };
  $("#anViolMix").innerHTML = Object.keys(counts).length ? Object.entries(counts).map(([k, n]) => `<div style="margin-bottom:14px"><div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px"><span>${k}</span><b class="mono">${n}</b></div><div style="height:9px;background:var(--line);border-radius:5px;overflow:hidden"><i style="display:block;height:100%;width:${n / max * 100}%;background:${C[k] || '#2B7BD4'}"></i></div></div>`).join("") : emptyState("No violations to chart.");
}
function emptyState(m) { return `<div class="empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4-4" stroke-linecap="round"/></svg><p>${m}</p></div>`; }

// =================== UPLOAD + POLYGON DRAWER (mirrors polygon.py) ===================
const drop = $("#drop"), fileInput = $("#fileInput"), runBtn = $("#runBtn");
let ctx = null, canvas = null;
// RUN.zones uses the SAME schema as polygon.py: {zone_id, zone_type, polygon:[[x,y]], legal_vector}
let RUN = { id: null, fw: 0, fh: 0, zones: [], mode: null, poly: [], closed: false, vecStart: null, awaitingVector: false };

if (drop) {
  drop.addEventListener("click", () => fileInput.click());
  ["dragover", "dragenter"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", ev => { if (ev.dataTransfer.files[0]) setFile(ev.dataTransfer.files[0]); });
  fileInput.addEventListener("change", () => fileInput.files[0] && setFile(fileInput.files[0]));
  runBtn.addEventListener("click", startProcessing);
  $("#demoBtn").addEventListener("click", () => { loadDemo(); setSteps(4); $("#statusMsg").innerHTML = "Demo run loaded with sample data."; });
  setupCanvas();
  $$(".mode-btn").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#closeShapeBtn").addEventListener("click", closeShape);
  $("#undoBtn").addEventListener("click", undo);
}

async function setFile(f) {
  $("#dropTitle").textContent = f.name;
  $("#dropSub").textContent = (f.size / 1e6).toFixed(1) + " MB · uploading…";
  runBtn.disabled = true; setSteps(0);
  const fd = new FormData(); fd.append("video", f);
  $("#statusMsg").innerHTML = `<span class="spinner"></span> Uploading & grabbing a frame…`;
  try {
    const resp = await fetch("/api/upload", { method: "POST", body: fd });
    if (!resp.ok) {
      let msg = `Upload failed (HTTP ${resp.status})`;
      try { const e = await resp.json(); if (e.detail) msg = e.detail; } catch {}
      $("#statusMsg").innerHTML = msg;
      return;
    }
    const res = await resp.json();
    if (!res.run_id) { $("#statusMsg").innerHTML = "Upload returned no run id — check the server log."; return; }
    RUN = { id: res.run_id, fw: res.width, fh: res.height, zones: [], mode: null, poly: [], closed: false, vecStart: null, awaitingVector: false };
    const img = $("#frameImg"); img.onload = () => sizeCanvas(); img.src = res.image;
    $("#zoneCard").style.display = "block";
    $("#dropSub").textContent = `${(f.size / 1e6).toFixed(1)} MB · ${res.width}×${res.height} · draw zones below (optional)`;
    $("#statusMsg").innerHTML = "Frame ready. Draw zones, or press Run pipeline.";
    runBtn.disabled = false; setSteps(1);
  } catch (err) { $("#statusMsg").innerHTML = "Upload failed — is the server running? " + err; }
}

function setupCanvas() {
  canvas = $("#zoneCanvas"); if (!canvas) return;
  ctx = canvas.getContext("2d");
  canvas.addEventListener("click", onClick);
  window.addEventListener("resize", sizeCanvas);
}
function sizeCanvas() { const img = $("#frameImg"); if (!canvas || !ctx || !img || !img.clientWidth) return; canvas.width = img.clientWidth; canvas.height = img.clientHeight; redraw(); }
function toFrame(dx, dy) { return [Math.round(dx * RUN.fw / canvas.width), Math.round(dy * RUN.fh / canvas.height)]; }
function toDisp(fx, fy) { return [fx * canvas.width / RUN.fw, fy * canvas.height / RUN.fh]; }

function setMode(m) {
  RUN.mode = m; RUN.poly = []; RUN.closed = false; RUN.vecStart = null; RUN.awaitingVector = false;
  $$(".mode-btn").forEach(b => b.classList.toggle("active", b.dataset.mode === m));
  $("#drawHint").textContent = m === "parking"
    ? "Parking mode — click points to outline the no-parking area, then Close shape."
    : "Road mode — outline the lane, Close shape, then click TWO points for the legal flow arrow (tail → head).";
  redraw();
}
function onClick(e) {
  if (!RUN.mode) { $("#drawHint").textContent = "Pick a mode first (Parking zone or Road lane)."; return; }
  const r = canvas.getBoundingClientRect();
  const pt = toFrame(e.clientX - r.left, e.clientY - r.top);
  if (RUN.awaitingVector) {           // road-lane arrow: two clicks
    if (!RUN.vecStart) { RUN.vecStart = pt; $("#drawHint").textContent = "Now click the arrowhead (direction of legal flow)."; }
    else { commitRoad(RUN.vecStart, pt); }
    redraw(); return;
  }
  if (RUN.closed) return;
  RUN.poly.push(pt); redraw();
}
function closeShape() {
  if (RUN.poly.length < 3) { $("#drawHint").textContent = "Need at least 3 points."; return; }
  RUN.closed = true;
  if (RUN.mode === "parking") { commitParking(); }
  else { RUN.awaitingVector = true; $("#drawHint").textContent = "Lane closed. Click TWO points for the flow arrow (tail → head)."; }
  redraw();
}
function commitParking() {
  RUN.zones.push({ zone_id: RUN.zones.length + 1, zone_type: "illegal_parking", polygon: RUN.poly.slice(), legal_vector: null });
  resetDrawing(); persist();
}
function commitRoad(start, end) {
  RUN.zones.push({ zone_id: RUN.zones.length + 1, zone_type: "road_lane", polygon: RUN.poly.slice(), legal_vector: { start, end } });
  resetDrawing(); persist();
}
function resetDrawing() { RUN.poly = []; RUN.closed = false; RUN.vecStart = null; RUN.awaitingVector = false; renderZoneList(); }
function undo() {
  if (RUN.awaitingVector && RUN.vecStart) { RUN.vecStart = null; }
  else if (RUN.awaitingVector) { RUN.awaitingVector = false; RUN.closed = false; }
  else if (RUN.poly.length) { RUN.poly.pop(); }
  else if (RUN.zones.length) { const z = RUN.zones.pop(); renderZoneList(); persist(); }
  redraw();
}
function redraw() {
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  RUN.zones.forEach(z => {
    const col = z.zone_type === "road_lane" ? "#E08A1E" : "#C8442E";
    drawPoly(z.polygon, col, true, `${z.zone_type === "road_lane" ? "Lane" : "Parking"} #${z.zone_id}`);
    if (z.legal_vector) drawArrow(z.legal_vector.start, z.legal_vector.end, "#1E5FA8");
  });
  if (RUN.poly.length) {
    const col = RUN.mode === "road" ? "#E08A1E" : "#C8442E";
    drawPoly(RUN.poly, col, RUN.closed, null, true);
  }
  if (RUN.vecStart) { const [x, y] = toDisp(...RUN.vecStart); ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fillStyle = "#1E5FA8"; ctx.fill(); }
}
function drawPoly(pts, color, fill, label, dots) {
  if (!pts.length) return;
  ctx.beginPath();
  pts.forEach((p, i) => { const [x, y] = toDisp(p[0], p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  if (fill) ctx.closePath();
  ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.stroke();
  if (fill) { ctx.fillStyle = color + "33"; ctx.fill(); }
  if (dots) pts.forEach(p => { const [x, y] = toDisp(p[0], p[1]); ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fillStyle = color; ctx.fill(); });
  if (label) { const [x, y] = toDisp(pts[0][0], pts[0][1]); ctx.fillStyle = color; ctx.font = "600 12px -apple-system,sans-serif"; ctx.fillText(label, x + 6, y - 6); }
}
function drawArrow(s, e, color) {
  const [x1, y1] = toDisp(...s), [x2, y2] = toDisp(...e);
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.stroke();
  const ang = Math.atan2(y2 - y1, x2 - x1), L = 12;
  ctx.beginPath(); ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - L * Math.cos(ang - 0.4), y2 - L * Math.sin(ang - 0.4));
  ctx.moveTo(x2, y2); ctx.lineTo(x2 - L * Math.cos(ang + 0.4), y2 - L * Math.sin(ang + 0.4));
  ctx.stroke();
}
function renderZoneList() {
  if (!RUN.zones.length) { $("#zoneList").innerHTML = `<div style="font-size:13px;color:var(--mist)">No zones yet. Drawing is optional.</div>`; return; }
  $("#zoneList").innerHTML = RUN.zones.map((z, i) => `<div style="display:flex;align-items:center;gap:9px;padding:9px 11px;border:1px solid var(--line);border-radius:8px;margin-bottom:8px;background:var(--blue-50)">
    <i style="width:11px;height:11px;border-radius:3px;background:${z.zone_type === 'road_lane' ? '#E08A1E' : '#C8442E'};flex:none"></i>
    <div style="flex:1"><div style="font-size:13px;font-weight:600">${z.zone_type === 'road_lane' ? 'Road lane' : 'Parking'} #${z.zone_id}</div><div style="font-size:11.5px;color:var(--mist)">${z.polygon.length} pts${z.legal_vector ? ' · has arrow' : ''}</div></div>
    <button onclick="removeZone(${i})" style="border:none;background:none;cursor:pointer;color:var(--mist);font-size:18px;line-height:1">×</button></div>`).join("");
}
window.removeZone = i => { RUN.zones.splice(i, 1); RUN.zones.forEach((z, k) => z.zone_id = k + 1); renderZoneList(); redraw(); persist(); };
function persist() { /* zones are sent at process time; nothing to persist mid-draw */ }

function setSteps(active) { $$("#pipeSteps .pstep").forEach(s => { const i = +s.dataset.s; s.classList.toggle("done", i < active); s.classList.toggle("active", i === active); }); }
async function startProcessing() {
  if (!RUN.id) return;
  runBtn.disabled = true; setSteps(2);
  $("#statusMsg").innerHTML = `<span class="spinner"></span> Running pipeline…${RUN.zones.length ? ` (${RUN.zones.length} zone${RUN.zones.length > 1 ? 's' : ''})` : ''}`;
  try {
    await fetch(`/api/runs/${RUN.id}/process`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ zones: RUN.zones }) });
  } catch { $("#statusMsg").innerHTML = "Could not start processing."; runBtn.disabled = false; return; }
  poll(RUN.id);
}
async function poll(id) {
  if (!id || id === "undefined") { $("#statusMsg").innerHTML = "No valid run to track. Re-upload the video."; runBtn.disabled = false; return; }
  const tick = async () => {
    const r = await fetch(`/api/runs/${id}`).then(r => r.json());
    const stepMap = { "Detection & tracking": 2, "Parked-vehicle extraction": 3, "Violation engines": 3, "Evidence harvesting": 4 };
    if (r.status === "done") {
      setSteps(4);
      $("#statusMsg").innerHTML = "✓ Pipeline complete." + (r.warn ? ` <span style="color:var(--amber)">(${r.warn})</span>` : "");
      if (r.annotated) $("#annoWrap").innerHTML = `<video controls src="/dash_output/${id}/${r.annotated}"></video>`;
      if (r.trajectory_map) $("#trajWrap").innerHTML = `<video controls src="/dash_output/${id}/${r.trajectory_map}"></video>`;
      await loadRuns(); await selectRun(id); return;
    }
    if (r.status === "error") { $("#statusMsg").innerHTML = `Pipeline failed — ${r.error}`; runBtn.disabled = false; return; }
    if (r.step && stepMap[r.step]) setSteps(stepMap[r.step]);
    $("#statusMsg").innerHTML = `<span class="spinner"></span> ${r.step || "Processing"}…`;
    setTimeout(tick, 1500);
  };
  tick();
}

(async function init() {
  const runs = await loadRuns();
  if (!runs.some(r => r.status === "done")) await loadDemo();
  showPage(document.body.dataset.active || "dashboard");
})();