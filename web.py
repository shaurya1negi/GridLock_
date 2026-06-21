"""
web.py  —  Traffic Violation Dashboard server
=============================================
Drop this file inside your GridLock_-main folder (next to main.py, polygon.py,
voilation_detector.py, etc.). It imports YOUR real modules directly and runs the
same 5-step pipeline main.py does, but headless — the browser draws the zones.

Run it with:
    uvicorn web:app
Then open http://127.0.0.1:8000

Needs two folders next to this file:
    templates/index.html
    static/dashboard.js
    static/dashboard.css
(the dashboard creates uploads/ and dash_output/ automatically)
"""
import base64
import json
import os
import shutil
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

UPLOADS = HERE / "uploads"
OUTPUTS = HERE / "dash_output"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="Traffic Violation Dashboard")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))

RUNS = {}
LOCK = threading.Lock()


def _set(rid, **kw):
    with LOCK:
        RUNS[rid].update(**kw)


def _latest(folder, prefix, suffix):
    files = sorted(folder.glob(f"{prefix}*{suffix}"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


# --------------------------------------------------------------------------- #
# Pipeline runner — same order as your main.py, headless
# --------------------------------------------------------------------------- #
def _run_pipeline(rid, video_path, out_dir, zones_json):
    try:
        import trajectory_collector
        import extract_parked_vehicles
        import voilation_detector as violator
        import evidence_harvester

        cap = cv2.VideoCapture(video_path)
        true_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if not true_fps or pd.isna(true_fps):
            true_fps = 30.0

        _set(rid, status="processing", step="Detection & tracking")
        trajectory_collector.process_video(
            video_path=video_path, output_dir=out_dir,
            fps_override=true_fps, polygon_json_path=zones_json)

        out = Path(out_dir)
        traj = _latest(out, "trajectory_data_", ".csv")
        if not traj:
            raise RuntimeError("trajectory CSV not produced")
        base = traj.stem

        # parked extraction
        _set(rid, step="Parked-vehicle extraction")
        df = pd.read_csv(traj)
        recs, blocks = [], []
        for tid, hist in df.groupby("tracker_id"):
            stops, sq = extract_parked_vehicles.extract_stationary_segments(
                hist, fps=true_fps, min_stop_duration_sec=3.0)
            if stops:
                vc = hist["vehicle_class"].iloc[0]
                for i, s in enumerate(stops):
                    recs.append({
                        "tracker_id": tid, "vehicle_class": vc, "stop_index": i + 1,
                        "median_position_x": s["median_position_x"],
                        "median_position_y": s["median_position_y"],
                        "spawn_frame": s["start_frame"], "death_frame": s["end_frame"],
                        "duration_sec": s["duration_sec"],
                        "lower_timestamp": round(s["start_frame"] / true_fps, 2),
                        "upper_timestamp": round(s["end_frame"] / true_fps, 2)})
                blocks.extend(sq)
        if recs:
            pd.DataFrame(recs).to_csv(out / f"summary_parked_{base}.csv", index=False)
            pd.concat(blocks, ignore_index=True).to_csv(out / f"timeline_parked_{base}.csv", index=False)

        # violation engines
        _set(rid, step="Violation engines")
        summ = _latest(out, "summary_parked_", ".csv")
        pc = wc = None
        if summ:
            try:
                pc = violator.detect_parking_violations(str(summ), zones_json, out_dir)
            except Exception as e:
                _set(rid, warn=f"parking: {e}")
        try:
            wc = violator.detect_wrong_side_violations(str(traj), zones_json, out_dir)
        except Exception as e:
            _set(rid, warn=f"wrong-side: {e}")

        # evidence
        _set(rid, step="Evidence harvesting")
        if pc and os.path.exists(pc):
            try:
                evidence_harvester.harvest_violation_patches(
                    video_path, pc, str(out / f"timeline_parked_{base}.csv"), out_dir)
            except Exception as e:
                _set(rid, warn=f"parking evidence: {e}")
        if wc and os.path.exists(wc):
            try:
                evidence_harvester.harvest_violation_patches(
                    video_path, wc, str(out / f"timeline_wrong_side_{base}.csv"), out_dir)
            except Exception as e:
                _set(rid, warn=f"wrong-side evidence: {e}")

        ann = _latest(out, "annotated_output_", ".mp4")
        tmap = _latest(out, "isolated_trajectory_map_", ".mp4")
        _set(rid, status="done", step="Complete",
             annotated=ann.name if ann else None,
             trajectory_map=tmap.name if tmap else None)
    except Exception as exc:
        _set(rid, status="error", error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Output readers
# --------------------------------------------------------------------------- #
def _rundir(rid):
    return OUTPUTS / rid


def summary_for(rid):
    out = _rundir(rid)
    traj = _latest(out, "trajectory_data_", ".csv")
    if not traj:
        return {}
    df = pd.read_csv(traj)
    real = df[df["frames_since_first_seen"] >= 5] if "frames_since_first_seen" in df else df
    parking = _latest(out, "violations_parking_", ".csv")
    wrong = _latest(out, "violations_wrong_side_", ".csv")
    return {
        "unique_vehicles": int(real["tracker_id"].nunique()),
        "total_detections": int(len(df)),
        "frames": int(df["frame_index"].max() + 1) if len(df) else 0,
        "class_breakdown": {k: int(v) for k, v in real.groupby("vehicle_class")["tracker_id"].nunique().items()},
        "n_parking": len(pd.read_csv(parking)) if parking else 0,
        "n_wrong_side": len(pd.read_csv(wrong)) if wrong else 0,
    }


def violations_for(rid):
    out = _rundir(rid)
    rows = []
    p = _latest(out, "violations_parking_", ".csv")
    w = _latest(out, "violations_wrong_side_", ".csv")
    if p:
        for v in pd.read_csv(p).to_dict("records"):
            rows.append({
                "tracker_id": v.get("tracker_id"), "type": "Illegal parking",
                "vehicle_class": v.get("vehicle_class"),
                "timestamp_sec": float(v.get("lower_timestamp", 0)),
                "detail": f"zone #{v.get('prohibited_zone_id')} · {v.get('points_inside_zone')} pts · {v.get('duration_sec')}s",
                "confidence": _conf(v.get("points_inside_zone"))})
    if w:
        for v in pd.read_csv(w).to_dict("records"):
            rows.append({
                "tracker_id": v.get("tracker_id"), "type": "Wrong-side driving",
                "vehicle_class": v.get("vehicle_class"),
                "timestamp_sec": float(v.get("timestamp_sec", 0)),
                "detail": f"lane #{v.get('road_zone_id')} · {v.get('angular_deviation')}° vs legal {v.get('legal_heading_deg')}°",
                "confidence": min(0.99, float(v.get("angular_deviation", 135)) / 180)})
    rows.sort(key=lambda r: r["timestamp_sec"])
    return rows


def _conf(s):
    try:
        a, b = str(s).split("/")
        return round(int(a) / int(b), 2)
    except Exception:
        return 0.6


def analytics_for(rid):
    out = _rundir(rid)
    traj = _latest(out, "trajectory_data_", ".csv")
    if not traj:
        return {}
    df = pd.read_csv(traj)
    df = df[df["frames_since_first_seen"] >= 5] if "frames_since_first_seen" in df else df
    per = df.groupby(df["timestamp_sec"].astype(int))["tracker_id"].nunique()
    moving = df[df["speed_px_sec"] > 0]["speed_px_sec"]
    return {
        "density_over_time": [{"t": int(t), "count": int(c)} for t, c in per.items()],
        "avg_speed_px_sec": round(float(moving.mean()), 1) if len(moving) else 0,
        "peak_concurrent": int(per.max()) if len(per) else 0,
    }


def evidence_for(rid):
    ev = _rundir(rid) / "evidence"
    if not ev.exists():
        return []
    items = []
    for jpg in sorted(ev.glob("*.jpg")):
        txt = jpg.with_suffix(".txt")
        items.append({"image": f"/dash_output/{rid}/evidence/{jpg.name}",
                      "filename": jpg.name,
                      "citation": txt.read_text() if txt.exists() else ""})
    return items


# --------------------------------------------------------------------------- #
# Pages + API
# --------------------------------------------------------------------------- #
PAGES = {"dashboard": "Overview", "upload": "Process video",
         "violations": "Violations", "evidence": "Evidence", "analytics": "Analytics"}


@app.get("/", response_class=HTMLResponse)
@app.get("/{page}", response_class=HTMLResponse)
def page(request: Request, page: str = "dashboard"):
    if page not in PAGES:
        raise HTTPException(404)
    return templates.TemplateResponse(request, "index.html", {"active": page, "pages": PAGES})


@app.post("/api/upload")
async def upload(video: UploadFile = File(...)):
    rid = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    dest = UPLOADS / f"{rid}_{video.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(video.file, f)
    (OUTPUTS / rid).mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(dest))
    ok, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok:
        raise HTTPException(400, "Could not read a frame from that video (codec issue?)")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    with LOCK:
        RUNS[rid] = {"status": "uploaded", "video": video.filename, "video_path": str(dest),
                     "step": "Awaiting zones", "annotated": None, "trajectory_map": None,
                     "error": None, "warn": None}
    return {"run_id": rid, "image": f"data:image/jpeg;base64,{b64}", "width": w, "height": h}


@app.post("/api/runs/{rid}/process")
async def process(rid: str, payload: dict):
    with LOCK:
        run = RUNS.get(rid)
    if not run:
        raise HTTPException(404, "Unknown run")
    out_dir = str(_rundir(rid))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zones_json = os.path.join(out_dir, f"parking_zones_{stamp}.json")
    with open(zones_json, "w") as f:
        json.dump(payload.get("zones", []), f)
    _set(rid, status="queued", step="Queued")
    threading.Thread(target=_run_pipeline, args=(rid, run["video_path"], out_dir, zones_json), daemon=True).start()
    return {"run_id": rid, "status": "queued", "zones": len(payload.get("zones", []))}


@app.get("/api/runs")
def list_runs():
    with LOCK:
        return [{"run_id": k, **v} for k, v in sorted(RUNS.items(), reverse=True)]


@app.get("/api/runs/{rid}")
def run_status(rid: str):
    with LOCK:
        if rid not in RUNS:
            raise HTTPException(404)
        return {"run_id": rid, **RUNS[rid]}


@app.get("/api/runs/{rid}/summary")
def api_summary(rid):
    return summary_for(rid)


@app.get("/api/runs/{rid}/violations")
def api_violations(rid):
    return violations_for(rid)


@app.get("/api/runs/{rid}/analytics")
def api_analytics(rid):
    return analytics_for(rid)


@app.get("/api/runs/{rid}/evidence")
def api_evidence(rid):
    return evidence_for(rid)


@app.get("/dash_output/{rid}/{path:path}")
def serve_output(rid: str, path: str):
    fp = _rundir(rid) / path
    if not fp.exists():
        raise HTTPException(404)
    return FileResponse(str(fp))


@app.get("/api/demo")
def demo():
    return {
        "summary": {"unique_vehicles": 59, "total_detections": 14211, "frames": 1192,
                    "class_breakdown": {"car": 53, "motorcycle": 1, "truck": 8},
                    "n_parking": 12, "n_wrong_side": 46},
        "violations": [
            {"tracker_id": 6, "type": "Illegal parking", "vehicle_class": "car", "timestamp_sec": 1.48,
             "detail": "zone #1 · 4/5 pts · 9.16s", "confidence": 0.8},
            {"tracker_id": 3, "type": "Wrong-side driving", "vehicle_class": "car", "timestamp_sec": 0.2,
             "detail": "lane #2 · 143.6° vs legal 0.0°", "confidence": 0.8},
            {"tracker_id": 19, "type": "Illegal parking", "vehicle_class": "car", "timestamp_sec": 11.48,
             "detail": "zone #1 · 5/5 pts · 15.64s", "confidence": 1.0}],
        "analytics": {"density_over_time": [{"t": i, "count": c} for i, c in enumerate(
            [3,4,5,7,8,9,11,12,10,9,8,9,11,13,12,10,8,6,7,9,10,12,11,9,8,6,5,4,6,7])],
            "avg_speed_px_sec": 104.4, "peak_concurrent": 22},
        "evidence": [],
    }