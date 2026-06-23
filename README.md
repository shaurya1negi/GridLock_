# RoadIQ — Traffic Violation Detection Pipeline

**Foundation, Analytical Engine & Live Web Dashboard**

This repository contains the core tracking and analytical orchestration framework for the automated traffic-violation detection pipeline, plus a **live web dashboard (RoadIQ)** that wraps the whole system in a browser UI. The architecture is engineered around a **Single-Pass, Multi-Stage Pipeline** that decouples expensive GPU computing from downstream multi-point geometric rule checking.

---

## Pipeline Overview

```
                                          ┌─────────────────────────────────────┐
                                          │       Input Raw Traffic Video        │
                                          └──────────────────┬──────────────────┘
                                                             │
                                                             ▼
                                          ┌─────────────────────────────────────┐
                                          │        main.py (Orchestrator)        │
                                          │   - Extracts True Hardware FPS       │
                                          │   - Captures Frame Dimensions        │
                                          └──────────────────┬──────────────────┘
                                                             │
                                     ┌───────────────────────┴───────────────────────┐
                                     ▼                                               ▼
                          ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
                          │      Step 1: Zone Tracing        │   │   Step 2: trajectory_collector   │
                          │  - Traces Interactive Polygons   │   │  - In-Memory YOLO & BoT-SORT     │
                          │  - Saves Structural JSON Map     │   │  - Dual-Stream Video Writer      │
                          └────────────────┬────────────────┘   └────────────────┬────────────────┘
                                           │                                     │
                                           └──────────────────┬──────────────────┘
                                                              │  (Pristine Telemetry CSV)
                                                              ▼
                                              ┌─────────────────────────────────┐
                                              │ Step 3: extract_parked_vehicles  │
                                              │  - Mid-Video Rolling Windows     │
                                              │  - Squelches Stop Lifecycles     │
                                              └────────────────┬────────────────┘
                                                               │ (Summary & Timeline Pairs)
                                                               ▼
                                              ┌─────────────────────────────────┐
                                              │     Step 4: Violation Engine     │
                                              │  - Multi-Point Parking Voting    │
                                              │  - Vector Field Alignments       │
                                              │  - Helmet & Triple-Riding Checks │
                                              └────────────────┬────────────────┘
                                                               │ (Citations Triggered)
                                                               ▼
                                              ┌─────────────────────────────────┐
                                              │     Evidence Harvesting Loop     │
                                              │  - Lifecycle Quality Grading     │
                                              │  - Max Box Area Patch Extraction │
                                              │  - Citation Metadata Sidecars    │
                                              └─────────────────────────────────┘
```

---

## Core System Architectural Principles

**Single Source of Truth (SSoT) Telemetry Sync.** The pipeline removes all hardcoded framing constants (`FPS = 30`). `main.py` opens the incoming video container header once at execution runtime, locks down the true hardware frame rate (`TRUE_FPS`), frame width, and frame height, and propagates these primitives across every sub-module in memory. This eliminates temporal drift and coordinate-mapping displacement during downstream processing.

**In-Memory Optimization.** The terminal-spawning subprocess wrapper inside `main.py` has been completely removed. All stages — from tracking-data ingestion to stationary-segment splitting — execute natively via Python module imports.

**Simultaneous Dual-Stream Rendering.** Instead of passing a finished video through a secondary decode/encode cycle to overlay zones, `trajectory_collector.py` parses the active `parking_zones.json` structure directly during tracking inference. While the raw pixels go into the YOLO framework to ensure zero model tracking interference, a concurrent background pixel-manipulation process paints boundaries, alpha-blended transparent forbidden areas, and direction arrows onto the `isolated_trajectory_map_*.mp4` black-canvas map in a single pass.

---

## Complete Pipeline Data Flow

```
                  [Raw Bounding Box from YOLO/BoT-SORT]
                                  │
                                  ▼
       [Layer 1: Bounding Box One-Euro Filter]  ──►  Smooths x1, y1, x2, y2 boundary matrices instantly
                                  │                   (Maintains responsive, zero-trailing lag)
                                  ▼
            [Calculate Velocity Math]            ──►  Compute derivative vectors: vx = Δx / Δt, vy = Δy / Δt
                                  │                   (Note: raw math amplifies sub-pixel jitter)
                                  ▼
        [Layer 2: Velocity Low-Pass Filter]      ──►  Low-alpha Vector EMA filter stabilizes tracking lines,
                                  │                   wiping out high-frequency derivative spikes
                                  ▼
     [Unified Ground-Truth Tracking CSV Matrix]  ──►  Distributed sequentially to analytical modules
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                 ┌───────────────┐ ┌───────────────┐
                 │ Parking Rule  │ │  Motion Rule  │
                 └───────┬───────┘ └───────┬───────┘
                         │                 │
                         ▼                 ▼
                 ┌─────────────────────────────────────────┐
                 │      Intelligent Evidence Harvester      │
                 │  - Drops entry/exit buffer frame rows    │
                 │  - Identifies target peak box area       │
                 │  - Crops JPG patch & writes sidecar TXT  │
                 └─────────────────────────────────────────┘
```

---

## CSV Schema

Every run generates a chronologically structured, tracking-lifecycle-grouped matrix sorted sequentially by `vehicle_class → tracker_id → frame_index`.

| Column | Type | Detailed Engineering Definition |
| :--- | :--- | :--- |
| `tracker_id` | `int` | Persistent identity tag assigned by BoT-SORT. Remains invariant across brief occlusions and visual tracking blocks. |
| `frame_index` | `int` | Zero-based incremental frame index counter. |
| `timestamp_sec` | `float` | Chronological timeline position computed via `frame_index / TRUE_FPS`. Independent of CPU wall-clock processing time. |
| `vehicle_class` | `str` | Stabilized classification string determined by class-history majority voting (e.g., `"car"`, `"motorcycle"`, `"person"`). |
| `class_id` | `int` | Original numerical YOLO classification tracking index. |
| `confidence` | `float` | Raw confidence coefficient, bounded from `0.0` to `1.0`. |
| `x1, y1, x2, y2` | `float` | One-Euro-filtered top-left and bottom-right pixel edge-bound coordinates. |
| `bottom_center_x` | `float` | Midpoint x-coordinate along the lower base-bound line. |
| `bottom_center_y` | `float` | Ground-plane road-contact point y-coordinate. **Required for spatial zone-intersection tests** to neutralize centroid vertical shifting on tall vehicles (trucks/buses). |
| `box_width` | `float` | Horizontal frame footprint width in pixels (`x2 - x1`). |
| `box_height` | `float` | Vertical frame footprint height in pixels (`y2 - y1`). |
| `velocity_x_px_sec` | `float` | Filtered horizontal velocity component in pixels per second. |
| `velocity_y_px_sec` | `float` | Filtered vertical velocity component in pixels per second. |
| `speed_px_sec` | `float` | Scalar velocity magnitude computed as $\sqrt{v_x^2 + v_y^2}$. Used for stationary checks and speed grading. |
| `heading_deg` | `float` | Angular trajectory direction ($\theta \in [0^\circ, 360^\circ)$), where $0^\circ = \text{Right}$, $90^\circ = \text{Down}$, $180^\circ = \text{Left}$, $270^\circ = \text{Up}$. Returns `-1.0` if stationary. |
| `is_stationary` | `int` | Boolean flag (`0` or `1`). Asserts `1` if the moving scalar speed remains below threshold for continuous frame cycles. |
| `frames_since_first_seen` | `int` | Rolling lifetime persistence tracker. Used to discard ghost artifacts ($\text{frames} < 5$) and set confirmation windows for complex violations. |

---

## Detailed Violation Rule-Engine Mechanics

### 1. Multi-Point Spatial Consensus Parking Check

The parking rule engine bypasses single-point boundary errors by setting up a **5-Point Consensus Voting Grid** mapped proportionally across the vehicle bounding-box footprint:

```
                            x1                         x2
                       y1  ┌───────────────────────────┐
                           │     o (Top Inset)         │
                           │                           │
                           │o(Left)  o(Center)  o(Right)│
                           │                           │
                           │     o (Bottom Center)     │
                       y2  └───────────────────────────┘
```

**The Logic:** When a vehicle enters a stationary state inside the tracking array, the engine extracts its tracking metrics and tests all 5 coordinates against the prohibited-zone polygon. An automated citation ticket is generated **if and only if at least 3 of the 5 points register completely inside the polygon boundary.** This eliminates false positives caused by overlapping shadow boundaries or adjacent-lane traffic.

### 2. Vector Field Alignment (Wrong-Side Driving)

Instead of relying on simple directional zone boundaries, the motion checker treats road-lane polygons as directional vector fields:

```
                  Legal Vector Direction (Heading: 90° Downward)
                        ┌──────────────────────────────┐
                        │              │               │
                        │              ▼               │
                        └──────────────────────────────┘
                                       ▲
                                       │
               Offending Vehicle Trajectory Vector (Heading: 270° Upward)
```

**The Logic:** When tracing a `road_lane` structural polygon, the operator explicitly registers a legal direction-vector arrow across the lane-center profile. The engine calculates the base heading angle of this vector. As vehicles traverse the polygon, their stabilized `heading_deg` values are continually evaluated against the lane's allowed direction. If the angular deviation exceeds the calibrated threshold ($\Delta\theta \ge 135^\circ$), the vehicle is instantly flagged for wrong-side driving.

### 3. Motorcycle Compliance (Helmet & Triple-Riding)

For every tracked motorcycle, the compliance layer crops the rider region and runs two checks:

- **Helmet check** — a dedicated helmet model classifies each rider's head; a bare head raises a **Helmet Violation**.
- **Rider-count check** — a person model counts occupants inside the motorcycle crop; **≥ 2 riders** raises a **Triple-Riding / Overloading** violation.

> Helmet & triple-riding detection require the `weights/helmet.pt` model. If it is absent, the pipeline gracefully skips this stage and continues with the other checks.

---

## Intelligent Evidence Harvesting Core

The system does not crop frames at the exact microsecond a rule condition is crossed. The evidence-harvesting layer performs automated trajectory-lifecycle quality grading:

- **Path Boundary Stabilization** — the harvester slices away the initial and closing 10% of the vehicle's violation tracking lifespan to omit entry/exit camera artifacts and edge occlusions.
- **Peak Area Extraction** — it iterates through the remaining timeline and isolates the frame where the vehicle achieves its **maximum bounding-box pixel area**, ensuring the captured asset is closest to the lens and free of motion blur.
- **Padded Citation Packaging** — it captures the frame patch with a 15% expanded padding perimeter, saves a crisp standalone `.jpg` evidence file, and writes a formatted `.txt` sidecar citation docket filled with vector-trajectory data, timestamp mappings, and confidence ratings.

---

## Directory Layout

```
├── main.py                     # Global Orchestrator & SSoT Initialization (routes video vs image)
├── polygon.py                  # Geometric Zone Configuration & Interactive UI Engine
├── trajectory_collector.py     # In-Memory Tracking, Telemetry Ingestion & Canvas Blending
├── extract_parked_vehicles.py  # Stationary Segment Slicing & Physics Squelching
├── voilation_detector.py       # Rule Engine (5-Point Consensus Grid, Vector Alignment, Compliance)
├── evidence_harvester.py       # Trajectory Quality Grading & Patch Extraction
├── video_engine.py             # Video pipeline orchestration
├── image_engine.py             # Static-photo audit pipeline (helmet & triple-riding)
├── config.py                   # Global Model Weight Declarations & Hyperparameters
│
├── web.py                      # RoadIQ web dashboard (FastAPI app)
├── templates/
│   └── index.html              # Dashboard markup
├── static/
│   ├── dashboard.js            # Dashboard logic & live SVG charts
│   └── dashboard.css           # Dashboard styling
│
├── weights/
│   ├── yolo11m.pt              # Detection / person model
│   └── helmet.pt               # Helmet model (required for helmet & triple-riding)
│
└── output/  (dash_output/ when run via the dashboard)
    ├── parking_zones_*.json        # Serialized Road Geometric Zone Databases
    ├── trajectory_data_*.csv       # High-Stability Primary Telemetry Matrix
    ├── summary_parked_*.csv        # Aggregated Log of Stationary Vehicle Incidents
    ├── timeline_parked_*.csv       # Filtered Microframe Records of Parked Life-cycles
    ├── violations_wrong_side_*     # Validated Vector Deviation Summary Reports
    ├── violations_compliance_*     # Helmet & Triple-Riding Violation Records
    ├── timeline_wrong_side_*       # Frame Matrix Tracks of Vehicles Driving Against Traffic
    └── evidence/                   # Automated Citations Repository Directory
        ├── ticket_id_*.jpg         # Maximum-Area Quality-Graded Image Patch Crops
        └── ticket_id_*.txt         # Complete Citation Metadata Ticket Sidecars
```

---

## Installation

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 2. install dependencies
pip install -r requirements.txt
```

The helmet model is **not** a pip package — fetch it once and place it at `weights/helmet.pt`:

```bash
pip install huggingface_hub
python download_helmet.py
```

> `download_helmet.py` prints the model's class order after download. The pipeline expects
> class `0 = With helmet`, class `1 = Without helmet`. If the printed order differs, adjust accordingly.

---

## A. Running the Pipeline (Command Line)

To execute the complete pipeline, pass the input raw file down through the configuration flags:

```bash
python main.py --input data/traffic_stream.mp4
```

To reuse existing polygon layouts across repeated evaluation runs and skip the interactive configuration drawer, append the skip modifier flag:

```bash
python main.py --input data/traffic_stream.mp4 --skip-polygon-drawing
```

`main.py` routes automatically by file type — pass a **photo** instead of a video to run the
static-image audit (helmet & triple-riding only; parking and wrong-side need motion):

```bash
python main.py --input data/scene_photo.jpg
```

All artifacts (telemetry CSV, violation reports, annotated videos, evidence crops) are written to the `output/` directory.

---

## B. Running the Web Dashboard (RoadIQ UI)

The dashboard wraps the same pipeline in a browser interface — upload footage, draw zones,
run the pipeline, and browse violations, evidence and analytics. It is served by **Uvicorn**
(an ASGI server) from the FastAPI app in `web.py`.

### 1. Install the web dependencies

These are already in `requirements.txt`, but if you installed only the core pipeline packages, add:

```bash
pip install fastapi "uvicorn[standard]" jinja2 python-multipart
```

### 2. (Optional) install ffmpeg for in-browser video playback

OpenCV writes `mp4v` video, which most browsers can't play. The dashboard auto-converts
finished videos to browser-friendly H.264 using **ffmpeg** if it's installed:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

If ffmpeg is absent the dashboard still runs — only the in-page video preview may not play.

### 3. Start the server

From the project root (the folder containing `web.py`):

```bash
uvicorn web:app --port 9123
```

Then open the dashboard in your browser:

```
http://localhost:9123
```

Useful variants:

```bash
# auto-reload while developing (restarts on file changes)
uvicorn web:app --port 9123 --reload

# expose to other devices on your network
uvicorn web:app --host 0.0.0.0 --port 9123
```

Here `web:app` means "the `app` object inside `web.py`". Stop the server with **Ctrl + C**.

### 4. Using the dashboard

- **Overview** — pipeline status and stage diagram.
- **Process video** — upload a video (or switch to **Photo** mode for image audits), draw zones, run the pipeline. Progress is polled automatically.
- **Violations** — every detected violation with type filters (parking / wrong-side / helmet / triple-riding) and CSV export.
- **Evidence** — harvested best-shot crops with their citation tickets.
- **Analytics** — six live charts (heading rose, density timeline, heatmap, speed distribution, fleet mix, violation breakdown) plus a motorcycle-compliance summary.

> After editing `static/` or `templates/` files, hard-refresh the browser (**Cmd/Ctrl + Shift + R**) so it loads the new assets instead of cached ones.

---

## Quick Reference

| Goal | Command |
| :--- | :--- |
| Process a video (CLI) | `python main.py --input data/clip.mp4` |
| Reuse saved zones | `python main.py --input data/clip.mp4 --skip-polygon-drawing` |
| Audit a photo (CLI) | `python main.py --input data/photo.jpg` |
| Download helmet model | `python download_helmet.py` |
| Launch the dashboard | `uvicorn web:app --port 9123` → `http://localhost:9123` |
