# Traffic Violation Detection Pipeline — Foundation & Analytical Engine
This repository contains the core tracking and analytical orchestration framework for the automated traffic violation detection pipeline. The architecture is engineered around a Single Pass, Multi-Stage Pipeline that decouples expensive GPU computing from downstream multi-point geometric rule checking.
```
                                                          ┌─────────────────────────────────────┐
                                                          │      Input Raw Traffic Video        │
                                                          └──────────────────┬──────────────────┘
                                                                             │
                                                                             ▼
                                                          ┌─────────────────────────────────────┐
                                                          │       main.py (Orchestrator)        │
                                                          │   - Extracts True Hardware FPS      │
                                                          │   - Captures Frame Dimensions       │
                                                          └──────────────────┬──────────────────┘
                                                                             │
                                                                             ▼
                                                 ┌───────────────────────────┴───────────────────────────┐
                                                 ▼                                                       ▼
                                      ┌─────────────────────────────────┐             ┌─────────────────────────────────┐
                                      │     Step 1: Zone Tracing        │             │   Step 2: trajectory_collector  │
                                      │  - Traces Interactive Polygons  │             │  - In-Memory YOLO & BoT-SORT    │
                                      │  - Saves Structural JSON Map    │             │  - Dual-Stream Video Writer     │
                                      └────────────────┬────────────────┘             └────────────────┬────────────────┘
                                                       │                                               │
                                                       └───────────────────────┬───────────────────────┘
                                                                               │  (Pristine Telemetry CSV)
                                                                               ▼
                                                              ┌─────────────────────────────────┐
                                                              │ Step 3: extract_parked_vehicles │
                                                              │  - Mid-Video Rolling Windows    │
                                                              │  - Squelches Stop Lifecycles    │
                                                              └────────────────┬────────────────┘
                                                                               │ (Summary & Timeline Pairs)
                                                                               ▼
                                                              ┌─────────────────────────────────┐
                                                              │     Step 4: Violation Engine    │
                                                              │  - Multi-Point Parking Voting   │
                                                              │  - Vector Field Alignments      │
                                                              └────────────────┬────────────────┘
                                                                               │ (Citations Triggered)
                                                                               ▼
                                                              ┌─────────────────────────────────┐
                                                              │    Evidence Harvesting Loop     │
                                                              │  - Lifecycle Quality Grading    │
                                                              │  - Max Box Area Patch Extraction│
                                                              │  - Citation Metadata Sidecars   │
                                                              └─────────────────────────────────┘  
```
                        
Core System Architectural PrinciplesSingle Source of Truth (SSoT) Telemetry Sync: The pipeline removes all hardcoded framing constants (FPS = 30). main.py opens the incoming video container header once at execution runtime, locks down the true hardware frame rate (TRUE_FPS), frame width, and frame height, and propagates these primitives across every sub-module in memory. This eliminates temporal drift and coordinate mapping displacement during downstream processing.In-Memory Optimization: The terminal-spawning subprocess wrapper inside main.py has been completely deleted. All stages—from tracking data ingestion to stationary segment splitting—are executed natively via Python module imports.Simultaneous Dual-Stream Rendering: Instead of passing a finished video through a secondary decoding/encoding cycle to overlay zones, trajectory_collector.py parses your active parking_zones.json structure directly during tracking inference. While the raw pixels go into the YOLO framework to ensure zero model tracking interference, a concurrent background pixel manipulation process paints boundaries, alpha-blended transparent forbidden areas, and direction arrows onto your isolated_trajectory_map_*.mp4 black canvas map in a single pass.Complete Pipeline Data Flow Diagram

```
                                                              [Raw Bounding Box from YOLO/BoT-SORT]
                                                                              │
                                                                              ▼
                                                              [Layer 1: Bounding Box One-Euro Filter] ──► Smooths x1, y1, x2, y2 boundary matrices instantly
                                                                              │                           (Maintains responsive, zero trailing lag)
                                                                              ▼
                                                                [Calculate Velocity Math]            ──► Compute derivative vectors: vx = Δx / Δt, vy = Δy / Δt
                                                                              │                           (Note: Raw math amplifies sub-pixel jitter)
                                                                              ▼
                                                             [Layer 2: Velocity Low-Pass Filter]     ──► Low-alpha Vector EMA Filter stabilizes tracking lines,
                                                                              │                           wiping out high-frequency derivative spikes directly
                                                                              ▼
                                                            [Unified Ground-Truth Tracking CSV Matrix] ──► Distributed sequentially out to analytical modules
                                                                              │
                                                                     ┌────────┴────────┐
                                                                     ▼                 ▼
                                                             ┌───────────────┐ ┌───────────────┐
                                                             │ Parking Rule  │ │  Motion Rule  │
                                                             └───────┬───────┘ └───────┬───────┘
                                                                     │                 │
                                                                     ▼                 ▼
                                                             ┌─────────────────────────────────────────┐
                                                             │      Intelligent Evidence Harvester     │
                                                             │  - Drops entry/exit buffer frame rows   │
                                                             │  - Identifies target peak box area      │
                                                             │  - Crops JPG patch & writes sidecar TXT │
                                                             └─────────────────────────────────────────┘
```

# CSV Schema
Every run generates a chronologically structured, tracking lifecycle-grouped matrix sorted sequentially by vehicle_class -> tracker_id -> frame_index.

# Detailed Violation Rule Engine Mechanics
1. Multi-Point Spatial Consensus Parking Check
The parking rule engine completely bypasses single-point boundary errors by setting up a 5-Point Consensus Voting Grid mapped proportionally across the vehicle bounding box footprint:

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

The Logic: When a vehicle enters a stationary state inside the tracking array, the engine extracts its tracking metrics and tests all 5 coordinates against the prohibited zone polygon. An automated citation ticket is generated if and only if at least 3 out of the 5 points register completely inside the polygon boundary. This eliminates false positives caused by overlapping shadow boundaries or adjacent lane traffic.

2. Vector Field Alignment (Wrong-Side Driving)Instead of relying on simple directional zone boundaries, the motion checker treats road lane polygons as directional vector fields: 

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
The Logic: When tracing out a road_lane structural polygon, the operator explicitly registers a legal direction vector arrow mapping directly across the lane center profile. The engine calculates the base heading angle of this vector. As vehicles traverse this polygon, their stabilized heading_deg values are continually evaluated against the lane's allowed direction angle. If the angular deviation exceeds your calibrated threshold ($\Delta\theta \ge 135^\circ$), the vehicle is instantly flagged for wrong-side driving.

# Intelligent Evidence Harvesting Core

Your system doesn't crop frames at the exact microsecond a rule condition is crossed. The evidence harvesting layer performs automated trajectory life-cycle quality grading:Path Boundary Stabilization: The harvester automatically slices away the initial and closing 10% blocks of the vehicle's violation tracking lifespan to omit entry/exit camera artifacts or edge occlusions.Peak Area Extraction: It iterates through the remaining timeline and isolates the frame where the vehicle achieves its maximum bounding box pixel area, ensuring the asset captured is closest to the lens and free from motion blur.Padded Citation Packaging: It captures the frame patch with a 15% expanded padding perimeter, saves a crisp standalone .jpg image evidence file, and outputs a formatted .txt official sidecar citation docket filled with vector trajectory data, timestamp mappings, and confidence ratings.Directory Pipeline Architecture Layout├── main.py                     

# Directory Pipeline Architecture Layout

```
                                      ├── main.py                     # Global Orchestrator & SSoT Initialization
                                      ├── polygon.py                  # Geometric Zone Configuration & Interactive UI Engine
                                      ├── trajectory_collector.py     # In-Memory Tracking, Telemetry Ingestion & Canvas Blending
                                      ├── extract_parked_vehicles.py  # Stationary Segment Slicing & Physics Squelching
                                      ├── voilation_detector.py       # Rule Engine (5-Point Consensus Grid & Vector Alignment)
                                      ├── evidence_harvester.py       # Trajectory Quality Grading & Patch Extraction
                                      ├── config.py                   # Global Model Weight Declarations & Hyperparameters
                                      └── output/
                                          ├── parking_zones_*.json     # Serialized Road Geometric Zone Databases
                                          ├── trajectory_data_*.csv   # High-Stability Primary Telemetry Matrix
                                          ├── summary_parked_*.csv    # Aggregated Log of Stationary Vehicle Incidents
                                          ├── timeline_parked_*.csv   # Filtered Microframe Records of Parked Life-cycles
                                          ├── violations_wrong_side_* # Validated Vector Deviation Summary Reports
                                          ├── timeline_wrong_side_* # Frame Matrix Tracks of Vehicles Driving Against Traffic
                                          └── evidence/               # Automated Citations Repository Directory
                                              ├── best_shot_id_*.jpg  # Maximum Area Quality-Graded Image Patch Crops
                                              └── best_shot_id_*.txt  # Complete Citation Metadata Ticket Sidecars
```

# Verification Execution Flow
To execute the complete pipeline, pass the input raw file string down through your configuration flags:

>>python main.py --video data/traffic_stream.mp4

To reuse existing polygon layouts across repeated evaluation runs and skip the interactive configuration drawer, append the skip modifier flag:

>>python main.py --video data/traffic_stream.mp4 --skip-polygon-drawing
