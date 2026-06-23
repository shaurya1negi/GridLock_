"""
video_engine.py
---------------
Dedicated Processing Engine for Spatiotemporal Stream Sequences.
Handles tracking lifecycle trajectories and temporal de-bounced analysis.
Upgraded with multi-model micro-compliance safety tracking layers.
"""

import os
import sys
import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import config  # References central parameters

# Core Pipeline Module Imports
import trajectory_collector
import polygon
import extract_parked_vehicles  
import voilation_detector as violator
import evidence_harvester

def extract_first_frame(video_path):
    """Extract the first frame of a video for polygon drawing."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame from video: {video_path}")
    
    return frame

def get_latest_trajectory_csv(output_dir):
    """Find the most recently created trajectory_data CSV."""
    csv_files = [f for f in os.listdir(output_dir) 
                 if f.startswith("trajectory_data_") and f.endswith(".csv")]
    if not csv_files:
        return None
    csv_files.sort(key=lambda f: os.path.getctime(os.path.join(output_dir, f)), reverse=True)
    return os.path.join(output_dir, csv_files[0])

def get_latest_summary_parked_csv(output_dir):
    """Find the most recently created summary_parked CSV."""
    csv_files = [f for f in os.listdir(output_dir) 
                 if f.startswith("summary_parked_") and f.endswith(".csv")]
    if not csv_files:
        return None
    csv_files.sort(key=lambda f: os.path.getctime(os.path.join(output_dir, f)), reverse=True)
    return os.path.join(output_dir, csv_files[0])

def get_latest_polygon_json(output_dir):
    """Locates the absolute newest zone configuration profile to enable skip reuse."""
    json_files = [f for f in os.listdir(output_dir)
                  if f.startswith("parking_zones_") and f.endswith(".json")]
    if not json_files:
        return None
    json_files.sort(key=lambda f: os.path.getctime(os.path.join(output_dir, f)), reverse=True)
    return os.path.join(output_dir, json_files[0])

def get_latest_compliance_csv(output_dir):
    """Find the most recently created violations_compliance CSV."""
    csv_files = [f for f in os.listdir(output_dir) 
                 if f.startswith("violations_compliance_") and f.endswith(".csv")]
    if not csv_files:
        return None
    csv_files.sort(key=lambda f: os.path.getctime(os.path.join(output_dir, f)), reverse=True)
    return os.path.join(output_dir, csv_files[0])


def run_video_pipeline(video_path, output_dir, skip_polygon_drawing , bike_compliance = False):
    """Executes the complete spatiotemporal tracking, checking, and harvest loop."""
    
    # Hardware Container Metadata Verification Hook
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not read video header stream: {video_path}")
        sys.exit(1)
        
    TRUE_FPS = cap.get(cv2.CAP_PROP_FPS)
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if TRUE_FPS == 0 or pd.isna(TRUE_FPS): 
        TRUE_FPS = 30.0
        
    print(f"Video Container Metadata Synced Globally:")
    print(f"   • Hardware Frame Rate : {round(TRUE_FPS, 2)} FPS")
    print(f"   • Total Frame Count   : {TOTAL_FRAMES} frames")
    print("\n" + "-"*70)

    # =========================================================================
    # STEP 1: POLYGON DRAWING VIA MODULAR IMPORT
    # =========================================================================
    print("\n[STEP 1/5] Drawing Prohibited Parking Zones...")
    print("-" * 70)
    
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    polygon_json_path = os.path.join(output_dir, f"parking_zones_{run_stamp}.json")
    polygons = []
    
    existing_config = get_latest_polygon_json(output_dir)
    
    if skip_polygon_drawing and existing_config:
        print(f"⏭  Reusing existing road zone topology profile natively: {existing_config}")
        polygon_json_path = existing_config
        polygons = polygon.load_polygons(polygon_json_path)
    else:
        if skip_polygon_drawing:
            print("⚠️ Notice: --skip-polygon-drawing fallback trigger: No pre-existing layout profile found to load.")
            
        try:
            frame = extract_first_frame(video_path)
            print(f"Extracted first frame: {frame.shape[0]}x{frame.shape[1]} pixels")
            
            drawer = polygon.PolygonDrawer(frame, window_name="Draw Prohibited Parking Zones")
            polygons = drawer.draw_polygons()
            
            if polygons:
                polygon.save_polygons(polygons, polygon_json_path)
                print(f"Polygon zones saved: {len(polygons)} zones")
            else:
                print("No polygons drawn. Continuing without zone constraints.")
        except Exception as e:
            print(f"Error during polygon drawing: {e}")
            sys.exit(1)
    
    # =========================================================================
    # STEP 2: TRAJECTORY COLLECTION (YOLO + TRACKING)
    # =========================================================================
    print("\n[STEP 2/5] Running YOLO Detection & Tracking...")
    print("-" * 70)
    
    try:
        trajectory_collector.process_video(
            video_path=video_path, 
            output_dir=output_dir, 
            fps_override=TRUE_FPS,
            polygon_json_path=polygon_json_path
        )
        print("Trajectory collection completed")
    except Exception as e:
        print(f"Error during native trajectory collection: {e}")
        sys.exit(1)
        
    # =========================================================================
    # STEP 3: NATIVE PARKED VEHICLE EXTRACTION
    # =========================================================================
    print("\n[STEP 3/5] Extracting Parked Vehicle Segments...")
    print("-" * 70)
    
    trajectory_csv = get_latest_trajectory_csv(output_dir)
    if not trajectory_csv:
        print("Error: No trajectory CSV found. Skipping parked vehicle extraction.")
        sys.exit(1)
    
    try:
        print(f"Reading data natively from: {trajectory_csv}")
        df = pd.read_csv(trajectory_csv)
        FPS = TRUE_FPS
        
        parked_summary_records = []
        all_parked_row_dfs = []
        all_tracks = df.groupby('tracker_id')

        for tracker_id, history in all_tracks:
            stops, squelched_blocks = extract_parked_vehicles.extract_stationary_segments(history, fps=FPS, min_stop_duration_sec=3.0)
            if stops:
                vehicle_class = history['vehicle_class'].iloc[0]
                for idx, stop in enumerate(stops):
                    lower_timestamp = round(stop['start_frame'] / FPS, 2)
                    upper_timestamp = round(stop['end_frame'] / FPS, 2)
                    
                    parked_summary_records.append({
                        "tracker_id": tracker_id,
                        "vehicle_class": vehicle_class,
                        "stop_index": idx + 1,
                        "median_position_x": stop['median_position_x'],
                        "median_position_y": stop['median_position_y'],
                        "spawn_frame": stop['start_frame'],
                        "death_frame": stop['end_frame'],
                        "duration_sec": stop['duration_sec'],
                        "lower_timestamp": lower_timestamp,
                        "upper_timestamp": upper_timestamp
                    })
                all_parked_row_dfs.extend(squelched_blocks)

        if parked_summary_records:
            base_name = os.path.splitext(os.path.basename(trajectory_csv))[0]
            
            summary_df = pd.DataFrame(parked_summary_records)
            summary_csv_output = os.path.join(output_dir, f"summary_parked_{base_name}.csv")
            summary_df.to_csv(summary_csv_output, index=False)
            
            detailed_df = pd.concat(all_parked_row_dfs, ignore_index=True)
            detailed_csv_output = os.path.join(output_dir, f"timeline_parked_{base_name}.csv")
            detailed_df.to_csv(detailed_csv_output, index=False)
            
            print(f"Parked vehicle extraction completed. Isolated {len(summary_df)} stop(s).")
        else:
            print("[INFO] No verified stop events discovered.")
    except Exception as e:
        print(f"Error during parked vehicle extraction: {e}")
        sys.exit(1)

    # =========================================================================
    # STEP 4: NATIVE VIOLATION RULE ENGINE (UPGRADED WITH COMPLIANCE)
    # =========================================================================
    print("\n[STEP 4/5] Running Traffic Violation Engine...")
    print("-" * 70)
    
    summary_parked_csv = get_latest_summary_parked_csv(output_dir)
    trajectory_csv = get_latest_trajectory_csv(output_dir)

    parking_citations = None
    wrong_side_citations = None
    compliance_citations = None
    # Initialize micro models for motorcycle safety compliance auditing
    person_model = YOLO("weights/yolo11m.pt")
    if(bike_compliance):
        helmet_model = YOLO("weights/helmet.pt")

    if summary_parked_csv:
        try:
            parking_citations = violator.detect_parking_violations(summary_parked_csv, polygon_json_path, output_dir)
        except Exception as e:
            print(f"Error during parking violation validation: {e}")

    if trajectory_csv:
        try:
            wrong_side_citations = violator.detect_wrong_side_violations(trajectory_csv, polygon_json_path, output_dir)
        except Exception as e:
            print(f"Error during wrong-side motion analysis: {e}")
            sys.exit(1)
        if(bike_compliance):
            try:
                # Run the new micro compliance video processor layer (Set to 2 for double riding check tests)
                compliance_citations = violator.detect_video_motorcycle_compliance(trajectory_csv_path=trajectory_csv,
                    video_path=video_path,
                    person_model=person_model,
                    helmet_model=helmet_model,
                    output_dir=output_dir,
                    rider_threshold=2
                )
                print("Violation processing completed")
            except Exception as e:
                print(f"Error during motorcycle compliance tracking: {e}")
                sys.exit(1)
    
    # =========================================================================
    # STEP 5: EVIDENCE HARVESTING LOOP
    # =========================================================================
    print("\n[STEP 5/5] Launching Evidence Harvester...")
    print("-" * 70)
    base_name = os.path.splitext(os.path.basename(trajectory_csv))[0]
    
    timeline_parked_path = os.path.join(output_dir, f"timeline_parked_{base_name}.csv")
    timeline_wrong_side_path = os.path.join(output_dir, f"timeline_wrong_side_{base_name}.csv")
    
    if parking_citations and os.path.exists(parking_citations):
        try:
            print(f"   -> Routing Parking Timeline Context: {timeline_parked_path}")
            evidence_harvester.harvest_violation_patches(video_path, parking_citations, timeline_parked_path, output_dir)
        except Exception as e:
            print(f"⚠️ Warning: Parking evidence harvest failed: {e}")

    if wrong_side_citations and os.path.exists(wrong_side_citations):
        try:
            print(f"   -> Routing Wrong-Side Timeline Context: {timeline_wrong_side_path}")
            evidence_harvester.harvest_violation_patches(video_path, wrong_side_citations, timeline_wrong_side_path, output_dir)
        except Exception as e:
            print(f"⚠️ Warning: Wrong-side evidence harvest failed: {e}")

    if bike_compliance and compliance_citations and os.path.exists(compliance_citations):
        try:
            print(f"   -> Routing Motorcycle Compliance Context (Static Crop Route)...")
            # Since compliance logs spatial bounding boxes directly inside the output file,
            # passing None as the timeline tracking CSV drops the harvester straight into static mode!
            evidence_harvester.harvest_violation_patches(video_path, compliance_citations, None, output_dir)
        except Exception as e:
            print(f"⚠️ Warning: Motorcycle compliance evidence harvest failed: {e}")

    print("\n" + "="*70)
    print("✅ VIDEO PIPELINE RUN COMPLETE!")
    print("="*70)
