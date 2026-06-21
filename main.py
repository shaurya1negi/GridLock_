"""
main.py
-------
Orchestrator for the traffic violation detection pipeline.
"""

import argparse
import os
import sys
import cv2
import subprocess
import pandas as pd
from datetime import datetime
import numpy as np

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


def get_latest_video_file(output_dir, prefix):
    """Find the most recently created video file."""
    video_files = [f for f in os.listdir(output_dir) 
                   if f.startswith(prefix) and f.endswith(".mp4")]
    
    if not video_files:
        return None
    
    video_files.sort(key=lambda f: os.path.getctime(os.path.join(output_dir, f)), reverse=True)
    return os.path.join(output_dir, video_files[0])


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection Pipeline - Main Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--video", required=True, help="Path to the traffic video file")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--skip-polygon-drawing", action="store_true", 
                       help="Skip polygon drawing if already exist")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # =========================================================================
    # CORE TELEMETRY EXTRACTOR: DYNAMIC HARDWARE METADATA HOOK
    # =========================================================================
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not read video header stream: {args.video}")
        sys.exit(1)
        
    TRUE_FPS = cap.get(cv2.CAP_PROP_FPS)
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Fallback safety validation if video container headers are slightly corrupt
    if TRUE_FPS == 0 or pd.isna(TRUE_FPS): 
        TRUE_FPS = 30.0
        
    print(f"Video Container Metadata Synced Globally:")
    print(f"   • Hardware Frame Rate : {round(TRUE_FPS, 2)} FPS")
    print(f"   • Total Frame Count   : {TOTAL_FRAMES} frames")

    print("\n" + "="*70)
    print("🚀 TRAFFIC VIOLATION DETECTION PIPELINE")
    print("="*70)
    
    # =========================================================================
    # STEP 1: POLYGON DRAWING VIA MODULAR IMPORT
    # =========================================================================
    print("\n[STEP 1/5] Drawing Prohibited Parking Zones...")
    print("-" * 70)
    
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    polygon_json_path = os.path.join(args.output_dir, f"parking_zones_{run_stamp}.json")
    polygons = []
    
    if args.skip_polygon_drawing and os.path.exists(polygon_json_path):
        print(f"⏭Using existing polygons: {polygon_json_path}")
        polygons = polygon.load_polygons(polygon_json_path)
    else:
        try:
            frame = extract_first_frame(args.video)
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
 # FIXED: Calling the imported module function natively in-memory!
        trajectory_collector.process_video(
            video_path=args.video, 
            output_dir=args.output_dir, 
            fps_override=TRUE_FPS,
            polygon_json_path=polygon_json_path
        )
        print("Trajectory collection completed")
    
    except Exception as e:
        print(f"Error during native trajectory collection: {e}")
        sys.exit(1)
        
    
    except subprocess.CalledProcessError as e:
        print(f"Error during trajectory collection: {e}")
        sys.exit(1)
    
    # =========================================================================
    # STEP 3: NATIVE PARKED VEHICLE EXTRACTION (NO TERMINAL CALL)
    # =========================================================================
    print("\n[STEP 3/5] Extracting Parked Vehicle Segments...")
    print("-" * 70)
    
    trajectory_csv = get_latest_trajectory_csv(args.output_dir)
    if not trajectory_csv:
        print("Error: No trajectory CSV found. Skipping parked vehicle extraction.")
        sys.exit(1)
    
    try:
        print(f"Reading data natively from: {trajectory_csv}")
        df = pd.read_csv(trajectory_csv)
        
        # Use global ground-truth TRUE_FPS parameter
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
            
            # Save Summary CSV
            summary_df = pd.DataFrame(parked_summary_records)
            summary_csv_output = os.path.join(args.output_dir, f"summary_parked_{base_name}.csv")
            summary_df.to_csv(summary_csv_output, index=False)
            
            # Save Timeline CSV
            detailed_df = pd.concat(all_parked_row_dfs, ignore_index=True)
            detailed_csv_output = os.path.join(args.output_dir, f"timeline_parked_{base_name}.csv")
            detailed_df.to_csv(detailed_csv_output, index=False)
            
            print(f"Parked vehicle extraction completed. Isolated {len(summary_df)} stop(s).")
        else:
            print("[INFO] No verified stop events discovered.")
            
    except Exception as e:
        print(f"Error during parked vehicle extraction: {e}")
        sys.exit(1)

    # =========================================================================
    # STEP 4: NATIVE VIOLATION RULE ENGINE (NO TERMINAL CALL)
    # =========================================================================
    print("\n[STEP 4/5] Running Traffic Violation Engine...")
    print("-" * 70)
    
    summary_parked_csv = get_latest_summary_parked_csv(args.output_dir)
    trajectory_csv = get_latest_trajectory_csv(args.output_dir)

    parking_citations = None
    wrong_side_citations = None
    
    # Running Engine Check A: Illegal Parking Consensus Check
    if summary_parked_csv:
        try:
            # FIXED: Pass dynamic TRUE_FPS parameter keyword arg
            parking_citations = violator.detect_parking_violations(summary_parked_csv, polygon_json_path, args.output_dir)
        except Exception as e:
            print(f"Error during parking violation validation: {e}")

    # Running Engine Check B: Vector-Field Wrong Side Driving Check
    if trajectory_csv:
        try:
            # FIXED: Pass dynamic TRUE_FPS parameter keyword arg
            wrong_side_citations = violator.detect_wrong_side_violations(trajectory_csv, polygon_json_path, args.output_dir)
            print("Violation processing completed")
        except Exception as e:
            print(f"Error during wrong-side motion analysis: {e}")
            sys.exit(1)
    
    # =========================================================================
    # NEW STEP: EVIDENCE HARVESTING LOOP
    # =========================================================================
    
    base_name = os.path.splitext(os.path.basename(trajectory_csv))[0]
    
    # Isolate the explicit separate timeline asset strings
    timeline_parked_path = os.path.join(args.output_dir, f"timeline_parked_{base_name}.csv")
    timeline_wrong_side_path = os.path.join(args.output_dir, f"timeline_wrong_side_{base_name}.csv")
    
    # Process Illegal Parking Evidence
    if parking_citations and os.path.exists(parking_citations):
        try:
            print(f"   -> Routing Parking Timeline Context: {timeline_parked_path}")
            evidence_harvester.harvest_violation_patches(
                args.video, parking_citations, timeline_parked_path, args.output_dir
            )
        except Exception as e:
            print(f" Warning: Parking evidence harvest failed: {e}")

    # Process Wrong-Side Driving Evidence
    if wrong_side_citations and os.path.exists(wrong_side_citations):
        try:
            print(f"   -> Routing Wrong-Side Timeline Context: {timeline_wrong_side_path}")
            evidence_harvester.harvest_violation_patches(
                args.video, wrong_side_citations, timeline_wrong_side_path, args.output_dir
            )
        except Exception as e:
            print(f" Warning: Wrong-side evidence harvest failed: {e}")



if __name__ == "__main__":
    main()