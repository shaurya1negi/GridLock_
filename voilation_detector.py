"""
voilation_detector.py
---------------------
Rule-Checking Engine using a Multi-Point Consensus Voting System and Direction Vectors.
"""
import os
import argparse
import math
import pandas as pd
from shapely.geometry import Point, Polygon
import polygon  
import cv2 

def detect_parking_violations(summary_csv_path, polygon_json_path, output_dir ):
    """
    ===========================================================================
    MULTI-POINT SPATIAL CONSENSUS DETECTION ENGINE
    ===========================================================================
    """
    print(f"\nRunning Multi-Point Parking Consensus Analysis on: {summary_csv_path}")
    
    zones_data = polygon.load_polygons(polygon_json_path)
    if not zones_data:
        print("No prohibited parking zones found in the configuration JSON. Skipping analysis.")
        return None

    parking_zones = []
    for zone in zones_data:
        if zone.get("zone_type") == "illegal_parking" and len(zone["polygon"]) >= 3:
            parking_zones.append((zone["zone_id"], Polygon(zone["polygon"])))

    if not parking_zones:
        print("No designated 'illegal_parking' zones configured to test.")
        return None

    if not os.path.exists(summary_csv_path):
        print(f"Error: Summary CSV not found at {summary_csv_path}")
        return None
        
    df_stops = pd.read_csv(summary_csv_path)
    
    timeline_csv_path = summary_csv_path.replace("summary_parked_", "timeline_parked_")
    if not os.path.exists(timeline_csv_path):
        print(f"Error: Cannot run consensus, matching timeline file missing at: {timeline_csv_path}")
        return None
    
    df_timeline = pd.read_csv(timeline_csv_path)
    violation_records = []

    for _, row in df_stops.iterrows():
        tid = int(row['tracker_id'])
        sf = int(row['spawn_frame'])
        df_frame = int(row['death_frame'])
        
        track_interval = df_timeline[
            (df_timeline['tracker_id'] == tid) & 
            (df_timeline['frame_index'] >= sf) & 
            (df_timeline['frame_index'] <= df_frame)
        ]
        
        if track_interval.empty:
            continue
            
        x1 = track_interval['x1'].median()
        y1 = track_interval['y1'].median()
        x2 = track_interval['x2'].median()
        y2 = track_interval['y2'].median()
        
        w = x2 - x1
        h = y2 - y1
        
        points_to_test = {
            "Bottom Center": Point((x1 + x2) / 2.0, y2),
            "Top Center (Inset)": Point((x1 + x2) / 2.0, y1 + (0.05 * h)),
            "Left Center (Inset)": Point(x1 + (0.05 * w), (y1 + y2) / 2.0),
            "Right Center (Inset)": Point(x2 - (0.05 * w), (y1 + y2) / 2.0),
            "Center Mass (Lower)": Point((x1 + x2) / 2.0, ((y1 + y2) / 2.0) + (0.05 * h))
        }
        
        for db_zone_id, zone_poly in parking_zones:
            points_inside_count = sum(1 for p_name, p_coord in points_to_test.items() if zone_poly.contains(p_coord))
            
            if points_inside_count >= 3:
                violation_records.append({
                    "tracker_id": tid,
                    "vehicle_class": row['vehicle_class'],
                    "violation_type": "Illegal Parking",
                    "prohibited_zone_id": db_zone_id,
                    "points_inside_zone": f"{points_inside_count}/5",
                    "stop_start_frame": sf,
                    "stop_end_frame": df_frame,
                    "duration_sec": row['duration_sec'],
                    "lower_timestamp": row['lower_timestamp'],
                    "upper_timestamp": row['upper_timestamp'],
                    "median_x": row['median_position_x'],
                    "median_y": row['median_position_y']
                })
                break 

    if not violation_records:
        print("Clean street check! No vehicles passed the 3/5 majority parking consensus vote.")
        return None

    df_violations = pd.DataFrame(violation_records)
    base_name = os.path.splitext(os.path.basename(summary_csv_path))[0]
    clean_base = base_name.replace("summary_parked_", "")
    
    output_csv_path = os.path.join(output_dir, f"violations_parking_{clean_base}.csv")
    df_violations.to_csv(output_csv_path, index=False)
    
    print(f"SUCCESS! Detected {len(df_violations)} consensus-verified parking incidents.")
    print(f" -> Parking Citations File written to: {output_csv_path}")
    return output_csv_path


def detect_wrong_side_violations(trajectory_csv_path, zone_json_path, output_dir, angle_tolerance_deg=135.0, violation_ratio_threshold=0.30):
    """
    ===========================================================================
    TRAJECTORY PROPORTIONAL VOTING ENGINE (PURE IMAGE-SPACE GEOMETRY)
    ===========================================================================
    """
    print(f"\n Running Trajectory Proportional Voting Analysis on: {trajectory_csv_path}")
    
    zones_data = polygon.load_polygons(zone_json_path)
    if not zones_data:
        print("No road configuration maps found. Skipping wrong side checks.")
        return None

    df_tracks = pd.read_csv(trajectory_csv_path)
    
    df_moving = df_tracks[(df_tracks['is_stationary'] == 0) & (df_tracks['heading_deg'] != -1.0)].copy()
    if df_moving.empty:
        print("Vector field clear! No moving vehicles found to evaluate.")
        return None

    violation_records = []
    all_wrong_side_timeline_rows = []

    for zone_info in zones_data:
        if zone_info.get("zone_type") != "road_lane":
            continue 
            
        poly_points = zone_info["polygon"]
        road_poly = Polygon(poly_points)
        zone_idx = zone_info["zone_id"]
        
        poly_minx, poly_miny, poly_maxx, poly_maxy = road_poly.bounds
        
        vec = zone_info["legal_vector"]
        if not vec:
            continue
            
        vx_legal = vec["end"][0] - vec["start"][0]
        vy_legal = vec["end"][1] - vec["start"][1]
        legal_heading = (math.degrees(math.atan2(vy_legal, vx_legal)) + 360) % 360

        df_zone_candidates = df_moving[
            (df_moving['bottom_center_x'] >= poly_minx) & 
            (df_moving['bottom_center_x'] <= poly_maxx) & 
            (df_moving['bottom_center_y'] >= poly_miny) & 
            (df_moving['bottom_center_y'] <= poly_maxy)
        ].copy()
        
        if df_zone_candidates.empty:
            continue

        df_zone_candidates['inside'] = df_zone_candidates.apply(
            lambda r: road_poly.contains(Point(r['bottom_center_x'], r['bottom_center_y'])), axis=1
        )
        df_inside = df_zone_candidates[df_zone_candidates['inside'] == True]
        
        if df_inside.empty:
            continue

        grouped_tracks = df_inside.groupby('tracker_id')
        
        for tid, track_history in grouped_tracks:
            if len(track_history) < 3:
                continue
                
            track_sorted = track_history.sort_values('frame_index')
            violating_frames_rows = []
            
            for _, row in track_sorted.iterrows():
                car_heading = row['heading_deg']
                
                angular_diff = abs(car_heading - legal_heading)
                if angular_diff > 180:
                    angular_diff = 360 - angular_diff
                    
                if angular_diff >= angle_tolerance_deg:
                    violating_frames_rows.append(row)
            
            total_frames_inside = len(track_sorted)
            total_violating_frames = len(violating_frames_rows)
            violation_ratio = total_violating_frames / total_frames_inside
            
            if violation_ratio >= violation_ratio_threshold:
                # 🌟 FIX: Append the raw row series objects directly to the master collection list
                all_wrong_side_timeline_rows.extend(violating_frames_rows)
                trigger_row = violating_frames_rows[total_violating_frames // 2]
                
                angle_deg = abs(track_sorted['heading_deg'].median() - legal_heading)
                if angle_deg > 180:
                    angle_deg = 360 - angle_deg
                
                violation_records.append({
                    "tracker_id": tid,
                    "vehicle_class": trigger_row['vehicle_class'],
                    "violation_type": "Wrong-Side Driving",
                    "road_zone_id": zone_idx,
                    "frame_index": int(trigger_row['frame_index']),
                    "timestamp_sec": trigger_row['timestamp_sec'],
                    "frames_inside_zone": total_frames_inside,
                    "frames_violating": total_violating_frames,
                    "violation_ratio": round(violation_ratio, 2),
                    "car_heading_deg": round(track_sorted['heading_deg'].median(), 1),
                    "legal_heading_deg": round(legal_heading, 1),
                    "angular_deviation": round(angle_deg, 1)
                })

    if not violation_records:
        print("Vector field clear! Proportional voting found zero wrong-side violations.")
        return None

    base_name = os.path.splitext(os.path.basename(trajectory_csv_path))[0]
    
    df_wrong_side = pd.DataFrame(violation_records)
    output_path = os.path.join(output_dir, f"violations_wrong_side_{base_name}.csv")
    df_wrong_side.to_csv(output_path, index=False)
    
    # 🌟 FIX: Instantiating directly from a flat list of row Series provides clean, structured data matrix parsing
    df_wrong_side_timeline = pd.DataFrame(all_wrong_side_timeline_rows)
    timeline_output_path = os.path.join(output_dir, f"timeline_wrong_side_{base_name}.csv")
    df_wrong_side_timeline.to_csv(timeline_output_path, index=False)
    
    print(f"SUCCESS! Detected {len(df_wrong_side)} trajectory-verified wrong-side violations.")
    print(f" -> Citation Summary written to: {output_path}")
    print(f" -> Continuous Timeline written to: {timeline_output_path}")
    return output_path


def process_motorcycle_compliance(frame, x1, y1, x2, y2, tracker_id, person_model, helmet_model, rider_threshold=2):
    """
    Core Compliance Module (Consolidated Dual-Model Edition): Extracts cleanly padded 
    motorcycle patches, evaluates rider counts and safety gear, and logs a single 
    consolidated citation to eliminate duplicate evidence harvesting.
    """
    img_h, img_w = frame.shape[:2]
    w, h = (x2 - x1), (y2 - y1)
    
    # 🎯 TUNED ASYMMETRICAL PADDING: Large enough for helmets, tight enough to exclude roadside noise
    x1_pad = max(0, int(x1 - (0.10 * w)))
    y1_pad = max(0, int(y1 - (0.15 * h)))  # Pulled down from 0.85 to isolate just the riders' head space
    x2_pad = min(img_w, int(x2 + (0.10 * w)))
    y2_pad = min(img_h, int(y2 + (0.10 * h)))
    
    cropped_patch = frame[y1_pad:y2_pad, x1_pad:x2_pad]
    if cropped_patch.size == 0:
        return None
        
    # -----------------------------------------------------------------
    # MICRO PASS 1: Count riders using the dedicated Person Model
    # -----------------------------------------------------------------
    person_results = person_model(cropped_patch, conf=0.2, iou=0.45, classes=[0], verbose=False)
    detected_persons = person_results[0].boxes
    total_riders = len(detected_persons) if detected_persons is not None else 0
        
    # -----------------------------------------------------------------
    # MICRO PASS 2: Detect helmet compliance using the Helmet Model
    # -----------------------------------------------------------------
    helmet_results = helmet_model(cropped_patch, conf=0.15, iou=0.80, verbose=False)
    detected_helmets = helmet_results[0].boxes
    helmet_violations = 0
    
    if detected_helmets is not None:
        for box in detected_helmets:
            if int(box.cls[0]) == 1:  # Index 1: without_helmet
                helmet_violations += 1
                
    # -----------------------------------------------------------------
    # 🎯 CONSOLIDATION LAYER: Prevent Duplicate File Generation
    # -----------------------------------------------------------------
    violations = []
    is_overloaded = total_riders >= rider_threshold
    has_helmet_infraction = helmet_violations > 0
    
    if is_overloaded or has_helmet_infraction:
        # Determine a singular offense name for clean file slug creation
        if is_overloaded and has_helmet_infraction:
            v_type = "Overloading and Helmet Violation"
        elif is_overloaded:
            v_type = "Multi-Passenger Overloading"
        else:
            v_type = "Helmet Violation"
            
        violations.append({
            "tracker_id": tracker_id,
            "vehicle_class": "motorcycle",
            "violation_type": v_type,
            "riders_detected": total_riders,
            "helmet_infractions": helmet_violations,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })
        
    return violations


#use audit_static_image_compliance for video_engine Motorcycle Compliance

def detect_video_motorcycle_compliance(trajectory_csv_path, video_path, person_model, helmet_model, output_dir, rider_threshold=2):
    """
    Spatiotemporal Video Evaluation: Scans full trajectory timelines, runs dual-model 
    compliance checking on maximum area motorcycle frames, and exports structured citation lists.
    Includes a 50-frame lifespan filter to reject split-second tracking glitches.
    """
    print(f"\n Running Video Motorcycle Compliance Audits on: {trajectory_csv_path}")
    if not os.path.exists(trajectory_csv_path):
        print(f"⚠️ Target trajectory track log missing at: {trajectory_csv_path}")
        return None
        
    df_tracks = pd.read_csv(trajectory_csv_path)
    
    # Filter timeline tracking rows strictly for motorcycles (Class ID 3)
    df_bikes = df_tracks[df_tracks['class_id'] == 3].copy()
    if df_bikes.empty:
        print("🏍️ No motorcycle trajectories discovered in video track log.")
        return None
        
    violation_records = []
    cap = cv2.VideoCapture(video_path)
    
    # Analyze the optimal high-res frame slice for each unique tracked motorcycle instance
    for tid, track_history in df_bikes.groupby('tracker_id'):
        
        # 🎯 TEMPORAL SQUELCH FILTER: Reject ghost tracking that exists for less than 50 frames
        if len(track_history) < 100:
            print(f"⏭️ Squelching Track ID {tid}: Existed for only {len(track_history)} frames (likely a glitch).")
            continue
            
        track_history['box_area'] = (track_history['x2'] - track_history['x1']) * (track_history['y2'] - track_history['y1'])
        best_row = track_history.loc[track_history['box_area'].idxmax()]
        
        target_frame = int(best_row['frame_index'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if ret:
            # Execute the dual-model compliance micro checks inside the padded patch frame matrix
            res = process_motorcycle_compliance(
                frame=frame,
                x1=best_row['x1'], y1=best_row['y1'], x2=best_row['x2'], y2=best_row['y2'],
                tracker_id=tid,
                person_model=person_model,
                helmet_model=helmet_model,
                rider_threshold=rider_threshold
            )
            
            if res:
                for viol in res:
                    violation_records.append({
                        "tracker_id": tid,
                        "vehicle_class": "motorcycle",
                        "violation_type": viol["violation_type"],
                        "frame_index": target_frame,
                        "timestamp_sec": best_row['timestamp_sec'],
                        "riders_detected": viol["riders_detected"],
                        "helmet_infractions": viol["helmet_infractions"],
                        "x1": viol["x1"], "y1": viol["y1"], "x2": viol["x2"], "y2": viol["y2"]
                    })

    cap.release()
    
    if not violation_records:
        print("✅ Safety check clear: All tracked video riders verified compliant.")
        return None
        
    base_name = os.path.splitext(os.path.basename(trajectory_csv_path))[0]
    df_viol = pd.DataFrame(violation_records)
    output_csv = os.path.join(output_dir, f"violations_compliance_{base_name}.csv")
    df_viol.to_csv(output_csv, index=False)
    
    print(f"SUCCESS! Logged {len(df_viol)} safety incidents across video stream tracks.")
    print(f" -> Compliance Citations written to: {output_csv}")
    return output_csv

#Motorcycle Compliance includes triple riding and no helmet
#use process_motorcycle_compliance for video_engine Motorcycle Compliance

def audit_static_image_compliance(image_path, macro_model, person_model, helmet_model, output_dir):
    """
    Executes the macro-to-micro dual compliance inspection loop over a static photographic asset.
    """
    frame = cv2.imread(image_path)
    if frame is None: 
        print(f"⚠️ Unable to open image frame for compliance audit: {image_path}")
        return None
    
    # Run Layer 1: Detect motorcycles across the entire scene canvas
    macro_results = macro_model(frame, conf=0.25, iou=0.3, verbose=False)
    all_violations = []
    
    for idx, box in enumerate(macro_results[0].boxes):
        class_id = int(box.cls[0])
        
        # Look specifically for the 'motorcycle' index (Class 3 in COCO)
        if class_id == 3: 
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            
            # Run Layer 2: Audit both micro compliance detectors inside this crop region
            res = process_motorcycle_compliance(
                frame=frame, 
                x1=x1, y1=y1, x2=x2, y2=y2, 
                tracker_id=idx + 1, 
                person_model=person_model, 
                helmet_model=helmet_model, 
                rider_threshold=2 # Testing threshold for double-riding validation
            )
            if res:
                all_violations.extend(res)
                
    if all_violations:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        df_viol = pd.DataFrame(all_violations)
        output_csv = os.path.join(output_dir, f"violations_compliance_{base_name}.csv")
        df_viol.to_csv(output_csv, index=False)
        return output_csv
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Traffic Violation Engine Postprocessor")
    parser.add_argument("--summary_csv", required=True, help="Path to summary_parked_*.csv file")
    parser.add_argument("--polygon_json", required=True, help="Path to matching parking_zones_*.json file")
    parser.add_argument("--output_dir", default="output", help="Directory destination path.")
    args = parser.parse_args()

    detect_parking_violations(args.summary_csv, args.polygon_json, args.output_dir)

if __name__ == "__main__":
    main()