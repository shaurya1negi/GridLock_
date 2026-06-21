"""
voilation_detector.py
---------------------
Upgraded Rule-Checking Engine using a Multi-Point Consensus Voting System and Direction Vectors.
"""
import os
import argparse
import math
import pandas as pd
from shapely.geometry import Point, Polygon
import polygon  

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


def detect_wrong_side_violations(trajectory_csv_path, zone_json_path, output_dir, angle_tolerance_deg=135.0, violation_ratio_threshold=0.50):
    """
    ===========================================================================
    TRAJECTORY PROPORTIONAL VOTING ENGINE (PURE IMAGE-SPACE GEOMETRY)
    ===========================================================================
    """
    print(f"\n🕵️ Running Trajectory Proportional Voting Analysis on: {trajectory_csv_path}")
    
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

def main():
    parser = argparse.ArgumentParser(description="Traffic Violation Engine Postprocessor")
    parser.add_argument("--summary_csv", required=True, help="Path to summary_parked_*.csv file")
    parser.add_argument("--polygon_json", required=True, help="Path to matching parking_zones_*.json file")
    parser.add_argument("--output_dir", default="output", help="Directory destination path.")
    args = parser.parse_args()

    detect_parking_violations(args.summary_csv, args.polygon_json, args.output_dir)

if __name__ == "__main__":
    main()
