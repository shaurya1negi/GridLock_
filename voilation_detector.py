"""
violator.py
-----------
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
    What this function does:
    1. Loads drawn illegal parking polygon boundaries from a detailed JSON file.
    2. Filters zones to process ONLY those flagged as 'illegal_parking'.
    3. Reads the high-level summary of vehicle stop events from a CSV.
    4. Cross-references each vehicle's timeline to calculate stabilized, 
       noise-filtered median bounding box dimensions (x1, y1, x2, y2).
    5. Generates a 5-point strategic layout matrix across the vehicle's body:
       - Bottom Center (where wheels contact the road surface)
       - Top Center (inset slightly below the roof line)
       - Left Center (inset slightly from the left edge)
       - Right Center (inset slightly from the right edge)
       - Center Mass (positioned slightly lower than the bounding box midpoint)
    6. Applies a MAJORITY CONSENSUS VOTE: A vehicle is only flagged for an 
       illegal parking violation if AT LEAST 3 OUT OF THE 5 points land 
       completely inside the prohibited polygon zone. 
    ===========================================================================
    """
    print(f"\nRunning Multi-Point Parking Consensus Analysis on: {summary_csv_path}")
    
    # 1. Load the detailed zone dictionary configurations
    zones_data = polygon.load_polygons(polygon_json_path)
    if not zones_data:
        print("No prohibited parking zones found in the configuration JSON. Skipping analysis.")
        return None

    # Filter and extract ONLY illegal parking zones from the detailed layout array
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
    
    # Locate the main tracking timeline file to find bounding box boundaries (x1, y1, x2, y2)
    timeline_csv_path = summary_csv_path.replace("summary_parked_", "timeline_parked_")
    if not os.path.exists(timeline_csv_path):
        print(f"Error: Cannot run consensus, matching timeline file missing at: {timeline_csv_path}")
        return None
    
    df_timeline = pd.read_csv(timeline_csv_path)
    violation_records = []

    # 2. Iterate over each localized stop segment
    for _, row in df_stops.iterrows():
        tid = int(row['tracker_id'])
        sf = int(row['spawn_frame'])
        df_frame = int(row['death_frame'])
        
        # Isolate the exact tracking row intervals for this vehicle's specific stop window
        track_interval = df_timeline[
            (df_timeline['tracker_id'] == tid) & 
            (df_timeline['frame_index'] >= sf) & 
            (df_timeline['frame_index'] <= df_frame)
        ]
        
        if track_interval.empty:
            continue
            
        # Compute the median spatial bounding box bounds across the stop duration to clear out jitter
        x1 = track_interval['x1'].median()
        y1 = track_interval['y1'].median()
        x2 = track_interval['x2'].median()
        y2 = track_interval['y2'].median()
        
        w = x2 - x1
        h = y2 - y1
        
        # --- GENERATE THE 5-POINT CONSENSUS MATRIX ---
        points_to_test = {
            "Bottom Center": Point((x1 + x2) / 2.0, y2),
            "Top Center (Inset)": Point((x1 + x2) / 2.0, y1 + (0.05 * h)),
            "Left Center (Inset)": Point(x1 + (0.05 * w), (y1 + y2) / 2.0),
            "Right Center (Inset)": Point(x2 - (0.05 * w), (y1 + y2) / 2.0),
            "Center Mass (Lower)": Point((x1 + x2) / 2.0, ((y1 + y2) / 2.0) + (0.05 * h))
        }
        
        # Check points against every validated prohibited parking zone polygon
        for db_zone_id, zone_poly in parking_zones:
            points_inside_count = sum(1 for p_name, p_coord in points_to_test.items() if zone_poly.contains(p_coord))
            
            # THE CONSENSUS TRIGGER: Violating only if a majority (>=3) of points are inside
            if points_inside_count >= 2:
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

    # 3. Compile outcomes to disk
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
    
    for _, viol in df_violations.iterrows():
        print(f"   • Vehicle ID {viol['tracker_id']} ({viol['vehicle_class']}) "
              f"Violated Zone #{viol['prohibited_zone_id']} [Confidence: {viol['points_inside_zone']} points inside] "
              f"from {viol['lower_timestamp']}s to {viol['upper_timestamp']}s")
              
    return output_csv_path


def detect_wrong_side_violations(trajectory_csv_path, zone_json_path, output_dir, angle_tolerance_deg=135.0):
    """
    ===========================================================================
    VECTOR ALIGNMENT TRAFFIC DIRECTION CHECKER (WITH TIMELINE EXPORT)
    ===========================================================================
    What this function does:
    1. Loads road boundary polygons and legal traffic vectors from JSON.
    2. Parses the master continuous vehicle tracking log.
    3. Calculates the angular difference between a moving vehicle's heading 
       and the lane's legal flow vector.
    4. Flags infractions where the vehicle moves against traffic (deviation >= 135°).
    5. NEW: Extracts and saves the FULL continuous history rows of the offending 
       vehicle while it was inside the wrong-side zone into a timeline CSV.
       
    Why this matters:
    Exporting the continuous wrong-side timeline allows our Evidence Harvester 
    to analyze every frame of the offense and select the absolute best, 
    highest-resolution, and least-blurry snapshot for the final citation.
    ===========================================================================
    """
    print(f"\n🕵️Running Vector Alignment Analysis on: {trajectory_csv_path}")
    
    # 1. Load the customized road profiles
    zones_data = polygon.load_polygons(zone_json_path)
    if not zones_data:
        print("No road configuration maps found. Skipping wrong side checks.")
        return None

    df_tracks = pd.read_csv(trajectory_csv_path)
    violation_records = []
    all_wrong_side_timeline_rows = []
    flagged_violations = set() 

    # 2. Extract road bounds safely filtering out plain parking zones
    for zone_info in zones_data:
        if zone_info.get("zone_type") != "road_lane":
            continue # Skip plain parking shapes completely
            
        poly_points = zone_info["polygon"]
        road_poly = Polygon(poly_points)
        zone_idx = zone_info["zone_id"]
        
        # Calculate the direction angle of the legal vector arrow safely
        vec = zone_info["legal_vector"]
        if not vec:
            continue
            
        vx_legal = vec["end"][0] - vec["start"][0]
        vy_legal = vec["end"][1] - vec["start"][1]
        legal_heading = (math.degrees(math.atan2(vy_legal, vx_legal)) + 360) % 360
        
        # 3. Filter data points to find vehicles currently inside this road polygon
        for idx, row in df_tracks.iterrows():
            if row['is_stationary'] == 1:
                continue # Ignore stopped cars
                
            car_point = Point(row['bottom_center_x'], row['bottom_center_y'])
            
            if road_poly.contains(car_point):
                tid = int(row['tracker_id'])
                fid = int(row['frame_index'])
                
                car_heading = row['heading_deg']
                
                # --- MATHEMATICAL VECTOR ALIGNMENT CHECK ---
                angular_diff = abs(car_heading - legal_heading)
                if angular_diff > 180:
                    angular_diff = 360 - angular_diff
                
                # Violation validation threshold logic execution
                if angular_diff >= angle_tolerance_deg:
                    # Capture EVERY frame row where the vehicle is breaking the law for the timeline
                    all_wrong_side_timeline_rows.append(row)
                    
                    # Only create ONE summary log entry per vehicle per zone session
                    if (tid, zone_idx) not in flagged_violations:
                        flagged_violations.add((tid, zone_idx))
                        violation_records.append({
                            "tracker_id": tid,
                            "vehicle_class": row['vehicle_class'],
                            "violation_type": "Wrong-Side Driving",
                            "road_zone_id": zone_idx,
                            "frame_index": fid, # Trigger frame
                            "timestamp_sec": row['timestamp_sec'],
                            "car_heading_deg": round(car_heading, 1),
                            "legal_heading_deg": round(legal_heading, 1),
                            "angular_deviation": round(angular_diff, 1)
                        })

    if not violation_records:
        print("Vector field clear! No wrong-side driving detected.")
        return None

    base_name = os.path.splitext(os.path.basename(trajectory_csv_path))[0]
    
    # Output A: Save the Summary File
    df_wrong_side = pd.DataFrame(violation_records)
    output_path = os.path.join(output_dir, f"violations_wrong_side_{base_name}.csv")
    df_wrong_side.to_csv(output_path, index=False)
    
    # Output B: Save the Timeline File matching the parking module's structural footprint!
    df_wrong_side_timeline = pd.DataFrame(all_wrong_side_timeline_rows)
    timeline_output_path = os.path.join(output_dir, f"timeline_wrong_side_{base_name}.csv")
    df_wrong_side_timeline.to_csv(timeline_output_path, index=False)
    
    print(f"SUCCESS! Detected {len(df_wrong_side)} wrong-side driving infractions.")
    print(f" -> Citation Summary written to: {output_path}")
    print(f" -> Continuous Timeline written to: {timeline_output_path}")
    return output_path

'''
Our system doesn't just crop a random frame when a vehicle violates a rule. The Evidence Harvesting Layer programmatically grades 
the entire lifecycle of the voilating vehicle's trajectory. It skips noisy entry/exit frames and isolates the frame with the maximum bounding 
box pixel area, automatically capturing the closest, sharpest, and least blurry image of the vehicle for optimal license plate visibility
'''

def main():
    parser = argparse.ArgumentParser(description="Traffic Violation Engine Postprocessor")
    parser.add_argument("--summary_csv", required=True, help="Path to summary_parked_*.csv file")
    parser.add_argument("--polygon_json", required=True, help="Path to matching parking_zones_*.json file")
    parser.add_argument("--output_dir", default="output", help="Directory destination path.")
    args = parser.parse_args()

    detect_parking_violations(args.summary_csv, args.polygon_json, args.output_dir)


if __name__ == "__main__":
    main()