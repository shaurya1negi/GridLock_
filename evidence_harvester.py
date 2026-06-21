"""
evidence_harvester.py
---------------------
Upgraded Evidence Harvesting Layer with Automated Frame Quality Grading and Companion Metadata Sidecars.
"""

import os
import cv2
import pandas as pd

def harvest_violation_patches(video_path, violation_csv_path, trajectory_csv_path, output_dir, pad_percent=0.15):
    """
    Analyzes full tracking timelines to automatically isolate and harvest the 
    highest-resolution, least-blurry frame patch for violating vehicle IDs, 
    and generates a matching metadata citation file.
    """
    print(f"\nInitializing Intelligent Evidence Harvesting with Sidecars for: {os.path.basename(violation_csv_path)}")
    
    # 1. --- DYNAMIC PATH RESOLUTION ---
    if "wrong_side" in violation_csv_path:
        timeline_csv_path = violation_csv_path.replace("violations_wrong_side_", "timeline_wrong_side_")
    else:
        timeline_csv_path = violation_csv_path.replace("violations_parking_", "timeline_parked_")

    if not os.path.exists(violation_csv_path) or not os.path.exists(timeline_csv_path):
        print(f"Required data timelines missing. Looked for:\n   -> {violation_csv_path}\n   -> {timeline_csv_path}\nSkipping intelligent harvest.")
        return

    df_violations = pd.read_csv(violation_csv_path)
    df_timeline = pd.read_csv(timeline_csv_path)
    
    if df_violations.empty:
        print("ℹ Violation log is empty. No evidence patches to extract.")
        return

    evidence_dir = os.path.join(output_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 2. Iterate over unique violating vehicle IDs
    for tid, group in df_violations.groupby('tracker_id'):
        
        # Isolate the vehicle's entire tracked lifecycle history inside the specific violation window
        is_parking = "stop_start_frame" in group.columns
        if is_parking:
            sf = int(group['stop_start_frame'].iloc[0])
            df_frame = int(group['stop_end_frame'].iloc[0])
            
            vehicle_history = df_timeline[
                (df_timeline['tracker_id'] == tid) & 
                (df_timeline['frame_index'] >= sf) & 
                (df_timeline['frame_index'] <= df_frame)
            ].copy()
        else:
            vehicle_history = df_timeline[df_timeline['tracker_id'] == tid].copy()

        if vehicle_history.empty:
            print(f"  Timeline history window empty for Tracker ID {tid}. Skipping.")
            continue

        # 3. --- THE AUTOMATED QUALITY GRADING ENGINE ---
        vehicle_history['box_area'] = (vehicle_history['x2'] - vehicle_history['x1']) * \
                                      (vehicle_history['y2'] - vehicle_history['y1'])
        
        if len(vehicle_history) > 10:
            trim_size = int(len(vehicle_history) * 0.1)
            stable_zone = vehicle_history.iloc[trim_size : -trim_size]
        else:
            stable_zone = vehicle_history

        best_entry = stable_zone.loc[stable_zone['box_area'].idxmax()]
        
        target_frame = int(best_entry['frame_index'])
        x1 = best_entry['x1']
        y1 = best_entry['y1']
        x2 = best_entry['x2']
        y2 = best_entry['y2']
        v_class = best_entry['vehicle_class']

        # Calculate bounding box dimensions
        w = x2 - x1
        h = y2 - y1

        # --- THE EXPANDED PADDING LAYER ---
        x1_padded = max(0, int(x1 - (pad_percent * w)))
        y1_padded = max(0, int(y1 - (pad_percent * h)))
        x2_padded = min(img_w, int(x2 + (pad_percent * w)))
        y2_padded = min(img_h, int(y2 + (pad_percent * h)))

        # 4. Fast video frame seek mechanics
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if ret:
            # Crop the expanded patch layout array matrix
            vehicle_patch = frame[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Save visual artifact down to disk
            violation_type = group['violation_type'].iloc[0].replace(" ", "_").lower()
            violation_type = violation_type.split("_(")[0] 
            
            base_filename = f"best_shot_id_{tid}_{v_class}_{violation_type}_frame_{target_frame}"
            
            # Export A: Save Image Patch
            patch_path = os.path.join(evidence_dir, f"{base_filename}.jpg")
            cv2.imwrite(patch_path, vehicle_patch)
            
            # Export B: --- NEW: GENERATE CITATION METADATA SIDECAR ---
            metadata_path = os.path.join(evidence_dir, f"{base_filename}.txt")
            with open(metadata_path, 'w') as meta_file:
                meta_file.write("==================================================\n")
                meta_file.write("          AUTOMATED TRAFFIC CITATION TICKET       \n")
                meta_file.write("==================================================\n")
                meta_file.write(f"VIOLATION TYPE  : {group['violation_type'].iloc[0].upper()}\n")
                meta_file.write(f"TRACKER ID      : {tid}\n")
                meta_file.write(f"VEHICLE CLASS   : {v_class.upper()}\n")
                meta_file.write("--------------------------------------------------\n")
                meta_file.write(f"EVIDENCE FRAME  : {target_frame}\n")
                meta_file.write(f"EVIDENCE TIME   : {round(best_entry['timestamp_sec'], 2)} seconds from video start\n")
                meta_file.write(f"RESOLUTION AREA : {int(best_entry['box_area'])} pixels\n")
                
                # Contextual Conditional Writing Block based on violation type constraints
                if is_parking:
                    meta_file.write("--------------------------------------------------\n")
                    meta_file.write("PARKING LOG METRICS:\n")
                    meta_file.write(f"  • Entry Timestamp: {group['lower_timestamp'].iloc[0]}s (Frame {sf})\n")
                    meta_file.write(f"  • Exit Timestamp : {group['upper_timestamp'].iloc[0]}s (Frame {df_frame})\n")
                    meta_file.write(f"  • Total Duration : {group['duration_sec'].iloc[0]} seconds\n")
                    meta_file.write(f"  • Target Zone ID : Prohibited Zone #{group['prohibited_zone_id'].iloc[0]}\n")
                else:
                    meta_file.write("--------------------------------------------------\n")
                    meta_file.write("VECTOR MOTION METRICS:\n")
                    meta_file.write(f"  • Vehicle Direction: {group['car_heading_deg'].iloc[0]}°\n")
                    meta_file.write(f"  • Allowed Direction: {group['legal_heading_deg'].iloc[0]}°\n")
                    meta_file.write(f"  • Deviation Angle  : {group['angular_deviation'].iloc[0]}° against legal lane flow\n")
                    meta_file.write(f"  • Target Zone ID   : Road Lane #{group['road_zone_id'].iloc[0]}\n")
                    
                meta_file.write("==================================================\n")
                
    print(f"evidence generated at: {evidence_dir}")

    cap.release()