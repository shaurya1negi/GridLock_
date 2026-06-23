"""
evidence_harvester.py
---------------------
Upgraded Evidence Harvesting Layer with Polymorphic Video/Image Canvas Support.
Patched to dynamically parse target video frames from static mode violation rows.
"""

import os
import cv2
import pandas as pd

def harvest_violation_patches(video_path, violation_csv_path, trajectory_csv_path, output_dir, pad_percent=0.10):
    """
    Analyzes tracking timelines or static frame metadata to automatically isolate 
    and harvest expanded bounding boxes for violations, generating automated dockets.
    """
    print(f"\nInitializing Intelligent Evidence Harvesting with Sidecars for: {os.path.basename(violation_csv_path)}")
    
    if not os.path.exists(violation_csv_path):
        print(f"Required violation logs missing at: {violation_csv_path}. Skipping harvest.")
        return

    df_violations = pd.read_csv(violation_csv_path)
    if df_violations.empty:
        print("ℹ Violation log is empty. No evidence patches to extract.")
        return

    # Check if we are processing a static image or a video timeline
    is_static_image = trajectory_csv_path is None or not os.path.exists(trajectory_csv_path)
    if not is_static_image:
        df_timeline = pd.read_csv(trajectory_csv_path)

    evidence_dir = os.path.join(output_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    # Polymorphic file canvas acquisition
    is_video = video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    
    if is_video:
        cap = cv2.VideoCapture(video_path)
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        static_frame = cv2.imread(video_path)
        if static_frame is None:
            print(f"⚠️ Unable to load base canvas file: {video_path}")
            return
        img_h, img_w = static_frame.shape[:2]

    # Iterate over unique violating instances
    for idx, row in df_violations.iterrows():
        tid = int(row['tracker_id']) if 'tracker_id' in row else idx
        v_class = row['vehicle_class'] if 'vehicle_class' in row else 'motorcycle'
        
        # Clean violation string safely
        raw_viol_type = str(row['violation_type']).strip()
        violation_type_slug = raw_viol_type.replace(" ", "_").lower().split("_(")[0]
        
        # Determine coordinate metrics from timeline or flat row
        if is_static_image:
            # 🎯 FIX: Check if the row contains a logged video frame index instead of hardcoding 0!
            if 'frame_index' in row and is_video:
                target_frame = int(row['frame_index'])
                timestamp_sec = f"{row['timestamp_sec']} seconds"
            else:
                target_frame = 0
                timestamp_sec = "N/A (Static Image)"
                
            x1, y1, x2, y2 = float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])
            box_area = int((x2 - x1) * (y2 - y1))
        else:
            is_parking = "stop_start_frame" in df_violations.columns
            if is_parking:
                sf = int(row['stop_start_frame'])
                df_frame = int(row['stop_end_frame'])
                vehicle_history = df_timeline[(df_timeline['tracker_id'] == tid) & (df_timeline['frame_index'] >= sf) & (df_timeline['frame_index'] <= df_frame)].copy()
            else:
                vehicle_history = df_timeline[df_timeline['tracker_id'] == tid].copy()

            if vehicle_history.empty:
                continue

            vehicle_history['box_area'] = (vehicle_history['x2'] - vehicle_history['x1']) * (vehicle_history['y2'] - vehicle_history['y1'])
            best_entry = vehicle_history.loc[vehicle_history['box_area'].idxmax()]
            
            target_frame = int(best_entry['frame_index'])
            x1, y1, x2, y2 = float(best_entry['x1']), float(best_entry['y1']), float(best_entry['x2']), float(best_entry['y2'])
            timestamp_sec = f"{round(best_entry['timestamp_sec'], 2)} seconds"
            box_area = int(best_entry['box_area'])

        w, h = (x2 - x1), (y2 - y1)

        # Apply standardized padding boundaries
        if v_class.lower() in ['motorcycle', 'bicycle', 'bike']:
            current_pad_x = 0.55  
            current_pad_y1 = 0.45 
            current_pad_y2 = 0.05 
        else:
            current_pad_x = pad_percent
            current_pad_y1 = pad_percent
            current_pad_y2 = pad_percent

        x1_padded = max(0, int(x1 - (current_pad_x * w)))
        y1_padded = max(0, int(y1 - (current_pad_y1 * h)))
        x2_padded = min(img_w, int(x2 + (current_pad_x * w)))
        y2_padded = min(img_h, int(y2 + (current_pad_y2 * h)))

        # Frame capture safely from the precise matching timestamp index
        if is_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret: continue
        else:
            frame = static_frame.copy()

        vehicle_patch = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Explicit naming layout scheme
        source_mode = "image" if not is_video else f"frame_{target_frame}"
        base_filename = f"ticket_id_{tid}_{v_class}_{violation_type_slug}_{source_mode}"
        
        # Save image patch securely
        if vehicle_patch.size > 0:
            cv2.imwrite(os.path.join(evidence_dir, f"{base_filename}.jpg"), vehicle_patch)
        
        # Generate companion text ticket cleanly
        ticket_path = os.path.join(evidence_dir, f"{base_filename}.txt")
        with open(ticket_path, 'w', encoding='utf-8') as meta_file:
            meta_file.write("==================================================\n")
            meta_file.write("          AUTOMATED TRAFFIC CITATION TICKET       \n")
            meta_file.write("==================================================\n")
            meta_file.write(f"OFFENSE REPORT  : {raw_viol_type.upper()}\n")
            meta_file.write(f"VEHICLE INDEX   : ID #{tid}\n")
            meta_file.write(f"VEHICLE CLASS   : {v_class.upper()}\n")
            meta_file.write("--------------------------------------------------\n")
            meta_file.write(f"CAPTURE MODE    : {'STATIC PHOTOGRAPHIC SNAPSHOT' if not is_video else 'TEMPORAL VIDEO STREAM'}\n")
            meta_file.write(f"EVIDENCE TIME   : {timestamp_sec}\n")
            if is_video:
                meta_file.write(f"TARGET FRAME    : {target_frame}\n")
            meta_file.write(f"CROP RESOLUTION : {box_area} pixels\n")
            meta_file.write("--------------------------------------------------\n")
            
            if "riders_detected" in row:
                meta_file.write(f"  • Total Riders Onboard : {int(row['riders_detected'])}\n")
            if "helmet_infractions" in row:
                meta_file.write(f"  • Unhelmeted Heads Found: {int(row['helmet_infractions'])}\n")
            if "points_inside_zone" in row:
                meta_file.write(f"  • Parking Consensus Score: {row['points_inside_zone']}\n")
            if "angular_deviation" in row:
                meta_file.write(f"  • Deviation Heading Angle: {row['angular_deviation']}°\n")
                
            meta_file.write("==================================================\n")
                
    print(f"Evidence patches cleanly compiled at: {evidence_dir}")
    if is_video: cap.release()
