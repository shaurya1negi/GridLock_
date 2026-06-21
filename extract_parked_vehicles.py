import argparse
import os
import pandas as pd
import numpy as np

def extract_stationary_segments(vehicle_history, fps=30, min_stop_duration_sec=3.0, speed_threshold=10.0):
    """
    Scans a single vehicle's timeline to find all individual intervals 
    where it came to a stop for a minimum duration, even if it moves later.
    """
    history = vehicle_history.sort_values('frame_index').copy()
    window_size = int(fps * min_stop_duration_sec)
    
    if len(history) < window_size:
        return [], []
        
    # --- MID-VIDEO STOP DETECTION VIA ROLLING WINDOW ---
    # Instead of averaging the vehicle's entire lifespan, a moving window analyzes 
    # localized velocity. This isolates temporary stops (e.g., at traffic lights or 
    # parking spots) even if the vehicle was moving before or after the interval.
    history['rolling_speed'] = history['speed_px_sec'].rolling(
        window=window_size, 
        min_periods=window_size, 
        center=True
    ).median()
    
    history['is_window_static'] = history['rolling_speed'] < speed_threshold
    static_blocks = (history['is_window_static'] != history['is_window_static'].shift()).cumsum()
    
    detected_stops = []
    squelched_frames_list = []
    
    for _, block in history.groupby(static_blocks):
        if block['is_window_static'].iloc[0]: 
            start_frame = int(block['frame_index'].iloc[0])
            end_frame = int(block['frame_index'].iloc[-1])
            duration_frames = end_frame - start_frame + 1
            
            # Temporal floor gate to reject glitches
            if duration_frames >= window_size:
                stop_x = np.median(block['bottom_center_x'])
                stop_y = np.median(block['bottom_center_y'])
                
                detected_stops.append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_sec": round(duration_frames / fps, 2),
                    "median_position_x": round(stop_x, 2),
                    "median_position_y": round(stop_y, 2)
                })
                
                # --- APPLY THE DEEP SQUELCH LAYER ---
                # Isolate only these stopped frames and flatten out their physics vectors
                block_copy = block.copy()
                block_copy['velocity_x_px_sec'] = 0.0
                block_copy['velocity_y_px_sec'] = 0.0
                block_copy['speed_px_sec'] = 0.0
                block_copy['heading_deg'] = -1.0
                block_copy['is_stationary'] = 1
                
                squelched_frames_list.append(block_copy)
            
    return detected_stops, squelched_frames_list

def main():
    parser = argparse.ArgumentParser(description="Parked Vehicle Trajectory Segregation Preprocessor")
    parser.add_argument("--csv", required=True, help="Path to the sorted ground-truth trajectory CSV file.")
    parser.add_argument("--output_dir", default="output", help="Directory to save the generated output files.")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: Target data file not found at {args.csv}")
        return

    df = pd.read_csv(args.csv)
    
    # Define FPS constant for calculation uniformity
    FPS = 30
    all_tracks = df.groupby('tracker_id')

    parked_summary_records = []
    all_parked_row_dfs = []

    print(f"--- Running Segmented Stop Exporter on {args.csv} ---")

    for tracker_id, history in all_tracks:
        stops, squelched_blocks = extract_stationary_segments(history, fps=FPS, min_stop_duration_sec=3.0)
        
        if stops:
            vehicle_class = history['vehicle_class'].iloc[0]
            print(f"\n🚗 Vehicle ID {tracker_id} ({vehicle_class}) registered {len(stops)} stop segment(s).")
            
            for idx, stop in enumerate(stops):
                # Calculate upper and lower timestamps based on the frames
                lower_timestamp = round(stop['start_frame'] / FPS, 2)
                upper_timestamp = round(stop['end_frame'] / FPS, 2)
                
                # Append high-level metadata matrix records with extra timestamp columns
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
            
            # Collect the modified dataframes for timeline compilation
            all_parked_row_dfs.extend(squelched_blocks)

    # 2. Compile and save arrays out to disk
    if not parked_summary_records:
        print("[INFO] Process complete: No verified stop events found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.csv))[0]
    
    # Output A: High-level metadata matrix of independent stop incidents
    summary_df = pd.DataFrame(parked_summary_records)
    summary_csv_output = os.path.join(args.output_dir, f"summary_parked_{base_name}.csv")
    summary_df.to_csv(summary_csv_output, index=False)
    
    # Output B: Master timeline containing only squelched frame intervals
    detailed_df = pd.concat(all_parked_row_dfs, ignore_index=True)
    detailed_csv_output = os.path.join(args.output_dir, f"timeline_parked_{base_name}.csv")
    detailed_df.to_csv(detailed_csv_output, index=False)
    
    print(f"\n⚡ Success! Parking isolation pipeline executed successfully.")
    print(f" -> Parked Metadata Summary written to:  {summary_csv_output}")
    print(f" -> Squelched Full Timelines written to:  {detailed_csv_output}")
    print(f" Isolated {len(summary_df)} total stop incidents across the dataset.")

if __name__ == "__main__":
    main()