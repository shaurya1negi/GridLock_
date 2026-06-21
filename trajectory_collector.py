"""
trajectory_collector.py
------------------------
FOUNDATION MODULE: Vehicle Trajectory Data Collection + Velocity + Black Canvas
"""
 
import argparse
import csv
import math
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd  

import polygon

import cv2
from ultralytics import YOLO
 
import config
 
 
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.filtered_value = None
        self.filtered_velocity = 0.0
        self.last_time = None
 
    def _smoothing_factor(self, cutoff, delta_time):
        if delta_time <= 0:
            return 1.0
        tau = 1.0 / (2 * 3.14159265359 * cutoff)
        return 1.0 / (1.0 + tau / delta_time)
 
    def filter(self, raw_value, timestamp):
        if self.filtered_value is None:
            self.filtered_value = raw_value
            self.filtered_velocity = 0.0
            self.last_time = timestamp
            return raw_value
 
        delta_time = timestamp - self.last_time
        if delta_time <= 0:
            return self.filtered_value
 
        raw_velocity = (raw_value - self.filtered_value) / delta_time
        alpha_velocity = self._smoothing_factor(self.d_cutoff, delta_time)
        self.filtered_velocity = (alpha_velocity * raw_velocity + (1.0 - alpha_velocity) * self.filtered_velocity)
        
        cutoff = self.min_cutoff + self.beta * abs(self.filtered_velocity)
        alpha = self._smoothing_factor(cutoff, delta_time)
        self.filtered_value = (alpha * raw_value + (1.0 - alpha) * self.filtered_value)
 
        self.last_time = timestamp
        return self.filtered_value
 
 
class EMAFilter:
    def __init__(self, alpha=0.85):
        self.alpha = alpha
        self.filtered_value = None
 
    def filter(self, raw_value):
        if self.filtered_value is None:
            self.filtered_value = raw_value
            return raw_value
        self.filtered_value = (self.alpha * raw_value + (1.0 - self.alpha) * self.filtered_value) 
        return self.filtered_value
 
 
def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
 
 
def build_output_paths(output_dir):
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{config.CSV_FILENAME_PREFIX}_{run_stamp}.csv")
    video_path = os.path.join(output_dir, f"{config.VIDEO_FILENAME_PREFIX}_{run_stamp}.mp4")
    black_canvas_path = os.path.join(output_dir, f"{config.BLACK_CANVAS_FILENAME_PREFIX}_{run_stamp}.mp4")
    return csv_path, video_path, black_canvas_path
 
 
def get_class_name(class_id):
    return config.TARGET_CLASSES.get(class_id, f"unknown_class_{class_id}")
 
 
def process_video(video_path, output_dir=None, fps_override=None , polygon_json_path=None):
    output_dir = output_dir or config.OUTPUT_DIR
    ensure_output_dir(output_dir)
    csv_path, annotated_video_path, black_canvas_path = build_output_paths(output_dir)
    
    model = YOLO(config.YOLO_MODEL_PATH)
 
    probe_capture = cv2.VideoCapture(video_path)
    if not probe_capture.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    
    frame_width = int(probe_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(probe_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe_capture.release()
 
    #  NATIVE HANDSHAKE OVERRIDE:
    if fps_override is not None:
        source_fps = float(fps_override)
        print(f"source_fps-> {source_fps:.2f} FPS")
    elif not source_fps or source_fps <= 0:
        source_fps = 30.0
    
    #  Load polygons directly from the JSON asset if provided
    polygons = []
    if polygon_json_path and os.path.exists(polygon_json_path):
        polygons = polygon.load_polygons(polygon_json_path)
        print(f"Loaded {len(polygons)} zone shapes to draw onto the active trajectory map canvas stream.")

    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    
    # Standard output video writer
    video_writer = cv2.VideoWriter(annotated_video_path, fourcc, source_fps, (frame_width, frame_height))
    
    # Black Canvas video writer
    black_canvas_writer = None
    if getattr(config, "SAVE_BLACK_CANVAS_VIDEO", True):
        black_canvas_writer = cv2.VideoWriter(black_canvas_path, fourcc, source_fps, (frame_width, frame_height))
 
    # CSV Generation with new Velocity Columns
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "tracker_id", "frame_index", "timestamp_sec", "vehicle_class", "class_id", "confidence",
        "x1", "y1", "x2", "y2", "bottom_center_x", "bottom_center_y", "box_width", "box_height",
        "velocity_x_px_sec", "velocity_y_px_sec",
        "speed_px_sec",        
        "heading_deg",         
        "is_stationary",       
        "frames_since_first_seen",  
    ])
 
    tracker_filters = defaultdict(lambda: {
        "x1": {"one_euro": OneEuroFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF), "ema": EMAFilter(config.EMA_ALPHA)},
        "y1": {"one_euro": OneEuroFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF), "ema": EMAFilter(config.EMA_ALPHA)},
        "x2": {"one_euro": OneEuroFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF), "ema": EMAFilter(config.EMA_ALPHA)},
        "y2": {"one_euro": OneEuroFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF), "ema": EMAFilter(config.EMA_ALPHA)},
    })
    
    # Historical state registries
    velocity_history = {}
    class_history = defaultdict(list)
    velocity_filters = defaultdict(lambda: {"vx": EMAFilter(alpha=config.VELOCITY_EMA_ALPHA), "vy": EMAFilter(alpha=config.VELOCITY_EMA_ALPHA)})
 
    stationary_frame_counts = defaultdict(int)
    first_seen_frame = {}
 
    frame_index = 0
    target_class_ids = list(config.TARGET_CLASSES.keys())
 
    print(f"Processing video: {video_path}")
    print(f"Source FPS: {source_fps:.2f} | Resolution: {frame_width}x{frame_height}")
 
    results_stream = model.track(
        source=video_path, tracker=config.TRACKER_CONFIG, conf=config.CONFIDENCE_THRESHOLD,
        iou=config.IOU_THRESHOLD, classes=target_class_ids, persist=True, stream=True, verbose=False
    )
 
    for result in results_stream:
        timestamp_sec = frame_index / source_fps
        annotated_frame = result.orig_img.copy()
        
        black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        for zone in polygons:
            pts = np.array(zone["polygon"], dtype=np.int32)
            # Differentiate color matching schemes natively
            color = (255, 128, 0) if zone["zone_type"] == "road_lane" else (0, 0, 255)
            
            # Burn background zones out with light alpha transparency blending
            cv2.polylines(black_frame, [pts], isClosed=True, color=color, thickness=2)
            overlay = black_frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.15, black_frame, 0.85, 0, black_frame)
            
            # Burn matching legal vectors if present
            if zone.get("legal_vector"):
                vec = zone["legal_vector"]
                cv2.arrowedLine(black_frame, tuple(vec["start"]), tuple(vec["end"]), (255, 0, 0), 2, tipLength=0.2)

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            tracker_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            xyxy_coords = boxes.xyxy.cpu().tolist()
 
            for tracker_id, class_id, confidence, (x1, y1, x2, y2) in zip(tracker_ids, class_ids, confidences, xyxy_coords):
                
                class_history[tracker_id].append(class_id)
                stable_class_id = max(set(class_history[tracker_id]), key=class_history[tracker_id].count)
                vehicle_class = get_class_name(stable_class_id)
 
                # Apply Filters
                if config.ENABLE_ONE_EURO_FILTER:
                    filters = tracker_filters[tracker_id]
                    x1 = filters["x1"]["one_euro"].filter(x1, timestamp_sec)
                    y1 = filters["y1"]["one_euro"].filter(y1, timestamp_sec)
                    x2 = filters["x2"]["one_euro"].filter(x2, timestamp_sec)
                    y2 = filters["y2"]["one_euro"].filter(y2, timestamp_sec)
 
                if config.ENABLE_EMA_FILTER:
                    filters = tracker_filters[tracker_id]
                    x1 = filters["x1"]["ema"].filter(x1)
                    y1 = filters["y1"]["ema"].filter(y1)
                    x2 = filters["x2"]["ema"].filter(x2)
                    y2 = filters["y2"]["ema"].filter(y2)
 
                bottom_center_x = (x1 + x2) / 2.0
                bottom_center_y = y2
                box_width = x2 - x1
                box_height = y2 - y1
 
                # ---- VELOCITY CALCULATION ----
                velocity_x = 0.0
                velocity_y = 0.0
 
                if tracker_id in velocity_history:
                    last_x, last_y, last_t = velocity_history[tracker_id]
                    dt = timestamp_sec - last_t
                    if dt > 0:
                        velocity_x = (bottom_center_x - last_x) / dt
                        velocity_y = (bottom_center_y - last_y) / dt
                
                velocity_x = velocity_filters[tracker_id]["vx"].filter(velocity_x)
                velocity_y = velocity_filters[tracker_id]["vy"].filter(velocity_y)
                
                velocity_history[tracker_id] = (bottom_center_x, bottom_center_y, timestamp_sec)
 
                # ---- DERIVED COLUMNS ----------------------------------------
                speed_px_sec = (velocity_x ** 2 + velocity_y ** 2) ** 0.5
 
                if speed_px_sec > 1.0:  
                    heading_deg = (math.degrees(math.atan2(velocity_y, velocity_x)) + 360) % 360
                else:
                    heading_deg = -1.0  
 
                if speed_px_sec < config.STATIONARY_SPEED_THRESHOLD_PX_SEC:
                    stationary_frame_counts[tracker_id] += 1
                else:
                    stationary_frame_counts[tracker_id] = 0  
 
                is_stationary = int(
                    stationary_frame_counts[tracker_id] >= config.STATIONARY_MIN_FRAMES
                )
 
                if tracker_id not in first_seen_frame:
                    first_seen_frame[tracker_id] = frame_index
                frames_since_first_seen = frame_index - first_seen_frame[tracker_id]
 
                csv_writer.writerow([
                    tracker_id, frame_index, round(timestamp_sec, 4), vehicle_class, stable_class_id, round(confidence, 4),
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                    round(bottom_center_x, 2), round(bottom_center_y, 2),
                    round(box_width, 2), round(box_height, 2),
                    round(velocity_x, 2), round(velocity_y, 2),
                    round(speed_px_sec, 2), round(heading_deg, 2),
                    is_stationary, frames_since_first_seen,
                ])
 
                # ---- STANDARD OUTPUT VISUALIZATION ----
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                label = f"ID:{tracker_id} {vehicle_class} {confidence:.2f}"
                cv2.putText(annotated_frame, label, (int(x1), max(int(y1) - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(annotated_frame, (int(bottom_center_x), int(bottom_center_y)), 4, (0, 0, 255), -1)
 
                # ---- BLACK CANVAS PLOTTING WRITER ----
                if black_canvas_writer is not None:
                    cx, cy = int(bottom_center_x), int(bottom_center_y)
                    cv2.circle(black_frame, (cx, cy), 5, (255, 255, 255), -1)
                    
                    vector_scale = 0.2 
                    vx_projected = int(cx + (velocity_x * vector_scale))
                    vy_projected = int(cy + (velocity_y * vector_scale))
                    
                    if abs(velocity_x) > 5 or abs(velocity_y) > 5:
                        cv2.line(black_frame, (cx, cy), (vx_projected, vy_projected), (255, 255, 255), 2)
                        cv2.circle(black_frame, (vx_projected, vy_projected), 2, (255, 255, 255), -1)
                    
                    canvas_label = f"ID:{tracker_id} ({cx},{cy})"
                    cv2.putText(black_frame, canvas_label, (cx + 10, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
 
        video_writer.write(annotated_frame)
        if black_canvas_writer is not None:
            black_canvas_writer.write(black_frame)
            
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"  Processed {frame_index} frames...")
 
    csv_file.close()
    video_writer.release()
    if black_canvas_writer is not None:
        black_canvas_writer.release()
 
    print("\nSorting CSV by vehicle class and tracking life cycles...")
    try:
        df = pd.read_csv(csv_path)
        df_sorted = df.sort_values(by=['vehicle_class', 'tracker_id', 'frame_index'], ascending=[True, True, True])
        df_sorted.to_csv(csv_path, index=False)
        print("CSV post-processing complete! Trajectories cleanly grouped.")
    except Exception as e:
        print(f"Warning: Post-processing sort failed: {e}")

    print(f"\nDone. Processed {frame_index} total frames.")
    print(f"CSV written to:             {csv_path}")
    print(f"Standard Video written to:   {annotated_video_path}")
    if black_canvas_writer is not None:
        print(f"Isolated Map written to:    {black_canvas_path}")
 
    return csv_path
 
 
def main():
    parser = argparse.ArgumentParser(description="Vehicle Trajectory Data Collection Upgraded Layer")
    parser.add_argument("--video", required=True, help="Path to the input traffic video file.")
    parser.add_argument("--output-dir", default=None, help="Output directory override.")
    args = parser.parse_args()
 
    process_video(video_path=args.video, output_dir=args.output_dir)
 
 
if __name__ == "__main__":
    main()
