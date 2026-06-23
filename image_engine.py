"""
image_engine.py
---------------
Processing Engine for Static Photographic Evidence.
Fixed to scan for macro vehicles and extract individual motorcycle patches separately.
"""

import os
import cv2
from ultralytics import YOLO
import config  # Dynamically references paths
import voilation_detector as violator  # Import the upgraded multi-model consensus checker
import evidence_harvester  # Import your polymorphic ticket generator

def analyze_single_frame(image_path, output_dir, model_backbone, automated_audit=False, save_annotated=True):
    """
    Core execution unit. Runs the multi-class YOLO model on a single frame,
    maps targets against macro vehicle constraints, extracts motorcycle crops, and logs reports.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"⚠️ Skipping corrupt or missing image: {image_path}")
        return False
        
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    img_h, img_w = frame.shape[:2]
    
    # 🎯 FIX: Use standard macro indices (like 3 for motorcycle) to filter the full scene pass
    macro_indices = list(config.TARGET_CLASSES.keys())
    results = model_backbone(frame, conf=config.CONFIDENCE_THRESHOLD, classes=macro_indices, iou=config.IOU_THRESHOLD)
    result_data = results[0]
    
    detected_violations = []
    motorcycle_count = 0

    if result_data.boxes is not None and len(result_data.boxes) > 0:
        for idx, box in enumerate(result_data.boxes):
            class_id = int(box.cls[0])
            class_name = result_data.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detected_violations.append({
                "raw_class": class_name,  
                "confidence": round(confidence, 4)
            })
            
            # 🌟 NEW FEATURE: Isolate and release all motorcycle patches separately
            if class_id == 3:  # 3 maps to motorcycle
                motorcycle_count += 1
                w, h = (x2 - x1), (y2 - y1)
                
                # Apply your exact asymmetrical upward padding to capture the riders completely
                x1_pad = max(0, int(x1 - (0.25 * w)))
                y1_pad = max(0, int(y1 - (0.55 * h)))
                x2_pad = min(img_w, int(x2 + (0.25 * w)))
                y2_pad = min(img_h, int(y2 + (0.40 * h)))
                
                bike_patch = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                if bike_patch.size > 0:
                    patch_name = f"{base_filename}_bike_patch_{motorcycle_count}.jpg"
                    patch_path = os.path.join(output_dir, patch_name)
                    cv2.imwrite(patch_path, bike_patch)
                    print(f"🏍️ Extracted motorcycle patch saved to: {patch_path}")
            
    # Save Visual Evidence Photo
    if save_annotated:
        if detected_violations: 
            annotated_frame = result_data.plot(line_width=1, font_size=3)
            img_output_path = os.path.join(output_dir, f"{base_filename}_annotated.jpg")
            cv2.imwrite(img_output_path, annotated_frame)
            print(f"📸 Visual Evidence saved securely to: {img_output_path}")
        else:
            img_output_path = os.path.join(output_dir, f"{base_filename}_clean.jpg")
            cv2.imwrite(img_output_path, frame)
            
    # Write out structured text report
    txt_output_path = os.path.join(output_dir, f"{base_filename}_report.txt")
    with open(txt_output_path, mode="w", encoding="utf-8") as report_file:
        report_file.write(f"==================================================\n")
        report_file.write(f"TRAFFIC ENFORCEMENT AUDIT DOCKET: {base_filename}\n")
        report_file.write(f"Source Context: {'AUTOMATED VIDEO RECON CROP' if automated_audit else 'MANUAL PHOTOGRAPHIC INPUT'}\n")
        report_file.write(f"==================================================\n\n")
        
        if detected_violations:
            report_file.write(f"STATUS: 🚨 TARGET VEHICLE ATTRIBUTE(S) DETECTED\n\n")
            report_file.write(f"{'Monitored Class':<25} | {'AI Confidence Score':<20}\n")
            report_file.write(f"--------------------------------------------------\n")
            for item in detected_violations:
                report_file.write(f"{item['raw_class']:<25} | {item['confidence']*100:.2f}%\n")
        else:
            report_file.write("STATUS: ✅ CLEAN SHEET (No monitored attributes identified).\n")
            
    print(f"📝 Audit Docket written securely to: {txt_output_path}")
    return len(detected_violations) > 0


def execute_compliance_pipeline(image_path, macro_model, person_model, helmet_model, output_dir):
    """Helper wrapper to execute the micro-compliance checking and harvest calls sequentially."""
    print(f"🔍 Running micro-compliance audits (Riders + Helmets) for: {os.path.basename(image_path)}")
    compliance_csv = violator.audit_static_image_compliance(
        image_path=image_path,
        macro_model=macro_model,
        person_model=person_model,
        helmet_model=helmet_model,
        output_dir=output_dir
    )
    
    if compliance_csv and os.path.exists(compliance_csv):
        evidence_harvester.harvest_violation_patches(
            video_path=image_path,
            violation_csv_path=compliance_csv,
            trajectory_csv_path=None,  
            output_dir=output_dir
        )


def run_image_pipeline(image_path, output_dir="output", automated_audit=False, save_annotated=True):
    """Intake Router Gateway. Executes macro vehicle detection and drops into micro compliance checks."""
    static_output_folder = os.path.join(output_dir, "static_image_output")
    os.makedirs(static_output_folder, exist_ok=True)
    
    print(f"\n📸 Initializing Photographic Audit Sequence...")
    print("-" * 70)
    
    # Load Weights
    macro_weights = "weights/yolo11m.pt"
    person_weights = "weights/yolo11m.pt"
    helmet_weights = "weights/helmet.pt"
    
    macro_model = YOLO(macro_weights)
    person_model = YOLO(person_weights)
    helmet_model = YOLO(helmet_weights)
        
    if os.path.isfile(image_path):
        analyze_single_frame(image_path, static_output_folder, macro_model, automated_audit, save_annotated)
        execute_compliance_pipeline(image_path, macro_model, person_model, helmet_model, static_output_folder)
        
    elif os.path.isdir(image_path):
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(supported_extensions)]
        
        for img_file_path in image_files:
            analyze_single_frame(img_file_path, static_output_folder, macro_model, automated_audit, save_annotated)
            execute_compliance_pipeline(img_file_path, macro_model, person_model, helmet_model, static_output_folder)