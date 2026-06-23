"""
main.py
-------
Dual-Mode Intake Orchestrator for the Traffic Violation Detection Suite.
Dynamically routes targets based on video containers vs photographic extensions.
"""

import argparse
import os
import sys

# Engine Module Imports
import video_engine
import image_engine

def get_file_extension(path):
    """Extracts lowercase file extension for precise interface routing."""
    return os.path.splitext(path)[1].lower()

def main():
    parser = argparse.ArgumentParser(
        description="Traffic Violation Suite - Multi-Mode Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Expose a single universal file intake argument flag
    parser.add_argument("--input", required=True, help="Path to input source file (video stream or photographic image asset)")
    parser.add_argument("--output-dir", default="output", help="Output destination directory path (default: output)")
    parser.add_argument("--skip-polygon-drawing", action="store_true", 
                        help="Skip polygon design mapping phase if configuration profile already exists")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Target input data file asset not discovered at: {args.input}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Supported Format Mappings
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    file_ext = get_file_extension(args.input)
    
    print("="*70)
    print("🚀 INITIALIZING TRAFFIC MONITORING & ENFORCEMENT INTERFACE")
    print("="*70)
    print(f"Target Source Detected: {args.input}")
    
    # --- ROUTING LOGIC BARRIER ---
    if file_ext in VIDEO_EXTENSIONS:
        print("Calling Spatiotemporal Video Stream Sequence Analytics Engine\n")
        video_engine.run_video_pipeline(
            video_path=args.input,
            output_dir=args.output_dir,
            skip_polygon_drawing=args.skip_polygon_drawing,
            bike_compliance=False
        )
        
    elif file_ext in IMAGE_EXTENSIONS:
        print("Calling Static Photographic Audit Engine\n")
        image_engine.run_image_pipeline(
            image_path=args.input,
            output_dir=args.output_dir,
            automated_audit=False,
            save_annotated=True
        )
        
    else:
        print(f"Structural Abort: Extension '{file_ext}' is not mapped to an active analytics engine wrapper.")
        print(f"Supported Video Streams: {list(VIDEO_EXTENSIONS)}")
        print(f"Supported Static Evidence Photos: {list(IMAGE_EXTENSIONS)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#python main.py --input snapshot.jpg 
#python main.py --input video.mp4