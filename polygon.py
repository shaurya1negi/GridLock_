"""
polygon.py
----------
Module handling interactive polygon drawing, zone typing, and detailed JSON saving.
"""

import os
import json
import numpy as np
import cv2

class PolygonDrawer:
    """An interactive OpenCV tool to draw zones, classify types, and assign direction vectors."""
    def __init__(self, frame, window_name="Setup Traffic Zones"):
        self.orig_frame = frame.copy()
        self.display_frame = frame.copy()
        self.window_name = window_name
        self.finalized_zones = []    # List of dicts: {"zone_type": ..., "polygon": ..., "legal_vector": ...}
        self.current_polygon = []    # Active polygon vertices
        
        # State Machine Flags
        self.active_zone_type = None
        self.drawing_vector = False
        self.vector_start = None

    def _mouse_callback(self, event, x, y, flags, param):
        # Discard clicks if a menu option hasn't been chosen yet
        if self.active_zone_type is None:
            return

        if not self.drawing_vector:
            # --- POLYGON MODE ---
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_polygon.append((x, y))
                self._redraw()
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.current_polygon) > 2:
                    if self.active_zone_type == "road_lane":
                        self.drawing_vector = True
                        self.vector_start = None
                        print("➡️ Road Lane closed. Now click TWO points to draw the LEGAL TRAFFIC FLOW ARROW.")
                    else:
                        # Parking zones don't need a vector, save immediately
                        self.finalized_zones.append({
                            "zone_id": len(self.finalized_zones) + 1,
                            "zone_type": "illegal_parking",
                            "polygon": self.current_polygon,
                            "legal_vector": None
                        })
                        print(f"✅ Saved Illegal Parking Zone #{len(self.finalized_zones)}")
                        self.current_polygon = []
                        self.active_zone_type = None
                        self._print_menu()
                else:
                    print("⚠️ A zone needs at least 3 points to close!")
                self._redraw()
        else:
            # --- VECTOR MODE (Only for Road Lanes) ---
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.vector_start is None:
                    self.vector_start = (x, y)
                    print("📍 Start point registered. Click the arrowhead direction next.")
                else:
                    # Save the complete Road Lane config
                    self.finalized_zones.append({
                        "zone_id": len(self.finalized_zones) + 1,
                        "zone_type": "road_lane",
                        "polygon": self.current_polygon,
                        "legal_vector": {
                            "start": self.vector_start,
                            "end": (x, y)
                        }
                    })
                    print(f"✅ Saved Road Lane Zone #{len(self.finalized_zones)} with Traffic Vector.")
                    self.current_polygon = []
                    self.drawing_vector = False
                    self.vector_start = None
                    self.active_zone_type = None
                    self._print_menu()
                    self._redraw()

    def _redraw(self):
        self.display_frame = self.orig_frame.copy()
        
        # 1. Draw all saved structural zones from database history
        for zone in self.finalized_zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            # Differentiate color schemes visually
            color = (255, 128, 0) if zone["zone_type"] == "road_lane" else (0, 0, 255)
            
            cv2.polylines(self.display_frame, [pts], isClosed=True, color=color, thickness=2)
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.20, self.display_frame, 0.80, 0, self.display_frame)
            
            # Render direction arrows if they exist
            if zone["legal_vector"]:
                vec = zone["legal_vector"]
                cv2.arrowedLine(self.display_frame, tuple(vec["start"]), tuple(vec["end"]), (255, 0, 0), 3, tipLength=0.3)

        # 2. Render active real-time sketching feedback lines
        if self.active_zone_type:
            color = (255, 128, 0) if self.active_zone_type == "road_lane" else (0, 0, 255)
            for pt in self.current_polygon:
                cv2.circle(self.display_frame, pt, 4, (0, 255, 0), -1)
            for i in range(len(self.current_polygon) - 1):
                cv2.line(self.display_frame, self.current_polygon[i], self.current_polygon[i+1], (0, 255, 0), 2)
                
            if self.drawing_vector:
                pts = np.array(self.current_polygon, dtype=np.int32)
                cv2.polylines(self.display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                if self.vector_start:
                    cv2.circle(self.display_frame, self.vector_start, 5, (0, 0, 255), -1)

        cv2.imshow(self.window_name, self.display_frame)

    def _print_menu(self):
        print("\n--- SELECT ZONE TYPE TO DRAW ---")
        print(" Press 'p' : Start drawing an ILLEGAL PARKING ZONE (Red)")
        print(" Press 'r' : Start drawing a ROAD TRAFFIC LANE (Blue/Orange + Direction Arrow)")
        print(" Press 'z' : UNDO last coordinate step / edit history")
        print(" Press 's' : SAVE EVERYTHING AND EXPORT DATA PIPELINE")
        print("--------------------------------")

    def draw_polygons(self):
        """
        ===========================================================================
        INTERACTIVE ROAD GEOMETRY & FLOW VECTOR STEP-BY-STEP USER INSTRUCTIONS
        ===========================================================================
        STEP 1: SELECT ZONE TYPE
        ------------------------
        When the window opens, focus your keyboard on the terminal or the image 
        and press one of these two mode selectors:
          - Press 'p' -> To configure an ILLEGAL PARKING ZONE boundary.
          - Press 'r' -> To configure a ROAD DIRECTION TRAFFIC LANE boundary.
        
        STEP 2: DRAWING THE POLYGON AREA
        -------------------------------
        - LEFT-CLICK with your mouse repeatedly to trace vertices around the lane 
          or the parking segment on the road frame.
        - RIGHT-CLICK to close the shape loop and finalize the spatial boundaries.
        
        STEP 3: MAPPING VECTOR DIRECTIONS (Only triggers if 'r' was selected)
        -------------------------------------------------------------------
        - After right-clicking a Road Lane, the polygon turns into a green preview frame.
        - Click Point A (The starting position tail of the direction arrow).
        - Click Point B (The ending direction arrowhead pointing along legal traffic flow).
        
        STEP 4: SAVING AND REPEATING / UNDO MECHANICS
        ---------------------------------------------
        - Press 'z' at any stage to step backward through individual points or shapes!
        - Once a shape is completed, the tool returns to the master menu. 
          You can select 'p' or 'r' again to trace additional zones as needed.
        - When all maps are ready, press the 's' key (or 'ESC') to serialize the structured
          configuration array directly to the JSON folder database.
        ===========================================================================
        """
        self._print_menu()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        img_h, img_w = self.orig_frame.shape[:2]
        cv2.resizeWindow(self.window_name, int(img_w * (720/img_h)), 720)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.imshow(self.window_name, self.display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('p'):
                self.active_zone_type = "illegal_parking"
                self.current_polygon = []
                self.drawing_vector = False
                print("🛑 Mode set: Drawing Illegal Parking Zone. Left-click points, Right-click to close.")
            elif key == ord('r'):
                self.active_zone_type = "road_lane"
                self.current_polygon = []
                self.drawing_vector = False
                print("🛣️ Mode set: Drawing Traffic Lane. Left-click points, Right-click to close, then trace flow arrow.")
                
            # --- NEW: DETAILED STATE MACHINE UNDO CORE LAYERS ---
            elif key == ord('z'):
                if self.drawing_vector:
                    if self.vector_start is not None:
                        # Case 1: Clear the registered arrow start tail, stay in arrow mode
                        self.vector_start = None
                        print("↩Undid vector start point. Click again to set the arrow base.")
                    else:
                        # Case 2: Escape arrow mode entirely and re-open the polygon sequence
                        self.drawing_vector = False
                        print("↩Returned back to polygon vertex placement mode.")
                    self._redraw()
                elif len(self.current_polygon) > 0:
                    # Case 3: Pop individual placement dots if drawing an unfinished polygon
                    removed_pt = self.current_polygon.pop()
                    print(f"↩Undid last shape vertex point: {removed_pt}")
                    self._redraw()
                elif len(self.finalized_zones) > 0:
                    # Case 4: If active sequence is empty, pop and edit the last completed database configuration
                    last_zone = self.finalized_zones.pop()
                    self.current_polygon = last_zone["polygon"]
                    self.active_zone_type = last_zone["zone_type"]
                    
                    if last_zone["legal_vector"]:
                        self.drawing_vector = True
                        self.vector_start = last_zone["legal_vector"]["start"]
                        print(f"↩Reopened Road Lane Zone #{last_zone['zone_id']} at the vector arrowhead assignment stage.")
                    else:
                        self.drawing_vector = False
                        print(f"↩Reopened Illegal Parking Zone #{last_zone['zone_id']} for vertex adjustments.")
                    self._redraw()
                else:
                    print("ℹState stack is clear. Nothing left to undo!")
                    
            elif key == ord('s') or key == 27:
                break
                
        cv2.destroyWindow(self.window_name)
        return self.finalized_zones
    


def save_polygons(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_polygons(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return json.load(f)