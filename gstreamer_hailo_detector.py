#!/usr/bin/env python3
"""
GStreamer Hailo Detector using official Hailo Python APIs
Based on hailo-rpi5-examples/basic_pipelines/detection.py
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import threading
import time
import numpy as np
import cv2
from queue import Queue
from typing import List, Dict
import hailo

# Check if Hailo is available
HAILO_AVAILABLE = True
try:
    # Use our custom app to handle HTTP sources
    from custom_hailo_app import CustomGStreamerDetectionApp
    from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
    from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class, GStreamerApp
except ImportError as e:
    print(f"[WARN] Hailo apps not available: {e}")
    HAILO_AVAILABLE = False


class HailoDetectorCallback(app_callback_class):
    """Callback class for Hailo detection results"""
    
    def __init__(self):
        super().__init__()
        # Use parent class for frame storage, just add detection storage
        self.latest_detections = []
        self.detection_lock = threading.Lock()
        # Crop areas for detection filtering
        self.crop_areas = []
        # Add our own frame storage for faster access
        self.latest_frame_copy = None
        self.frame_copy_lock = threading.Lock()
        self.frame_count = 0
    
    def update_detections(self, detections):
        """Update the latest detections"""
        with self.detection_lock:
            self.latest_detections = detections.copy()
    
    def get_detections(self):
        """Get the latest detections"""
        with self.detection_lock:
            return self.latest_detections.copy()
    
    def update_frame_copy(self, frame):
        """Store a copy of the latest frame for fast retrieval"""
        with self.frame_copy_lock:
            if frame is not None:
                # Store reference directly - frame is already a copy from the callback
                self.latest_frame_copy = frame
                self.frame_count += 1
    
    def get_frame_copy(self):
        """Get the latest frame copy (returns reference, caller must not modify)"""
        with self.frame_copy_lock:
            # Return the reference directly instead of making another copy
            # This is much faster but caller must not modify the frame
            return self.latest_frame_copy


def hailo_detection_callback(pad, info, user_data):
    """
    Callback function for Hailo detection pipeline.
    Extracts detections and video frames from the GStreamer buffer.
    """
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    try:
        # Get the caps from the pad
        format, width, height = get_caps_from_pad(pad)

        # Retrieve the frame if requested
        frame = None
        if (getattr(user_data, "use_frame", False) and
                format is not None and width is not None and height is not None):
            frame = get_numpy_from_buffer(buffer, format, width, height)
            # Convert to BGR for OpenCV drawing/streaming
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Extract detections from the buffer
        parsed_detections = []
        roi = hailo.get_roi_from_buffer(buffer)
        hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION) if roi else []

        for det_obj in hailo_detections:
            label = det_obj.get_label()
            confidence = float(det_obj.get_confidence())
            bbox = det_obj.get_bbox()

            raw_x1 = float(bbox.xmin())
            raw_y1 = float(bbox.ymin())
            raw_x2 = float(bbox.xmax())
            raw_y2 = float(bbox.ymax())

            # Hailo BBox can be normalized (0-1) or absolute. Scale when values look normalized.
            if width and height and max(abs(raw_x1), abs(raw_y1), abs(raw_x2), abs(raw_y2)) <= 1.5:
                x1 = int(raw_x1 * width)
                y1 = int(raw_y1 * height)
                x2 = int(raw_x2 * width)
                y2 = int(raw_y2 * height)
            else:
                x1 = int(raw_x1)
                y1 = int(raw_y1)
                x2 = int(raw_x2)
                y2 = int(raw_y2)

            # Clip coordinates to frame bounds when available
            if width is not None and height is not None:
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

            track_id = None
            track_objs = det_obj.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if track_objs:
                track_id = track_objs[0].get_id()

            # Determine which configured areas this detection touches
            hit_areas = []
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            for area in getattr(user_data, "crop_areas", []):
                ax, ay, aw, ah = area.get('x', 0), area.get('y', 0), area.get('w', 0), area.get('h', 0)
                if aw <= 0 or ah <= 0:
                    continue
                if ax <= cx <= ax + aw and ay <= cy <= ay + ah:
                    area_name = area.get('name') or area.get('door') or area.get('door_id') or 'unknown'
                    hit_areas.append(area_name)

            parsed_detections.append({
                'class_name': label,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'track_id': track_id,
                'areas': hit_areas,
            })

        # Draw detections on the frame before storing it
        if frame is not None and parsed_detections:
            for det in parsed_detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class_name']} ({det['confidence']:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Store frame copies for streaming
        if frame is not None:
            try:
                user_data.set_frame(frame)
                user_data.update_frame_copy(frame)
            except Exception as e:
                print(f"[ERROR] Frame storage failed: {e}")

        # Update detections
        user_data.update_detections(parsed_detections)

        # Increment frame counter
        user_data.increment()
        frame_count = user_data.get_count()

        # Log detections immediately
        if parsed_detections:
            for det in parsed_detections:
                areas_str = f" in areas: {det['areas']}" if det['areas'] else ""
                print(f"[HAILO DETECTION] Frame {frame_count}: {det['class_name']} detected! "
                      f"Confidence={det['confidence']:.2f}{areas_str} "
                      f"bbox={det['bbox']}")

        # Periodic status log
        if frame_count % 30 == 0:
            has_frame = user_data.get_frame() is not None
            print(f"[HAILO STATUS] Frame {frame_count}: {len(parsed_detections)} detections, Frame: {'OK' if has_frame else 'None'}")
        
    except Exception as e:
        print(f"[ERROR] Callback error: {e}")
        import traceback
        traceback.print_exc()
    
    return Gst.PadProbeReturn.OK


class GStreamerHailoDetector:
    """
    GStreamer-based Hailo AI detector using official Hailo APIs.
    Provides simple interface for cat door application.
    """
    
    def __init__(self, hef_path="/usr/share/hailo-models/yolov8s_h8l.hef", input_source="rpi"):
        """
        Initialize the detector.
        
        Args:
            hef_path: Path to the Hailo model file
            input_source: Either 'rpi' for camera or HTTP URL for video stream
        """
        self.hef_path = hef_path
        self.input_source = input_source
        self.app = None
        self.user_data = None
        self.app_thread = None
        self.initialized = False
        self.running = False
        self.crop_areas = []  # List of crop regions [x, y, w, h]
        
        if not HAILO_AVAILABLE:
            print("[ERROR] Hailo apps not available")
            return
        
        print(f"[HAILO] Initializing detector with {hef_path}, input={input_source}")
        self.initialized = True
    
    def set_detection_areas(self, areas):
        """
        Set the detection areas to focus inference on.
        
        Args:
            areas: List of area dicts with 'rect' key containing [x, y, w, h]
        """
        self.crop_areas = []
        for area in areas:
            rect = area.get('rect', [0, 0, 0, 0])
            if len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
                self.crop_areas.append({
                    'x': rect[0],
                    'y': rect[1],
                    'w': rect[2],
                    'h': rect[3],
                    'door': area.get('door', 'unknown')
                })
        print(f"[HAILO] Set {len(self.crop_areas)} detection areas")
        for i, area in enumerate(self.crop_areas):
            print(f"  Area {i+1}: door={area['door']}, rect=[{area['x']}, {area['y']}, {area['w']}, {area['h']}]")
    
    def start(self):
        """Start the Hailo detection pipeline"""
        if not self.initialized:
            print("[ERROR] Cannot start - not initialized")
            return False
        
        if self.running:
            print("[WARN] Already running")
            return True
        
        try:
            # Create callback data object
            self.user_data = HailoDetectorCallback()
            self.user_data.use_frame = True  # Enable frame capture
            self.user_data.crop_areas = self.crop_areas  # Pass crop areas to callback
            
            # to simulate command-line execution. This is cleaner than manipulating sys.argv.
            from hailo_apps.hailo_app_python.core.common.core import get_default_parser
            import argparse
            parser = get_default_parser()

            args_list = [
                '--input', self.input_source,
                '--hef-path', self.hef_path,
                '--disable-sync', 
                '--show-fps', 
                '--use-frame',
            ]
            
            # Save original sys.argv and temporarily replace it
            import sys
            original_argv = sys.argv
            sys.argv = ['detection'] + args_list
            
            try:
                # Create the app - the CustomGStreamerDetectionApp expects callback, user_data, and parser
                self.app = CustomGStreamerDetectionApp(hailo_detection_callback, self.user_data, parser)
            finally:
                # Restore original sys.argv
                sys.argv = original_argv
            
            # Set video_sink to fakesink to disable HDMI output
            # This fixes the garbage display on the connected monitor
            if hasattr(self.app, 'video_sink'):
                self.app.video_sink = 'fakesink'
                print("[HAILO] Video output disabled (using fakesink)")
            
            # Run the app in a separate thread
            self.app_thread = threading.Thread(target=self.app.run, daemon=True)
            self.app_thread.start()
            
            self.running = True
            print("[HAILO] Detection pipeline started")
            
            # Give it a moment to initialize
            time.sleep(2)
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop(self):
        """Stop the detection pipeline"""
        if not self.running:
            return
        
        try:
            if self.app:
                self.app.shutdown()
            self.running = False
            print("[HAILO] Detection pipeline stopped")
        except Exception as e:
            print(f"[ERROR] Error stopping: {e}")
    
    def get_latest_frame(self):
        """Get the most recent video frame"""
        if self.user_data:
            # Use our cached copy for faster, more reliable retrieval
            return self.user_data.get_frame_copy()
        return None
    
    def get_latest_detections(self) -> List[Dict]:
        """
        Get the most recent detections.
        
        Returns:
            List of detection dictionaries with keys: class_name, confidence, bbox
        """
        if self.user_data:
            return self.user_data.get_detections()
        return []
    
    def detect(self, target_classes=['cat'], confidence_threshold=0.5) -> List[Dict]:
        """
        Get detections filtered by class and confidence.
        
        Args:
            target_classes: List of class names to detect
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of filtered detections
        """
        all_detections = self.get_latest_detections()
        filtered = []
        
        for det in all_detections:
            if (det['class_name'] in target_classes and 
                det['confidence'] >= confidence_threshold):
                filtered.append(det)
        
        return filtered


# Test code
if __name__ == "__main__":
    print("Testing GStreamer Hailo Detector...")
    
    detector = GStreamerHailoDetector()
    if detector.start():
        print("Detector started, running for 10 seconds...")
        
        for i in range(10):
            time.sleep(1)
            frame = detector.get_latest_frame()
            detections = detector.get_latest_detections()
            print(f"[{i+1}] Frame: {'OK' if frame is not None else 'None'}, Detections: {len(detections)}")
        
        detector.stop()
        print("Test complete!")
    else:
        print("Failed to start detector")
