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
    from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
    from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class
    from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import GStreamerDetectionApp
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
        
        # Get video frame if available - use parent class method
        frame = None
        if format is not None and width is not None and height is not None:
            try:
                frame = get_numpy_from_buffer(buffer, format, width, height)
                if frame is not None:
                    # Convert RGB to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # Store using parent class method
                    user_data.set_frame(frame)
                    # Also store in our own cache for faster retrieval
                    user_data.update_frame_copy(frame)
            except Exception as e:
                print(f"[ERROR] Frame extraction failed: {e}")
                pass  # Continue even if frame extraction fails
        
        # Get the detections from the buffer using Hailo API
        roi = hailo.get_roi_from_buffer(buffer)
        hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        # Parse detections into our format
        detections = []
        for detection in hailo_detections:
            label = detection.get_label()
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            
            # Only include cat detections with reasonable confidence
            # COCO class 15 is 'cat' - filter to only cats
            if confidence > 0.3 and label.lower() in ['cat', 'cats']:
                det_x1 = int(bbox.xmin() * width)
                det_y1 = int(bbox.ymin() * height)
                det_x2 = int(bbox.xmax() * width)
                det_y2 = int(bbox.ymax() * height)
                
                detection_dict = {
                    'class_name': label,
                    'confidence': confidence,
                    'bbox': [det_x1, det_y1, det_x2, det_y2],
                    'areas': []  # Which detection areas this overlaps with
                }
                
                # Check if detection overlaps with any configured areas
                if hasattr(user_data, 'crop_areas') and user_data.crop_areas:
                    for area in user_data.crop_areas:
                        # Check if detection center is within area
                        det_center_x = (det_x1 + det_x2) / 2
                        det_center_y = (det_y1 + det_y2) / 2
                        
                        area_x1 = area['x']
                        area_y1 = area['y']
                        area_x2 = area['x'] + area['w']
                        area_y2 = area['y'] + area['h']
                        
                        if (area_x1 <= det_center_x <= area_x2 and 
                            area_y1 <= det_center_y <= area_y2):
                            detection_dict['areas'].append(area['door'])
                
                detections.append(detection_dict)
        
        # Update detections
        user_data.update_detections(detections)
        
        # Increment frame counter
        user_data.increment()
        
        # Print detection info periodically
        frame_count = user_data.get_count()
        if frame_count % 30 == 0:  # Every 30 frames
            # Use parent class method to check frame availability
            has_frame = user_data.get_frame() is not None
            has_frame_copy = user_data.latest_frame_copy is not None
            frame_status = f"OK (parent={has_frame}, copy={has_frame_copy}, stored={user_data.frame_count})"
            print(f"[HAILO] Frame {frame_count}: {len(detections)} detections, Frame: {frame_status}")
            for det in detections[:3]:  # Only show first 3
                areas_str = f" in areas: {det['areas']}" if det['areas'] else ""
                print(f"  - {det['class_name']}: {det['confidence']:.2f}{areas_str}")
        
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
    
    def __init__(self, hef_path="/usr/local/hailo/resources/models/hailo8l/yolov6n.hef"):
        self.hef_path = hef_path
        self.app = None
        self.user_data = None
        self.app_thread = None
        self.initialized = False
        self.running = False
        self.crop_areas = []  # List of crop regions [x, y, w, h]
        
        if not HAILO_AVAILABLE:
            print("[ERROR] Hailo apps not available")
            return
        
        print(f"[HAILO] Initializing detector with {hef_path}")
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
            
            # Create custom argument parser to configure the pipeline
            from hailo_apps.hailo_app_python.core.common.core import get_default_parser
            import argparse
            parser = get_default_parser()
            
            # Override command line args to use camera
            import sys
            old_argv = sys.argv
            sys.argv = [
                sys.argv[0],
                '--input', 'rpi',  # Use Raspberry Pi camera
                '--hef-path', self.hef_path,
                '--disable-sync',  # Disable sync for lower latency
                '--show-fps',
                '--use-frame',  # CRITICAL: Enable frame capture in callback
            ]
            
            # Create the detection app with our callback
            self.app = GStreamerDetectionApp(hailo_detection_callback, self.user_data, parser)
            
            # Restore original argv
            sys.argv = old_argv
            
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
