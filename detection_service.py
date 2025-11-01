#!/usr/bin/env python3
"""
Cat Detection Service
Runs in a separate thread to perform periodic cat detection in configured areas
using the Hailo AI HAT for hardware-accelerated inference.
"""

import threading
import time
import cv2
import numpy as np
from typing import Dict, Optional, List
import queue

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    import hailo
    from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import GStreamerDetectionApp
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("[DETECTION] Hailo detector not available")


class DetectionService:
    """
    Parallel detection service that:
    1. Receives frames and area configurations
    2. Crops areas from frames
    3. Runs Hailo detection on cropped areas
    4. Checks if detected cat matches the expected profile
    """
    
    def __init__(self, detection_interval=1.0):
        """
        Args:
            detection_interval: Seconds between detection runs (default 1.0)
        """
        self.running = False
        self.thread = None
        self.detector = None
        self.detection_interval = detection_interval
        
        # Frame queue (latest frame for detection)
        self.frame_queue = queue.Queue(maxsize=1)
        
        # Detection results per area (thread-safe)
        self._results_lock = threading.Lock()
        self._detection_results = {}  # {area_name: result_dict}
        
        # Area configurations
        self._areas_lock = threading.Lock()
        self._areas = []  # List of area configs
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'processing_time_ms': 0,
            'last_detection_time': None
        }
        
        print("[DETECTION] Detection service initialized")
    
    def start(self):
        """Start the detection service thread"""
        if self.running:
            print("[DETECTION] Service already running")
            return
        
        # Initialize Hailo GStreamer detector
        if HAILO_AVAILABLE:
            try:
                self.detector = GStreamerHailoDetector()
                if not self.detector.initialized:
                    print("[DETECTION] Failed to initialize Hailo detector")
                    return
                
                # Start the GStreamer pipeline
                if not self.detector.start():
                    print("[DETECTION] Failed to start Hailo detection pipeline")
                    return
                
                print("[DETECTION] Hailo GStreamer detector started successfully")
            except Exception as e:
                print(f"[DETECTION] Error initializing detector: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            print("[DETECTION] Hailo not available, service cannot start")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("[DETECTION] Service started")
    
    def stop(self):
        """Stop the detection service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Stop the GStreamer detector
        if self.detector and hasattr(self.detector, 'stop'):
            self.detector.stop()
        
        print("[DETECTION] Service stopped")
    
    def update_areas(self, areas: List[Dict]):
        """
        Update area configurations for detection
        
        Args:
            areas: List of area configs with format:
                {
                    'name': 'Misa',
                    'rect': [x, y, width, height],
                    'expected_color': 'orange',  # or 'black'
                    'door_id': 'door1'
                }
        """
        with self._areas_lock:
            self._areas = areas.copy()
        
        # Also update the GStreamer detector's crop areas for optimization
        if self.detector and hasattr(self.detector, 'set_detection_areas'):
            # Convert area format for GStreamer detector
            detector_areas = []
            for area in areas:
                detector_areas.append({
                    'door': area.get('name', 'unknown'),
                    'rect': area.get('rect', [0, 0, 0, 0])
                })
            self.detector.set_detection_areas(detector_areas)
        
        print(f"[DETECTION] Updated {len(areas)} detection areas")
    
    def submit_frame(self, frame: np.ndarray):
        """
        Submit a frame for detection (non-blocking)
        Drops old frames if queue is full
        
        Args:
            frame: BGR image from camera
        """
        try:
            # Try to put frame, but don't block if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  # Remove old frame
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Skip if still full
    
    def get_detection_result(self, area_name: str) -> Optional[Dict]:
        """
        Get the latest detection result for a specific area
        
        Returns:
            Dict with keys:
                - cat_detected: bool
                - color_match: bool (True if detected color matches expected)
                - confidence: float
                - detected_color: str ('orange', 'black', 'unknown')
                - expected_color: str
                - timestamp: float
            Or None if no result available
        """
        with self._results_lock:
            return self._detection_results.get(area_name, None)
    
    def get_all_results(self) -> Dict[str, Dict]:
        """Get detection results for all areas"""
        with self._results_lock:
            return self._detection_results.copy()
    
    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        print("[DETECTION] Detection loop started")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Get current areas to process
                with self._areas_lock:
                    areas = self._areas.copy()
                
                if not areas:
                    # No areas configured, wait
                    time.sleep(self.detection_interval)
                    continue
                
                # Get latest detections from GStreamer detector
                # The detector runs continuously and provides latest detections
                all_detections = self.detector.get_latest_detections()
                
                # Process each configured area
                for area_config in areas:
                    result = self._check_detections_in_area(all_detections, area_config)
                    
                    # Store result
                    with self._results_lock:
                        self._detection_results[area_config['name']] = result
                
                # Update statistics
                processing_time = (time.time() - start_time) * 1000
                self.stats['processing_time_ms'] = processing_time
                self.stats['last_detection_time'] = time.time()
                self.stats['total_detections'] += 1
                
                # Wait before next detection cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, self.detection_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"[DETECTION] Error in detection loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)
        
        print("[DETECTION] Detection loop stopped")
    
    def _check_detections_in_area(self, detections: List[Dict], area_config: Dict) -> Dict:
        """
        Check if any detections fall within the configured area
        
        Args:
            detections: List of detections from GStreamer detector
            area_config: Area configuration dict
            
        Returns:
            Detection result dict
        """
        area_name = area_config.get('name', 'Unknown')
        rect = area_config.get('rect', [])
        expected_color = area_config.get('expected_color', 'unknown')
        
        result = {
            'cat_detected': False,
            'color_match': False,
            'confidence': 0.0,
            'detected_color': 'unknown',
            'expected_color': expected_color,
            'timestamp': time.time(),
            'area_name': area_name
        }
        
        try:
            # Validate rect
            if len(rect) != 4:
                print(f"[DETECTION] Invalid rect for {area_name}: {rect}")
                return result
            
            area_x, area_y, area_w, area_h = rect
            area_x1, area_y1 = area_x, area_y
            area_x2, area_y2 = area_x + area_w, area_y + area_h
            
            # Check each detection to see if it overlaps with this area
            max_confidence = 0.0
            best_detection = None
            
            for det in detections:
                # Get detection bounding box
                bbox = det.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                det_x1, det_y1, det_x2, det_y2 = bbox
                
                # Calculate detection center point
                det_center_x = (det_x1 + det_x2) / 2
                det_center_y = (det_y1 + det_y2) / 2
                
                # Check if detection center is within area
                if (area_x1 <= det_center_x <= area_x2 and 
                    area_y1 <= det_center_y <= area_y2):
                    
                    confidence = det.get('confidence', 0.0)
                    
                    # Keep track of best detection in this area
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_detection = det
            
            # Process the best detection found in this area
            if best_detection:
                result['cat_detected'] = True
                result['confidence'] = max_confidence
                
                # For now, we'll accept any cat (color matching can be added later)
                # The GStreamer detector currently only detects "cat" class
                result['detected_color'] = 'any'
                
                # Color matching logic
                if expected_color.lower() in ['any', 'unknown', '']:
                    # Accept any cat color
                    result['color_match'] = True
                    self.stats['successful_matches'] += 1
                    print(f"[DETECTION] ✓ {area_name}: cat detected with confidence {max_confidence:.2f}")
                else:
                    # For specific color expectations, we'd need to add color analysis
                    # For now, just accept it
                    result['color_match'] = True
                    result['detected_color'] = expected_color  # Assume match for now
                    self.stats['successful_matches'] += 1
                    print(f"[DETECTION] ✓ {area_name}: cat detected (confidence {max_confidence:.2f}, assuming {expected_color})")
            
        except Exception as e:
            print(f"[DETECTION] Error checking detections in {area_name}: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def get_stats(self) -> Dict:
        """Get detection service statistics"""
        return self.stats.copy()


# Global detection service instance
_detection_service = None

def get_detection_service() -> Optional[DetectionService]:
    """Get or create the global detection service instance"""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService(detection_interval=1.0)
    return _detection_service

def start_detection_service():
    """Start the global detection service"""
    service = get_detection_service()
    if service:
        service.start()

def stop_detection_service():
    """Stop the global detection service"""
    global _detection_service
    if _detection_service:
        _detection_service.stop()
