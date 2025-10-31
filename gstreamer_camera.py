# gstreamer_camera.py
"""
Camera wrapper that uses GStreamer Hailo detector for both video and detection.
This replaces the picamera2-based Camera class to avoid camera conflicts.
"""

import cv2
import numpy as np
import time
from typing import Optional


class GStreamerCamera:
    """
    Camera interface that wraps the GStreamer Hailo detector.
    Provides the same API as the original Camera class but uses GStreamer frames.
    """
    
    def __init__(self, detector=None, width=640, height=480):
        """
        Initialize camera wrapper.
        
        Args:
            detector: GStreamerHailoDetector instance
            width: Video width
            height: Video height
        """
        self.detector = detector
        self.width = width
        self.height = height
        self._dummy_frame = None
        
        # Create a dummy frame for when detector isn't ready
        self._create_dummy_frame()
    
    def _create_dummy_frame(self):
        """Create a placeholder frame."""
        self._dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            self._dummy_frame,
            "Initializing Camera...",
            (self.width // 4, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
    
    def capture(self):
        """
        Capture a single frame.
        
        Returns:
            numpy array (BGR format) or dummy frame
        """
        if self.detector is None:
            return self._dummy_frame.copy()
        
        try:
            frame = self.detector.get_latest_frame()
            if frame is None:
                # Uncomment for debugging: print("[WARN] Detector returned None frame")
                return self._dummy_frame.copy()
            
            # Update dimensions if frame size differs (use native camera resolution)
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                self.height, self.width = frame.shape[:2]
                print(f"[CAMERA] Updated dimensions to {self.width}x{self.height}")
            
            return frame
        except Exception as e:
            print(f"[ERROR] Frame capture failed: {e}")
            # Handle errors and return dummy frame
            return self._dummy_frame.copy()
    
    def frames(self):
        """
        Generator that yields frames continuously.
        
        Yields:
            numpy arrays (BGR format)
        """
        last_frame = None
        last_frame_id = None
        frame_count = 0
        import hashlib
        import time
        
        while True:
            try:
                frame = self.capture()
                if frame is not None:
                    # Create a hash to detect if frame content actually changed
                    frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]
                    
                    # Log every 30 frames to verify frames are changing
                    if frame_count % 30 == 0:
                        print(f"[STREAM] Frame {frame_count}, hash: {frame_hash}")
                    
                    # Always yield the frame
                    yield frame
                    last_frame = frame
                    last_frame_id = frame_hash
                    frame_count += 1
                elif last_frame is not None:
                    # If we can't get a new frame, yield the last one we got
                    yield last_frame
                else:
                    # Yield dummy frame if we've never gotten a real frame
                    yield self._dummy_frame.copy()
                
                # Minimal sleep to prevent tight loop
                time.sleep(0.001)
                
            except Exception as e:
                print(f"[ERROR] Frame generation error: {e}")
                if last_frame is not None:
                    yield last_frame
                else:
                    yield self._dummy_frame.copy()
                time.sleep(0.01)
            time.sleep(0.001)  # Very small sleep
    
    def encode_jpeg(self, frame, quality=85):
        """
        Encode frame as JPEG.
        
        Args:
            frame: numpy array
            quality: JPEG quality (0-100)
            
        Returns:
            Tuple of (success, jpeg_bytes)
        """
        if frame is None:
            return False, None
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', frame, encode_param)
        
        if result:
            return True, encimg.tobytes()
        return False, None
    
    def close(self):
        """Close camera (no-op for GStreamer wrapper)."""
        pass
