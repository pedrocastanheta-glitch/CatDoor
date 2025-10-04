# camera.py
import time
import numpy as np
import cv2

class Camera:
    """
    Simple camera helper:
    - Tries Picamera2 first (RGB888), then falls back to V4L2 (/dev/video0).
    - Provides capture() -> BGR ndarray, and encode_jpeg() -> (ok, bytes)
    """
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = None
        self.picam2 = None
        self.cap = None
        self._init_backend()

    # def _init_backend(self):
    #     # Try Picamera2
    #     try:
    #         from picamera2 import Picamera2
    #         self.picam2 = Picamera2()
    #         cfg = self.picam2.create_video_configuration(
    #             main={"size": (self.width, self.height), "format": "BGR888"}
    #         )
    #         self.picam2.configure(cfg)
    #         self.picam2.start()
    #         self.backend = "picamera2"
    #         return
    #     except Exception as e:
    #         # No Picamera2 or failed to start → fallback to V4L2
    #         self.picam2 = None

    #     # Fallback: V4L2 (USB or bcm2835-v4l2)
    #     self.cap = cv2.VideoCapture(0)
    #     self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
    #     self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    #     self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    #     if not self.cap.isOpened():
    #         raise RuntimeError("No camera backend available (Picamera2 failed, /dev/video0 not opened).")
    #     self.backend = "v4l2"

    # def capture(self):
    #     if self.backend == "picamera2" and self.picam2 is not None:
    #         # Picamera2 now returns BGR directly
    #         arr = self.picam2.capture_array()
    #         return arr # <-- This part is correct from your last change
        
    #     elif self.backend == "v4l2" and self.cap is not None:
    #         # This indented block was likely missing or misaligned
    #         ok, bgr = self.cap.read()
    #         if not ok:
    #             return None
    #         return bgr
        
    #     else:
    #         return None
# camera.py

    def _init_backend(self):
        # Try Picamera2
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            # 1. Request RGB888, which we know works without crashing the camera.
            cfg = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(cfg)
            self.picam2.start()
            self.backend = "picamera2"
            return
        except Exception as e:
            # No Picamera2 or failed to start → fallback to V4L2
            self.picam2 = None

        # Fallback: V4L2 (USB or bcm2835-v4l2)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError("No camera backend available (Picamera2 failed, /dev/video0 not opened).")
        self.backend = "v4l2"

# camera.py

  # camera.py

# camera.py

    # def capture(self):
    #     if self.backend == "picamera2" and self.picam2 is not None:
    #         # The error confirms the camera is sending a 3-channel image.
    #         # The color distortion points to the YCrCb color space.
    #         # This line converts YCrCb -> BGR.
    #         ycrcb_frame = self.picam2.capture_array()
    #         return cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)
    #     elif self.backend == "v4l2" and self.cap is not None:
    #         ok, bgr = self.cap.read()
    #         if not ok:
    #             return None
    #         return bgr
    #     else:
    #         return None

# camera.py

# camera.py

    def capture(self):
        if self.backend == "picamera2" and self.picam2 is not None:
            # The visual evidence proves the camera is already providing
            # the BGR frame that OpenCV needs. No conversion is necessary.
            arr = self.picam2.capture_array()
            return arr
        elif self.backend == "v4l2" and self.cap is not None:
            ok, bgr = self.cap.read()
            if not ok:
                return None
            return bgr
        else:
            return None

    def frames(self):
        while True:
            f = self.capture()
            if f is None:
                time.sleep(0.05)
                continue
            yield f

    def encode_jpeg(self, frame, quality=85):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buf = cv2.imencode(".jpg", frame, params)
        return ok, (buf.tobytes() if ok else None)
