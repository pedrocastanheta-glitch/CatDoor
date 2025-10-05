# data_collector.py
import cv2
import time
import os
import threading
from camera import Camera
from gpiozero import MotionSensor, LED

# --- Configuration ---
SAVE_PATH = "/media/pedrocastanheta/2606-4C43/cat_images"
MOTION_THRESHOLD = 1000
CAPTURE_COOLDOWN_SEC = 3.0

# --- GPIO Configuration (from app.py) ---
PIR_PIN_1 = 22  # <-- MODIFIED: First PIR sensor
# PIR_PIN_2 = 10  # <-- NEW: Second PIR sensor
LED_PIN = 27
LED_HOLD_SECONDS = 30

# --- Global timer for the LED ---
_pir_off_timer = None

def run_collector():
    """
    Detects motion and saves a snapshot, with PIR-controlled lighting.
    """
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Ready to capture. Saving images to: {SAVE_PATH}")

    # --- Initialize GPIO devices ---
    led = None
    try:
        led = LED(LED_PIN)
        pir1 = MotionSensor(PIR_PIN_1) # <-- MODIFIED: Initialize first sensor
        # pir2 = MotionSensor(PIR_PIN_2) # <-- NEW: Initialize second sensor
        print("PIR sensors and LED initialized.")

        def handle_motion():
            """Turns on LED and sets a timer to turn it off."""
            global _pir_off_timer
            if not led.is_lit:
                print("PIR Motion Detected -> LED ON")
                led.on()
            
            if _pir_off_timer:
                _pir_off_timer.cancel()
            
            _pir_off_timer = threading.Timer(LED_HOLD_SECONDS, lambda: (print("LED Timeout -> LED OFF"), led.off()))
            _pir_off_timer.daemon = True
            _pir_off_timer.start()

        # Assign the same function to both PIR sensors' motion events
        pir1.when_motion = handle_motion # <-- MODIFIED
        # pir2.when_motion = handle_motion # <-- NEW

    except Exception as e:
        print(f"Could not initialize GPIO. Lighting will be disabled. Error: {e}")

    # Initialize camera and background subtractor
    cam = Camera()
    subtractor = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=75, detectShadows=False)
    
    last_capture_time = 0
    
    print("Starting visual motion detection... Press Ctrl+C to stop.")

    try:
        for frame in cam.frames():
            if frame is None:
                continue

            motion_mask = subtractor.apply(frame)
            motion_pixels = cv2.countNonZero(motion_mask)

            if motion_pixels > MOTION_THRESHOLD:
                current_time = time.time()
                
                if (current_time - last_capture_time) > CAPTURE_COOLDOWN_SEC:
                    timestamp = int(current_time)
                    filename = f"capture_{timestamp}.jpg"
                    full_path = os.path.join(SAVE_PATH, filename)
                    
                    cv2.imwrite(full_path, frame)
                    
                    print(f"Visual Motion Detected! Saved image to {full_path}")
                    
                    last_capture_time = current_time
    finally:
        if led:
            led.off()
            print("LED turned off during exit.")


if __name__ == "__main__":
    try:
        run_collector()
    except KeyboardInterrupt:
        print("\nStopping data collection.")