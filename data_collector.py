# data_collector.py
import cv2
import time
import os
from camera import Camera # Assuming your camera.py is in the same folder

# --- Configuration ---
SAVE_PATH = "/media/pedrocastanheta/2606-4C43/cat_images"  # The folder on the USB drive to save images
MOTION_THRESHOLD = 1000  # The number of moving pixels to trigger a capture
CAPTURE_COOLDOWN_SEC = 3.0  # Wait at least 3 seconds between captures

def run_collector():
    """
    Detects motion and saves a snapshot to the specified path.
    """
    # Create the directory on the USB drive if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Ready to capture. Saving images to: {SAVE_PATH}")

    # Initialize camera and background subtractor
    cam = Camera()
    subtractor = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=75, detectShadows=False)
    
    last_capture_time = 0
    
    print("Starting motion detection... Press Ctrl+C to stop.")

    for frame in cam.frames():
        if frame is None:
            continue

        # Create a motion mask
        motion_mask = subtractor.apply(frame)
        
        # Count the number of white pixels (moving pixels)
        motion_pixels = cv2.countNonZero(motion_mask)

        # If motion is detected above the threshold
        if motion_pixels > MOTION_THRESHOLD:
            current_time = time.time()
            
            # Check if the cooldown period has passed
            if (current_time - last_capture_time) > CAPTURE_COOLDOWN_SEC:
                # Create a unique filename using a timestamp
                timestamp = int(current_time)
                filename = f"capture_{timestamp}.jpg"
                full_path = os.path.join(SAVE_PATH, filename)
                
                # Save the original color frame
                cv2.imwrite(full_path, frame)
                
                print(f"Motion detected! Saved image to {full_path}")
                
                # Update the last capture time
                last_capture_time = current_time

        # Optional: To see a live preview, uncomment the lines below
        # cv2.imshow("Live Feed", frame)
        # cv2.imshow("Motion Mask", motion_mask)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run_collector()
    except KeyboardInterrupt:
        print("\nStopping data collection.")