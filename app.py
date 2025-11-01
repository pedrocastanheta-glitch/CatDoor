# app.py
import os, json, time, threading
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from functools import wraps
import base64
# Attempt to import gpiozero; fall back if unavailable
try:
    from gpiozero import MotionSensor, LED
    GPIOZERO_AVAILABLE = True
except Exception as e:
    MotionSensor = None
    LED = None
    GPIOZERO_AVAILABLE = False
    print(f"[WARN] gpiozero not available: {e}. PIR/LED feature will be disabled.")
from signal import pause
import servo_control as servo
import numpy as np
import cv2

# Camera and Detection - Single unified GStreamer pipeline
cam = None
detector = None
DETECTION_AVAILABLE = False

try:
    from gstreamer_hailo_detector import GStreamerHailoDetector
    
    # Create ONE detector that owns the camera
    detector = GStreamerHailoDetector(
        hef_path="/usr/share/hailo-models/yolov8s_h8l.hef",
        input_source="rpi"  # Use Raspberry Pi camera
    )
    
    # Start the GStreamer pipeline
    if detector.start():
        DETECTION_AVAILABLE = True
        print("[INFO] GStreamer Hailo detector started - single unified pipeline")
    else:
        detector = None
        print("[ERROR] Failed to start GStreamer Hailo detector")
        
except Exception as e:
    detector = None
    print(f"[ERROR] GStreamer Hailo detector initialization failed: {e}")
    import traceback
    traceback.print_exc()

# No separate detection service needed - detector handles everything
detection_service = None

# --- CONFIGURATION ---
CONFIG_FILE = "areas_config.json"
STATE_FILE = "door_state.json"
AUTH_FILE = "remote_auth.json"
EVENT_LOG_FILE = "event_log.json"
MAX_LOG_ENTRIES = 1000
DETECT_INTERVAL_SEC = 0.2
OPEN_COOLDOWN_SEC = 4.0
CLOSE_DELAY_SEC = 5.0
STARTUP_GRACE_PERIOD_SEC = 5.0

# --- Remote Access Authentication ---
REMOTE_AUTH_ENABLED = os.environ.get('CATDOOR_REMOTE_AUTH', 'true').lower() == 'true'
REMOTE_USERNAME = os.environ.get('CATDOOR_USERNAME', 'catdoor')
REMOTE_PASSWORD = os.environ.get('CATDOOR_PASSWORD', 'meow2024')

def check_auth(username, password):
    """Check if username/password is valid"""
    return username == REMOTE_USERNAME and password == REMOTE_PASSWORD

def requires_auth(f):
    """Decorator for routes that require authentication when remote access is enabled"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Skip auth if not enabled or accessing locally
        if not REMOTE_AUTH_ENABLED or request.remote_addr in ['127.0.0.1', 'localhost', '::1']:
            return f(*args, **kwargs)
        
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response(
                'Authentication required for remote access\n'
                'Default: username=catdoor, password=meow2024', 
                401,
                {'WWW-Authenticate': 'Basic realm="Cat Door Remote Access"'}
            )
        return f(*args, **kwargs)
    return decorated

# --- PIR LED constants ---
PIR_PIN_1 = 22
PIR_PIN_2 = 10
LED_PIN = 27
LED_HOLD_SECONDS = 120

# --- Flask & Camera ---
app = Flask(__name__, template_folder="templates", static_folder="static")

# Disable Flask's default request logging
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- STATE VARIABLES ---
_config_lock = threading.Lock()
_state_lock = threading.Lock()  # Add dedicated state lock
_door_operation_lock = threading.Lock()  # Prevent simultaneous door operations
_event_log_lock = threading.Lock()  # Lock for event log access
_last_open_ts = {}
_last_detection_ts = {}
_manual_override = {}
_door_state = {}  # Tracks both state and mode for each door
_startup_time = time.time()  # Track when the system started
_pir_status = {"PIR1": {"active": False, "last_motion": 0}, "PIR2": {"active": False, "last_motion": 0}}
_pir_timers = {"PIR1": None, "PIR2": None}
_pending_door_actions = {}  # Track pending door actions to prevent duplicate triggers

# --- EVENT LOGGING ---
def log_event(event_type, message, area=None, cat_color=None, door_id=None):
    """Log an event to the event log file with timestamp"""
    try:
        with _event_log_lock:
            # Load existing log
            events = []
            if os.path.exists(EVENT_LOG_FILE):
                try:
                    with open(EVENT_LOG_FILE, "r") as f:
                        events = json.load(f)
                except:
                    events = []
            
            # Create new event entry
            event = {
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "type": event_type,  # 'cat_detected', 'door_action', 'manual_action', 'color_mismatch'
                "message": message,
                "area": area,
                "cat_color": cat_color,
                "door_id": door_id
            }
            
            # Add to beginning of list (newest first)
            events.insert(0, event)
            
            # Keep only last MAX_LOG_ENTRIES entries
            events = events[:MAX_LOG_ENTRIES]
            
            # Save back to file
            with open(EVENT_LOG_FILE, "w") as f:
                json.dump(events, f, indent=2)
                
    except Exception as e:
        print(f"[ERROR] Failed to log event: {e}")

def get_recent_events(count=20):
    """Get the most recent events from the log"""
    try:
        with _event_log_lock:
            if os.path.exists(EVENT_LOG_FILE):
                with open(EVENT_LOG_FILE, "r") as f:
                    events = json.load(f)
                    return events[:count]
            return []
    except Exception as e:
        print(f"[ERROR] Failed to read event log: {e}")
        return []

def load_door_state():
    """Load door state and synchronize with actual physical switch positions"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                loaded_state = json.load(f)
                # Synchronize with actual physical switch positions
                try:
                    # Get all switch statuses
                    switch_statuses = servo.get_switch_status()
                    
                    for door_id in loaded_state.keys():
                        # Get actual switch status (switch pressed = door closed)
                        switch_pressed = switch_statuses.get(door_id, {}).get("is_pressed", True)
                        actual_state = "close" if switch_pressed else "open"
                        
                        # Update state to match physical reality
                        loaded_state[door_id]["state"] = actual_state
                        
                        # Set proper mode: closed doors should be automatic, open doors manual
                        if actual_state == "close":
                            loaded_state[door_id]["last_mode"] = "automatic"
                        else:
                            loaded_state[door_id]["last_mode"] = "manual"
                        
                        print(f"[STARTUP] Door {door_id}: physical={actual_state}, mode={loaded_state[door_id]['last_mode']}")
                    
                    return loaded_state
                except Exception as e:
                    print(f"[STARTUP] Error syncing switch status: {e}")
                    # Fall back to loaded state if switch reading fails
                    return loaded_state
        except Exception:
            pass
    
    # Default: both doors closed and automatic (ready for detection)
    default_state = {k: {"state": "close", "last_mode": "automatic"} for k in servo.DOORS.keys()}
    print("[STARTUP] Using default door state")
    return default_state

def save_door_state():
    with _state_lock:
        with open(STATE_FILE, "w") as f:
            json.dump(_door_state, f)
_pir_off_timer = None
_last_snapshot_bgr = None # For storing the snapshot for HSV sampling

# --- CONFIG LOADING ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f: return json.load(f)
        except Exception: pass
    return {"areas": []}

def save_config(cfg):
    with open(CONFIG_FILE + ".tmp", "w") as f: json.dump(cfg, f, indent=2)
    os.replace(CONFIG_FILE + ".tmp", CONFIG_FILE)

config = load_config()

# Configure detector with detection areas
if detector and DETECTION_AVAILABLE and config.get("areas"):
    try:
        detector.set_detection_areas(config["areas"])
        print(f"[INFO] Configured {len(config['areas'])} detection areas in Hailo detector")
    except Exception as e:
        print(f"[WARN] Failed to configure detection areas: {e}")
else:
    print("[INFO] No detection areas configured or detector unavailable")

# --- HELPERS ---
def read_pi_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except Exception: return None

# --- VIDEO STREAMING ---
def gen_frames(area_name=None):
    """
    Video streaming from unified GStreamer/Hailo detector.
    The detector owns the camera and provides annotated frames.
    
    Args:
        area_name: If provided, crop to this specific area. If None, show full frame with all areas.
    """
    global config, detector
    frame_count = 0
    last_frame_time = 0
    target_fps = 20  # Target FPS for video stream
    frame_interval = 1.0 / target_fps
    
    # Check if detector is available
    if detector is None or not DETECTION_AVAILABLE:
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ERROR: Detector not available", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1.0)
        return
    
    # If cropping to specific area, get the area config
    crop_rect = None
    if area_name:
        with _config_lock:
            areas = config.get("areas", [])
            for area in areas:
                if area.get("name") == area_name:
                    rect = area.get("rect", [0, 0, 0, 0])
                    x, y, w, h = [int(v) for v in rect]
                    if w > 0 and h > 0:
                        crop_rect = (x, y, w, h)
                    break
        
        if crop_rect is None:
            # Area not found, return error frame
            error_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Area '{area_name}' not found", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            _, buffer = cv2.imencode('.jpg', error_frame)
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1.0)
            return
    
    while True:
        # Rate limit the stream
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
            continue
        
        last_frame_time = current_time
        frame_count += 1
        if frame_count % 100 == 0:  # Less frequent logging
            stream_type = f"area '{area_name}'" if area_name else "full"
            print(f"[STREAM] {stream_type}: Frame {frame_count} at ~{1.0/frame_interval:.1f} FPS")
        
        # Get frame from detector - it's already annotated with bounding boxes
        frame = None
        try:
            frame = detector.get_latest_frame()
        except Exception as e:
            if frame_count <= 5:
                print(f"[ERROR] Failed to get frame from detector: {e}")
        
        # Skip if no frame available
        if frame is None:
            time.sleep(0.1)
            continue
        
        # If cropping to specific area, crop now
        if crop_rect:
            x, y, w, h = crop_rect
            # Ensure crop is within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)
            
            frame = frame[y:y+h, x:x+w]
            
            # Draw area name on cropped frame
            cv2.putText(frame, area_name, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Full frame mode - draw configured areas as rectangles
            with _config_lock:
                areas = config.get("areas", [])
            
            for area in areas:
                rect = area.get("rect", [0, 0, 0, 0])
                x, y, w, h = [int(v) for v in rect]
                
                if w <= 0 or h <= 0:
                    continue
                
                area_name_full = area.get("name", "Area")
                color = (255, 0, 0)  # Blue rectangles for areas
                    
                # Draw area rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw area label
                cv2.putText(frame, area_name_full, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Door action threading helper ---
def threaded_door_action(action, door_id, set_mode=None):
    """
    Unified door action function that uses proper synchronization to prevent race conditions.
    """
    def worker():
        # Use door operation lock to prevent simultaneous operations on the same door
        with _door_operation_lock:
            try:
                # Use the same request_door_action path to prevent race conditions
                if servo.request_door_action(door_id, action):
                    # Wait for servo operation to complete
                    time.sleep(2)
                    
                    # Get the actual state from servo system
                    actual_state = servo.get_door_status(door_id)
                    
                    with _state_lock:
                        _door_state[door_id]["state"] = actual_state
                        if set_mode:
                            _door_state[door_id]["last_mode"] = set_mode
                    
                    # Force save the state immediately
                    save_door_state()
                    print(f"[AUTO] Door '{door_id}' {action} completed, state: {actual_state}, mode: {set_mode or 'unchanged'}")
                else:
                    print(f"[AUTO] Door '{door_id}' {action} request ignored - operation in progress")
            except Exception as e:
                print(f"[ERROR] Door action {action} failed for {door_id}: {e}")
            finally:
                # Clear the pending action flag
                _pending_door_actions.pop(door_id, None)
    
    t = threading.Thread(target=worker, daemon=True)
    t.start()

# --- CORE DETECTION LOGIC ---
def detection_loop():
    """
    Core detection logic using the unified GStreamer/Hailo detector.
    Gets detections directly from the detector that owns the camera.
    """
    global config, _last_open_ts, _last_detection_ts, _manual_override, _door_state
    
    while True:
        time.sleep(DETECT_INTERVAL_SEC)
        with _config_lock:
            areas = list(config.get("areas", []))
        if not areas: 
            continue

        now = time.time()
        door_detection_states = {}

        # Initialize detection states for all doors
        for area in areas:
            door_id = area.get("door_id", "door1")
            if door_id not in door_detection_states:
                door_detection_states[door_id] = {
                    "cat_detected": False,
                    "detected_cat": None,
                    "detected_color": None,
                    "color_ratio": 0.0,
                }
            
            # Skip if manual override
            with _config_lock:
                if _manual_override.get(door_id, False):
                    door_detection_states[door_id]["manual"] = True
                    continue

        # Get detections directly from the detector
        if detector and DETECTION_AVAILABLE:
            try:
                # Get all detections from the unified pipeline
                detections = detector.get_latest_detections()
                frame_for_color = None
                
                # Check each detection against configured areas
                for det in detections:
                    det_x1, det_y1, det_x2, det_y2 = det['bbox']
                    det_center_x = (det_x1 + det_x2) / 2
                    det_center_y = (det_y1 + det_y2) / 2
                    
                    # Skip detections that are not cats; prevents people triggering the door
                    if det.get('class_name', '').lower() != 'cat':
                        continue

                    # Find which area(s) this detection overlaps with
                    for area in areas:
                        rect = area.get("rect", [0, 0, 0, 0])
                        x, y, w, h = [int(v) for v in rect]
                        
                        if w <= 0 or h <= 0:
                            continue
                        
                        # Check if detection center is in this area
                        if x <= det_center_x <= x + w and y <= det_center_y <= y + h:
                            door_id = area.get("door_id", "door1")
                            area_name = area.get("name", "Unknown")
                            matched_profile = None
                            match_ratio = None
                            profiles = _get_area_color_profiles(area)

                            if profiles:
                                if frame_for_color is None:
                                    frame_for_color = detector.get_latest_frame()
                                color_ok, match_ratio, matched_profile = detection_matches_color(frame_for_color, det, area)
                                if not color_ok:
                                    expected_names = ", ".join(p.get("name", "unknown") for p in profiles)
                                    ratio_text = f"{match_ratio:.2f}" if match_ratio is not None else "0.00"
                                    print(f"[COLOR] Rejecting detection in area '{area_name}': ratio={ratio_text}, expected={expected_names}")
                                    log_event(
                                        'color_mismatch',
                                        f"Color mismatch in {area_name} (expected {expected_names})",
                                        area=area_name,
                                        cat_color=matched_profile or 'unknown',
                                        door_id=door_id
                                    )
                                    continue
                                if matched_profile and match_ratio is not None:
                                    print(f"[COLOR] Matched profile '{matched_profile}' in area '{area_name}' (ratio={match_ratio:.2f})")

                            # Cat detected in this area!
                            detected_cat_name = f"{det['class_name'].title()} ({det['confidence']:.2f})"
                            if matched_profile:
                                detected_cat_name += f" [{matched_profile}]"

                            door_detection_states[door_id]["cat_detected"] = True
                            door_detection_states[door_id]["detected_cat"] = detected_cat_name
                            door_detection_states[door_id]["detected_color"] = matched_profile
                            if match_ratio is not None:
                                door_detection_states[door_id]["color_ratio"] = match_ratio
                            
                            print(f"[AI_DETECT] {detected_cat_name} detected in area '{area_name}'")
                            
                            # Log the detection
                            log_event(
                                'cat_detected',
                                f"Cat observed in {area_name} area (Door can open)",
                                area=area_name,
                                cat_color=matched_profile,
                                door_id=door_id
                            )
                            
            except Exception as e:
                print(f"[ERROR] Detection query failed: {e}")
        else:
            # No detector available
            continue


        # Process detection results
        for door_id, state in door_detection_states.items():
            # Skip detection if door is in manual override mode
            if state.get("manual", False) or _manual_override.get(door_id, False): 
                continue

            # Check startup grace period to prevent unwanted actions after boot
            if now - _startup_time < STARTUP_GRACE_PERIOD_SEC:
                remaining = int(STARTUP_GRACE_PERIOD_SEC - (now - _startup_time))
                print(f"[STARTUP] Grace period: {remaining}s remaining - detection active but door actions disabled")
                continue

            cat_detected = state.get("cat_detected", False)
            detected_cat = state.get("detected_cat", None)
            is_forbidden = state.get("forbidden", False)

            if is_forbidden and not cat_detected:
                if _door_state[door_id]["state"] == "open":
                    print(f"[DETECT] Door '{door_id}' FORBIDDEN match -> close NOW")
                    threaded_door_action("close", door_id, "automatic")
                    _last_detection_ts.pop(door_id, None)
            elif cat_detected:
                last_open = _last_open_ts.get(door_id, 0)
                
                # Don't auto-open if door was manually closed (respect user intent)
                if (_door_state[door_id]["state"] == "close" and 
                    _door_state[door_id].get("last_mode") == "manual"):
                    print(f"[DETECT] Door '{door_id}' manually closed - skipping auto-open for {detected_cat}")
                    _last_detection_ts[door_id] = now  # Update detection time but don't open
                    continue
                
                # Check if an open action is already pending to prevent duplicate triggers
                if _pending_door_actions.get(door_id) != "open":
                    if now - last_open >= OPEN_COOLDOWN_SEC:
                        if _door_state[door_id]["state"] == "close":
                            print(f"[DETECT] Door '{door_id}' detected {detected_cat} -> open")
                            _pending_door_actions[door_id] = "open"
                            threaded_door_action("open", door_id, "automatic")
                            _last_open_ts[door_id] = now
                            # Log automatic door opening
                            log_event(
                                'door_action',
                                f"Door {door_id} opened automatically ({detected_cat})",
                                door_id=door_id
                            )
                _last_detection_ts[door_id] = now
            else:
                last_seen = _last_detection_ts.get(door_id, 0)
                
                # Don't auto-close if door was manually opened (respect user intent)
                if (_door_state[door_id]["state"] == "open" and 
                    _door_state[door_id].get("last_mode") == "manual"):
                    print(f"[DETECT] Door '{door_id}' manually opened - skipping auto-close")
                    continue
                
                # Check if a close action is already pending to prevent duplicate triggers
                if _pending_door_actions.get(door_id) != "close":
                    if _door_state[door_id]["state"] == "open" and now - last_seen > CLOSE_DELAY_SEC:
                        print(f"[DETECT] Door '{door_id}' not seen for {CLOSE_DELAY_SEC}s -> close")
                        _pending_door_actions[door_id] = "close"
                        threaded_door_action("close", door_id, "automatic")
                        _last_detection_ts.pop(door_id, None)
                        _last_open_ts.pop(door_id, None)
                        # Log automatic door closing
                        log_event(
                            'door_action',
                            f"Door {door_id} closed automatically (no cat detected for {CLOSE_DELAY_SEC}s)",
                            door_id=door_id
                        )

# --- PIR LED WORKER ---
def pir_led_manager():
    global _pir_off_timer
    if not GPIOZERO_AVAILABLE:
        print("[PIR_LED] gpiozero unavailable, PIR manager disabled.")
        return
        
    try:
        led = LED(LED_PIN)
        pir1 = MotionSensor(PIR_PIN_1)
        pir2 = MotionSensor(PIR_PIN_2)
        print("PIR sensors and LED initialized.")

        def handle_motion(sensor_name):
            global _pir_off_timer, _pir_status, _pir_timers
            try:
                # Determine which PIR sensor triggered
                pir_key = "PIR1" if "PIR 1" in sensor_name else "PIR2"
                
                if not led.is_lit:
                    print(f"PIR Motion Detected by {sensor_name} -> LED ON")
                    led.on()
                
                # Update PIR status immediately
                _pir_status[pir_key]["active"] = True
                _pir_status[pir_key]["last_motion"] = time.time()
                print(f"[PIR] {pir_key} status set to ACTIVE - timer reset")
                
                # Cancel existing timer for this PIR
                if _pir_timers[pir_key]:
                    _pir_timers[pir_key].cancel()
                
                # Set new timer to reset PIR status after 5 seconds
                def reset_pir_status():
                    _pir_status[pir_key]["active"] = False
                    _pir_timers[pir_key] = None
                    print(f"[PIR] {pir_key} status set to INACTIVE after 5 seconds")
                
                _pir_timers[pir_key] = threading.Timer(5.0, reset_pir_status)
                _pir_timers[pir_key].daemon = True
                _pir_timers[pir_key].start()
                
                # Handle LED timer (keep existing logic)
                if _pir_off_timer:
                    _pir_off_timer.cancel()
                _pir_off_timer = threading.Timer(LED_HOLD_SECONDS, lambda: (print("LED Timeout -> LED OFF"), led.off()))
                _pir_off_timer.daemon = True
                _pir_off_timer.start()
                
            except Exception as e:
                print(f"[ERROR] PIR motion handling failed: {e}")

        pir1.when_motion = lambda: handle_motion(f"PIR 1 (Pin {PIR_PIN_1})")
        pir2.when_motion = lambda: handle_motion(f"PIR 2 (Pin {PIR_PIN_2})")

        print("[PIR_LED] PIR manager started.")
        try:
            pause()
        except:
            pass
        finally:
            if 'led' in locals():
                led.off()
            print("[PIR_LED] PIR manager stopped.")

    except Exception as e:
        print(f"[PIR_LED] Failed to initialize GPIO: {e}. Feature disabled.")
        return

# --- HELPER HSV Function ---
def hsv_in_range(hsv, lo, hi):
    lo = np.array(lo, dtype=np.uint8); hi = np.array(hi, dtype=np.uint8)
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    else:
        mask1 = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]]), np.array([179, hi[1], hi[2]]))
        mask2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), np.array([hi[0], hi[1], hi[2]]))
        return cv2.bitwise_or(mask1, mask2)


def _get_area_color_profiles(area):
    profiles = area.get("color_profiles")
    if isinstance(profiles, dict):
        profiles = [profiles]
    if not profiles:
        single = area.get("color_profile")
        if single:
            profiles = [single]
    return profiles or []


def detection_matches_color(frame, detection, area):
    profiles = _get_area_color_profiles(area)
    if not profiles:
        return True, None, None
    if frame is None:
        area_name = area.get("name", "Unknown")
        print(f"[COLOR] Frame unavailable; skipping color check for area '{area_name}'")
        return True, None, None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in detection.get('bbox', [0, 0, 0, 0])]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return False, 0.0, None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0, None

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    best_ratio = 0.0
    best_profile = None
    for profile in profiles:
        lo = profile.get("hsv_lo")
        hi = profile.get("hsv_hi")
        if lo is None or hi is None:
            continue

        mask = hsv_in_range(roi_hsv, lo, hi)
        ratio = float(cv2.countNonZero(mask)) / float(mask.size)
        min_ratio = float(profile.get("min_ratio", 0.12))

        if ratio >= min_ratio:
            return True, ratio, profile.get("name", "unknown")

        if ratio > best_ratio:
            best_ratio = ratio
            best_profile = profile.get("name", "unknown")

    return False, best_ratio, best_profile

# --- FLASK API ROUTES ---
@app.route("/")
@requires_auth
def index():
    from flask import request
    
    # Check if user is accessing locally (for shutdown button visibility)
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
    
    # Allow only local access (localhost, 127.0.0.1, or local Pi IP)
    local_ips = ['127.0.0.1', 'localhost', '::1']
    
    # Get Pi's local IP to allow direct Pi access
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        local_ips.append(local_ip)
    except:
        pass
    
    # Check if client is accessing from the Pi itself
    is_local_access = any(client_ip.startswith(ip) for ip in local_ips) or client_ip in local_ips
    
    return render_template("index.html", local_access=is_local_access, client_ip=client_ip)

@app.route('/video_feed')
@requires_auth
def video_feed():
    """Video streaming route - full camera view with all areas."""
    response = Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/video_feed/<area_name>')
@requires_auth
def video_feed_area(area_name):
    """Video streaming route for a specific area - shows only that cropped area."""
    response = Response(gen_frames(area_name=area_name), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/setup', methods=['GET', 'POST'])
@requires_auth
def setup_page():
    if request.method == 'POST':
        # Handle configuration update
        try:
            new_config = request.get_json()
            with _config_lock:
                config.update(new_config)
                save_config(config)
                
                # Update detector areas if config changed
                if detector and DETECTION_AVAILABLE and 'areas' in new_config:
                    detector.set_detection_areas(config["areas"])
                    print(f"[CONFIG] Updated {len(config['areas'])} detection areas in Hailo detector")
                    
            print(f"[CONFIG] Updated config: {new_config}")
        except Exception as e:
            print(f"[ERROR] Failed to update config: {e}")
            
        return redirect(url_for('setup_page'))

    return render_template("setup.html")

@app.route('/get_areas_config')
def get_areas_config_json():
    with _config_lock:
        return jsonify(config.get("areas", []))

@app.route("/api/temp")
@requires_auth
def api_temp():
    return jsonify({"temp_c": read_pi_temp()})

@app.route("/api/door_status")
@requires_auth
def api_door_status():
    with _state_lock:
        # Sync states from servo control module
        for door_id in servo.DOORS:
            servo_state = servo.get_door_status(door_id)
            if door_id in _door_state:
                _door_state[door_id]["state"] = servo_state
        return jsonify(_door_state)

@app.route("/api/override_status")
@requires_auth
def api_override_status(): 
    with _config_lock: 
        return jsonify(_manual_override)

@app.route("/api/config", methods=["GET", "POST"])
@requires_auth
def api_config():
    global config
    if request.method == "POST":
        with _config_lock:
            config = request.get_json()
            save_config(config)
        return jsonify({"ok": True})
    
    with _config_lock:
        return jsonify(config)

@app.route("/api/door/<door_id>/<action>", methods=["POST"])
@requires_auth
def api_door(door_id, action):
    global _manual_override
    if door_id not in servo.DOORS: return jsonify({"ok": False, "error": f"Unknown door {door_id}"}), 400
    
    # Check current door state (with minimal locking)
    with _state_lock:
        current_state = _door_state[door_id]["state"]
    
    if action == "open":
        if current_state == "open":
            print(f"[MANUAL] Door '{door_id}' is already open. No action needed.")
            return jsonify({"ok": True, "status": current_state, "message": "Door already open"})
        
        # Initiate door open action (don't hold locks during operation)
        if servo.request_door_action(door_id, "open"):
            _pending_door_actions[door_id] = "open"
            with _state_lock:
                _manual_override[door_id] = True  # Set manual override to disable detection
                _door_state[door_id]["last_mode"] = "manual"
            save_door_state()
            print(f"[MANUAL] Door '{door_id}' manually opening - detection disabled.")
            # Log manual door opening
            log_event(
                'manual_action',
                f"Door {door_id} opened manually",
                door_id=door_id
            )
            
            # Wait for servo operation to complete WITHOUT holding locks
            time.sleep(6)
            
            with _state_lock:
                _door_state[door_id]["state"] = servo.get_door_status(door_id)
            save_door_state()
            print(f"[MANUAL] Door '{door_id}' open operation completed, state: {_door_state[door_id]['state']}")
            _pending_door_actions.pop(door_id, None)
        else:
            print(f"[MANUAL] Open request for '{door_id}' ignored - already opening.")
            return jsonify({"ok": False, "error": "Door already opening"}), 400
    
    elif action == "close":
        if current_state == "close":
            # If door is already closed, ensure it's in automatic mode for detection
            with _state_lock:
                _manual_override[door_id] = False
                _door_state[door_id]["last_mode"] = "automatic"
            save_door_state()
            print(f"[MANUAL] Door '{door_id}' is already closed - set to automatic mode.")
            _pending_door_actions.pop(door_id, None)
            return jsonify({"ok": True, "status": current_state, "message": "Door already closed, automatic mode enabled"})
        
        # Initiate door close action (don't hold locks during operation)
        if servo.request_door_action(door_id, "close"):
            _pending_door_actions[door_id] = "close"
            with _state_lock:
                _manual_override[door_id] = False
                _door_state[door_id]["last_mode"] = "automatic"
            save_door_state()
            print(f"[MANUAL] Door '{door_id}' manually closing - automatic detection enabled.")
            # Log manual door closing
            log_event(
                'manual_action',
                f"Door {door_id} closed manually",
                door_id=door_id
            )
            
            # Wait for servo operation to complete WITHOUT holding locks
            time.sleep(8)
            
            with _state_lock:
                _door_state[door_id]["state"] = servo.get_door_status(door_id)
            save_door_state()
            print(f"[MANUAL] Door '{door_id}' close operation completed, state: {_door_state[door_id]['state']}")
            _pending_door_actions.pop(door_id, None)
        else:
            print(f"[MANUAL] Close request for '{door_id}' ignored - already closing.")
            return jsonify({"ok": False, "error": "Door already closing"}), 400
    
    return jsonify({"ok": True, "status": servo.get_door_status(door_id)})

@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    global _last_snapshot_bgr, detector
    # Get frame from detector
    frame = None
    if detector and DETECTION_AVAILABLE:
        try:
            frame = detector.get_latest_frame()
        except Exception as e:
            print(f"[ERROR] Failed to capture snapshot: {e}")
    
    if frame is None: 
        return jsonify({"ok": False, "error": "No frame available from detector"}), 500
    
    # Store for HSV sampling
    _last_snapshot_bgr = frame.copy()
    
    # Encode to JPEG
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret: 
        return jsonify({"ok": False, "error": "encode failed"}), 500
    
    return Response(buffer.tobytes(), mimetype="image/jpeg")

@app.route("/api/hsv_from_rect", methods=["POST"])
def api_hsv_from_rect():
    global _last_snapshot_bgr
    rect = request.get_json().get("rect")
    frame = _last_snapshot_bgr
    if frame is None: return jsonify({"ok": False, "error": "No snapshot taken yet"}), 400
    x, y, w, h = [int(v) for v in rect]
    roi_bgr = frame[y:y+h, x:x+w]
    if roi_bgr.size == 0: return jsonify({"ok": False, "error": "Crop is empty"}), 400
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(roi_hsv)
    h_lo, h_hi = int(np.percentile(h, 5)), int(np.percentile(h, 95))
    s_lo, s_hi = int(np.percentile(s, 10)), int(np.percentile(s, 90))
    v_lo, v_hi = int(np.percentile(v, 10)), int(np.percentile(v, 90))
    return jsonify({"ok": True, "hsv_lo": [h_lo, max(s_lo, 30), max(v_lo, 30)], "hsv_hi": [h_hi, s_hi, 255]})

@app.route("/api/video_info")
def api_video_info():
    # Default resolution for Raspberry Pi camera
    return jsonify({"width": 1280, "height": 720})

@app.route("/api/areas")
@requires_auth
def api_areas():
    """Return list of configured detection areas"""
    with _config_lock:
        areas = config.get("areas", [])
    # Return simplified area info for frontend
    area_list = []
    for area in areas:
        area_list.append({
            "name": area.get("name", "Unknown"),
            "door_id": area.get("door_id", "door1"),
            "expected_color": area.get("expected_color", "any")
        })
    return jsonify({"areas": area_list})

@app.route("/api/switch_status")
def api_switch_status():
    return jsonify(servo.get_switch_status())

@app.route("/api/pir_status")
def api_pir_status():
    global _pir_status
    return jsonify({
        "PIR1": {
            "active": _pir_status["PIR1"]["active"],
            "status": "Yes" if _pir_status["PIR1"]["active"] else "No"
        },
        "PIR2": {
            "active": _pir_status["PIR2"]["active"], 
            "status": "Yes" if _pir_status["PIR2"]["active"] else "No"
        }
    })

@app.route("/api/detection_status")
def api_detection_status():
    """Get current AI detection status for all areas"""
    if not detection_service or not DETECTION_AVAILABLE:
        return jsonify({
            "enabled": False,
            "message": "Detection service not available"
        })
    
    results = detection_service.get_all_results()
    stats = detection_service.get_stats()
    
    return jsonify({
        "enabled": True,
        "results": results,
        "stats": stats,
        "service_running": detection_service.running
    })

@app.route("/api/event_log")
def api_event_log():
    """Get recent events from the log"""
    count = request.args.get('count', 20, type=int)
    count = min(count, 100)  # Cap at 100 entries
    events = get_recent_events(count)
    return jsonify({
        "ok": True,
        "events": events,
        "count": len(events)
    })

@app.route("/api/shutdown", methods=["POST"])
@requires_auth
def api_shutdown():
    """Safely shutdown the Raspberry Pi - only available for local access"""
    import subprocess
    import time
    from flask import request
    
    # Check if request is from localhost/local network
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
    
    # Allow only local access (localhost, 127.0.0.1, or local Pi IP)
    local_ips = ['127.0.0.1', 'localhost', '::1']
    
    # Get Pi's local IP to allow direct Pi access
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        local_ips.append(local_ip)
    except:
        pass
    
    # Check if client is accessing from the Pi itself
    is_local_access = any(client_ip.startswith(ip) for ip in local_ips) or client_ip in local_ips
    
    if not is_local_access:
        print(f"[SHUTDOWN] Denied remote shutdown request from {client_ip}")
        return jsonify({
            "ok": False,
            "error": "Shutdown is only available when accessing from the Raspberry Pi directly"
        }), 403
    
    try:
        # Log the shutdown request
        print(f"[SHUTDOWN] Local shutdown requested from {client_ip}")
        
        # Give a small delay to send response
        def delayed_shutdown():
            time.sleep(2)
            print("[SHUTDOWN] Executing shutdown command...")
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
        
        # Start shutdown in background thread
        import threading
        threading.Thread(target=delayed_shutdown, daemon=True).start()
        
        return jsonify({
            "ok": True, 
            "message": "Shutdown initiated. The Raspberry Pi will power off in 2 seconds."
        })
        
    except Exception as e:
        print(f"[SHUTDOWN] Error: {e}")
        return jsonify({
            "ok": False, 
            "error": f"Shutdown failed: {str(e)}"
        }), 500

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    _door_state = load_door_state()
    
    # Initialize door states and sync with servo control module
    for door_id in servo.DOORS:
        _manual_override[door_id] = False
        _pending_door_actions[door_id] = None  # Initialize pending actions
        if door_id not in _door_state:
            _door_state[door_id] = {"state": "open", "last_mode": "manual"}
        elif "last_mode" not in _door_state[door_id]:
            _door_state[door_id]["last_mode"] = "manual"
    
    # Sync states between app.py and servo_control.py
    print("[INIT] Synchronizing door states on startup...")
    for door_id in servo.DOORS:
        app_state = _door_state[door_id]["state"]
        servo_state = servo.get_door_status(door_id)
        
        if app_state != servo_state:
            print(f"[INIT] State mismatch for {door_id}: app={app_state}, servo={servo_state}")
            # Update app state to match servo reality
            _door_state[door_id]["state"] = servo_state
            print(f"[INIT] Updated app state for {door_id} to {servo_state}")
        else:
            print(f"[INIT] Door {door_id} state synchronized: {app_state}")
        
        # Initialize last detection time for open doors to prevent immediate closure on startup
        if _door_state[door_id]["state"] == "open":
            _last_detection_ts[door_id] = time.time()
            print(f"[INIT] Door {door_id} is open - initialized last_detection_ts to prevent immediate closure")
    
    save_door_state()
    
    if detector and DETECTION_AVAILABLE:
        print("[INIT] Starting detection loop thread")
        threading.Thread(target=detection_loop, daemon=True).start()
    else:
        print("[INIT] Detection loop not started; detector unavailable")
    threading.Thread(target=pir_led_manager, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, threaded=True)