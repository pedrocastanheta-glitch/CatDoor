# app.py
import os, json, time, threading
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
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
from camera import Camera
import servo_control as servo
import numpy as np
import cv2

# --- CONFIGURATION ---
CONFIG_FILE = "areas_config.json"
STATE_FILE = "door_state.json"
DETECT_INTERVAL_SEC = 0.2
OPEN_COOLDOWN_SEC = 4.0
CLOSE_DELAY_SEC = 5.0

# --- PIR LED constants ---
PIR_PIN_1 = 22
PIR_PIN_2 = 10
LED_PIN = 27
LED_HOLD_SECONDS = 120

# --- Flask & Camera ---
app = Flask(__name__, template_folder="templates", static_folder="static")
cam = Camera()

# --- STATE VARIABLES ---
_config_lock = threading.Lock()
_state_lock = threading.Lock()  # Add dedicated state lock
_last_open_ts = {}
_last_detection_ts = {}
_manual_override = {}
_door_state = {}  # Tracks both state and mode for each door
_pir_status = {"PIR1": {"active": False, "last_motion": 0}, "PIR2": {"active": False, "last_motion": 0}}
_pir_timers = {"PIR1": None, "PIR2": None}
def load_door_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Default: both doors open/manual
    return {k: {"state": "open", "last_mode": "manual"} for k in servo.DOORS.keys()}

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

# --- HELPERS ---
def read_pi_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except Exception: return None

# --- VIDEO STREAMING ---
def gen_frames():
    global config
    try:
        for frame in cam.frames():
            try:
                # Flip the frame horizontally (180 degrees horizontal rotation)
                frame = cv2.flip(frame, 1)
                overlay_frame = frame.copy()
                
                with _config_lock:
                    areas = list(config.get("areas", []))

                if areas:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    for area in areas:
                        rect = area.get("rect", [0,0,0,0])
                        x, y, w, h = [int(v) for v in rect]
                        
                        if w <= 0 or h <= 0:
                            continue
                        
                        # Draw the detection rectangle
                        cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # Cyan rectangle

                        # --- MODIFICATION START ---
                        # Calculate pixel counts for display
                        sub_hsv = hsv[y:y+h, x:x+w]
                        allowed_pixels = 0
                        forbidden_pixels = 0

                        if sub_hsv.size > 0:
                            for p in area.get("profiles", []):
                                if p and p.get("hsv_lo") and p.get("hsv_hi"):
                                    mask = hsv_in_range(sub_hsv, p.get("hsv_lo"), p.get("hsv_hi"))
                                    allowed_pixels += cv2.countNonZero(mask)
                            
                            for p in area.get("forbidden_profiles", []):
                                if p and p.get("hsv_lo") and p.get("hsv_hi"):
                                    mask = hsv_in_range(sub_hsv, p.get("hsv_lo"), p.get("hsv_hi"))
                                    forbidden_pixels += cv2.countNonZero(mask)

                        # Create the new text label with pixel counts
                        area_name = area.get("name", "Unnamed Area")
                        # --- THIS LINE IS CHANGED ---
                        area_text = f"{area_name} (All: {allowed_pixels}, Forb: {forbidden_pixels})"
                        # --- MODIFICATION END ---
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        font_thickness = 2
                        text_color = (0, 255, 255) # Yellow-ish
                        text_background_color = (0, 0, 0) # Black for contrast

                        # Adjust background size for the new, longer text
                        (text_width, text_height), baseline = cv2.getTextSize(area_text, font, font_scale, font_thickness)
                        
                        cv2.rectangle(overlay_frame, (x, y - text_height - baseline - 5), (x + text_width + 5, y), text_background_color, -1)
                        cv2.putText(overlay_frame, area_text, (x + 5, y - baseline - 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                        # Combine both allowed and forbidden profiles for visualization
                        all_profiles = area.get("profiles", []) + area.get("forbidden_profiles", [])
                        
                        if sub_hsv.size > 0 and all_profiles:
                            combined_mask = None
                            for p in all_profiles:
                                if p and p.get("hsv_lo") and p.get("hsv_hi"):
                                    mask = hsv_in_range(sub_hsv, p.get("hsv_lo"), p.get("hsv_hi"))
                                    if combined_mask is None:
                                        combined_mask = mask
                                    else:
                                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                            
                            if combined_mask is not None and cv2.countNonZero(combined_mask) > 0:
                                roi = overlay_frame[y:y+h, x:x+w]
                                green_highlight = np.full(roi.shape, (0, 255, 0), dtype=np.uint8)
                                masked_highlight = cv2.bitwise_and(green_highlight, green_highlight, mask=combined_mask)
                                cv2.addWeighted(masked_highlight, 0.4, roi, 0.6, 0, roi)

                ret, jpg = cam.encode_jpeg(overlay_frame)
                if not ret:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            except Exception as e:
                print(f"[ERROR] Frame processing failed: {e}")
                continue
    except Exception as e:
        print(f"[ERROR] Camera stream failed: {e}")
        return

# --- Door action threading helper ---
def threaded_door_action(action, door_id, set_mode=None):
    def worker():
        try:
            if action == "open":
                servo.open_door(door_id)
                with _state_lock:
                    _door_state[door_id]["state"] = servo.get_door_status(door_id)
                    if set_mode:
                        _door_state[door_id]["last_mode"] = set_mode
            elif action == "close":
                servo.close_door(door_id)
                with _state_lock:
                    _door_state[door_id]["state"] = servo.get_door_status(door_id)
                    if set_mode:
                        _door_state[door_id]["last_mode"] = set_mode
            save_door_state()
        except Exception as e:
            print(f"[ERROR] Door action {action} failed for {door_id}: {e}")
    t = threading.Thread(target=worker, daemon=True)
    t.start()

# --- CORE DETECTION LOGIC ---
def detection_loop():
    global config, _last_open_ts, _last_detection_ts, _manual_override, _door_state
    while True:
        time.sleep(DETECT_INTERVAL_SEC)
        with _config_lock:
            areas = list(config.get("areas", []))
        if not areas: continue

        try:
            frame = cam.capture()
            if frame is None: 
                continue

            # Flip the frame horizontally to match video feed
            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            now = time.time()
        except Exception as e:
            print(f"[ERROR] Camera capture failed: {e}")
            continue

        door_detection_states = {}
        for area in areas:
            door_id = area.get("door_id", "door1")
            if door_id not in door_detection_states:
                door_detection_states[door_id] = {"allowed": False, "forbidden": False}
            with _config_lock:
                if _manual_override.get(door_id, False):
                    door_detection_states[door_id]["manual"] = True
                    continue
            rect = area.get("rect", [0, 0, 0, 0]); x, y, w, h = [int(v) for v in rect]
            min_pixels = int(area.get("min_pixels", 50))
            if w <= 0 or h <= 0: continue
            sub_hsv = hsv[max(0, y):y+h, max(0, x):x+w]
            if sub_hsv.size == 0: continue

            if not door_detection_states[door_id]["allowed"]:
                for p in area.get("profiles", []):
                    mask = hsv_in_range(sub_hsv, p.get("hsv_lo"), p.get("hsv_hi"))
                    if cv2.countNonZero(mask) >= min_pixels:
                        door_detection_states[door_id]["allowed"] = True; break

            if not door_detection_states[door_id]["forbidden"]:
                for p in area.get("forbidden_profiles", []):
                    mask = hsv_in_range(sub_hsv, p.get("hsv_lo"), p.get("hsv_hi"))
                    if cv2.countNonZero(mask) >= min_pixels:
                        door_detection_states[door_id]["forbidden"] = True; break

        for door_id, state in door_detection_states.items():
            if state.get("manual", False): continue

            is_allowed = state["allowed"]
            is_forbidden = state["forbidden"]

            # Only act if there are detection profiles configured for this door
            area_profiles = []
            for area in areas:
                if area.get("door_id") == door_id:
                    area_profiles += area.get("profiles", []) + area.get("forbidden_profiles", [])
            if not area_profiles:
                continue

            if is_forbidden and not is_allowed:
                if _door_state[door_id]["state"] == "open":
                    print(f"[DETECT] Door '{door_id}' FORBIDDEN match -> close NOW")
                    # Don't update state here - let threaded_door_action handle it
                    threaded_door_action("close", door_id, "automatic")
                    _last_detection_ts.pop(door_id, None)
            elif is_allowed:
                last_open = _last_open_ts.get(door_id, 0)
                if now - last_open >= OPEN_COOLDOWN_SEC:
                    if _door_state[door_id]["state"] == "close":
                        print(f"[DETECT] Door '{door_id}' allowed match -> open")
                        # Don't update state here - let threaded_door_action handle it
                        threaded_door_action("open", door_id, "automatic")
                        _last_open_ts[door_id] = now
                _last_detection_ts[door_id] = now
            else:
                last_seen = _last_detection_ts.get(door_id, 0)
                if _door_state[door_id]["state"] == "open" and now - last_seen > CLOSE_DELAY_SEC:
                    print(f"[DETECT] Door '{door_id}' not seen for {CLOSE_DELAY_SEC}s -> close")
                    # Don't update state here - let threaded_door_action handle it
                    threaded_door_action("close", door_id, "automatic")
                    _last_detection_ts.pop(door_id, None)
                    _last_open_ts.pop(door_id, None)

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

# --- FLASK API ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/setup")
def setup():
    return render_template("setup.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/temp")
def api_temp():
    return jsonify({"temp_c": read_pi_temp()})

@app.route("/api/door_status")
def api_door_status():
    with _state_lock:
        return jsonify(_door_state)

@app.route("/api/override_status")
def api_override_status(): 
    with _config_lock: 
        return jsonify(_manual_override)

@app.route("/api/config", methods=["GET", "POST"])
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
def api_door(door_id, action):
    global _manual_override
    if door_id not in servo.DOORS: return jsonify({"ok": False, "error": f"Unknown door {door_id}"}), 400
    with _config_lock:
        if action == "open":
            threaded_door_action("open", door_id, "manual")
            _manual_override[door_id] = True
            print(f"[MANUAL] Door '{door_id}' is now in manual hold mode.")
        elif action == "close":
            threaded_door_action("close", door_id, "automatic")
            _manual_override[door_id] = False
            print(f"[MANUAL] Door '{door_id}' has resumed automatic mode.")
        else: return jsonify({"ok": False, "error": "action must be open|close"}), 400
    return jsonify({"ok": True, "status": servo.get_door_status(door_id)})

@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    global _last_snapshot_bgr
    frame = cam.capture()
    if frame is None: return jsonify({"ok": False, "error": "capture failed"}), 500
    # Flip the frame horizontally to match video feed
    frame = cv2.flip(frame, 1)
    _last_snapshot_bgr = frame.copy()
    ok, jpg = cam.encode_jpeg(frame)
    if not ok: return jsonify({"ok": False, "error": "encode failed"}), 500
    return Response(jpg, mimetype="image/jpeg")

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
    return jsonify({"width": cam.width, "height": cam.height})

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

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    _door_state = load_door_state()
    for door_id in servo.DOORS:
        _manual_override[door_id] = False
        if door_id not in _door_state:
            _door_state[door_id] = {"state": "open", "last_mode": "manual"}
        elif "last_mode" not in _door_state[door_id]:
            _door_state[door_id]["last_mode"] = "manual"
    save_door_state()
    
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=pir_led_manager, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, threaded=True)