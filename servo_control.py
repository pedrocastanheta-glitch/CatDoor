# servo_control.py
#test
import threading
import time, os, json
import numpy as np

# --- Hardware Configuration ---
# Using manual PWM pulse width calculation
DOORS = {
    "door1": {
        # Updated wiring: latch1 -> channel 4, latch2 -> channel 5, door -> channel 6
        "latch1": 4, "latch2": 5, "door": 6,
        "latch_us": lambda angle: int(420 + (angle / 180.0) * (2620 - 420)),
        "door_us":  lambda angle: int(360 + (angle / 180.0) * (2670 - 360)),
        "open_seq":  {"latch1": 10, "latch2": 45, "door": 130}, # Final angle
        "close_seq": {"door": 5, "latch1": 50, "latch2": 10},   # Relaxed close angle
    },
    "door2": {
        "latch1": 8, "latch2": 9, "door": 10,
        "latch_us": lambda angle: int(420 + (angle / 180.0) * (2620 - 420)),
        "door_us":  lambda angle: int(360 + (angle / 180.0) * (2670 - 360)),
        "open_seq":  {"latch1": 10, "latch2": 45, "door": 130},
        "close_seq": {"door": 5, "latch1": 50, "latch2": 10},
    },
}

# ----- PCA9685 Hardware Initialization -----
pca = None
i2c = None
HARDWARE_AVAILABLE = False

try:
    from board import SCL, SDA
    import busio
    from adafruit_pca9685 import PCA9685

    print("[SERVO] Initializing I2C bus...")
    i2c = busio.I2C(SCL, SDA)

    print("[SERVO] Initializing PCA9685...")
    pca = PCA9685(i2c)
    pca.frequency = 50

    HARDWARE_AVAILABLE = True
    print("[SERVO] PCA9685 initialized successfully")
except Exception as e:
    print(f"[SERVO] WARNING: PCA9685 not available: {e}")
    print("[SERVO] Servo control will be disabled. App will run in camera-only mode.")
    HARDWARE_AVAILABLE = False


STATE_FILE = "door_state.json"

try:
    from gpiozero import Button
    # Defines a switch for each door that is physically connected.
    DOOR_SWITCHES = {
        "door1": Button(23, pull_up=True),
        "door2": Button(24, pull_up=True)
    }
except Exception as e:
    DOOR_SWITCHES = {}
    print(f"[SERVO] WARN: Could not initialize GPIO for limit switches: {e}")

# --- Helper Functions ---
def disable_servos(door_id: str) -> None:
    """ The user-provided working logic to disable servos. """
    if not HARDWARE_AVAILABLE or door_id not in DOORS: return
    cfg = DOORS[door_id]
    print(f"[SERVO] Disabling servos for {door_id} (power off via duty_cycle=0).")
    try:
        pca.channels[cfg["door"]].duty_cycle = 0
        pca.channels[cfg["latch1"]].duty_cycle = 0
        pca.channels[cfg["latch2"]].duty_cycle = 0
        print(f"[SERVO] Servos for {door_id} disabled.")
    except Exception as e:
        print(f"[SERVO] ERROR disabling servos: {e}")

def _set_pwm_us(channel: int, microseconds: int) -> None:
    """Helper to set PWM duty cycle based on microseconds."""
    duty_12 = int(microseconds * 4096 / 20000)
    duty_16 = min(max(duty_12 << 4, 0), 0xFFFF)
    pca.channels[channel].duty_cycle = duty_16

def _move_angle(door_id: str, which: str, angle: int) -> None:
    """Adapted to use manual PWM pulse width calculation."""
    if not HARDWARE_AVAILABLE: return
    try:
        cfg = DOORS[door_id]
        us = cfg["door_us"](angle) if which == "door" else cfg["latch_us"](angle)
        _set_pwm_us(cfg[which], us)
    except Exception as e:
        print(f"[SERVO] ERROR in _move_angle: {e}")

def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f: 
                data = json.load(f)
                if isinstance(list(data.values())[0], dict):
                    return {k: v.get("state", "close") for k, v in data.items()}
                else:
                    return data
        except Exception: pass
    return {k: "close" for k in DOORS.keys()}

def _save_state() -> None:
    """Save the current door states to file"""
    try:
        # Load existing state file to preserve last_mode
        existing_data = {}
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                existing_data = json.load(f)
        
        # Update with current states
        for door_id, state in _state.items():
            if door_id not in existing_data:
                existing_data[door_id] = {}
            existing_data[door_id]["state"] = state
            if "last_mode" not in existing_data[door_id]:
                existing_data[door_id]["last_mode"] = "manual"
        
        with open(STATE_FILE, "w") as f:
            json.dump(existing_data, f)
    except Exception as e:
        print(f"[SERVO] ERROR saving state: {e}")

_state = _load_state()
_active_operations = {}

# --- State and Operation Management (from user's code) ---
def cancel_active_operation(door_id: str) -> int:
    if door_id in _active_operations:
        operation_info = _active_operations[door_id]
        operation_info["cancel_flag"].set()
        current_angle = operation_info.get("current_angle", None)
        print(f"[SERVO] Cancelling active {operation_info['operation']} for {door_id} at angle {current_angle}")
        return current_angle
    return None

def register_operation(door_id: str, operation: str, thread_obj: threading.Thread, cancel_flag: threading.Event):
    _active_operations[door_id] = {
        "operation": operation,
        "thread": thread_obj, 
        "cancel_flag": cancel_flag,
        "current_angle": None
    }

def update_current_angle(door_id: str, angle: int):
    if door_id in _active_operations:
        _active_operations[door_id]["current_angle"] = angle

def unregister_operation(door_id: str):
    if door_id in _active_operations:
        del _active_operations[door_id]

# --- Core Door Functions (from user's code) ---
def open_door(door_id: str, cancel_flag: threading.Event = None) -> None:
    if cancel_flag is None: cancel_flag = threading.Event()
    cfg = DOORS[door_id]
    if cancel_flag.is_set(): return

    _state[door_id] = "opening"
    print(f"[SERVO] Door {door_id} state set to 'opening'")
    
    _move_angle(door_id, "latch1", cfg["open_seq"]["latch1"])
    _move_angle(door_id, "latch2", cfg["open_seq"]["latch2"])
    time.sleep(0.5)

    start_angle = cfg["close_seq"]["door"]
    end_angle = cfg["open_seq"]["door"]
    step = 1 if end_angle > start_angle else -1
    print(f"[SERVO] Opening {door_id} slowly...")
 
    for angle in range(start_angle, end_angle + step, step):
        if cancel_flag.is_set():
            print(f"[SERVO] Open cancelled for {door_id} at angle {angle}")
            unregister_operation(door_id)
            return
        _move_angle(door_id, "door", angle)
        update_current_angle(door_id, angle)
        time.sleep(0.05) # Increased delay to reduce humming during movement
    
    time.sleep(1.0)
    _state[door_id] = "open"
    _save_state()
    print(f"[SERVO] Door {door_id} state set to 'open'")
    threading.Timer(1.0, disable_servos, args=[door_id]).start()
    unregister_operation(door_id)

def close_door(door_id: str, cancel_flag: threading.Event = None, from_angle: int = None) -> bool:
    if cancel_flag is None: cancel_flag = threading.Event()
    _state[door_id] = "closing"
    print(f"[SERVO] Door {door_id} state set to 'closing'")
    
    MAX_CLOSE_ATTEMPTS = 5
    for attempt in range(MAX_CLOSE_ATTEMPTS):
        cfg = DOORS[door_id]
        start_angle = from_angle if from_angle is not None else cfg["open_seq"]["door"]
        end_angle = cfg["close_seq"]["door"]
        halfway_angle = int((start_angle + end_angle) / 2)
        fast_delay, slow_delay = 0.02, 0.04
        step = -1 if start_angle > end_angle else 1
        
        if cancel_flag.is_set(): return False
        
        print(f"[SERVO] Closing {door_id} door (Attempt {attempt + 1})...")
        for angle in range(start_angle, halfway_angle, step):
            if cancel_flag.is_set(): return False
            _move_angle(door_id, "door", angle)
            update_current_angle(door_id, angle)
            time.sleep(fast_delay)
            
        for angle in range(halfway_angle, end_angle + step, step):
            if cancel_flag.is_set(): return False
            _move_angle(door_id, "door", angle)
            update_current_angle(door_id, angle)
            time.sleep(slow_delay)
        
        time.sleep(1.0)

        is_shut_correctly = True
        switch = DOOR_SWITCHES.get(door_id)
        if switch:
            try:
                is_shut = switch.is_pressed
                print(f"[SERVO] Switch status for {door_id}: is_pressed={is_shut}")
                is_shut_correctly = is_shut
            except Exception as e:
                print(f"[SERVO] WARN: Could not check switch status: {e}")

        if is_shut_correctly:
            print(f"[SERVO] Door '{door_id}' is shut. Engaging latches.")
            _move_angle(door_id, "latch1", cfg["close_seq"]["latch1"])
            _move_angle(door_id, "latch2", cfg["close_seq"]["latch2"])
            time.sleep(1.0)
            _state[door_id] = "close"
            _save_state()
            threading.Timer(1.0, disable_servos, args=[door_id]).start()
            unregister_operation(door_id)
            return True

        print(f"[!!ERROR!!] Door '{door_id}' failed to shut on attempt {attempt + 1}.")
        if attempt < MAX_CLOSE_ATTEMPTS - 1:
            open_door(door_id)
            time.sleep(2.0)
        
    print(f"[SERVO] All {MAX_CLOSE_ATTEMPTS} attempts to close '{door_id}' failed.")
    threading.Timer(1.0, disable_servos, args=[door_id]).start()
    unregister_operation(door_id)
    return False

# --- High-Level Control (from user's code) ---
def request_door_action(door_id: str, action: str) -> bool:
    current_operation = _active_operations.get(door_id, {}).get("operation")
    
    if current_operation == action + "ing":
        return False
    
    current_angle = None
    if current_operation:
        current_angle = cancel_active_operation(door_id)
        time.sleep(0.2)
    
    cancel_flag = threading.Event()
    
    if action == "open":
        worker = lambda: open_door(door_id, cancel_flag)
        operation_name = "opening"
    elif action == "close":
        worker = lambda: close_door(door_id, cancel_flag, from_angle=current_angle if current_operation == "opening" else None)
        operation_name = "closing"
    else:
        return False
    
    thread = threading.Thread(target=worker, daemon=True)
    register_operation(door_id, operation_name, thread, cancel_flag)
    thread.start()
    return True

# --- Status Functions (from user's code) ---
def get_door_status(door_id: str) -> str:
    return _state.get(door_id, "close")

def get_servo_status() -> dict:
    return {door: get_door_status(door) for door in DOORS.keys()}

def get_switch_status() -> dict:
    return {door_id: {"is_pressed": switch.is_pressed} for door_id, switch in DOOR_SWITCHES.items()}

def shutdown_sequence():
    print("[SERVO] Shutdown sequence initiated...")
    for door_id in DOORS.keys():
        if get_door_status(door_id) != 'close':
            close_door(door_id)
    print("[SERVO] Shutdown sequence complete.")

