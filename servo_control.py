# servo_control.py
import threading
import time, os, json
import board, busio
from adafruit_pca9685 import PCA9685
import numpy as np

# ----- PCA9685 -----
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

STATE_FILE = "door_state.json"

# --- Hardware Configuration ---
DOORS = {
    "door1": {
        "latch1": 13, "latch2": 14, "door": 12,
        "latch_us": lambda angle: int(420 + (angle / 180.0) * (2620 - 420)),
        "door_us":  lambda angle: int(360 + (angle / 180.0) * (2670 - 360)),
        "open_seq":  {"latch1": 15, "latch2": 45, "door": 95},
        "close_seq": {"door": 0, "latch1": 50, "latch2": 10},
    },
    "door2": {
        "latch1": 9, "latch2": 10, "door": 8,
        "latch_us": lambda angle: int(420 + (angle / 180.0) * (2620 - 420)),
        "door_us":  lambda angle: int(360 + (angle / 180.0) * (2670 - 360)),
        "open_seq":  {"latch1": 10, "latch2": 45, "door": 120},
        "close_seq": {"door": 5, "latch1": 50, "latch2": 10},
    },
}

try:
    from gpiozero import Button
    # Defines a switch for each door that is physically connected.
    DOOR_SWITCHES = {
        "door1": Button(23, pull_up=True),
        "door2": Button(24, pull_up=True)  # <-- Temporarily disabled. Uncomment when switch is connected.
    }
except Exception:
    DOOR_SWITCHES = {}
    print("[SERVO] WARN: Could not initialize GPIO for limit switches.")

# --- Helper Functions ---
def disable_servos(door_id: str) -> None:
    if door_id not in DOORS: return
    cfg = DOORS[door_id]
    print(f"[SERVO] Disabling servos for {door_id} (power off).")
    pca.channels[cfg["door"]].duty_cycle = 0
    pca.channels[cfg["latch1"]].duty_cycle = 0
    pca.channels[cfg["latch2"]].duty_cycle = 0

def _set_pwm_us(channel: int, microseconds: int) -> None:
    duty_12 = int(microseconds * 4096 / 20000)
    duty_16 = min(max(duty_12 << 4, 0), 0xFFFF)
    pca.channels[channel].duty_cycle = duty_16

def _move_angle(door_id: str, which: str, angle: int) -> None:
    cfg = DOORS[door_id]
    us = cfg["door_us"](angle) if which == "door" else cfg["latch_us"](angle)
    _set_pwm_us(cfg[which], us)

def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f: 
                data = json.load(f)
                # Handle both old and new format
                if isinstance(list(data.values())[0], dict):
                    # New format: {"door1": {"state": "open", "last_mode": "manual"}}
                    return {k: v.get("state", "close") for k, v in data.items()}
                else:
                    # Old format: {"door1": "open"}
                    return data
        except Exception: pass
    return {k: "close" for k in DOORS.keys()}

def _save_state(state: dict) -> None:
    # Don't save - let app.py handle state management
    pass

_state = _load_state()
_active_operations = {}  # Track ongoing operations: {door_id: {"operation": "opening/closing", "thread": thread_obj, "cancel_flag": threading.Event(), "current_angle": int}}

# --- Helper Functions ---
def cancel_active_operation(door_id: str) -> int:
    """Cancel any active operation for the specified door. Returns current angle if available."""
    if door_id in _active_operations:
        operation_info = _active_operations[door_id]
        operation_info["cancel_flag"].set()  # Signal cancellation
        current_angle = operation_info.get("current_angle", None)
        print(f"[SERVO] Cancelling active {operation_info['operation']} operation for {door_id} at angle {current_angle}")
        return current_angle
    return None

def register_operation(door_id: str, operation: str, thread_obj: threading.Thread, cancel_flag: threading.Event):
    """Register an active operation for tracking."""
    _active_operations[door_id] = {
        "operation": operation,
        "thread": thread_obj, 
        "cancel_flag": cancel_flag,
        "current_angle": None
    }

def update_current_angle(door_id: str, angle: int):
    """Update the current angle for an active operation."""
    if door_id in _active_operations:
        _active_operations[door_id]["current_angle"] = angle

def unregister_operation(door_id: str):
    """Remove operation from active tracking."""
    if door_id in _active_operations:
        del _active_operations[door_id]

# --- Core Door Functions ---
def open_door(door_id: str, cancel_flag: threading.Event = None) -> None:
    """Open door with optional cancellation support."""
    if cancel_flag is None:
        cancel_flag = threading.Event()
    
    cfg = DOORS[door_id]
    
    # Check for cancellation before starting
    if cancel_flag.is_set():
        print(f"[SERVO] Open operation for {door_id} cancelled before starting")
        return
    
    # Update state immediately to "opening"
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
            print(f"[SERVO] Open operation for {door_id} cancelled at angle {angle}")
            unregister_operation(door_id)
            return
        _move_angle(door_id, "door", angle)
        update_current_angle(door_id, angle)  # Track current position
        time.sleep(0.03)
    
    time.sleep(1.0)
    _state[door_id] = "open"
    print(f"[SERVO] Door {door_id} state set to 'open'")
    # Don't save state here - let app.py manage it
    threading.Timer(1.0, disable_servos, args=[door_id]).start()
    unregister_operation(door_id)

def close_door(door_id: str, cancel_flag: threading.Event = None, from_angle: int = None) -> bool:
    """Close door with optional cancellation support and custom start angle."""
    if cancel_flag is None:
        cancel_flag = threading.Event()
    
    # Update state immediately to "closing"
    _state[door_id] = "closing"
    print(f"[SERVO] Door {door_id} state set to 'closing'")
    
    MAX_CLOSE_ATTEMPTS = 5
    for attempt in range(MAX_CLOSE_ATTEMPTS):
        cfg = DOORS[door_id]
        
        # Use custom start angle if provided, otherwise use normal sequence
        if from_angle is not None:
            start_angle = from_angle
        else:
            start_angle = cfg["open_seq"]["door"]
            
        end_angle = cfg["close_seq"]["door"]
        halfway_angle = int((start_angle + end_angle) / 2)
        fast_delay = 0.02; slow_delay = 0.04
        step = -1 if start_angle > end_angle else 1
        
        # Check for cancellation before starting
        if cancel_flag.is_set():
            print(f"[SERVO] Close operation for {door_id} cancelled before starting")
            unregister_operation(door_id)
            return False
        
        print(f"[SERVO] Closing {door_id} door (Attempt {attempt + 1})...")
        for angle in range(start_angle, halfway_angle, step):
            if cancel_flag.is_set():
                print(f"[SERVO] Close operation for {door_id} cancelled at angle {angle}")
                unregister_operation(door_id)
                return False
            _move_angle(door_id, "door", angle)
            update_current_angle(door_id, angle)  # Track current position
            time.sleep(fast_delay)
            
        for angle in range(halfway_angle, end_angle + step, step):
            if cancel_flag.is_set():
                print(f"[SERVO] Close operation for {door_id} cancelled at angle {angle}")
                unregister_operation(door_id)
                return False
            _move_angle(door_id, "door", angle)
            update_current_angle(door_id, angle)  # Track current position
            time.sleep(slow_delay)
        
        time.sleep(1.0)

        is_shut_correctly = False
        try:
            switch = DOOR_SWITCHES.get(door_id)
            if switch is None:
                # If no switch is defined for this door, assume it closed correctly.
                is_shut_correctly = True
            elif switch.is_pressed:
                is_shut_correctly = True
        except Exception as e:
            print(f"[SERVO] WARN: Could not check switch status: {e}")
            is_shut_correctly = True

        if is_shut_correctly:
            print(f"[SERVO] Door '{door_id}' is shut. Engaging latches.")
            _move_angle(door_id, "latch1", cfg["close_seq"]["latch1"])
            _move_angle(door_id, "latch2", cfg["close_seq"]["latch2"])
            time.sleep(1.0)
            
            _state[door_id] = "close"
            print(f"[SERVO] Door {door_id} state set to 'close'")
            # Don't save state here - let app.py manage it
            threading.Timer(1.0, disable_servos, args=[door_id]).start()
            unregister_operation(door_id)
            return True

        print(f"[!!ERROR!!] Door '{door_id}' failed to shut on attempt {attempt + 1}. Latches not engaged.")
        if attempt < MAX_CLOSE_ATTEMPTS - 1:
            print(f"[SERVO] Retrying: opening door...")
            open_door(door_id)
            time.sleep(2.0)
        
    print(f"[SERVO] All {MAX_CLOSE_ATTEMPTS} attempts to close '{door_id}' failed.")
    threading.Timer(1.0, disable_servos, args=[door_id]).start()
    unregister_operation(door_id)
    return False

# --- High-Level Door Control with Interruption ---
def request_door_action(door_id: str, action: str) -> bool:
    """
    Request a door action with intelligent handling of ongoing operations.
    Returns True if action was initiated, False if ignored (same operation already running).
    """
    current_operation = _active_operations.get(door_id, {}).get("operation")
    
    # If same operation is already running, ignore the request
    if current_operation == action + "ing":  # "opening" or "closing"
        print(f"[SERVO] Ignoring {action} request for {door_id} - already {current_operation}")
        return False
    
    # If different operation is running, cancel it and start reversal from current position
    current_angle = None
    if current_operation:
        print(f"[SERVO] Cancelling {current_operation} and starting {action} for {door_id}")
        current_angle = cancel_active_operation(door_id)
        time.sleep(0.2)  # Brief pause to let cancellation take effect
    
    # Start the new operation in a separate thread
    cancel_flag = threading.Event()
    
    if action == "open":
        def worker():
            # If we have a current angle from cancelled operation, start from there
            if current_angle is not None and current_operation == "closing":
                print(f"[SERVO] Starting open from angle {current_angle} (reversed from close)")
                # For opening from mid-close, we can start directly from current position
                cfg = DOORS[door_id]
                _state[door_id] = "opening"
                end_angle = cfg["open_seq"]["door"]
                step = 1 if current_angle < end_angle else -1
                
                for angle in range(current_angle, end_angle + step, step):
                    if cancel_flag.is_set():
                        unregister_operation(door_id)
                        return
                    _move_angle(door_id, "door", angle)
                    update_current_angle(door_id, angle)
                    time.sleep(0.03)
                
                _state[door_id] = "open"
                threading.Timer(1.0, disable_servos, args=[door_id]).start()
                unregister_operation(door_id)
            else:
                open_door(door_id, cancel_flag)
        operation_name = "opening"
    elif action == "close":
        def worker():
            # If we have a current angle from cancelled operation, start from there
            if current_angle is not None and current_operation == "opening":
                print(f"[SERVO] Starting close from angle {current_angle} (reversed from open)")
                close_door(door_id, cancel_flag, from_angle=current_angle)
            else:
                close_door(door_id, cancel_flag)
        operation_name = "closing"
    else:
        print(f"[SERVO] Invalid action: {action}")
        return False
    
    thread = threading.Thread(target=worker, daemon=True)
    register_operation(door_id, operation_name, thread, cancel_flag)
    thread.start()
    
    print(f"[SERVO] Started {operation_name} operation for {door_id}")
    return True

# --- Status Functions ---
def get_door_status(door_id: str) -> str:
    return _state.get(door_id, "close")

def get_servo_status() -> dict:
    return {door: get_door_status(door) for door in DOORS.keys()}

def get_switch_status() -> dict:
    status = {}
    for door_id, switch in DOOR_SWITCHES.items():
        status[door_id] = {"is_pressed": switch.is_pressed}
    return status

def shutdown_sequence():
    print("[SERVO] Shutdown sequence initiated...")
    all_doors_closed = True
    for door_id in DOORS.keys():
        if get_door_status(door_id) == 'open':
            print(f"[SERVO] Attempting to close {door_id} for shutdown.")
            if not close_door(door_id):
                all_doors_closed = False
    
    if all_doors_closed:
        print("[SERVO] All doors confirmed closed or were already closed.")
    else:
        print("[SERVO] WARN: One or more doors failed to close during shutdown.")
    
    print("[SERVO] Shutdown sequence complete.")

