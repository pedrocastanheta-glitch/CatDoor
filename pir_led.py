# pir_led.py
# Works on Raspberry Pi 5 (Bookworm) with gpiozero + lgpio
from gpiozero import MotionSensor, LED
import threading
import servo_control as servo

# If you wired 2× AM312 in parallel, they share the same PIR pin.
PIR_PIN = 22          # your parallel PIR line
LED_PIN = 27          # LED array control via NPN/MOSFET gate
HOLD_SECONDS = 30
DOOR_ID = "door1"     # optional: auto-open door when motion

pir = MotionSensor(PIR_PIN, queue_len=1, sample_rate=100, threshold=0.2)
led = LED(LED_PIN)

off_timer = None
door_open = (servo.get_door_status(DOOR_ID) == "open")

def motion_detected():
    global off_timer, door_open
    print("PIR: motion → LED ON, (re)start 30s")
    led.on()
    if not door_open:
        try:
            servo.open_door(DOOR_ID)
            door_open = True
        except Exception as e:
            print(f"Open {DOOR_ID} failed: {e}")

    if off_timer:
        off_timer.cancel()
    t = threading.Timer(HOLD_SECONDS, no_motion_timeout)
    t.daemon = True
    t.start()

    # keep reference so we can cancel on next motion
    globals()["off_timer"] = t

def no_motion_timeout():
    global door_open
    print("PIR: no motion → LED OFF (30s)")
    led.off()
    if door_open:
        try:
            servo.close_door(DOOR_ID)
            door_open = False
        except Exception as e:
            print(f"Close {DOOR_ID} failed: {e}")

pir.when_motion = motion_detected

print("PIR LED controller running. Ctrl+C to stop.")
try:
    from signal import pause
    pause()
except KeyboardInterrupt:
    pass
finally:
    try:
        led.off()
    except Exception:
        pass
