#!/bin/bash
# stop_catdoor.sh - Properly stop the Cat Door application and clean up resources

echo "Stopping Cat Door Application..."

# Try to stop service if it exists
if systemctl is-active --quiet catdoor.service 2>/dev/null; then
    echo "Stopping catdoor service..."
    sudo systemctl stop catdoor.service
    sleep 2
fi

# Kill the Flask app
echo "Stopping Flask app..."
pkill -f "python3 app.py"
sleep 1

# Kill any remaining Python processes related to catdoor
echo "Stopping any remaining Python processes..."
pkill -f "catdoor"
sleep 1

# Kill any GStreamer processes
echo "Stopping GStreamer processes..."
pkill -f "gst-launch"
pkill -f "GstAppSrc"
sleep 1

# Kill any Hailo processes
echo "Stopping Hailo processes..."
pkill -f "hailo"
sleep 1

# Force kill if still running
if pgrep -f "python3 app.py" > /dev/null; then
    echo "Force killing Flask app..."
    pkill -9 -f "python3 app.py"
    sleep 1
fi

# Reset Hailo device if available
if [ -c /dev/hailo0 ]; then
    echo "Resetting Hailo device..."
    sudo chmod 666 /dev/hailo0 2>/dev/null || true
fi

# Kill any zombie picamera2 processes
echo "Cleaning up camera processes..."
pkill -f "picamera2"
pkill -f "libcamera"
sleep 1

echo "Cat Door Application stopped successfully!"
echo "Resources cleaned up."
