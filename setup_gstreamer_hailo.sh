#!/bin/bash
# setup_gstreamer_hailo.sh
# Setup script for GStreamer-based Hailo detection

set -e

echo "=========================================="
echo "GStreamer Hailo Detector Setup"
echo "=========================================="
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "WARNING: This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update package list
echo "[1/6] Updating package list..."
sudo apt-get update

# Install GStreamer packages
echo "[2/6] Installing GStreamer packages..."
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

# Install Python GStreamer bindings
echo "[3/6] Installing Python GStreamer bindings..."
sudo apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0

# Check if Hailo TAPPAS is installed
echo "[4/6] Checking Hailo TAPPAS installation..."
if dpkg -l | grep -q hailo-tappas-core; then
    echo "✓ Hailo TAPPAS core is already installed"
else
    echo "✗ Hailo TAPPAS core not found"
    echo "Please install hailo-tappas-core package"
    echo "Visit: https://github.com/hailo-ai/hailo-rpi5-examples"
    exit 1
fi

# Install Python packages in virtual environment
echo "[5/6] Installing Python packages..."
if [ -d "$HOME/.virtualenvs/catdoor" ]; then
    source "$HOME/.virtualenvs/catdoor/bin/activate"
    pip install PyGObject>=3.42.0
    echo "✓ Python packages installed in catdoor virtualenv"
else
    echo "WARNING: catdoor virtualenv not found at $HOME/.virtualenvs/catdoor"
    echo "Please create it first with: mkvirtualenv catdoor"
fi

# Verify GStreamer plugins
echo "[6/6] Verifying GStreamer Hailo plugins..."
if gst-inspect-1.0 hailonet > /dev/null 2>&1; then
    echo "✓ hailonet plugin found"
else
    echo "✗ hailonet plugin not found"
    echo "Please check Hailo TAPPAS installation"
fi

if gst-inspect-1.0 hailofilter > /dev/null 2>&1; then
    echo "✓ hailofilter plugin found"
else
    echo "✗ hailofilter plugin not found"
    echo "Please check Hailo TAPPAS installation"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy the updated files to your Raspberry Pi:"
echo "   scp gstreamer_hailo_detector.py pi@raspberrypi.local:~/catdoor/"
echo "   scp app.py pi@raspberrypi.local:~/catdoor/"
echo ""
echo "2. Restart the application:"
echo "   ssh pi@raspberrypi.local"
echo "   workon catdoor"
echo "   cd ~/catdoor"
echo "   python3 app.py"
echo ""
