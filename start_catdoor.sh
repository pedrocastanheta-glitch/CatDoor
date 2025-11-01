#!/bin/bash
# Startup script for Cat Door with Hailo GStreamer detector

# Activate Hailo virtual environment (has all required dependencies)
source /home/pedrocastanheta/hailo-rpi5-examples/venv_hailo_rpi_examples/bin/activate

# Set environment variables
export HAILO_ENV_FILE="/home/pedrocastanheta/hailo-rpi5-examples/.env"

# Change to catdoor directory
cd /home/pedrocastanheta/catdoor

# Run the app
python3 app.py
