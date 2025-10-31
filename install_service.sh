#!/bin/bash
# install_service.sh - Install Cat Door as a systemd service

echo "Installing Cat Door Service..."

# Copy service file to systemd
echo "Copying service file..."
sudo cp catdoor.service /etc/systemd/system/

# Reload systemd to recognize new service
echo "Reloading systemd..."
sudo systemctl daemon-reload

# Enable service to start on boot
echo "Enabling service to start on boot..."
sudo systemctl enable catdoor.service

# Stop any running instances
echo "Stopping any running instances..."
bash stop_catdoor.sh 2>/dev/null || true
sleep 2

# Start the service
echo "Starting Cat Door service..."
sudo systemctl start catdoor.service

# Check status
echo ""
echo "Service status:"
sudo systemctl status catdoor.service --no-pager

echo ""
echo "Installation complete!"
echo ""
echo "Useful commands:"
echo "  Start:   sudo systemctl start catdoor"
echo "  Stop:    sudo systemctl stop catdoor"
echo "  Restart: sudo systemctl restart catdoor"
echo "  Status:  sudo systemctl status catdoor"
echo "  Logs:    sudo journalctl -u catdoor -f"
