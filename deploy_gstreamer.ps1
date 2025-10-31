# deploy_gstreamer.ps1
# Deploy GStreamer-based Hailo detector to Raspberry Pi

$ErrorActionPreference = "Stop"

$PI_HOST = "pedrocastanheta@raspberrypi.local"
$REMOTE_DIR = "/home/pedrocastanheta/catdoor"
$LOCAL_DIR = "d:\CatDoor\CatDoor"

Write-Host "=========================================="
Write-Host "Deploying GStreamer Hailo Detector"
Write-Host "=========================================="
Write-Host ""

# Check if files exist
$files = @(
    "$LOCAL_DIR\gstreamer_hailo_detector.py",
    "$LOCAL_DIR\app.py",
    "$LOCAL_DIR\setup_gstreamer_hailo.sh"
)

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        Write-Host "ERROR: File not found: $file" -ForegroundColor Red
        exit 1
    }
}

# Stop running application
Write-Host "[1/5] Stopping running application..." -ForegroundColor Cyan
ssh $PI_HOST "pkill -9 -f 'python3.*app.py' 2>/dev/null || true"
Start-Sleep -Seconds 2

# Copy files
Write-Host "[2/5] Copying files to Raspberry Pi..." -ForegroundColor Cyan
scp "$LOCAL_DIR\gstreamer_hailo_detector.py" "${PI_HOST}:${REMOTE_DIR}/"
scp "$LOCAL_DIR\app.py" "${PI_HOST}:${REMOTE_DIR}/"
scp "$LOCAL_DIR\setup_gstreamer_hailo.sh" "${PI_HOST}:${REMOTE_DIR}/"

# Make setup script executable
Write-Host "[3/5] Making setup script executable..." -ForegroundColor Cyan
ssh $PI_HOST "chmod +x ${REMOTE_DIR}/setup_gstreamer_hailo.sh"

# Run setup script
Write-Host "[4/5] Running setup script on Raspberry Pi..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
ssh $PI_HOST "cd ${REMOTE_DIR} && ./setup_gstreamer_hailo.sh"

# Start application
Write-Host "[5/5] Starting application..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Cat Door application with GStreamer Hailo detector..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop monitoring (application will keep running)" -ForegroundColor Yellow
Write-Host ""

# Start the application and show output
ssh $PI_HOST "bash -c 'source ~/.virtualenvs/catdoor/bin/activate && cd ${REMOTE_DIR} && python3 app.py'"
