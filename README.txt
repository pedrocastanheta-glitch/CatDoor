CatDoor2 • Multi Sub-Area Setup
=================================

What you got
------------
- app.py                -> Flask app with /setup page and API endpoints
- templates/setup.html  -> Full UI to draw multiple sub-areas inside an Area and sample HSV
- camera.py             -> Minimal CatDoorCamera helper (replace with your pipeline if you have one)
- areas_config.json     -> Stores areas and subareas (+ sampled HSVs)

Quick start
-----------
1) Create & activate a venv (optional):
   python3 -m venv .venv && source .venv/bin/activate

2) Install deps:
   pip install flask opencv-python numpy

3) Run:
   python app.py

4) Open:
   http://<raspberrypi>:5000/setup

Tips
----
- Hold SHIFT and drag on the snapshot to create sub-areas (ROIs).
- Click "Sample color" to store the HSV mean for that ROI.
- "Save All" writes to /mnt/data/areas_config.json.
- Replace camera.py with your existing CatDoor camera pipeline that implements:
    def get_bgr_frame(self) -> np.ndarray  # HxWx3 BGR
    def get_jpeg(self) -> bytes            # optional fast path

Integrating with detection
--------------------------
Load areas_config.json in your detection loop and iterate subareas for the selected Area.
Compute the current ROI HSV mean and compare to the stored sample with your tolerance.
