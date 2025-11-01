import sys
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


class CustomGStreamerDetectionApp(GStreamerDetectionApp):
    """Custom detection app with HTTP source support and headless display."""

    def __init__(self, app_callback, user_data, parser):
        super().__init__(app_callback, user_data, parser)

    def create_pipeline(self):
        """Ensure the display sink is headless before building the pipeline."""
        self.video_sink = "fakesink"
        print("[CUSTOM APP] Forcing video-sink=fakesink for headless operation")
        super().create_pipeline()
    
    def _get_source_bin(self):
        if not self.args.input.startswith(('http://', 'https://')):
            # If it's not a web stream, use the parent class's logic.
            return super()._get_source_bin()

        print("[CUSTOM APP] Detected HTTP stream. Building source bin with souphttpsrc for MJPEG.")
        
        # Create a new bin to hold the source elements.
        source_bin = Gst.Bin.new("source-bin")
        
        # 1. Create the souphttpsrc element for HTTP streaming.
        source = Gst.ElementFactory.make("souphttpsrc", "http-source")
        source.set_property("location", self.args.input)
        source.set_property("is-live", True)

        # 2. The stream is MJPEG, so we need jpegdec and videoconvert.
        jpegdec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")
        videoconvert = Gst.ElementFactory.make("videoconvert", "video-converter")
        
        # 3. Add all elements to the bin.
        source_bin.add(source)
        source_bin.add(jpegdec)
        source_bin.add(videoconvert)
        
        # 4. Link the elements together.
        if not source.link(jpegdec):
            print("[CUSTOM APP] ERROR: Could not link souphttpsrc to jpegdec.", file=sys.stderr)
            return None
        if not jpegdec.link(videoconvert):
            print("[CUSTOM APP] ERROR: Could not link jpegdec to videoconvert.", file=sys.stderr)
            return None

        # 5. A GhostPad is used to expose the 'src' pad of the last element in our bin
        # so it can be linked to the next element in the main pipeline.
        pad = Gst.GhostPad.new("src", videoconvert.get_static_pad("src"))
        source_bin.add_pad(pad)
        
        return source_bin