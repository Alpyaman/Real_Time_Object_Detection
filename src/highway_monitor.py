"""
Highway Monitor Web App
Real-time highway vehicle monitoring with RTSP/HTTP stream support.

Usage:
    streamlit run web_apps/highway_monitor.py
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from detector import ObjectDetector
from tracker import ByteTracker
from highway_counter import VehicleCounter


class HighwayProcessor(VideoProcessorBase):
    """Video processor for highway monitoring."""
    
    def __init__(self):
        """Initialize processor."""
        self.detector = None
        self.tracker = None
        self.counter = None
        self.frame_count = 0
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process video frame."""
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Initialize components if needed
        if self.detector is None:
            self._initialize(img.shape[:2])
        
        # Detect vehicles
        detections = self.detector.detect(img)
        
        # Filter for vehicles only
        vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        vehicle_detections = [d for d in detections if int(d[5]) in vehicle_classes]
        
        # Update tracker
        tracks = self.tracker.update(vehicle_detections, img.shape[:2])
        
        # Update counter
        self.counter.update(tracks, img.shape[:2], fps=30.0)
        
        # Draw visualizations
        img = self.detector.draw_detections(img, vehicle_detections)
        img = self.tracker.draw_tracks(img, tracks)
        img = self.counter.draw(img)
        
        self.frame_count += 1
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _initialize(self, frame_shape):
        """Initialize components."""
        # Get settings from session state
        model = st.session_state.get('model', 'yolov8n.pt')
        conf = st.session_state.get('conf_threshold', 0.3)
        line_pos = st.session_state.get('line_position', 0.5)
        line_dir = st.session_state.get('line_direction', 'horizontal')
        
        self.detector = ObjectDetector(
            model_name=model,
            conf_threshold=conf,
            iou_threshold=0.5
        )
        
        self.tracker = ByteTracker(
            track_thresh=0.4,
            track_buffer=60
        )
        
        self.counter = VehicleCounter(
            line_position=line_pos,
            line_direction=line_dir,
            min_confidence=conf
        )


def main():
    """Main app."""
    st.set_page_config(
        page_title="Highway Monitor",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Highway Vehicle Monitor")
    st.markdown("Real-time vehicle counting from highway cameras")
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    
    # Video source
    st.sidebar.subheader("Video Source")
    source_type = st.sidebar.radio(
        "Source Type",
        ["Webcam", "RTSP Stream", "HTTP Stream", "Video File"]
    )
    
    if source_type == "RTSP Stream":
        st.sidebar.info(
            "📡 RTSP Stream Format:\n\n"
            "`rtsp://username:password@ip:port/stream`\n\n"
            "Example:\n"
            "`rtsp://admin:password123@192.168.1.100:554/cam/realmonitor?channel=1`"
        )
        
        stream_url = st.sidebar.text_input(
            "RTSP URL",
            value="rtsp://",
            help="Enter full RTSP stream URL"
        )
        
    elif source_type == "HTTP Stream":
        st.sidebar.info(
            "🌐 HTTP Stream Format:\n\n"
            "`http://ip:port/video.mjpg`\n\n"
            "Example:\n"
            "`http://192.168.1.100:8080/video`"
        )
        
        stream_url = st.sidebar.text_input(
            "HTTP URL",
            value="http://",
            help="Enter HTTP/HTTPS stream URL"
        )
    
    # Model settings
    st.sidebar.subheader("Detection Settings")
    
    model = st.sidebar.selectbox(
        "Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="n=fastest, s=balanced, m=accurate"
    )
    st.session_state['model'] = model
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05
    )
    st.session_state['conf_threshold'] = conf_threshold
    
    # Counting line settings
    st.sidebar.subheader("Counting Line")
    
    line_direction = st.sidebar.radio(
        "Line Direction",
        ["horizontal", "vertical"],
        help="horizontal = left-right traffic, vertical = up-down traffic"
    )
    st.session_state['line_direction'] = line_direction
    
    line_position = st.sidebar.slider(
        "Line Position",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0.0=top/left, 1.0=bottom/right"
    )
    st.session_state['line_position'] = line_position
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        
        if source_type == "Webcam":
            # Use webrtc_streamer for webcam
            webrtc_ctx = webrtc_streamer(
                key="highway-webcam",
                video_processor_factory=HighwayProcessor,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
            )
            
        else:
            st.warning(
                "⚠️ External streams (RTSP/HTTP) require running the CLI version:\n\n"
                f"```bash\n"
                f"python highway_cam_counter.py --source {stream_url if source_type != 'Video File' else 'video.mp4'}\n"
                f"```"
            )
            
            st.info(
                "**Why?**\n\n"
                "WebRTC (browser-based) doesn't support RTSP/HTTP streams directly. "
                "Use the CLI version for external camera feeds."
            )
    
    with col2:
        st.subheader("📊 Statistics")
        
        # Placeholder for statistics
        stats_placeholder = st.empty()
        
        # Example statistics (would be updated in real app)
        with stats_placeholder.container():
            st.metric("Total Vehicles", "0", "0")
            st.metric("Cars", "0", "0")
            st.metric("Trucks", "0", "0")
            
            st.markdown("---")
            
            st.markdown("**Directional Counts**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("⬆️ Upward", "0")
            with col_b:
                st.metric("⬇️ Downward", "0")
    
    # Instructions
    st.markdown("---")
    st.subheader("📖 How to Use")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown(
            "**1️⃣ Webcam Mode**\n\n"
            "Use built-in or USB webcam for testing. "
            "Grant camera permissions when prompted."
        )
    
    with col_b:
        st.markdown(
            "**2️⃣ RTSP Stream**\n\n"
            "For IP cameras and DVRs. Requires camera credentials. "
            "Use CLI version for best performance."
        )
    
    with col_c:
        st.markdown(
            "**3️⃣ HTTP Stream**\n\n"
            "For MJPEG or HTTP video streams. "
            "Test URL in browser first to verify accessibility."
        )
    
    # Common RTSP examples
    with st.expander("📡 Common RTSP URL Formats"):
        st.markdown(
            """
            **Generic IP Camera:**
            ```
            rtsp://username:password@192.168.1.100:554/stream1
            ```
            
            **Hikvision:**
            ```
            rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
            ```
            
            **Dahua:**
            ```
            rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
            ```
            
            **Axis:**
            ```
            rtsp://root:password@192.168.1.100/axis-media/media.amp
            ```
            
            **Amcrest:**
            ```
            rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1
            ```
            
            **Reolink:**
            ```
            rtsp://admin:password@192.168.1.100:554/h264Preview_01_main
            ```
            """
        )
    
    # CLI Instructions
    with st.expander("💻 CLI Version (Recommended for RTSP/HTTP)"):
        st.markdown(
            """
            For better performance with external streams, use the CLI version:
            
            **Basic Usage:**
            ```bash
            # RTSP stream
            python highway_cam_counter.py --source "rtsp://user:pass@ip:port/stream"
            
            # HTTP stream
            python highway_cam_counter.py --source "http://camera-ip/video.mjpg"
            
            # Video file
            python highway_cam_counter.py --source highway_traffic.mp4
            ```
            
            **With Custom Settings:**
            ```bash
            python highway_cam_counter.py \
                --source "rtsp://camera.url" \
                --model yolov8s.pt \
                --line-pos 0.6 \
                --direction vertical \
                --output counted_video.mp4 \
                --save-stats stats.json
            ```
            
            **Available Options:**
            - `--model`: Model size (yolov8n/s/m/l/x)
            - `--line-pos`: Counting line position (0.0-1.0)
            - `--direction`: Line direction (horizontal/vertical)
            - `--conf`: Detection confidence (0.0-1.0)
            - `--output`: Save output video
            - `--save-stats`: Export statistics to JSON
            - `--no-display`: Run headless (for servers)
            """
        )


if __name__ == "__main__":
    main()
