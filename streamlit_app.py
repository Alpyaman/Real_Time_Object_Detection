"""
Real-Time Object Detection Web Interface
Built with Streamlit and streamlit-webrtc for efficient video streaming.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from typing import Optional
import logging

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import ObjectDetector, ByteTracker, SimpleDeepSORT, draw_tracks, draw_detections, draw_fps
from src.utils import print_system_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    """Process video frames with object detection and tracking."""
    
    def __init__(self):
        self.detector: Optional[ObjectDetector] = None
        self.tracker = None
        self.tracker_type = None
        self.frame_count = 0
        self.show_fps = True
        self.show_labels = True
        self.show_confidence = True
        
    def initialize_detector(self, model_name, confidence, device):
        """Initialize or reinitialize the detector."""
        try:
            self.detector = ObjectDetector(
                model_name=f"{model_name}.pt",
                confidence_threshold=confidence,
                device=device
            )
            logger.info(f"Detector initialized: {model_name} on {device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            return False
    
    def initialize_tracker(self, tracker_type, config):
        """Initialize or reinitialize the tracker."""
        if tracker_type is None or tracker_type == "None":
            self.tracker = None
            self.tracker_type = None
            return
        
        try:
            if tracker_type == "ByteTrack":
                self.tracker = ByteTracker(**config)
            elif tracker_type == "DeepSORT":
                self.tracker = SimpleDeepSORT(**config)
            self.tracker_type = tracker_type
            logger.info(f"Tracker initialized: {tracker_type}")
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            self.tracker = None
            self.tracker_type = None
    
    def recv(self, frame):
        """
        Process incoming video frame.
        This method is called by streamlit-webrtc for each frame.
        """
        img = frame.to_ndarray(format="bgr24")
        
        # Skip processing if detector not initialized
        if self.detector is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        try:
            self.frame_count += 1
            
            # Run detection
            detections, inference_time = self.detector.detect(img)
            
            # Apply tracking if enabled
            if self.tracker is not None and detections:
                if self.tracker_type == "DeepSORT":
                    detections = self.tracker.update(detections, img)
                else:
                    detections = self.tracker.update(detections)
            
            # Draw visualizations
            if self.show_labels and detections:
                if self.tracker is not None:
                    img = draw_tracks(img, detections)
                else:
                    img = draw_detections(
                        img, detections, 
                        show_confidence=self.show_confidence
                    )
            
            # Draw FPS
            if self.show_fps:
                fps = 1.0 / inference_time if inference_time > 0 else 0
                img = draw_fps(img, fps)
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🎯 Real-Time Object Detection")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    model_name = st.sidebar.selectbox(
        "YOLO Model",
        ["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
        help="Smaller models are faster but less accurate"
    )
    
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    device = st.sidebar.selectbox(
        "Device",
        ["cpu", "cuda"],
        help="Use CUDA for GPU acceleration"
    )
    
    # Tracking settings
    st.sidebar.subheader("Tracking Settings")
    enable_tracking = st.sidebar.checkbox("Enable Object Tracking", value=False)
    
    tracker_type = None
    tracker_config = {}
    
    if enable_tracking:
        tracker_type = st.sidebar.selectbox(
            "Tracker",
            ["ByteTrack", "DeepSORT"],
            help="ByteTrack is faster, DeepSORT is more robust"
        )
        
        if tracker_type == "ByteTrack":
            st.sidebar.markdown("**ByteTrack Config**")
            tracker_config = {
                'track_thresh': st.sidebar.slider("Track Threshold", 0.0, 1.0, 0.5, 0.05),
                'track_buffer': st.sidebar.slider("Track Buffer", 10, 60, 30, 5),
                'match_thresh': st.sidebar.slider("Match Threshold", 0.0, 1.0, 0.8, 0.05),
                'min_box_area': 10.0
            }
        else:  # DeepSORT
            st.sidebar.markdown("**DeepSORT Config**")
            tracker_config = {
                'max_age': st.sidebar.slider("Max Age", 10, 60, 30, 5),
                'min_hits': st.sidebar.slider("Min Hits", 1, 5, 3, 1),
                'iou_threshold': st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.3, 0.05),
                'appearance_weight': st.sidebar.slider("Appearance Weight", 0.0, 1.0, 0.3, 0.05)
            }
    
    # Display settings
    st.sidebar.subheader("Display Settings")
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence", value=True)
    show_fps = st.sidebar.checkbox("Show FPS", value=True)
    
    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = VideoProcessor()
    
    processor = st.session_state.processor
    
    # Initialize button
    if st.sidebar.button("🚀 Initialize/Update Model", type="primary"):
        with st.spinner("Initializing model..."):
            success = processor.initialize_detector(model_name, confidence, device)
            if success:
                st.sidebar.success("✅ Model initialized!")
                
                # Initialize tracker if enabled
                if enable_tracking:
                    processor.initialize_tracker(tracker_type, tracker_config)
                    st.sidebar.success(f"✅ {tracker_type} initialized!")
                else:
                    processor.initialize_tracker(None, {})
            else:
                st.sidebar.error("❌ Failed to initialize model")
    
    # Update display settings
    processor.show_labels = show_labels
    processor.show_confidence = show_confidence
    processor.show_fps = show_fps
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        
        # WebRTC streamer
        ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: processor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Status indicator
        if ctx.state.playing:
            st.success("🟢 Stream Active")
        else:
            st.info("⚪ Stream Inactive - Click 'START' to begin")
    
    with col2:
        st.subheader("📊 Statistics")
        
        if processor.detector is not None:
            st.metric("Frames Processed", processor.frame_count)
            
            if processor.tracker is not None:
                st.metric("Tracker", processor.tracker_type)
                if hasattr(processor.tracker, 'tracked_tracks'):
                    st.metric("Active Tracks", len(processor.tracker.tracked_tracks))
        else:
            st.warning("⚠️ Model not initialized. Click 'Initialize/Update Model' in sidebar.")
        
        # Instructions
        st.subheader("📖 Instructions")
        st.markdown("""
        1. Configure settings in the sidebar
        2. Click **Initialize/Update Model**
        3. Click **START** to begin detection
        4. Allow camera access when prompted
        
        **Tips:**
        - Use `yolov8n` for fastest performance
        - Enable GPU (CUDA) for 5-10x speedup
        - ByteTrack is faster, DeepSORT handles occlusions better
        """)
    
    # Additional info in expander
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### Real-Time Object Detection System
        
        This web interface provides real-time object detection using:
        - **YOLO** (You Only Look Once) for fast object detection
        - **ByteTrack** or **DeepSORT** for object tracking
        - **streamlit-webrtc** for efficient video streaming
        
        **Why streamlit-webrtc?**
        - Processes video in a separate thread (no lag)
        - Doesn't re-run the entire Python script per frame
        - Efficient for real-time applications
        
        **Performance:**
        - CPU: 20-30 FPS (yolov8n)
        - GPU: 100+ FPS (yolov8n)
        
        **Note:** First run will download model weights (~6-130 MB depending on model).
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, YOLO, and streamlit-webrtc")


if __name__ == "__main__":
    main()
