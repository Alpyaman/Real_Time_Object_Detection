"""
Simple Detection-Only Streamlit App
Minimal interface focused on object detection without tracking.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src import ObjectDetector, draw_detections, draw_fps

# Page config
st.set_page_config(
    page_title="Object Detection",
    page_icon="🔍",
    layout="centered"
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class SimpleDetector:
    """Simple video processor for detection only."""
    
    def __init__(self):
        self.detector = None
        self.confidence = 0.5
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.detector is not None:
            try:
                detections, inference_time = self.detector.detect(img)
                
                if detections:
                    img = draw_detections(img, detections, show_confidence=True)
                
                fps = 1.0 / inference_time if inference_time > 0 else 0
                img = draw_fps(img, fps)
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("🔍 Object Detection")
    st.markdown("Simple real-time object detection with YOLO")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    model = st.sidebar.radio(
        "Model",
        ["yolov8n (Fast)", "yolov8s (Balanced)", "yolov8m (Accurate)"],
        help="Smaller = faster, larger = more accurate"
    )
    model_name = model.split()[0]
    
    confidence = st.sidebar.slider(
        "Confidence",
        0.0, 1.0, 0.5, 0.05
    )
    
    # Initialize
    if 'processor' not in st.session_state:
        st.session_state.processor = SimpleDetector()
    
    processor = st.session_state.processor
    processor.confidence = confidence
    
    if st.sidebar.button("Initialize", type="primary"):
        with st.spinner("Loading model..."):
            try:
                processor.detector = ObjectDetector(
                    model_name=f"{model_name}.pt",
                    confidence_threshold=confidence,
                    device="cpu"
                )
                st.sidebar.success("✅ Ready!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    
    # Video stream
    webrtc_streamer(
        key="detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: processor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Info
    st.info("👆 Click 'START' and allow camera access")


if __name__ == "__main__":
    main()
