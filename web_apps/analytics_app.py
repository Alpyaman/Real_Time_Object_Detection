"""
Analytics Dashboard
Real-time object detection with live charts and statistics.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import sys
from pathlib import Path
from collections import defaultdict, deque
import time
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src import ObjectDetector, draw_detections, draw_fps

# Page config
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class AnalyticsProcessor:
    """Video processor with analytics collection."""
    
    def __init__(self):
        self.detector = None
        
        # Analytics data
        self.frame_count = 0
        self.fps_history = deque(maxlen=100)
        self.detection_counts = defaultdict(int)
        self.class_counts = defaultdict(int)
        self.fps_samples = deque(maxlen=30)
        
        self.start_time = time.time()
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.detector is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        try:
            self.frame_count += 1
            
            # Detect
            detections, inference_time = self.detector.detect(img)
            
            # Collect analytics
            self.detection_counts[len(detections)] += 1
            
            for det in detections:
                self.class_counts[det['class_name']] += 1
            
            # FPS tracking
            fps = 1.0 / inference_time if inference_time > 0 else 0
            self.fps_samples.append(fps)
            self.fps_history.append(fps)
            
            # Draw
            if detections:
                img = draw_detections(img, detections, show_confidence=True)
            
            img = draw_fps(img, fps)
            
        except Exception as e:
            st.error(f"Error: {e}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def get_stats(self):
        """Get current statistics."""
        avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0
        runtime = time.time() - self.start_time
        
        return {
            'frames': self.frame_count,
            'avg_fps': avg_fps,
            'runtime': runtime,
            'total_detections': sum(self.detection_counts.values()),
            'class_counts': dict(self.class_counts)
        }


def main():
    st.title("📊 Real-Time Analytics Dashboard")
    st.markdown("Live object detection with statistics and charts")
    
    # Sidebar
    st.sidebar.header("Settings")
    model = st.sidebar.selectbox("Model", ["yolov8n", "yolov8s", "yolov8m"])
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
    
    # Initialize
    if 'processor' not in st.session_state:
        st.session_state.processor = AnalyticsProcessor()
    
    processor = st.session_state.processor
    
    if st.sidebar.button("🚀 Start", type="primary"):
        with st.spinner("Loading..."):
            try:
                processor.detector = ObjectDetector(
                    model_name=f"{model}.pt",
                    confidence_threshold=confidence,
                    device="cpu"
                )
                processor.frame_count = 0
                processor.detection_counts.clear()
                processor.class_counts.clear()
                processor.fps_history.clear()
                processor.start_time = time.time()
                
                st.sidebar.success("✅ Running!")
            except Exception as e:
                st.sidebar.error(f"❌ {e}")
    
    # Reset button
    if st.sidebar.button("🔄 Reset Stats"):
        processor.frame_count = 0
        processor.detection_counts.clear()
        processor.class_counts.clear()
        processor.fps_history.clear()
        processor.start_time = time.time()
        st.sidebar.success("Stats reset!")
    
    # Main layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📹 Live Feed")
        webrtc_streamer(
            key="analytics",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: processor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.subheader("📈 Live Metrics")
        
        if processor.detector:
            stats = processor.get_stats()
            
            # Key metrics
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Frames", stats['frames'])
                st.metric("Avg FPS", f"{stats['avg_fps']:.1f}")
            
            with metric_cols[1]:
                st.metric("Detections", stats['total_detections'])
                mins = int(stats['runtime'] // 60)
                secs = int(stats['runtime'] % 60)
                st.metric("Runtime", f"{mins}m {secs}s")
            
            # FPS chart
            if len(processor.fps_history) > 10:
                st.subheader("FPS Over Time")
                fps_data = pd.DataFrame({
                    'FPS': list(processor.fps_history)
                })
                st.line_chart(fps_data)
            
            # Class distribution
            if processor.class_counts:
                st.subheader("Detected Objects")
                class_data = pd.DataFrame({
                    'Class': list(processor.class_counts.keys()),
                    'Count': list(processor.class_counts.values())
                }).sort_values('Count', ascending=False)
                
                st.bar_chart(class_data.set_index('Class'))
                
                # Detailed table
                with st.expander("📋 Details"):
                    st.dataframe(class_data, use_container_width=True)
        else:
            st.warning("Click 'Start' to begin")
    
    # Additional info
    st.markdown("---")
    
    info_cols = st.columns(3)
    
    with info_cols[0]:
        st.metric("Model", model)
    
    with info_cols[1]:
        st.metric("Confidence", f"{confidence:.2f}")
    
    with info_cols[2]:
        if processor.detector:
            device = processor.detector.device
            st.metric("Device", device.upper())


if __name__ == "__main__":
    main()
