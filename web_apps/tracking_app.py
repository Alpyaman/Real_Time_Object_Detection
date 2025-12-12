"""
Object Tracking Streamlit App
Real-time object tracking with persistent IDs.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src import ObjectDetector, ByteTracker, SimpleDeepSORT, draw_tracks, draw_fps

# Page config
st.set_page_config(
    page_title="Object Tracking",
    page_icon="🎯",
    layout="wide"
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class TrackingProcessor:
    """Video processor with object tracking."""
    
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.tracker_type = None
        self.track_history = defaultdict(int)
        self.frame_count = 0
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.detector is None or self.tracker is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        try:
            self.frame_count += 1
            
            # Detect and track
            detections, inference_time = self.detector.detect(img)
            
            if detections:
                if self.tracker_type == "DeepSORT":
                    tracked = self.tracker.update(detections, img)
                else:
                    tracked = self.tracker.update(detections)
                
                # Update history
                for obj in tracked:
                    self.track_history[obj['track_id']] += 1
                
                # Draw
                img = draw_tracks(img, tracked)
            
            fps = 1.0 / inference_time if inference_time > 0 else 0
            img = draw_fps(img, fps)
            
        except Exception as e:
            st.error(f"Error: {e}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("🎯 Real-Time Object Tracking")
    st.markdown("Track objects with persistent IDs across frames")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model
    model = st.sidebar.selectbox(
        "Model",
        ["yolov8n", "yolov8s", "yolov8m"]
    )
    
    # Tracker
    tracker_type = st.sidebar.radio(
        "Tracker",
        ["ByteTrack", "DeepSORT"],
        help="ByteTrack: Fast | DeepSORT: Robust"
    )
    
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
    
    # Tracker config
    if tracker_type == "ByteTrack":
        st.sidebar.subheader("ByteTrack Settings")
        track_config = {
            'track_thresh': st.sidebar.slider("Track Thresh", 0.0, 1.0, 0.5, 0.05),
            'track_buffer': st.sidebar.slider("Track Buffer", 10, 60, 30, 5),
            'match_thresh': 0.8,
            'min_box_area': 10.0
        }
    else:
        st.sidebar.subheader("DeepSORT Settings")
        track_config = {
            'max_age': st.sidebar.slider("Max Age", 10, 60, 30, 5),
            'min_hits': st.sidebar.slider("Min Hits", 1, 5, 3, 1),
            'iou_threshold': 0.3,
            'appearance_weight': 0.3
        }
    
    # Initialize
    if 'processor' not in st.session_state:
        st.session_state.processor = TrackingProcessor()
    
    processor = st.session_state.processor
    
    if st.sidebar.button("🚀 Initialize", type="primary"):
        with st.spinner("Initializing..."):
            try:
                # Detector
                processor.detector = ObjectDetector(
                    model_name=f"{model}.pt",
                    confidence_threshold=confidence,
                    device="cpu"
                )
                
                # Tracker
                if tracker_type == "ByteTrack":
                    processor.tracker = ByteTracker(**track_config)
                else:
                    processor.tracker = SimpleDeepSORT(**track_config)
                
                processor.tracker_type = tracker_type
                processor.track_history.clear()
                processor.frame_count = 0
                
                st.sidebar.success("✅ Ready to track!")
                
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Tracking")
        webrtc_streamer(
            key="tracking",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: processor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.subheader("📊 Statistics")
        
        if processor.detector and processor.tracker:
            st.metric("Frames", processor.frame_count)
            st.metric("Unique Objects", len(processor.track_history))
            
            if hasattr(processor.tracker, 'tracked_tracks'):
                st.metric("Active Tracks", len(processor.tracker.tracked_tracks))
            
            # Top tracked objects
            if processor.track_history:
                st.subheader("Top Tracked")
                sorted_tracks = sorted(
                    processor.track_history.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                for track_id, count in sorted_tracks:
                    st.text(f"ID {track_id}: {count} frames")
        else:
            st.warning("Initialize first")
    
    # Info
    with st.expander("ℹ️ About Tracking"):
        st.markdown(f"""
        **Current Tracker: {tracker_type}**
        
        **ByteTrack:**
        - Very fast (~1-2ms overhead)
        - Good for traffic, sports
        - Uses motion only
        
        **DeepSORT:**
        - Robust (~3-5ms overhead)
        - Better for crowds, occlusions
        - Uses appearance features
        
        **Tips:**
        - Each detected object gets a unique ID
        - IDs persist across frames
        - Track history shows how long each object was tracked
        """)


if __name__ == "__main__":
    main()
