# Streamlit Web Apps

This directory contains Streamlit applications for web-based object detection and tracking.

## Available Apps

### 1. Main App (`streamlit_app.py`)
Full-featured application with all options.

```bash
streamlit run streamlit_app.py
```

**Features:**
- Real-time object detection
- Object tracking (ByteTrack/DeepSORT)
- Full configuration options
- Live statistics

---

### 2. Detection App (`detection_app.py`)
Simple detection-only interface.

```bash
streamlit run web_apps/detection_app.py
```

**Features:**
- Clean, minimal UI
- Fast object detection
- No tracking overhead
- Perfect for quick demos

---

### 3. Tracking App (`tracking_app.py`)
Focused on object tracking.

```bash
streamlit run web_apps/tracking_app.py
```

**Features:**
- Persistent object IDs
- ByteTrack or DeepSORT
- Track history and statistics
- Great for traffic/crowd analysis

---

### 4. Analytics Dashboard (`analytics_app.py`)
Real-time analytics and charts.

```bash
streamlit run web_apps/analytics_app.py
```

**Features:**
- Live FPS charts
- Object class distribution
- Detection statistics
- Export-ready data

---

## Quick Start

1. **Install dependencies:**
```bash
pip install streamlit streamlit-webrtc av
```

2. **Run an app:**
```bash
streamlit run streamlit_app.py
```

3. **Allow camera access** when prompted by your browser

4. **Configure settings** in the sidebar

5. **Click START** to begin detection

---

## Tips

- **Performance:** Use `yolov8n` for fastest FPS
- **GPU:** Select CUDA device for 5-10x speedup
- **Tracking:** ByteTrack is faster, DeepSORT handles occlusions better
- **Browser:** Works best in Chrome/Edge (Firefox may have WebRTC issues)

---

## Deployment

### Local Network
```bash
streamlit run streamlit_app.py --server.address=0.0.0.0
```
Then access from other devices: `http://YOUR_IP:8501`

### Cloud (Streamlit Cloud)
1. Push to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy (Note: webcam may not work on cloud, use video files)

---

## Troubleshooting

**Camera not detected:**
- Check browser permissions
- Try HTTPS (camera access restricted on HTTP)
- Use `localhost` instead of `127.0.0.1`

**Low FPS:**
- Use smaller model (yolov8n)
- Reduce confidence threshold
- Enable GPU if available
- Disable tracking

**Connection issues:**
- Check firewall settings
- Ensure port 8501 is open
- Try different STUN server

---

## Architecture

```
Browser (WebRTC) ←→ streamlit-webrtc ←→ VideoProcessor ←→ YOLO Detector
                                              ↓
                                         Object Tracker
                                              ↓
                                          Analytics
```

**Key Point:** `streamlit-webrtc` processes video in a separate thread, avoiding the performance bottleneck of standard Streamlit's re-run behavior.

---

Built with Streamlit, YOLO, and streamlit-webrtc
