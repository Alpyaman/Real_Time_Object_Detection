# Web Interface Quick Start

## 🌐 Launch the Web Interface

```bash
streamlit run streamlit_app.py
```

Your browser will open to `http://localhost:8501`

## 🚀 First Time Setup

1. **Allow Camera Access**
   - Browser will prompt for camera permissions
   - Click "Allow" to enable webcam

2. **Configure in Sidebar**
   - Select YOLO model (start with `yolov8n` for speed)
   - Adjust confidence threshold (default 0.5 is good)
   - Enable tracking if needed (optional)

3. **Initialize**
   - Click **"🚀 Initialize/Update Model"** button
   - Wait for model to load (first time downloads weights)
   - You'll see "✅ Model initialized!" when ready

4. **Start Detection**
   - Click **"START"** button on video feed
   - Detection runs in real-time!

## 📱 Available Web Apps

### Main App (Full Features)
```bash
streamlit run streamlit_app.py
```
- All detection and tracking features
- Complete configuration options
- Live statistics

### Simple Detection
```bash
streamlit run web_apps/detection_app.py
```
- Minimal UI for quick demos
- Fast object detection only
- No tracking overhead

### Tracking Focus
```bash
streamlit run web_apps/tracking_app.py
```
- Object tracking with persistent IDs
- ByteTrack or DeepSORT selection
- Track history and statistics

### Analytics Dashboard
```bash
streamlit run web_apps/analytics_app.py
```
- Real-time FPS charts
- Object class distribution graphs
- Detection statistics
- Export-ready data

## 🔧 Configuration Options

### Model Settings
- **YOLO Model**: Choose speed vs accuracy
  - `yolov8n` - Fastest (recommended for webcam)
  - `yolov8s` - Balanced
  - `yolov8m` - More accurate
  - `yolov8l` - High accuracy

- **Confidence Threshold**: 0.0-1.0 (default 0.5)
  - Lower = more detections (may include false positives)
  - Higher = fewer, more confident detections

- **Device**: CPU or CUDA
  - Use CPU for compatibility
  - Use CUDA for 5-10x speedup (requires NVIDIA GPU)

### Tracking Settings (Optional)

**ByteTrack** (Fast)
- Track Threshold: Confidence to start tracking
- Track Buffer: Frames to keep lost tracks
- Match Threshold: IoU for matching detections

**DeepSORT** (Robust)
- Max Age: Frames to keep track without detection
- Min Hits: Detections before confirming track
- Appearance Weight: How much to use appearance vs motion

### Display Settings
- **Show Labels**: Display class names
- **Show Confidence**: Show confidence scores
- **Show FPS**: Display FPS counter

## 🌍 Access from Other Devices

### Local Network Access

```bash
# Run on all network interfaces
streamlit run streamlit_app.py --server.address=0.0.0.0
```

Then access from any device on your network:
```
http://YOUR_LOCAL_IP:8501
```

Find your IP:
- **Windows**: `ipconfig` (look for IPv4 Address)
- **Mac/Linux**: `ifconfig` (look for inet)

### Example
If your computer's IP is `192.168.1.100`:
- From phone: `http://192.168.1.100:8501`
- From tablet: `http://192.168.1.100:8501`

## 🎯 Tips for Best Performance

1. **Use Smallest Model First**
   - Start with `yolov8n`
   - Upgrade to larger models if needed

2. **Optimize for Your Hardware**
   - CPU: yolov8n, no tracking
   - GPU: yolov8s/m with tracking

3. **Close Other Applications**
   - Free up camera and CPU resources

4. **Good Lighting**
   - Better lighting = better detection

5. **Stable Internet** (for first run)
   - Model downloads on first use

## 🐛 Troubleshooting

### Camera Not Working
- **Check Permissions**: Browser settings → Camera
- **Try Different Browser**: Chrome/Edge recommended
- **Use HTTPS**: Camera requires secure context
- **Check URL**: Use `localhost` not `127.0.0.1`

### Low FPS
- Use `yolov8n` model
- Disable tracking
- Lower confidence threshold
- Close other apps
- Use GPU if available

### Stream Freezes
- Refresh browser page
- Click STOP then START
- Restart Streamlit app
- Check CPU usage (Task Manager)

### Model Won't Load
- Check internet connection (first download)
- Wait 30-60 seconds for download
- Check disk space
- Try smaller model

### Connection Errors
- Check firewall settings
- Ensure port 8501 is open
- Try different STUN server
- Check antivirus settings

## 📊 Understanding the Interface

### Video Feed
- Live camera feed with detection overlays
- Bounding boxes around detected objects
- Labels showing class and confidence
- FPS counter in corner

### Statistics Panel
- **Frames Processed**: Total frames analyzed
- **Active Tracks**: Currently tracked objects (if tracking enabled)
- **Average FPS**: Performance metric
- **Track History**: Longest tracked objects

### Controls
- **START/STOP**: Control video stream
- **Initialize**: Load/reload model with new settings
- **Sidebar Sliders**: Real-time configuration

## 🎓 Example Workflows

### Traffic Monitoring
1. Use `yolov8n` for speed
2. Enable ByteTrack tracking
3. Focus on vehicles (car, truck, bus)
4. Monitor statistics for counting

### People Counting
1. Use `yolov8s` for accuracy
2. Enable DeepSORT for robustness
3. Filter to "person" class only
4. Track IDs for in/out counting

### Security Monitoring
1. Use `yolov8m` for accuracy
2. Enable DeepSORT
3. Higher confidence threshold (0.6-0.7)
4. Monitor analytics dashboard

### Quick Demo
1. Use detection_app for simplicity
2. `yolov8n` model
3. Default settings
4. Just show detections

## 🔐 Security Notes

- Web interface runs locally by default
- Camera feed stays on your machine
- No data sent to external servers
- Use HTTPS for production deployments

## 📚 Additional Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [streamlit-webrtc Guide](https://github.com/whitphx/streamlit-webrtc)
- [YOLO Documentation](https://docs.ultralytics.com)
- Main README: [README.md](../README.md)

---

**Enjoy real-time object detection in your browser! 🎉**
