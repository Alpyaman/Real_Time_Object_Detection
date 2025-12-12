# Highway Camera Setup Guide

Complete guide for connecting to highway cameras and counting vehicles.

## 🎯 Quick Start

```bash
# Test with webcam
python highway_cam_counter.py --source 0

# Connect to RTSP camera
python highway_cam_counter.py --source "rtsp://username:password@192.168.1.100:554/stream"

# Process video file
python highway_cam_counter.py --source highway_traffic.mp4
```

## 📡 Connecting to IP Cameras

### RTSP Stream URL Format

Most IP cameras use RTSP (Real-Time Streaming Protocol):

```
rtsp://[username]:[password]@[ip-address]:[port]/[path]
```

**Example:**
```
rtsp://admin:password123@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

### Finding Your Camera's RTSP URL

1. **Check Camera Documentation**
   - Look for "RTSP URL" or "Streaming URL" in manual
   - Check manufacturer's website

2. **Common Default Paths by Brand:**

   **Hikvision:**
   ```
   rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
   rtsp://admin:password@192.168.1.64:554/Streaming/Channels/102 (sub-stream)
   ```

   **Dahua:**
   ```
   rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
   rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1 (sub-stream)
   ```

   **Axis:**
   ```
   rtsp://root:password@192.168.1.100/axis-media/media.amp
   rtsp://root:password@192.168.1.100/axis-media/media.amp?videocodec=h264
   ```

   **Amcrest:**
   ```
   rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
   ```

   **Reolink:**
   ```
   rtsp://admin:password@192.168.1.100:554/h264Preview_01_main
   rtsp://admin:password@192.168.1.100:554/h264Preview_01_sub (sub-stream)
   ```

   **Foscam:**
   ```
   rtsp://admin:password@192.168.1.100:554/videoMain
   rtsp://admin:password@192.168.1.100:554/videoSub (sub-stream)
   ```

   **Generic:**
   ```
   rtsp://admin:password@192.168.1.100:554/stream1
   rtsp://admin:password@192.168.1.100:554/live
   rtsp://admin:password@192.168.1.100:554/ch01
   ```

3. **Use Discovery Tools:**
   - ONVIF Device Manager (Windows)
   - VLC Media Player (Test → Network Stream)
   - ffplay (command line): `ffplay "rtsp://url"`

### HTTP/MJPEG Streams

Some cameras support HTTP MJPEG streams:

```
http://[ip-address]:[port]/[path]
```

**Examples:**
```
http://192.168.1.100:8080/video
http://192.168.1.100/mjpg/video.mjpg
http://admin:password@192.168.1.100/cgi-bin/mjpg/video.cgi
```

## 🔧 Configuration

### Basic Settings

```bash
python highway_cam_counter.py \
    --source "rtsp://cam.url" \
    --model yolov8n.pt \
    --conf 0.3 \
    --line-pos 0.5 \
    --direction horizontal
```

### Advanced Settings

```bash
python highway_cam_counter.py \
    --source "rtsp://cam.url" \
    --model yolov8s.pt \
    --conf 0.4 \
    --iou 0.5 \
    --line-pos 0.6 \
    --direction vertical \
    --track-thresh 0.4 \
    --track-buffer 60 \
    --output counted_video.mp4 \
    --save-stats stats.json
```

**Parameter Explanations:**

- `--source`: Video source (camera index, file, RTSP/HTTP URL)
- `--model`: YOLO model (n=fastest, x=most accurate)
- `--conf`: Detection confidence (0.1-0.9, lower=more detections)
- `--iou`: Intersection over Union for NMS (0.3-0.7)
- `--line-pos`: Counting line position (0.0=top/left, 1.0=bottom/right)
- `--direction`: Line direction (horizontal for left-right, vertical for up-down)
- `--track-thresh`: Minimum confidence for tracking (0.3-0.6)
- `--track-buffer`: Frames to keep lost tracks (30-120)
- `--output`: Save output video (optional)
- `--save-stats`: Export statistics to JSON (optional)
- `--no-display`: Run without GUI (for servers)
- `--max-frames`: Process only N frames (for testing)

## 📍 Setting Up the Counting Line

### Horizontal Line (Left-Right Traffic)

```bash
# Line in middle (default)
python highway_cam_counter.py --source video.mp4 --direction horizontal --line-pos 0.5

# Line at 60% from top (common for highway cameras)
python highway_cam_counter.py --source video.mp4 --direction horizontal --line-pos 0.6

# Line near top
python highway_cam_counter.py --source video.mp4 --direction horizontal --line-pos 0.3
```

**Best for:**
- Highway views (side angle)
- Traffic moving left-to-right or right-to-left
- Lanes perpendicular to camera

### Vertical Line (Up-Down Traffic)

```bash
# Line in middle
python highway_cam_counter.py --source video.mp4 --direction vertical --line-pos 0.5

# Line at 40% from left
python highway_cam_counter.py --source video.mp4 --direction vertical --line-pos 0.4
```

**Best for:**
- Overhead/aerial views
- Traffic moving toward/away from camera
- Lanes parallel to camera

### Tips for Line Placement:

1. **Position where vehicles are fully visible**
   - Not at frame edges
   - Not where vehicles are occluded
   - Clear separation between lanes

2. **Consider perspective**
   - Closer to camera = larger vehicles = better detection
   - Too far = small vehicles = missed detections

3. **Test and adjust**
   - Run with `--max-frames 500` to test
   - Adjust `--line-pos` based on results
   - Verify counts match manual counting

## 🎥 Testing Your Setup

### Step 1: Test Camera Connection

```bash
# Test if stream is accessible
ffplay "rtsp://username:password@camera-ip:554/stream"

# Or use VLC:
# Media → Open Network Stream → Enter RTSP URL
```

### Step 2: Test Detection

```bash
# Process 100 frames to verify detection
python highway_cam_counter.py \
    --source "rtsp://cam.url" \
    --max-frames 100
```

### Step 3: Calibrate Settings

```bash
# Start with fast model for testing
python highway_cam_counter.py \
    --source "rtsp://cam.url" \
    --model yolov8n.pt \
    --conf 0.3 \
    --line-pos 0.5 \
    --max-frames 500
```

**Adjust based on results:**
- Too many false detections? → Increase `--conf` to 0.4-0.5
- Missing vehicles? → Decrease `--conf` to 0.2-0.3
- Wrong counts? → Adjust `--line-pos`
- Slow performance? → Use yolov8n.pt
- Better accuracy needed? → Use yolov8s.pt or yolov8m.pt

### Step 4: Full Deployment

```bash
# Run with optimized settings
python highway_cam_counter.py \
    --source "rtsp://cam.url" \
    --model yolov8s.pt \
    --conf 0.35 \
    --line-pos 0.6 \
    --direction horizontal \
    --save-stats hourly_stats.json
```

## 📊 Understanding Statistics

### Real-Time Display

While running, you'll see:
```
Total Vehicles: 150
Cars: 120 (80.0%)
Trucks: 30 (20.0%)
Active Tracks: 8

Direction: Up 75 | Down 75
```

### Exported Statistics (JSON)

```json
{
  "total": 150,
  "cars": 120,
  "trucks": 30,
  "cars_up": 60,
  "cars_down": 60,
  "trucks_up": 15,
  "trucks_down": 15,
  "car_percentage": 80.0,
  "truck_percentage": 20.0,
  "hourly_stats": {
    "0": {"cars": 10, "trucks": 2},
    "1": {"cars": 8, "trucks": 1},
    ...
  },
  "active_tracks": 8,
  "timestamp": "2025-12-12T10:30:00",
  "speeds": {
    "1": 65.5,
    "2": 58.3,
    ...
  }
}
```

## 🚀 Production Deployment

### Running 24/7

Use screen/tmux to keep running in background:

```bash
# Start screen session
screen -S highway_counter

# Run counter
python highway_cam_counter.py \
    --source "rtsp://cam.url" \
    --no-display \
    --save-stats /var/log/highway_stats.json

# Detach: Ctrl+A, then D
# Reattach: screen -r highway_counter
```

### Automatic Restart on Disconnect

Create a bash script `run_counter.sh`:

```bash
#!/bin/bash
while true; do
    python highway_cam_counter.py \
        --source "rtsp://cam.url" \
        --no-display \
        --save-stats stats_$(date +%Y%m%d_%H%M%S).json
    
    echo "Connection lost. Restarting in 5 seconds..."
    sleep 5
done
```

Run it:
```bash
chmod +x run_counter.sh
./run_counter.sh
```

### Scheduled Statistics Export

Use cron to export statistics hourly:

```bash
# Edit crontab
crontab -e

# Add line (exports stats every hour)
0 * * * * /usr/bin/python /path/to/highway_cam_counter.py --source "rtsp://cam.url" --max-frames 3600 --save-stats /var/log/stats_$(date +\%Y\%m\%d_\%H).json
```

## 🐛 Troubleshooting

### Connection Issues

**Problem:** "Failed to connect to video source"

**Solutions:**
1. Verify camera is online: `ping camera-ip`
2. Test RTSP URL in VLC or ffplay
3. Check credentials (username/password)
4. Verify port is not blocked (554 for RTSP)
5. Try sub-stream URL (lower resolution)
6. Check camera RTSP settings are enabled

### Performance Issues

**Problem:** Low FPS, laggy video

**Solutions:**
1. Use faster model: `--model yolov8n.pt`
2. Use camera's sub-stream (lower resolution)
3. Lower confidence: `--conf 0.25`
4. Run on GPU if available
5. Reduce `--track-buffer` to 30

### Counting Issues

**Problem:** Incorrect vehicle counts

**Solutions:**
1. Adjust counting line position: `--line-pos 0.6`
2. Change line direction if needed
3. Increase confidence: `--conf 0.4`
4. Use better model: `--model yolov8s.pt`
5. Ensure vehicles fully cross line
6. Check for occlusions at line position

### Stream Drops

**Problem:** Stream disconnects frequently

**Solutions:**
1. Check network stability
2. Use wired connection if possible
3. Lower stream resolution (use sub-stream)
4. Increase camera buffer settings
5. Use automatic restart script (see above)

## 📈 Best Practices

1. **Camera Placement**
   - Mount at 30-45° angle for best vehicle detection
   - Ensure good lighting (avoid direct sunlight)
   - Minimize occlusions

2. **Network**
   - Use wired connection when possible
   - Ensure sufficient bandwidth (2-4 Mbps per camera)
   - Set up QoS for camera traffic

3. **Configuration**
   - Start with default settings
   - Test for 30-60 minutes
   - Adjust based on actual results
   - Document your final settings

4. **Monitoring**
   - Check logs regularly
   - Validate counts periodically
   - Monitor system resources (CPU/RAM)
   - Keep statistics for trend analysis

## 🆘 Getting Help

If you encounter issues:

1. Test camera with VLC/ffplay first
2. Run with `--max-frames 100` for quick testing
3. Check system requirements (CPU, RAM, network)
4. Review logs and error messages
5. Try different model sizes
6. Verify RTSP URL format for your camera brand

## 📝 Example Deployments

### City Traffic Monitoring

```bash
python highway_cam_counter.py \
    --source "rtsp://traffic-cam-1:554/stream" \
    --model yolov8m.pt \
    --conf 0.4 \
    --line-pos 0.55 \
    --direction horizontal \
    --save-stats /var/log/traffic/hourly_$(date +%Y%m%d_%H).json
```

### Toll Booth Counting

```bash
python highway_cam_counter.py \
    --source "rtsp://toll-cam:554/stream" \
    --model yolov8s.pt \
    --conf 0.35 \
    --line-pos 0.6 \
    --direction horizontal \
    --track-buffer 90 \
    --output /var/recordings/toll_$(date +%Y%m%d).mp4
```

### Parking Lot Monitoring

```bash
python highway_cam_counter.py \
    --source "rtsp://parking-cam:554/stream" \
    --model yolov8n.pt \
    --conf 0.3 \
    --line-pos 0.5 \
    --direction vertical \
    --no-display
```

---

**Ready to count vehicles on highways! 🚗🚛**
