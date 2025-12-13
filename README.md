# Real-Time Object Detection 🎯

A high-performance real-time object detection system using YOLOv8/YOLOv11 architecture. Optimized for speed and accuracy, achieving 20+ FPS on standard hardware.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎬 Demo

![Highway Vehicle Counter Demo](Animation.gif)

*Real-time vehicle counting on highway camera feed with separate tracking for cars and trucks*

## 🌟 Features

- **Real-Time Performance**: Achieves 20-30+ FPS on CPU, 100+ FPS on GPU
- **Multiple YOLO Models**: Support for YOLOv8n/s/m/l/x variants
- **Object Tracking**: ByteTrack and DeepSORT for persistent object IDs
- **Web Interface**: Streamlit dashboard with streamlit-webrtc
- **Flexible Input**: Webcam, video files, or RTSP streams
- **Easy Configuration**: YAML-based configuration system
- **Performance Monitoring**: Real-time FPS counter and statistics
- **Video Recording**: Save processed videos with detections
- **Detection Export**: Save detection results to JSON
- **GPU Acceleration**: CUDA and Apple Silicon (MPS) support
- **Class Filtering**: Detect only specific object classes
- **Optimized Pipeline**: Frame skipping and resolution control

## 📋 System Requirements

- Python 3.8+
- OpenCV 4.8+
- PyTorch 2.0+
- Webcam (for live detection)
- Optional: CUDA-capable GPU (NVIDIA) or Apple Silicon

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Real_Time_Object_Detection.git
cd Real_Time_Object_Detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

The first time you run detection, YOLO will automatically download the model weights.

### Basic Usage

**Webcam Detection (Default)**
```bash
python main.py
```

**Video File Detection**
```bash
python main.py --source video.mp4
```

**Use GPU (CUDA)**
```bash
python main.py --device cuda
```

**Save Output Video**
```bash
python main.py --save --output output/result.mp4
```

**Enable Object Tracking (ByteTrack - Fast)**
```bash
python main.py --tracker bytetrack
```

**Enable Object Tracking (DeepSORT - Robust)**
```bash
python main.py --tracker deepsort
```

**Launch Web Interface**
```bash
streamlit run streamlit_app.py
```

### Controls

- **q** - Quit application
- **s** - Toggle recording
- **r** - Reset statistics

## 📖 Documentation

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Video Input Source                    │
│           (Webcam / Video File / RTSP Stream)           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Frame Preprocessing                     │
│         (Resize, Normalize, Frame Skipping)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                YOLO Detection Engine                     │
│         (YOLOv8n/s/m/l/x - Ultralytics)                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Post-Processing & NMS                       │
│    (Confidence Filtering, IoU-based Suppression)        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Visualization & Output                     │
│     (Bounding Boxes, Labels, FPS, Video Recording)      │
└─────────────────────────────────────────────────────────┘
```├── tracker.py            # Object tracking (ByteTrack, DeepSORT)
│   └── utils.py              # Utility functions
├── examples/
│   ├── webcam_detection.py         # Simple webcam example
│   ├── video_file_detection.py     # Video file processing
│   ├── benchmark.py                # Performance benchmarking
│   ├── detect_specific_objects.py  # Class filtering example
│   ├── tracking_bytetrack.py       # ByteTrack tracking demo
│   ├── tracking_deepsort.py        # DeepSORT tracking demo
│   ├── tracking_comparison.py      # Compare tracking algorithms
│   └── people_counter.py           # People counting with tracking
├── src/
│   ├── __init__.py           # Package initialization
│   ├── detector.py           # Core YOLO detection module
│   ├── video_processor.py    # Video stream processing
│   └── utils.py              # Utility functions
├── examples/
│   ├── webcam_detection.py         # Simple webcam example
│   ├── video_file_detection.py     # Video file processing
│   ├── benchmark.py                # Performance benchmarking
│   └── detect_specific_objects.py  # Class filtering example
├── config.yaml               # Configuration file
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                # This file
```

### YOLO Model Variants

| Model | Size | Speed (CPU) | Speed (GPU) | Use Case |
|-------|------|-------------|-------------|----------|
| YOLOv8n | 6 MB | 25-30 FPS | 100+ FPS | Real-time, resource-constrained |
| YOLOv8s | 22 MB | 15-20 FPS | 80+ FPS | Balanced speed/accuracy |
| YOLOv8m | 52 MB | 8-12 FPS | 60+ FPS | Higher accuracy needed |
| YOLOv8l | 87 MB | 5-8 FPS | 40+ FPS | Maximum accuracy |
| YOLOv8x | 136 MB | 3-5 FPS | 30+ FPS | Research, offline processing |

### Configuration

Edit [config.yaml](config.yaml) to customize behavior:

```yaml
model:
  name: "yolov8n"              # Model variant
  confidence_threshold: 0.5    # Min confidence for detections
  iou_threshold: 0.45          # IoU threshold for NMS
  device: "cpu"                # cpu, cuda, or mps

video:
  input_source: 0              # 0=webcam, or video file path
  frame_skip: 0                # Skip frames for performance
  resize_width: 640            # Processing width
  resize_height: 480           # Processing height

display:
  show_fps: true
  show_labels: true
  show_confidence: true
```

### Command-Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --source SOURCE          Video source (0 for webcam, or video path)
  --model MODEL           YOLO model (yolov8n/s/m/l/x)
  --tracker TYPE         Enable tracking (bytetrack/deepsort)
  --track-thresh T       Tracking confidence threshold
  --track-buffer N       Frames to keep lost tracks
  --save                 Save output video
  --output PATH          Output video path
  --info                 Show system information
  --max-frames N         Process max N frames
```

## 🎯 Object Tracking

The system supports two tracking algorithms for maintaining persistent object IDs across frames:

### ByteTrack (Recommended for Speed)

**Best for:**
- High-speed scenarios (traffic, sports, drones)
- When objects don't frequently occlude each other
- Resource-constrained environments

**Advantages:**
- Very fast (minimal overhead)
- Simple and robust
- Works well for 90% of use cases

**Usage:**
```bash
python main.py --tracker bytetrack
```

### DeepSORT (Recommended for Robustness)

**Best for:**
- Scenarios with heavy occlusion (people in crowds)
- When objects temporarily disappear (behind pillars, cars)
- Need for appearance-based re-identification

**Advantages:**
- Better handling of occlusions
- Appearance features help re-identify objects
- More robust in complex scenarios

**Usage:**
```bash
python main.py --tracker deepsort
```

### Tracking Comparison

| Feature | ByteTrack | DeepSORT |
|---------|-----------|----------|
| Speed | ⚡⚡⚡ Very Fast | ⚡⚡ Fast |
| Occlusion Handling | ⭐⭐ Good | ⭐⭐⭐ Excellent |
| Re-identification | ❌ No | ✅ Yes (appearance) |
| Memory Usage | Low | Medium |
| Best For | Traffic, Sports | Crowds, Complex scenes |

### Tracking Examples

```python
# ByteTrack Example
from src import ObjectDetector, VideoProcessor

detector = ObjectDetector(model_name="yolov8n.pt")
processor = VideoProcessor(
    detector=detector,
    input_source=0,
    tracker='bytetrack',
  # 5. Object Tracking

```python
from src import ObjectDetector, VideoProcessor

detector = ObjectDetector(model_name="yolov8n.pt")
processor = VideoProcessor(
    detector=detector,
    input_source=0,
    tracker='bytetrack'  # or 'deepsort'
)
processor.run()
```

### 6. People Counter

```python
# See examples/people_counter.py for a complete implementation
# Uses tracking to count people crossing a line
python examples/people_counter.py
```

##  tracker_config={
        'track_thresh': 0.5,
        'track_buffer': 30,
        'match_thresh': 0.8
    }
)
processor.run()
```

```python
# DeepSORT Example
processor = VideoProcessor(
    detector=detector,
    input_source=0,
    tracker='deepsort',
    tracker_config={
        'max_age': 30,
        'min_hits': 3,
        'iou_threshold': 0.3,
        'appearance_weight': 0.3
    }
)
processor.run()
```

See [examples/tracking_bytetrack.py](examples/tracking_bytetrack.py) and [examples/tracking_deepsort.py](examples/tracking_deepsort.py) for complete examples.-skip N               Skip N frames
  --save                 Save output video
  --output PATH          Output video path
  --info                 Show system information
  --max-frames N         Process max N frames
```

## 💻 Examples

### 1. Basic Webcam Detection

```python
from src import ObjectDetector, VideoProcessor

detector = ObjectDetector(
    model_name="yolov8n.pt",
    confidence_threshold=0.5,
    device="cpu"
)

processor = VideoProcessor(
    detector=detector,
    input_source=0,  # Webcam
    resize_width=640,
    resize_height=480
)

processor.run()
```

### 2. Video File with Recording

```python
from src import ObjectDetector, VideoProcessor

detector = ObjectDetector(model_name="yolov8s.pt", device="cuda")
processor = VideoProcessor(detector=detector, input_source="video.mp4")

processor.run(save_video=True, output_path="output/result.mp4")
```

### 3. Detect Specific Objects

```python
from src import ObjectDetector, VideoProcessor

detector = ObjectDetector(model_name="yolov8n.pt")

# Only detect people and vehicles
class_names = detector.get_class_names()
target_ids = [k for k, v in class_names.items() 
              if v in ['person', 'car', 'truck']]

# Use filtered detection
detections, _ = detector.detect(frame, classes=target_ids)
```

### 4. Performance Benchmarking

```python
from src import ObjectDetector
from src.utils import benchmark_model

detector = ObjectDetector(model_name="yolov8n.pt")
results = benchmark_model(detector, num_frames=100)

print(f"Average FPS: {results['average_fps']:.2f}")
```

## ⚡ Performance Optimization

### Tips for Maximum FPS

1. **Use Nano Model**: YOLOv8n is optimized for speed
2. **GPU Acceleration**: 5-10x speedup with CUDA
3. x] Real-time object detection with YOLO
- [x] Object tracking (ByteTrack & DeepSORT)
- [x] Multi-camera support preparation
- [ ] Web interface with Flask/FastAPI
- [ ] Advanced tracking features (trajectory prediction)
- [ ] Custom model training pipeline
- [ ] Docker containerization
- [ ] REST API for remote inference
- [ ] Mobile app deployment (TFLite/ONNX)
- [ ] Cloud deployment guide (AWS/Azure/GCP)
- [ ] Action recognition and behavior analysis
# Maximum speed (may sacrifice accuracy)
python main.py --model yolov8n --resize 416 416 --skip 1 --confidence 0.6

# Balanced (recommended)
python main.py --model yolov8s --resize 640 480 --device cuda

# Maximum accuracy
python main.py --model yolov8x --resize 1280 720 --device cuda --confidence 0.3
```

## 🔧 Advanced Features

### Custom Callbacks

```python
def on_frame_callback(frame, detections):
    # Custom logic for each frame
    if detections:
        print(f"Found {len(detections)} objects")
        # Send alert, log to database, etc.

processor.run(on_frame=on_frame_callback)
```

### Detection Export

```python
# Save all detections to JSON
processor.save_detections("output/detections.json")

# Format:
# {
#   "frame": 123,
#   "timestamp": 1234567890.123,
#   "detections": [
#     {
#       "bbox": [x1, y1, x2, y2],
#       "confidence": 0.95,
#       "class_id": 0,
#       "class_name": "person"
#     }
#   ]
# }
```

## 🐛 Troubleshooting

### Common Issues

**1. Low FPS on CPU**
- Use YOLOv8n (nano) model
- Reduce resolution: `--resize 416 416`
- Enable frame skipping: `--skip 1`
- Close other applications

**2. CUDA Out of Memory**
- Use smaller model (yolov8n or yolov8s)
- Reduce batch size in config
- Lower resolution

**3. Webcam Not Opening**
- Check camera permissions
- Try different source: `--source 1` or `--source 2`
- Ensure no other app is using camera

**4. Model Download Fails**
- Check internet connection
- Manually download from [Ultralytics](https://github.com/ultralytics/assets/releases)
- Place in project root or specify full path

**5. Streamlit WebRTC Issues**
- Use Chrome/Edge (better WebRTC support)
- Access via `localhost` not `127.0.0.1`
- Check browser camera permissions
- Try HTTPS for secure context

## 📊 Benchmarks

Tested on Intel i7-10700K (8 cores, 3.8 GHz), 16GB RAM, NVIDIA RTX 3070

| Configuration | Model | Resolution | FPS (CPU) | FPS (GPU) |
|--------------|-------|------------|-----------|-----------|
| Ultra Fast | yolov8n | 416x416 | 32 | 145 |
| Fast | yolov8n | 640x480 | 28 | 120 |
| Balanced | yolov8s | 640x480 | 18 | 95 |
| Accurate | yolov8m | 640x480 | 10 | 70 |
| Maximum | yolov8x | 1280x720 | 3 | 35 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - YOLO implementation
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🗺️ Roadmap

- [ ] Web interface with Flask/FastAPI
- [ ] Multi-camera support
- [ ] Object tracking (DeepSORT integration)
- [ ] Custom model training pipeline
- [ ] Docker containerization
- [ ] REST API for remote inference
- [ ] Mobile app deployment (TFLite/ONNX)
- [ ] Cloud deployment guide (AWS/Azure/GCP)

---

**Made with ❤️ for Computer Vision and Real-Time ML**
