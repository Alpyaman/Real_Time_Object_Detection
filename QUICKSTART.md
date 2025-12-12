# Quick Start Guide

## Installation

1. **Install Python 3.8+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify: `python --version`

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **First Run** (downloads model automatically)
   ```bash
   python main.py
   ```

## Common Commands

### Webcam Detection
```bash
python main.py
```

### Video File
```bash
python main.py --source path/to/video.mp4
```

### Use GPU (CUDA)
```bash
python main.py --device cuda
```

### Save Output
```bash
python main.py --save --output output/result.mp4
```

### Different Model (more accurate but slower)
```bash
python main.py --model yolov8s
```

### Optimize for Speed
```bash
python main.py --model yolov8n --resize 416 416 --skip 1
```

## Keyboard Controls

- **q** - Quit
- **s** - Toggle recording
- **r** - Reset statistics

## Troubleshooting

### "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### "Camera not found"
```bash
# Try different camera index
python main.py --source 1
```

### Low FPS
```bash
# Use faster model and lower resolution
python main.py --model yolov8n --resize 416 416
```

### GPU not detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

## Examples

Run the example scripts:

```bash
# Simple webcam detection
python examples/webcam_detection.py

# Process video file
python examples/video_file_detection.py

# Benchmark models
python examples/benchmark.py

# Detect specific objects only
python examples/detect_specific_objects.py
```

## System Information

Check your system capabilities:
```bash
python main.py --info
```

## Next Steps

1. Review [README.md](README.md) for detailed documentation
2. Customize [config.yaml](config.yaml) for your needs
3. Check [examples/](examples/) folder for more use cases
4. Enable GPU for 5-10x performance boost
