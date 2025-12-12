# Installation Guide

Complete guide for installing the Real-Time Object Detection system.

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Camera**: Webcam (for live detection)

### Recommended for Best Performance
- **GPU**: NVIDIA GPU with CUDA support (RTX 2060 or better)
- **RAM**: 16GB
- **Storage**: 5GB free space (for models and outputs)
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)

## 🚀 Quick Installation

### Option 1: Standard Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Real_Time_Object_Detection.git
cd Real_Time_Object_Detection

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Minimal Installation (Detection Only)

```bash
# Install only core dependencies
pip install ultralytics opencv-python torch torchvision numpy pyyaml
```

### Option 3: Full Installation with GPU Support

```bash
# Install PyTorch with CUDA (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## 📦 Dependency Breakdown

### Core ML and Computer Vision
```
ultralytics>=8.0.0         # YOLOv8/YOLOv11 (~50MB)
opencv-python>=4.8.0       # Computer vision (~80MB)
opencv-contrib-python>=4.8.0  # Extended CV features
torch>=2.0.0               # PyTorch (~2GB)
torchvision>=0.15.0        # Vision utilities
```

### Object Tracking
```
filterpy>=1.4.5            # Kalman filter
scipy>=1.10.0              # Scientific computing
lap>=0.4.0                 # Linear assignment problem
```

### Web Interface
```
streamlit>=1.28.0          # Web framework
streamlit-webrtc>=0.47.0   # Real-time video streaming
av>=10.0.0                 # Video processing
```

### Utilities
```
numpy>=1.24.0              # Numerical operations
pillow>=10.0.0             # Image processing
pyyaml>=6.0                # Configuration files
tqdm>=4.65.0               # Progress bars
```

### Optional (for advanced features)
```
flask>=2.3.0               # REST API
flask-cors>=4.0.0          # CORS support
pandas>=1.5.0              # Data analysis (for analytics)
```

## 🔧 Platform-Specific Instructions

### Windows

```bash
# 1. Install Python 3.8+ from python.org

# 2. Open Command Prompt or PowerShell

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Test installation
python main.py --info
```

**GPU Setup (NVIDIA):**
```bash
# Install CUDA Toolkit from NVIDIA website
# Then install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### macOS

```bash
# 1. Install Python via Homebrew
brew install python@3.9

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test installation
python main.py --info
```

**Apple Silicon (M1/M2/M3):**
```bash
# PyTorch with Metal (MPS) support
pip install torch torchvision

# Verify MPS availability
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### Linux (Ubuntu/Debian)

```bash
# 1. Install Python and dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv

# 2. Install system libraries
sudo apt install libgl1-mesa-glx libglib2.0-0

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Test installation
python main.py --info
```

**GPU Setup (NVIDIA):**
```bash
# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall

# Install CUDA Toolkit
# Visit: https://developer.nvidia.com/cuda-downloads

# Install cuDNN
# Visit: https://developer.nvidia.com/cudnn

# Verify CUDA
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## ✅ Verify Installation

### Test Core Components

```python
# test_installation.py
import sys
print("Python:", sys.version)

import cv2
print("OpenCV:", cv2.__version__)

import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

from ultralytics import YOLO
print("Ultralytics: OK")

import streamlit
print("Streamlit:", streamlit.__version__)

print("\n✅ All core dependencies installed successfully!")
```

Run test:
```bash
python test_installation.py
```

### Test Detection

```bash
# Test with system info
python main.py --info

# Test basic detection (will download model on first run)
python main.py --max-frames 10
```

### Test Web Interface

```bash
# Launch web app
streamlit run streamlit_app.py

# Should open browser to http://localhost:8501
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "No module named 'cv2'"
```bash
pip install opencv-python
```

#### 2. "No module named 'ultralytics'"
```bash
pip install ultralytics
```

#### 3. PyTorch CUDA not working
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Streamlit import error
```bash
pip install --upgrade streamlit streamlit-webrtc
```

#### 5. Camera not accessible
- **Windows**: Check camera privacy settings
- **Linux**: Add user to video group: `sudo usermod -a -G video $USER`
- **macOS**: Grant camera permission in System Preferences

#### 6. Import error with lap
```bash
# On Windows, may need Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Then reinstall
pip install lap
```

## 🔍 Dependency Issues

### Resolve version conflicts
```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

### Check installed versions
```bash
pip list
```

### Update all dependencies
```bash
pip install --upgrade -r requirements.txt
```

## 💾 Storage Requirements

### Model Weights (downloaded on first use)
- YOLOv8n: ~6 MB
- YOLOv8s: ~22 MB
- YOLOv8m: ~52 MB
- YOLOv8l: ~87 MB
- YOLOv8x: ~136 MB

### Installation Size
- Python packages: ~3 GB
- Model weights: 6-136 MB (depending on model)
- Virtual environment: ~500 MB

**Total**: ~3.5-4 GB

## 🚀 Performance Optimization

### For CPU Only
```bash
# Install CPU-only PyTorch (smaller size)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### For GPU
```bash
# Install with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU usage
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

## 📚 Next Steps

After successful installation:

1. **Read Documentation**
   - [README.md](README.md) - Main documentation
   - [QUICKSTART.md](QUICKSTART.md) - CLI quick start
   - [WEB_QUICKSTART.md](WEB_QUICKSTART.md) - Web UI guide

2. **Run Examples**
   ```bash
   python examples/webcam_detection.py
   ```

3. **Try Web Interface**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Benchmark Your System**
   ```bash
   python examples/benchmark.py
   ```

## 🆘 Getting Help

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section
2. Verify [system requirements](#system-requirements)
3. Run `python main.py --info` for system diagnostics
4. Check GitHub Issues for similar problems
5. Create new issue with error details

## 📝 Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed from requirements.txt
- [ ] CUDA installed (if using GPU)
- [ ] Installation verified with test script
- [ ] Basic detection test successful
- [ ] Web interface launches successfully
- [ ] Camera access working (for webcam use)

---

**You're ready to start detecting and tracking objects in real-time! 🎉**
