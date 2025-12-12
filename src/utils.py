"""
Utility functions for the real-time object detection system.
"""

import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return get_default_config()
    except Exception as e:
        print(f"Error loading config: {e}, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'model': {
            'name': 'yolov8n',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'device': 'cpu'
        },
        'video': {
            'input_source': 0,
            'frame_skip': 0,
            'target_fps': 30,
            'resize_width': 640,
            'resize_height': 480
        },
        'display': {
            'show_fps': True,
            'show_labels': True,
            'show_confidence': True,
            'bbox_thickness': 2,
            'font_scale': 0.6
        },
        'optimization': {
            'use_half_precision': False,
            'use_tensorrt': False,
            'batch_size': 1,
            'num_threads': 4
        },
        'output': {
            'save_video': False,
            'output_path': 'output',
            'save_detections': False
        }
    }


def setup_output_directory(output_path: str = "output") -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path: Path to output directory
        
    Returns:
        Path object
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {output_dir}")
    return output_dir


def save_detection_results(
    detections: List[Dict],
    output_path: str = "output/detections.json"
):
    """
    Save detection results to JSON file.
    
    Args:
        detections: List of detection dictionaries
        output_path: Path to save results
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)
    
    print(f"Detection results saved to {output_path}")


def benchmark_model(
    detector,
    num_frames: int = 100,
    resolution: tuple = (640, 480)
) -> Dict[str, float]:
    """
    Benchmark model performance.
    
    Args:
        detector: ObjectDetector instance
        num_frames: Number of frames to test
        resolution: Resolution to test (width, height)
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n🔬 Running benchmark with {num_frames} frames at {resolution}...")
    
    # Create dummy frames
    dummy_frame = np.random.randint(0, 255, (*resolution[::-1], 3), dtype=np.uint8)
    
    inference_times = []
    
    for i in range(num_frames):
        start = time.time()
        _, inference_time = detector.detect(dummy_frame)
        total_time = time.time() - start
        inference_times.append(total_time)
        
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{num_frames} frames")
    
    # Calculate statistics
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    results = {
        'average_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'average_fps': fps,
        'num_frames': num_frames,
        'resolution': resolution
    }
    
    print("\n📊 Benchmark Results:")
    print(f"Average FPS: {fps:.2f}")
    print(f"Average time: {avg_time * 1000:.2f} ms/frame")
    print(f"Std deviation: {std_time * 1000:.2f} ms")
    print(f"Min time: {min_time * 1000:.2f} ms")
    print(f"Max time: {max_time * 1000:.2f} ms")
    
    return results


def get_video_properties(video_path: str) -> Dict[str, Any]:
    """
    Get properties of a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    props = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    
    return props


def print_system_info():
    """Print system information relevant to object detection."""
    import torch
    import platform
    
    print("\n" + "="*60)
    print("💻 SYSTEM INFORMATION")
    print("="*60)
    
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"PyTorch: {torch.__version__}")
    
    # CUDA info
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA Available: No")
    
    # MPS info (Apple Silicon)
    if torch.backends.mps.is_available():
        print("Apple MPS (Metal) Available: Yes")
    
    print("="*60 + "\n")


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of recent frames to track for metrics
        """
        self.window_size = window_size
        self.frame_times = []
        self.inference_times = []
        self.detection_counts = []
    
    def update(self, frame_time: float, inference_time: float, num_detections: int):
        """Update metrics with new frame data."""
        self.frame_times.append(frame_time)
        self.inference_times.append(inference_time)
        self.detection_counts.append(num_detections)
        
        # Keep only recent frames
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            self.inference_times.pop(0)
            self.detection_counts.pop(0)
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if not self.frame_times:
            return 0.0
        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time in ms."""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times) * 1000
    
    def get_avg_detections(self) -> float:
        """Get average number of detections per frame."""
        if not self.detection_counts:
            return 0.0
        return np.mean(self.detection_counts)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {
            'fps': self.get_fps(),
            'avg_inference_ms': self.get_avg_inference_time(),
            'avg_detections': self.get_avg_detections(),
            'frames_tracked': len(self.frame_times)
        }


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters())
