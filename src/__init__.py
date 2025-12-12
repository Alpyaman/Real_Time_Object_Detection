"""Initialize the src package."""

from .detector import ObjectDetector, draw_detections, draw_fps
from .video_processor import VideoProcessor
from .utils import load_config, setup_output_directory, save_detection_results

__all__ = [
    'ObjectDetector',
    'draw_detections',
    'draw_fps',
    'VideoProcessor',
    'load_config',
    'setup_output_directory',
    'save_detection_results'
]
