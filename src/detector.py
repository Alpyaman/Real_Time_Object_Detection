"""
Core object detection module using YOLO architecture.
Provides real-time object detection capabilities with optimized performance.
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import time


class ObjectDetector:
    """
    Real-time object detection using YOLOv8/YOLOv11 architecture.
    Optimized for speed and accuracy balance.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize the object detector.
        
        Args:
            model_name: YOLO model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
                       'n' = nano (fastest), 'x' = extra large (most accurate)
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for non-maximum suppression
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = self._setup_device(device)
        
        print(f"Initializing {model_name} on {self.device}...")
        self.model = self._load_model()
        
        # Performance tracking
        self.inference_times = []
        self.frame_count = 0
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate the compute device."""
        if device == "cuda" and torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            print("Apple Silicon GPU (MPS) available")
            return "mps"
        else:
            if device != "cpu":
                print(f"Warning: {device} not available, falling back to CPU")
            return "cpu"
    
    def _load_model(self) -> YOLO:
        """Load the YOLO model with optimizations."""
        model = YOLO(self.model_name)
        
        # Move model to device
        model.to(self.device)
        
        # Warmup inference for consistent timing
        print("Warming up model...")
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            _ = model(dummy_input, verbose=False)
        
        print("Model loaded and ready!")
        return model
    
    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> Tuple[List[Dict], float]:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            classes: Optional list of class IDs to detect (None = all classes)
            
        Returns:
            Tuple of (detections, inference_time)
            detections: List of dictionaries containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
            inference_time: Time taken for inference in seconds
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False
        )[0]
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        # Parse results
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': results.names[cls_id]
                })
        
        return detections, inference_time
    
    def get_average_inference_time(self) -> float:
        """Get average inference time across all frames."""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
    
    def get_fps(self) -> float:
        """Calculate average FPS."""
        avg_time = self.get_average_inference_time()
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_times = []
        self.frame_count = 0
    
    def get_class_names(self) -> Dict[int, str]:
        """Get dictionary of class ID to class name mapping."""
        return self.model.names


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict],
    show_confidence: bool = True,
    bbox_thickness: int = 2,
    font_scale: float = 0.6,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the frame.
    
    Args:
        frame: Input image
        detections: List of detection dictionaries
        show_confidence: Whether to show confidence scores
        bbox_thickness: Thickness of bounding box lines
        font_scale: Scale of text labels
        colors: Optional dictionary mapping class_id to BGR color tuple
        
    Returns:
        Frame with drawn detections
    """
    if colors is None:
        # Generate distinct colors for each class
        np.random.seed(42)
        colors = {i: tuple(map(int, np.random.randint(0, 255, 3))) 
                 for i in range(100)}
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        class_id = det['class_id']
        class_name = det['class_name']
        confidence = det['confidence']
        
        # Get color for this class
        color = colors.get(class_id, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thickness)
        
        # Prepare label
        if show_confidence:
            label = f"{class_name}: {confidence:.2f}"
        else:
            label = class_name
        
        # Draw label background
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2
        )
    
    return frame


def draw_fps(frame: np.ndarray, fps: float, position: str = "top-left") -> np.ndarray:
    """
    Draw FPS counter on the frame.
    
    Args:
        frame: Input image
        fps: FPS value to display
        position: Position of FPS counter ('top-left', 'top-right')
        
    Returns:
        Frame with FPS counter
    """
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    if position == "top-left":
        x, y = 10, 30
    else:  # top-right
        x = frame.shape[1] - text_width - 10
        y = 30
    
    # Draw background
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (0, 255, 0),
        thickness
    )
    
    return frame
