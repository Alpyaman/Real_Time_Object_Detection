"""
Video processing module for real-time object detection.
Handles video streams from webcams, video files, and RTSP streams.
"""

import cv2
import numpy as np
import time
from typing import Optional, Callable, Dict, List
from pathlib import Path
import json

from .detector import ObjectDetector, draw_detections, draw_fps


class VideoProcessor:
    """
    Process video streams for real-time object detection.
    Supports webcams, video files, and performance optimization.
    """
    
    def __init__(
        self,
        detector: ObjectDetector,
        input_source: int | str = 0,
        resize_width: int = 640,
        resize_height: int = 480,
        frame_skip: int = 0,
        show_fps: bool = True,
        show_labels: bool = True,
        show_confidence: bool = True,
        bbox_thickness: int = 2,
        font_scale: float = 0.6
    ):
        """
        Initialize video processor.
        
        Args:
            detector: ObjectDetector instance
            input_source: Video source (0 for webcam, path to video file)
            resize_width: Width to resize frames for processing
            resize_height: Height to resize frames for processing
            frame_skip: Number of frames to skip for performance (0 = no skip)
            show_fps: Whether to display FPS counter
            show_labels: Whether to show object labels
            show_confidence: Whether to show confidence scores
            bbox_thickness: Thickness of bounding boxes
            font_scale: Scale of text labels
        """
        self.detector = detector
        self.input_source = input_source
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.frame_skip = frame_skip
        self.show_fps = show_fps
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        
        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_webcam = isinstance(input_source, int)
        
        # Performance tracking
        self.frame_times = []
        self.total_frames = 0
        self.processed_frames = 0
        
        # Recording
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        
        # Detection storage
        self.all_detections: List[Dict] = []
    
    def start(self) -> bool:
        """
        Start video capture.
        
        Returns:
            True if successfully started, False otherwise
        """
        self.cap = cv2.VideoCapture(self.input_source)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.input_source}")
            return False
        
        # Set camera properties for webcam
        if self.is_webcam:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resize_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resize_height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        print(f"Video source opened: {self.input_source}")
        return True
    
    def stop(self):
        """Stop video capture and release resources."""
        if self.cap is not None:
            self.cap.release()
        
        if self.video_writer is not None:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        print("Video processor stopped")
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, List[Dict], float]:
        """
        Process a single frame with object detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, detections, processing_time)
        """
        start_time = time.time()
        
        # Resize frame if needed
        if frame.shape[1] != self.resize_width or frame.shape[0] != self.resize_height:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        
        # Detect objects
        detections, inference_time = self.detector.detect(frame)
        
        # Draw detections if labels are enabled
        if self.show_labels and detections:
            frame = draw_detections(
                frame,
                detections,
                show_confidence=self.show_confidence,
                bbox_thickness=self.bbox_thickness,
                font_scale=self.font_scale
            )
        
        processing_time = time.time() - start_time
        
        return frame, detections, processing_time
    
    def run(
        self,
        on_frame: Optional[Callable[[np.ndarray, List[Dict]], None]] = None,
        max_frames: Optional[int] = None,
        save_video: bool = False,
        output_path: str = "output/output.mp4"
    ):
        """
        Run real-time object detection on video stream.
        
        Args:
            on_frame: Optional callback function called on each frame
                     Signature: on_frame(frame, detections)
            max_frames: Maximum number of frames to process (None = unlimited)
            save_video: Whether to save processed video
            output_path: Path to save output video
        """
        if not self.start():
            return
        
        # Setup video writer if saving
        if save_video:
            self._setup_video_writer(output_path)
        
        print("\n🎥 Starting real-time object detection...")
        print("Press 'q' to quit, 's' to toggle recording, 'r' to reset stats\n")
        
        frame_count = 0
        skip_counter = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    if not self.is_webcam:
                        print("End of video file reached")
                    break
                
                self.total_frames += 1
                
                # Frame skipping for performance
                if skip_counter < self.frame_skip:
                    skip_counter += 1
                    continue
                skip_counter = 0
                
                # Process frame
                processed_frame, detections, processing_time = self.process_frame(frame)
                self.processed_frames += 1
                self.frame_times.append(processing_time)
                
                # Store detections
                if detections:
                    self.all_detections.append({
                        'frame': self.processed_frames,
                        'timestamp': time.time(),
                        'detections': detections
                    })
                
                # Draw FPS if enabled
                if self.show_fps:
                    current_fps = self._calculate_fps()
                    processed_frame = draw_fps(processed_frame, current_fps)
                
                # Call custom callback
                if on_frame is not None:
                    on_frame(processed_frame, detections)
                
                # Display frame
                cv2.imshow('Real-Time Object Detection', processed_frame)
                
                # Write to video if recording
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    self._toggle_recording(output_path)
                elif key == ord('r'):
                    self._reset_stats()
                
                # Check max frames limit
                if max_frames is not None and frame_count >= max_frames:
                    print(f"\nReached maximum frame limit: {max_frames}")
                    break
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self._print_statistics()
            self.stop()
    
    def _setup_video_writer(self, output_path: str):
        """Setup video writer for recording."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0  # Default FPS for output
        
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (self.resize_width, self.resize_height)
        )
        
        if not self.video_writer.isOpened():
            print(f"Warning: Could not create video writer at {output_path}")
            self.video_writer = None
    
    def _toggle_recording(self, output_path: str):
        """Toggle video recording."""
        if not self.recording:
            if self.video_writer is None:
                self._setup_video_writer(output_path)
            self.recording = True
            print("🔴 Recording started")
        else:
            self.recording = False
            print("⏸️  Recording paused")
    
    def _reset_stats(self):
        """Reset performance statistics."""
        self.frame_times = []
        self.detector.reset_stats()
        print("📊 Statistics reset")
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.frame_times) < 10:
            return 0.0
        
        # Use last 30 frames for FPS calculation
        recent_times = self.frame_times[-30:]
        avg_time = sum(recent_times) / len(recent_times)
        
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def _print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*60)
        print("📊 PROCESSING STATISTICS")
        print("="*60)
        
        print(f"Total frames: {self.total_frames}")
        print(f"Processed frames: {self.processed_frames}")
        
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
            
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Average processing time: {avg_time*1000:.2f} ms/frame")
        
        avg_inference = self.detector.get_average_inference_time()
        if avg_inference > 0:
            print(f"Average inference time: {avg_inference*1000:.2f} ms/frame")
            print(f"Inference FPS: {1.0/avg_inference:.2f}")
        
        print(f"Total detections recorded: {len(self.all_detections)}")
        print("="*60 + "\n")
    
    def save_detections(self, output_path: str = "output/detections.json"):
        """
        Save all detections to JSON file.
        
        Args:
            output_path: Path to save detections
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.all_detections, f, indent=2)
        
        print(f"Detections saved to {output_path}")
