"""
Example: Real-Time Object Detection from Webcam
Simple example showing how to use the object detection system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import ObjectDetector, VideoProcessor


def main():
    """Run real-time object detection on webcam."""
    
    print("🎥 Starting webcam object detection...")
    print("Press 'q' to quit\n")
    
    # Initialize detector with lightweight model for speed
    detector = ObjectDetector(
        model_name="yolov8n.pt",  # Nano model - fastest
        confidence_threshold=0.5,
        device="cpu"  # Change to "cuda" if you have GPU
    )
    
    # Initialize video processor for webcam (source=0)
    processor = VideoProcessor(
        detector=detector,
        input_source=0,  # 0 = default webcam
        resize_width=640,
        resize_height=480,
        show_fps=True,
        show_labels=True,
        show_confidence=True
    )
    
    # Custom callback to print detection info
    def on_frame(frame, detections):
        if detections:
            objects = [det['class_name'] for det in detections]
            unique_objects = set(objects)
            print(f"Detected: {', '.join(unique_objects)} ({len(detections)} objects)")
    
    # Run detection
    processor.run(on_frame=on_frame)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
