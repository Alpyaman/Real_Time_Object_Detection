"""
Example: Detect Specific Objects
Filter detections to only show specific object classes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import ObjectDetector, VideoProcessor


def main():
    """Detect only specific object classes."""
    
    # Initialize detector
    detector = ObjectDetector(
        model_name="yolov8n.pt",
        confidence_threshold=0.6,  # Higher confidence for better accuracy
        device="cpu"
    )
    
    # Get all available classes
    class_names = detector.get_class_names()
    print("\n📋 Available object classes:")
    print(", ".join(sorted(class_names.values())))
    print()
    
    # Define classes we want to detect (example: people and vehicles)
    target_classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
    
    # Get class IDs for our targets
    class_ids = [k for k, v in class_names.items() if v in target_classes]
    
    print(f"🎯 Detecting only: {', '.join(target_classes)}\n")
    print("Press 'q' to quit\n")
    
    # Create custom detection function that filters classes
    original_detect = detector.detect
    
    def filtered_detect(frame, classes=None):
        return original_detect(frame, classes=class_ids)
    
    detector.detect = filtered_detect
    
    # Initialize video processor
    processor = VideoProcessor(
        detector=detector,
        input_source=0,  # Webcam
        resize_width=640,
        resize_height=480,
        show_fps=True,
        show_labels=True,
        show_confidence=True
    )
    
    # Track detections
    def on_frame(frame, detections):
        if detections:
            detected = [f"{det['class_name']} ({det['confidence']:.2f})" 
                       for det in detections]
            print(f"Found: {', '.join(detected)}")
    
    # Run detection
    processor.run(on_frame=on_frame)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
