"""
Example: Object Detection on Video File
Process a video file and optionally save the output.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import ObjectDetector, VideoProcessor, get_video_properties


def main():
    """Run object detection on a video file."""
    
    # Path to your video file
    video_path = "path/to/your/video.mp4"
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        print("\nPlease update the video_path variable with your video file path.")
        return
    
    print(f"📹 Processing video: {video_path}\n")
    
    # Get video properties
    try:
        props = get_video_properties(video_path)
        print("Video Properties:")
        print(f"  Resolution: {props['width']}x{props['height']}")
        print(f"  FPS: {props['fps']:.2f}")
        print(f"  Frames: {props['frame_count']}")
        print(f"  Duration: {props['duration']:.2f}s\n")
    except Exception as e:
        print(f"Could not read video properties: {e}\n")
    
    # Initialize detector
    detector = ObjectDetector(
        model_name="yolov8n.pt",  # Use nano for speed, or yolov8s for better accuracy
        confidence_threshold=0.5,
        device="cpu"  # Change to "cuda" for GPU acceleration
    )
    
    # Initialize video processor
    processor = VideoProcessor(
        detector=detector,
        input_source=video_path,
        resize_width=640,  # Resize for faster processing
        resize_height=480,
        frame_skip=0,  # Set to 1-2 to skip frames for faster processing
        show_fps=True,
        show_labels=True,
        show_confidence=True
    )
    
    # Track statistics
    detection_stats = {}
    
    def on_frame(frame, detections):
        """Callback to track detection statistics."""
        for det in detections:
            class_name = det['class_name']
            detection_stats[class_name] = detection_stats.get(class_name, 0) + 1
    
    # Run detection
    output_path = "output/processed_video.mp4"
    print(f"Processing... (Output will be saved to {output_path})")
    print("Press 'q' to stop early\n")
    
    processor.run(
        on_frame=on_frame,
        save_video=True,  # Save the processed video
        output_path=output_path
    )
    
    # Print detection statistics
    if detection_stats:
        print("\n📊 Detection Statistics:")
        for class_name, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} detections")
    
    # Save detections to JSON
    processor.save_detections("output/video_detections.json")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
