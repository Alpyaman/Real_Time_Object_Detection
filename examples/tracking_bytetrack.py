"""
Example: Object Tracking with ByteTrack
Track objects across frames with persistent IDs using ByteTrack.
Fast and efficient tracking, ideal for scenarios without heavy occlusion.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import ObjectDetector, VideoProcessor
from collections import defaultdict


def main():
    """Run object tracking with ByteTrack."""
    
    print("🎯 Starting Object Tracking with ByteTrack")
    print("="*60)
    print("ByteTrack is optimized for speed and works well when objects")
    print("don't frequently cross paths (e.g., highway traffic, drones).")
    print("="*60)
    print("\nControls:")
    print("  q - Quit")
    print("  s - Toggle recording")
    print("  r - Reset statistics\n")
    
    # Initialize detector
    detector = ObjectDetector(
        model_name="yolov8n.pt",  # Fast model for real-time tracking
        confidence_threshold=0.5,
        device="cpu"  # Change to "cuda" for GPU
    )
    
    # ByteTrack configuration
    bytetrack_config = {
        'track_thresh': 0.5,      # Confidence threshold for track creation
        'track_buffer': 30,       # Frames to keep lost tracks
        'match_thresh': 0.8,      # IoU threshold for matching
        'min_box_area': 10.0      # Minimum box area to track
    }
    
    # Initialize video processor with tracking
    processor = VideoProcessor(
        detector=detector,
        input_source=0,  # Webcam (change to video path for file)
        resize_width=640,
        resize_height=480,
        show_fps=True,
        show_labels=True,
        show_confidence=True,
        tracker='bytetrack',
        tracker_config=bytetrack_config
    )
    
    # Track statistics
    track_history = defaultdict(int)  # Count frames each ID appears
    max_track_id = 0
    
    def on_frame(frame, tracked_objects):
        """Callback to track statistics."""
        nonlocal max_track_id
        
        if tracked_objects:
            for obj in tracked_objects:
                track_id = obj['track_id']
                track_history[track_id] += 1
                max_track_id = max(max_track_id, track_id)
            
            # Print current tracks
            active_ids = [obj['track_id'] for obj in tracked_objects]
            unique_classes = set(obj['class_name'] for obj in tracked_objects)
            
            print(f"Frame {processor.processed_frames}: "
                  f"{len(tracked_objects)} objects tracked "
                  f"(IDs: {sorted(active_ids)}, Classes: {unique_classes})")
    
    # Run tracking
    try:
        processor.run(
            on_frame=on_frame,
            save_video=False  # Set to True to save tracked video
        )
        
        # Print final statistics
        print("\n" + "="*60)
        print("📊 TRACKING STATISTICS")
        print("="*60)
        print(f"Total unique objects tracked: {len(track_history)}")
        print(f"Highest track ID: {max_track_id}")
        print("\nTop 10 longest-tracked objects:")
        
        sorted_tracks = sorted(track_history.items(), key=lambda x: x[1], reverse=True)
        for track_id, frame_count in sorted_tracks[:10]:
            print(f"  ID {track_id}: {frame_count} frames")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTracking stopped by user")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
