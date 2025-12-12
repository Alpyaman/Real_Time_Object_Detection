"""
Example: Object Tracking with DeepSORT
Track objects using appearance features for robust re-identification.
Better for scenarios with occlusions (people walking behind objects).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import ObjectDetector, VideoProcessor
from collections import defaultdict


def main():
    """Run object tracking with DeepSORT."""
    
    print("🎯 Starting Object Tracking with DeepSORT")
    print("="*60)
    print("DeepSORT uses appearance features (color histograms) to")
    print("remember objects. Great when objects temporarily disappear")
    print("behind obstacles (e.g., person behind a pillar).")
    print("="*60)
    print("\nControls:")
    print("  q - Quit")
    print("  s - Toggle recording")
    print("  r - Reset statistics\n")
    
    # Initialize detector
    detector = ObjectDetector(
        model_name="yolov8n.pt",
        confidence_threshold=0.5,
        device="cpu"  # Change to "cuda" for GPU
    )
    
    # DeepSORT configuration
    deepsort_config = {
        'max_age': 30,            # Max frames to keep track without detection
        'min_hits': 3,            # Min detections before track is confirmed
        'iou_threshold': 0.3,     # IoU threshold for matching
        'appearance_weight': 0.3  # Weight of appearance vs motion (0-1)
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
        tracker='deepsort',
        tracker_config=deepsort_config
    )
    
    # Track statistics
    track_history = defaultdict(lambda: {'frames': 0, 'class': None, 'max_age': 0})
    track_switches = 0  # Count ID switches (potential re-identifications)
    prev_active_ids = set()
    
    def on_frame(frame, tracked_objects):
        """Callback to track statistics."""
        nonlocal track_switches, prev_active_ids
        
        if tracked_objects:
            current_active_ids = set()
            
            for obj in tracked_objects:
                track_id = obj['track_id']
                current_active_ids.add(track_id)
                
                # Update history
                track_history[track_id]['frames'] += 1
                track_history[track_id]['class'] = obj['class_name']
                track_history[track_id]['max_age'] = max(
                    track_history[track_id]['max_age'],
                    obj.get('track_age', 0)
                )
            
            # Detect potential re-identifications (new IDs appearing)
            new_ids = current_active_ids - prev_active_ids
            if new_ids and prev_active_ids:
                track_switches += len(new_ids)
            
            prev_active_ids = current_active_ids
            
            # Print current state
            class_counts = defaultdict(int)
            for obj in tracked_objects:
                class_counts[obj['class_name']] += 1
            
            print(f"Frame {processor.processed_frames}: "
                  f"{len(tracked_objects)} tracked | "
                  f"Classes: {dict(class_counts)} | "
                  f"Active IDs: {sorted(current_active_ids)}")
    
    # Run tracking
    try:
        processor.run(
            on_frame=on_frame,
            save_video=False  # Set to True to save tracked video
        )
        
        # Print final statistics
        print("\n" + "="*60)
        print("📊 DEEPSORT TRACKING STATISTICS")
        print("="*60)
        print(f"Total unique objects tracked: {len(track_history)}")
        print(f"Potential re-identifications: {track_switches}")
        
        # Group by class
        class_tracks = defaultdict(list)
        for track_id, info in track_history.items():
            if info['class']:
                class_tracks[info['class']].append((track_id, info['frames']))
        
        print("\nTracked objects by class:")
        for class_name, tracks in sorted(class_tracks.items()):
            print(f"\n  {class_name}: {len(tracks)} unique objects")
            # Show top 5 longest tracks for this class
            sorted_tracks = sorted(tracks, key=lambda x: x[1], reverse=True)[:5]
            for track_id, frame_count in sorted_tracks:
                print(f"    ID {track_id}: {frame_count} frames")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTracking stopped by user")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
