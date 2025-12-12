"""
Example: People Counter with Tracking
Count people entering/exiting a defined region using object tracking.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import ObjectDetector, VideoProcessor
import cv2

class PeopleCounter:
    """Count people crossing a line using tracking."""
    
    def __init__(self, line_position=0.5, direction='up'):
        """
        Initialize people counter.
        
        Args:
            line_position: Vertical position of counting line (0-1)
            direction: 'up' or 'down' - which direction counts
        """
        self.line_position = line_position
        self.direction = direction
        
        self.tracked_positions = {}  # track_id -> previous y position
        self.count_in = 0
        self.count_out = 0
        self.counted_ids = set()  # Prevent double counting
    
    def update(self, frame, tracked_objects):
        """Update counter with new frame."""
        height = frame.shape[0]
        line_y = int(height * self.line_position)
        
        # Draw counting line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (10, line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Process tracked objects
        for obj in tracked_objects:
            # Only count people
            if obj['class_name'] != 'person':
                continue
            
            track_id = obj['track_id']
            
            # Get center y position
            bbox = obj['bbox']
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # Check if this object crossed the line
            if track_id in self.tracked_positions:
                prev_y = self.tracked_positions[track_id]
                
                # Crossing from top to bottom
                if prev_y < line_y and center_y >= line_y:
                    if track_id not in self.counted_ids:
                        self.count_in += 1
                        self.counted_ids.add(track_id)
                        print(f"✅ Person IN: ID {track_id} (Total IN: {self.count_in})")
                
                # Crossing from bottom to top
                elif prev_y > line_y and center_y <= line_y:
                    if track_id not in self.counted_ids:
                        self.count_out += 1
                        self.counted_ids.add(track_id)
                        print(f"⬅️ Person OUT: ID {track_id} (Total OUT: {self.count_out})")
            
            # Update position
            self.tracked_positions[track_id] = center_y
        
        # Draw counter
        cv2.rectangle(frame, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"IN: {self.count_in}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self.count_out}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def get_stats(self):
        """Get counting statistics."""
        return {
            'in': self.count_in,
            'out': self.count_out,
            'net': self.count_in - self.count_out,
            'total': len(self.counted_ids)
        }


def main():
    """Run people counter with tracking."""
    
    print("👥 People Counter with Tracking")
    print("="*60)
    print("This example counts people crossing a line.")
    print("Uses ByteTrack to maintain consistent IDs.")
    print("="*60)
    print("\nControls:")
    print("  q - Quit")
    print("  r - Reset counter\n")
    
    # Initialize detector
    detector = ObjectDetector(
        model_name="yolov8n.pt",
        confidence_threshold=0.6,  # Higher confidence for people
        device="cpu"
    )
    
    # Initialize video processor with ByteTrack
    processor = VideoProcessor(
        detector=detector,
        input_source=0,  # Change to video path
        resize_width=640,
        resize_height=480,
        show_fps=True,
        show_labels=True,
        tracker='bytetrack',
        tracker_config={
            'track_thresh': 0.6,
            'track_buffer': 30,
            'match_thresh': 0.8
        }
    )
    
    # Initialize counter
    counter = PeopleCounter(line_position=0.5)
    
    def on_frame(frame, tracked_objects):
        """Process frame with counter."""
        # Filter to only people
        people = [obj for obj in tracked_objects if obj['class_name'] == 'person']
        
        # Update counter (it modifies frame in-place)
        counter.update(frame, people)
    
    # Run counting
    try:
        processor.run(on_frame=on_frame)
        
        # Print final statistics
        stats = counter.get_stats()
        print("\n" + "="*60)
        print("📊 COUNTING RESULTS")
        print("="*60)
        print(f"People entered (IN): {stats['in']}")
        print(f"People exited (OUT): {stats['out']}")
        print(f"Net count: {stats['net']}")
        print(f"Total unique people: {stats['total']}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nCounting stopped by user")
        stats = counter.get_stats()
        print(f"\nCurrent count: IN={stats['in']}, OUT={stats['out']}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
