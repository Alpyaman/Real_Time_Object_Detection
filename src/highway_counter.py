"""
Highway Vehicle Counter
Counts trucks and cars crossing a virtual line in highway camera feeds.
"""

import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import json


class VehicleCounter:
    """Counts vehicles crossing a virtual line with separate tracking for trucks and cars."""
    
    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {
        'car': 2,
        'motorcycle': 3,
        'bus': 5,
        'truck': 7
    }
    
    # Group into two categories
    CARS = ['car', 'motorcycle']  # Small vehicles
    TRUCKS = ['bus', 'truck']     # Large vehicles
    
    def __init__(
        self,
        line_position: float = 0.5,  # Position of counting line (0.0 to 1.0)
        line_direction: str = 'horizontal',  # 'horizontal' or 'vertical'
        min_confidence: float = 0.4,
        track_memory: int = 30  # Frames to remember track positions
    ):
        """
        Initialize vehicle counter.
        
        Args:
            line_position: Position of counting line (0.0=top/left, 1.0=bottom/right)
            line_direction: 'horizontal' for left-right traffic, 'vertical' for up-down
            min_confidence: Minimum detection confidence to count
            track_memory: Number of frames to remember track positions
        """
        self.line_position = line_position
        self.line_direction = line_direction
        self.min_confidence = min_confidence
        self.track_memory = track_memory
        
        # Counting statistics
        self.cars_count = 0
        self.trucks_count = 0
        self.total_count = 0
        
        # Direction-specific counts (for bidirectional traffic)
        self.cars_up = 0
        self.cars_down = 0
        self.trucks_up = 0
        self.trucks_down = 0
        
        # Track history: {track_id: deque of positions}
        self.track_positions: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.track_memory)
        )
        
        # Tracks that have crossed: {track_id: (vehicle_type, direction)}
        self.crossed_tracks = {}
        
        # Hourly statistics
        self.hourly_stats = defaultdict(lambda: {'cars': 0, 'trucks': 0})
        self.last_hour = datetime.now().hour
        
        # Speed estimation (if available)
        self.track_speeds = {}  # {track_id: speed_mph}
        
    def update(
        self,
        tracks: List,
        frame_shape: Tuple[int, int],
        fps: float = 30.0
    ) -> Dict:
        """
        Update counter with new tracks.
        
        Args:
            tracks: List of tracks from tracker (each with id, bbox, class_id)
            frame_shape: (height, width) of the frame
            fps: Frames per second for speed estimation
            
        Returns:
            Dictionary with counting statistics
        """
        height, width = frame_shape
        
        # Calculate line position in pixels
        if self.line_direction == 'horizontal':
            line_pos = int(height * self.line_position)
        else:
            line_pos = int(width * self.line_position)
        
        # Update hourly stats if hour changed
        current_hour = datetime.now().hour
        if current_hour != self.last_hour:
            self.last_hour = current_hour
        
        # Process each track
        for track in tracks:
            track_id = track[4]  # Assuming track = [x1, y1, x2, y2, track_id, class_id, conf]
            if len(track) < 7:
                continue
                
            class_id = int(track[5])
            conf = track[6]
            
            # Skip if confidence too low
            if conf < self.min_confidence:
                continue
            
            # Skip if not a vehicle
            if class_id not in self.VEHICLE_CLASSES.values():
                continue
            
            # Get vehicle type
            vehicle_type = self._get_vehicle_type(class_id)
            
            # Get center point of bounding box
            x1, y1, x2, y2 = track[:4]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Add to track history
            if self.line_direction == 'horizontal':
                position = cy
            else:
                position = cx
            
            self.track_positions[track_id].append(position)
            
            # Check if track has crossed the line
            if track_id not in self.crossed_tracks and len(self.track_positions[track_id]) >= 2:
                positions = list(self.track_positions[track_id])
                
                # Check for line crossing
                crossed, direction = self._check_crossing(positions, line_pos)
                
                if crossed:
                    # Mark as crossed
                    self.crossed_tracks[track_id] = (vehicle_type, direction)
                    
                    # Update counts
                    self._update_counts(vehicle_type, direction)
                    
                    # Estimate speed if possible
                    if len(positions) >= 10:  # Need enough history for speed
                        speed = self._estimate_speed(positions, fps, height)
                        self.track_speeds[track_id] = speed
        
        # Clean old tracks
        self._cleanup_old_tracks(tracks)
        
        return self.get_statistics()
    
    def _get_vehicle_type(self, class_id: int) -> str:
        """Determine if vehicle is car or truck."""
        # Find class name
        class_name = None
        for name, cid in self.VEHICLE_CLASSES.items():
            if cid == class_id:
                class_name = name
                break
        
        if class_name in self.CARS:
            return 'car'
        elif class_name in self.TRUCKS:
            return 'truck'
        else:
            return 'car'  # Default to car
    
    def _check_crossing(
        self,
        positions: List[int],
        line_pos: int
    ) -> Tuple[bool, str]:
        """
        Check if track crossed the counting line.
        
        Returns:
            (crossed, direction) where direction is 'up' or 'down'
        """
        # Check last two positions
        prev_pos = positions[-2]
        curr_pos = positions[-1]
        
        # Check if crossed from up to down
        if prev_pos < line_pos <= curr_pos:
            return True, 'down'
        
        # Check if crossed from down to up
        if prev_pos > line_pos >= curr_pos:
            return True, 'up'
        
        return False, ''
    
    def _update_counts(self, vehicle_type: str, direction: str):
        """Update counting statistics."""
        self.total_count += 1
        
        if vehicle_type == 'car':
            self.cars_count += 1
            if direction == 'up':
                self.cars_up += 1
            else:
                self.cars_down += 1
            self.hourly_stats[self.last_hour]['cars'] += 1
        else:
            self.trucks_count += 1
            if direction == 'up':
                self.trucks_up += 1
            else:
                self.trucks_down += 1
            self.hourly_stats[self.last_hour]['trucks'] += 1
    
    def _estimate_speed(
        self,
        positions: List[int],
        fps: float,
        frame_height: int,
        real_height_meters: float = 20.0  # Assumed real height covered by frame
    ) -> float:
        """
        Estimate vehicle speed in mph.
        
        Args:
            positions: List of pixel positions
            fps: Frames per second
            frame_height: Height of frame in pixels
            real_height_meters: Real-world height covered by frame
            
        Returns:
            Speed in mph
        """
        if len(positions) < 10:
            return 0.0
        
        # Calculate pixel movement over time
        pixel_distance = abs(positions[-1] - positions[-10])
        time_seconds = 10 / fps
        
        # Convert to real-world distance
        meters_per_pixel = real_height_meters / frame_height
        distance_meters = pixel_distance * meters_per_pixel
        
        # Calculate speed
        speed_mps = distance_meters / time_seconds  # meters per second
        speed_mph = speed_mps * 2.237  # convert to mph
        
        return speed_mph
    
    def _cleanup_old_tracks(self, active_tracks: List):
        """Remove tracks that are no longer active."""
        active_ids = {int(track[4]) for track in active_tracks}
        
        # Remove from track_positions
        old_ids = set(self.track_positions.keys()) - active_ids
        for track_id in old_ids:
            if track_id in self.track_positions:
                del self.track_positions[track_id]
    
    def get_statistics(self) -> Dict:
        """Get current counting statistics."""
        return {
            'total': self.total_count,
            'cars': self.cars_count,
            'trucks': self.trucks_count,
            'cars_up': self.cars_up,
            'cars_down': self.cars_down,
            'trucks_up': self.trucks_up,
            'trucks_down': self.trucks_down,
            'car_percentage': (self.cars_count / self.total_count * 100) if self.total_count > 0 else 0,
            'truck_percentage': (self.trucks_count / self.total_count * 100) if self.total_count > 0 else 0,
            'hourly_stats': dict(self.hourly_stats),
            'active_tracks': len(self.track_positions)
        }
    
    def draw(
        self,
        frame: np.ndarray,
        show_line: bool = True,
        show_stats: bool = True
    ) -> np.ndarray:
        """
        Draw counting line and statistics on frame.
        
        Args:
            frame: Input frame
            show_line: Whether to draw the counting line
            show_stats: Whether to draw statistics
            
        Returns:
            Frame with visualizations
        """
        height, width = frame.shape[:2]
        
        # Draw counting line
        if show_line:
            if self.line_direction == 'horizontal':
                y = int(height * self.line_position)
                cv2.line(frame, (0, y), (width, y), (0, 255, 255), 3)
                cv2.putText(
                    frame, "COUNTING LINE",
                    (10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2
                )
            else:
                x = int(width * self.line_position)
                cv2.line(frame, (x, 0), (x, height), (0, 255, 255), 3)
                cv2.putText(
                    frame, "COUNTING LINE",
                    (x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2
                )
        
        # Draw statistics
        if show_stats:
            stats = self.get_statistics()
            
            # Create semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Draw text
            y_offset = 40
            texts = [
                f"Total Vehicles: {stats['total']}",
                f"Cars: {stats['cars']} ({stats['car_percentage']:.1f}%)",
                f"Trucks: {stats['trucks']} ({stats['truck_percentage']:.1f}%)",
                f"Active Tracks: {stats['active_tracks']}",
                "",
                f"Direction: Up {self.cars_up + self.trucks_up} | Down {self.cars_down + self.trucks_down}"
            ]
            
            for i, text in enumerate(texts):
                cv2.putText(
                    frame, text,
                    (20, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )
        
        return frame
    
    def reset(self):
        """Reset all counters."""
        self.cars_count = 0
        self.trucks_count = 0
        self.total_count = 0
        self.cars_up = 0
        self.cars_down = 0
        self.trucks_up = 0
        self.trucks_down = 0
        self.track_positions.clear()
        self.crossed_tracks.clear()
        self.hourly_stats.clear()
        self.track_speeds.clear()
        self.last_hour = datetime.now().hour
    
    def export_statistics(self, filepath: str):
        """Export statistics to JSON file."""
        stats = self.get_statistics()
        stats['timestamp'] = datetime.now().isoformat()
        stats['speeds'] = self.track_speeds
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics exported to {filepath}")
