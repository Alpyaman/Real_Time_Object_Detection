"""
Object tracking module supporting ByteTrack and DeepSORT algorithms.
Assigns persistent IDs to detected objects across frames.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from filterpy.kalman import KalmanFilter
import cv2
from scipy.optimize import linear_sum_assignment
import lap


class KalmanBBoxTracker:
    """
    Kalman Filter for tracking bounding boxes in image space.
    State: [x_center, y_center, area, aspect_ratio, vx, vy, va, vr]
    """
    
    count = 0  # Global track ID counter
    
    def __init__(self, bbox: np.ndarray):
        """
        Initialize Kalman filter with initial bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        # Define Kalman filter: 8 states, 4 measurements
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement function (observe position and size)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R *= 10.0
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initial state covariance
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        # Initialize state
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBBoxTracker.count
        KalmanBBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """
        Convert [x1, y1, x2, y2] to [x_center, y_center, area, aspect_ratio].
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        a = w * h
        r = w / float(h) if h != 0 else 1.0
        return np.array([x, y, a, r]).reshape((4, 1))
    
    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """
        Convert [x_center, y_center, area, aspect_ratio] to [x1, y1, x2, y2].
        """
        w = np.sqrt(z[2] * z[3])
        h = z[2] / w if w != 0 else 0
        x1 = z[0] - w / 2.0
        y1 = z[1] - h / 2.0
        x2 = z[0] + w / 2.0
        y2 = z[1] + h / 2.0
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    
    def update(self, bbox: np.ndarray):
        """
        Update the state with observed bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
    
    def predict(self) -> np.ndarray:
        """
        Predict the next state.
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.kf.x[:4]))
        
        return self.history[-1][0]
    
    def get_state(self) -> np.ndarray:
        """
        Get current bounding box.
        
        Returns:
            Current bounding box [x1, y1, x2, y2]
        """
        return self._z_to_bbox(self.kf.x[:4])[0]


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.
    
    Args:
        bboxes1: (N, 4) array of [x1, y1, x2, y2]
        bboxes2: (M, 4) array of [x1, y1, x2, y2]
        
    Returns:
        (N, M) array of IoU values
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))
    
    # Expand dims for broadcasting
    bboxes1 = np.expand_dims(bboxes1, 1)  # (N, 1, 4)
    bboxes2 = np.expand_dims(bboxes2, 0)  # (1, M, 4)
    
    # Compute intersection
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    
    # Compute areas
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    
    # Compute IoU
    union = area1 + area2 - intersection
    iou = intersection / np.maximum(union, 1e-6)
    
    return iou


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform linear assignment using LAP (Hungarian algorithm).
    
    Args:
        cost_matrix: (N, M) cost matrix
        thresh: Threshold for valid matches
        
    Returns:
        Tuple of (matches, unmatched_a, unmatched_b)
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])
    
    # Use LAP for assignment
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.array(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)
    
    return matches, unmatched_a, unmatched_b


class ByteTracker:
    """
    ByteTrack: Fast and simple online multi-object tracking.
    Paper: https://arxiv.org/abs/2110.06864
    
    Key idea: Use both high and low confidence detections for association.
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: float = 10.0
    ):
        """
        Initialize ByteTracker.
        
        Args:
            track_thresh: Detection confidence threshold for track creation
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching
            min_box_area: Minimum bounding box area
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        
        self.frame_id = 0
        KalmanBBoxTracker.count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox', 'confidence', 'class_id', 'class_name'
            
        Returns:
            List of tracked objects with added 'track_id' field
        """
        self.frame_id += 1
        
        # Separate high and low confidence detections
        if len(detections) > 0:
            bboxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            
            # Filter by minimum box area
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            valid = areas > self.min_box_area
            
            detections = [det for i, det in enumerate(detections) if valid[i]]
            if len(detections) == 0:
                return self._process_no_detections()
            
            bboxes = bboxes[valid]
            scores = scores[valid]
            
            # Split into high and low confidence
            high_mask = scores >= self.track_thresh
            low_mask = scores < self.track_thresh
            
            high_dets = [det for i, det in enumerate(detections) if high_mask[i]]
            low_dets = [det for i, det in enumerate(detections) if low_mask[i]]
        else:
            return self._process_no_detections()
        
        # Predict current locations of existing tracks
        for track in self.tracked_tracks:
            track.predict()
        
        # First association with high confidence detections
        tracked_output = []
        
        if len(high_dets) > 0:
            high_bboxes = np.array([det['bbox'] for det in high_dets])
            track_bboxes = np.array([track.get_state() for track in self.tracked_tracks])
            
            if len(track_bboxes) > 0:
                iou_matrix = iou_batch(track_bboxes, high_bboxes)
                matches, unmatched_tracks, unmatched_dets = linear_assignment(
                    1 - iou_matrix, 1 - self.match_thresh
                )
                
                # Update matched tracks
                for track_idx, det_idx in matches:
                    track = self.tracked_tracks[track_idx]
                    det = high_dets[det_idx]
                    track.update(np.array(det['bbox']))
                    
                    tracked_output.append({
                        **det,
                        'track_id': track.id,
                        'track_age': track.age,
                        'hits': track.hits
                    })
                
                # Handle unmatched detections - create new tracks
                for det_idx in unmatched_dets:
                    det = high_dets[det_idx]
                    track = KalmanBBoxTracker(np.array(det['bbox']))
                    self.tracked_tracks.append(track)
                    
                    tracked_output.append({
                        **det,
                        'track_id': track.id,
                        'track_age': track.age,
                        'hits': track.hits
                    })
                
                # Move unmatched tracks to lost
                for track_idx in unmatched_tracks:
                    track = self.tracked_tracks[track_idx]
                    if track.time_since_update <= self.track_buffer:
                        self.lost_tracks.append(track)
                    else:
                        self.removed_tracks.append(track)
                
                self.tracked_tracks = [t for i, t in enumerate(self.tracked_tracks) 
                                      if i not in unmatched_tracks]
            else:
                # No existing tracks, create new ones
                for det in high_dets:
                    track = KalmanBBoxTracker(np.array(det['bbox']))
                    self.tracked_tracks.append(track)
                    
                    tracked_output.append({
                        **det,
                        'track_id': track.id,
                        'track_age': track.age,
                        'hits': track.hits
                    })
        
        # Second association with low confidence detections for lost tracks
        if len(low_dets) > 0 and len(self.lost_tracks) > 0:
            low_bboxes = np.array([det['bbox'] for det in low_dets])
            lost_bboxes = np.array([track.get_state() for track in self.lost_tracks])
            
            iou_matrix = iou_batch(lost_bboxes, low_bboxes)
            matches, _, _ = linear_assignment(1 - iou_matrix, 1 - 0.5)  # Lower threshold
            
            for track_idx, det_idx in matches:
                track = self.lost_tracks[track_idx]
                det = low_dets[det_idx]
                track.update(np.array(det['bbox']))
                self.tracked_tracks.append(track)
                
                tracked_output.append({
                    **det,
                    'track_id': track.id,
                    'track_age': track.age,
                    'hits': track.hits
                })
            
            # Remove recovered tracks from lost
            recovered_indices = [m[0] for m in matches]
            self.lost_tracks = [t for i, t in enumerate(self.lost_tracks) 
                               if i not in recovered_indices]
        
        # Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks 
                           if t.time_since_update <= self.track_buffer]
        
        return tracked_output
    
    def _process_no_detections(self) -> List[Dict]:
        """Process frame with no detections."""
        for track in self.tracked_tracks:
            track.predict()
            if track.time_since_update > self.track_buffer:
                self.removed_tracks.append(track)
        
        self.tracked_tracks = [t for t in self.tracked_tracks 
                              if t.time_since_update <= self.track_buffer]
        
        return []
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        KalmanBBoxTracker.count = 0


class SimpleDeepSORT:
    """
    Simplified DeepSORT-style tracker with appearance features.
    Uses color histograms as a simple appearance descriptor.
    
    For full DeepSORT with deep Re-ID features, use a dedicated library.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        appearance_weight: float = 0.3
    ):
        """
        Initialize SimpleDeepSORT tracker.
        
        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
            appearance_weight: Weight of appearance similarity (0-1)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        
        self.tracks = []
        self.track_features = {}  # Store appearance features
        self.frame_id = 0
        KalmanBBoxTracker.count = 0
    
    def _extract_appearance(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract simple appearance features (color histogram).
        
        Args:
            frame: Image frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Feature vector
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(48)  # 16 bins * 3 channels
        
        roi = frame[y1:y2, x1:x2]
        
        # Compute color histogram for each channel
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([roi], [i], None, [16], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-6)  # Normalize
            hist_features.append(hist)
        
        return np.concatenate(hist_features)
    
    def _appearance_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute appearance distance (1 - cosine similarity)."""
        dot = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        cosine_sim = dot / (norm1 * norm2)
        return 1.0 - cosine_sim
    
    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts
            frame: Current frame (required for appearance features)
            
        Returns:
            List of tracked objects with 'track_id'
        """
        self.frame_id += 1
        
        # Predict current locations
        for track in self.tracks:
            track.predict()
        
        if len(detections) == 0:
            # Remove dead tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return []
        
        # Extract appearance features if frame provided
        det_features = []
        if frame is not None:
            for det in detections:
                feat = self._extract_appearance(frame, np.array(det['bbox']))
                det_features.append(feat)
        
        # Compute cost matrix
        if len(self.tracks) > 0:
            track_bboxes = np.array([t.get_state() for t in self.tracks])
            det_bboxes = np.array([det['bbox'] for det in detections])
            
            # IoU distance
            iou_matrix = iou_batch(track_bboxes, det_bboxes)
            iou_distance = 1 - iou_matrix
            
            # Appearance distance
            if frame is not None and len(det_features) > 0:
                app_distance = np.zeros((len(self.tracks), len(detections)))
                for i, track in enumerate(self.tracks):
                    if track.id in self.track_features:
                        track_feat = self.track_features[track.id]
                        for j, det_feat in enumerate(det_features):
                            app_distance[i, j] = self._appearance_distance(track_feat, det_feat)
                
                # Combined distance
                cost_matrix = (1 - self.appearance_weight) * iou_distance + \
                             self.appearance_weight * app_distance
            else:
                cost_matrix = iou_distance
            
            # Hungarian algorithm
            matches, unmatched_tracks, unmatched_dets = linear_assignment(
                cost_matrix, 1 - self.iou_threshold
            )
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_tracks = []
            unmatched_dets = np.arange(len(detections))
        
        # Update matched tracks
        tracked_output = []
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            
            track.update(np.array(det['bbox']))
            
            # Update appearance feature
            if frame is not None and len(det_features) > 0:
                # Exponential moving average
                if track.id in self.track_features:
                    old_feat = self.track_features[track.id]
                    new_feat = det_features[det_idx]
                    self.track_features[track.id] = 0.9 * old_feat + 0.1 * new_feat
                else:
                    self.track_features[track.id] = det_features[det_idx]
            
            if track.hits >= self.min_hits or self.frame_id <= self.min_hits:
                tracked_output.append({
                    **det,
                    'track_id': track.id,
                    'track_age': track.age,
                    'hits': track.hits
                })
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track = KalmanBBoxTracker(np.array(det['bbox']))
            self.tracks.append(track)
            
            # Store appearance feature
            if frame is not None and len(det_features) > 0:
                self.track_features[track.id] = det_features[det_idx]
            
            if self.min_hits == 1:
                tracked_output.append({
                    **det,
                    'track_id': track.id,
                    'track_age': track.age,
                    'hits': track.hits
                })
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Clean up feature cache
        active_ids = {t.id for t in self.tracks}
        self.track_features = {k: v for k, v in self.track_features.items() if k in active_ids}
        
        return tracked_output
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.track_features = {}
        self.frame_id = 0
        KalmanBBoxTracker.count = 0


def draw_tracks(
    frame: np.ndarray,
    tracked_objects: List[Dict],
    show_trajectory: bool = True,
    trajectory_length: int = 30,
    bbox_thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw tracked objects with persistent IDs and trajectories.
    
    Args:
        frame: Input image
        tracked_objects: List of tracked object dicts with 'track_id'
        show_trajectory: Whether to show object trajectories
        trajectory_length: Number of points in trajectory
        bbox_thickness: Thickness of bounding boxes
        font_scale: Scale of text labels
        
    Returns:
        Frame with drawn tracks
    """
    # Generate consistent colors for track IDs
    np.random.seed(42)
    colors = {}
    
    for obj in tracked_objects:
        track_id = obj['track_id']
        bbox = obj['bbox']
        class_name = obj['class_name']
        confidence = obj.get('confidence', 0)
        
        # Get consistent color for this track ID
        if track_id not in colors:
            colors[track_id] = tuple(map(int, np.random.randint(50, 255, 3)))
        color = colors[track_id]
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thickness)
        
        # Draw label with track ID
        label = f"ID:{track_id} {class_name} {confidence:.2f}"
        
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        
        # Background for label
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2
        )
        
        # Draw center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)
    
    return frame
