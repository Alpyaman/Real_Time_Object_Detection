"""
Example: Compare ByteTrack vs DeepSORT Performance
Side-by-side comparison of tracking algorithms.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import ByteTracker, SimpleDeepSORT
import numpy as np


def benchmark_tracker(tracker, tracker_name, num_frames=100):
    """Benchmark tracker performance."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {tracker_name}")
    print('='*60)
    
    # Simulate detections
    times = []
    
    for i in range(num_frames):
        # Generate random detections (3-5 objects)
        num_objects = np.random.randint(3, 6)
        detections = []
        
        for _ in range(num_objects):
            x1 = np.random.randint(0, 400)
            y1 = np.random.randint(0, 300)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            
            detections.append({
                'bbox': [x1, y1, x1 + w, y1 + h],
                'confidence': np.random.uniform(0.5, 0.95),
                'class_id': np.random.randint(0, 3),
                'class_name': ['person', 'car', 'bicycle'][np.random.randint(0, 3)]
            })
        
        # Time the update
        start = time.time()
        
        if tracker_name == "DeepSORT":
            # DeepSORT needs a frame
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            tracked = tracker.update(detections, dummy_frame)
        else:
            tracked = tracker.update(detections)
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{num_frames} frames")
    
    # Calculate statistics
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    max_time = np.max(times) * 1000
    
    print("\nResults:")
    print(f"  Average time: {avg_time:.2f} ms/frame")
    print(f"  Std deviation: {std_time:.2f} ms")
    print(f"  Max time: {max_time:.2f} ms")
    print(f"  Estimated FPS: {1000 / avg_time:.2f}")
    
    return avg_time


def main():
    """Compare tracking algorithms."""
    
    print("\n🔬 TRACKING ALGORITHM COMPARISON")
    print("="*60)
    print("This benchmark compares ByteTrack and DeepSORT")
    print("in terms of processing speed.")
    print("="*60)
    
    num_frames = 100
    
    # Benchmark ByteTrack
    bytetrack = ByteTracker(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8
    )
    bytetrack_time = benchmark_tracker(bytetrack, "ByteTrack", num_frames)
    
    # Benchmark DeepSORT
    deepsort = SimpleDeepSORT(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        appearance_weight=0.3
    )
    deepsort_time = benchmark_tracker(deepsort, "DeepSORT", num_frames)
    
    # Print comparison
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Algorithm':<15} {'Avg Time (ms)':<18} {'Est. FPS':<12} {'Speed'}")
    print("-"*60)
    print(f"{'ByteTrack':<15} {bytetrack_time:<18.2f} {1000/bytetrack_time:<12.2f} {'⚡ Fastest'}")
    print(f"{'DeepSORT':<15} {deepsort_time:<18.2f} {1000/deepsort_time:<12.2f} {'🎯 More Robust'}")
    print("="*60)
    
    speedup = deepsort_time / bytetrack_time
    print(f"\nByteTrack is {speedup:.2f}x faster than DeepSORT")
    
    print("\n💡 Recommendations:")
    print("  • ByteTrack: Use for high-speed tracking (traffic, sports)")
    print("  • DeepSORT: Use when objects frequently occlude each other")
    print("  • Both work well for most real-time applications")
    

if __name__ == "__main__":
    main()
