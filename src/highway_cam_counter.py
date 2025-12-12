"""
Highway Camera Vehicle Counter
Connect to highway camera feeds and count vehicles in real-time.

Usage:
    # Local video file
    python highway_cam_counter.py --source highway_video.mp4
    
    # RTSP stream
    python highway_cam_counter.py --source rtsp://username:password@ip:port/stream
    
    # HTTP/HTTPS stream
    python highway_cam_counter.py --source http://camera-ip/video.mjpg
    
    # With custom settings
    python highway_cam_counter.py --source rtsp://cam.url --line-pos 0.6 --direction vertical
"""

import argparse
import cv2
from datetime import datetime
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detector import ObjectDetector, draw_detections
from tracker import ByteTracker, draw_tracks
from highway_counter import VehicleCounter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Highway Vehicle Counter')
    
    # Source
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: camera index (0), video file, or RTSP/HTTP stream URL'
    )
    
    # Model settings
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLO model to use (n=fastest, x=most accurate)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.3,
        help='Detection confidence threshold (0.0-1.0)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='IOU threshold for NMS (0.0-1.0)'
    )
    
    # Counting line settings
    parser.add_argument(
        '--line-pos',
        type=float,
        default=0.5,
        help='Position of counting line (0.0=top/left, 1.0=bottom/right)'
    )
    
    parser.add_argument(
        '--direction',
        type=str,
        default='horizontal',
        choices=['horizontal', 'vertical'],
        help='Direction of counting line'
    )
    
    # Tracker settings
    parser.add_argument(
        '--track-thresh',
        type=float,
        default=0.4,
        help='Tracking confidence threshold'
    )
    
    parser.add_argument(
        '--track-buffer',
        type=int,
        default=60,
        help='Number of frames to keep lost tracks'
    )
    
    # Output settings
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video file path (optional)'
    )
    
    parser.add_argument(
        '--save-stats',
        type=str,
        default=None,
        help='Path to save statistics JSON (optional)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without display window (for headless servers)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=-1,
        help='Maximum number of frames to process (-1 for unlimited)'
    )
    
    return parser.parse_args()


def connect_to_stream(source: str, retry_attempts: int = 3) -> cv2.VideoCapture:
    """
    Connect to video source with retry logic.
    
    Args:
        source: Video source (camera index, file, or stream URL)
        retry_attempts: Number of connection attempts
        
    Returns:
        VideoCapture object
    """
    # Convert camera index to int if numeric
    if source.isdigit():
        source = int(source)
    
    print(f"\nConnecting to video source: {source}")
    
    for attempt in range(retry_attempts):
        cap = cv2.VideoCapture(source)
        
        if cap.isOpened():
            # Test read
            ret, frame = cap.read()
            if ret:
                print("✓ Connected successfully!")
                print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                
                # Get FPS if available
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    print(f"  FPS: {fps:.1f}")
                
                return cap
        
        print(f"✗ Connection attempt {attempt + 1}/{retry_attempts} failed")
        cap.release()
    
    raise RuntimeError(f"Failed to connect to video source: {source}")


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 60)
    print("Highway Camera Vehicle Counter")
    print("=" * 60)
    
    # Connect to video source
    try:
        cap = connect_to_stream(args.source)
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  - Check if camera/stream is accessible")
        print("  - Verify RTSP URL format: rtsp://username:password@ip:port/stream")
        print("  - Test HTTP stream in browser first")
        print("  - Check network connectivity")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default if FPS not available
    
    # Initialize detector
    print(f"\nInitializing detector: {args.model}")
    detector = ObjectDetector(
        model_name=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Initialize tracker
    print("Initializing tracker: ByteTrack")
    tracker = ByteTracker(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer
    )
    
    # Initialize vehicle counter
    print("Initializing counter:")
    print(f"  Line position: {args.line_pos}")
    print(f"  Line direction: {args.direction}")
    counter = VehicleCounter(
        line_position=args.line_pos,
        line_direction=args.direction,
        min_confidence=args.conf
    )
    
    # Initialize video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"\nSaving output to: {args.output}")
    
    # Processing
    print("\n" + "=" * 60)
    print("Processing started - Press 'q' to quit, 'r' to reset counters")
    print("=" * 60)
    
    frame_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\nEnd of video or connection lost")
                break
            
            frame_count += 1
            
            # Check max frames
            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"\nReached max frames: {args.max_frames}")
                break
            
            # Detect objects
            detections, inference_time = detector.detect(frame)
            
            # Filter for vehicles only (cars, trucks, buses, motorcycles)
            vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
            vehicle_detections = [d for d in detections if d['class_id'] in vehicle_classes]
            
            # Update tracker
            tracks = tracker.update(vehicle_detections)
            
            # Update counter
            counter.update(tracks, frame.shape[:2], fps)
            
            # Draw visualizations
            frame = draw_detections(frame, vehicle_detections)
            frame = draw_tracks(frame, tracks)
            frame = counter.draw(frame)
            
            # Draw FPS
            elapsed = (datetime.now() - start_time).total_seconds()
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                frame,
                f"FPS: {current_fps:.1f}",
                (width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
            
            # Write frame if output specified
            if writer:
                writer.write(frame)
            
            # Display frame
            if not args.no_display:
                cv2.imshow('Highway Vehicle Counter', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    print("\nResetting counters...")
                    counter.reset()
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                stats = counter.get_statistics()
                print(f"Frame {frame_count}: Total={stats['total']}, "
                      f"Cars={stats['cars']}, Trucks={stats['trucks']}, "
                      f"FPS={current_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("Final Statistics")
        print("=" * 60)
        
        stats = counter.get_statistics()
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\nProcessing Summary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Time elapsed: {elapsed:.1f}s")
        print(f"  Average FPS: {frame_count / elapsed:.1f}")
        
        print("\nVehicle Counts:")
        print(f"  Total vehicles: {stats['total']}")
        print(f"  Cars: {stats['cars']} ({stats['car_percentage']:.1f}%)")
        print(f"  Trucks: {stats['trucks']} ({stats['truck_percentage']:.1f}%)")
        
        print("\nDirectional Counts:")
        print(f"  Upward: {stats['cars_up'] + stats['trucks_up']}")
        print(f"  Downward: {stats['cars_down'] + stats['trucks_down']}")
        
        # Save statistics if requested
        if args.save_stats:
            counter.export_statistics(args.save_stats)
        
        print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
