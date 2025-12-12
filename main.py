"""
Real-Time Object Detection - Main Entry Point
Detects objects in real-time from webcam or video file using YOLO.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import ObjectDetector, VideoProcessor, load_config, setup_output_directory, print_system_info


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection using YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with webcam (default)
  python main.py

  # Run with video file
  python main.py --source video.mp4

  # Use different YOLO model
  python main.py --model yolov8s

  # Save output video
  python main.py --save --output output/result.mp4

  # Show system info
  python main.py --info

  # Use GPU (CUDA)
  python main.py --device cuda

Controls:
  q - Quit
  s - Toggle recording
  r - Reset statistics
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source (0 for webcam, or path to video file)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n',
        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        help='YOLO model variant (n=nano/fastest, x=extra large/most accurate)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run inference on'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detections (0.0-1.0)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (0.0-1.0)'
    )
    
    parser.add_argument(
        '--resize',
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=('WIDTH', 'HEIGHT'),
        help='Resize frames to WIDTH HEIGHT for processing'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        default=0,
        help='Skip frames for better performance (0 = no skip)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save output video'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/output.mp4',
        help='Output video path'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without display window (for headless systems)'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Print system information and exit'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process'
    )
    
    args = parser.parse_args()
    
    # Print system info and exit if requested
    if args.info:
        print_system_info()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config['model']['name'] = args.model + '.pt'
    if args.device:
        config['model']['device'] = args.device
    if args.confidence:
        config['model']['confidence_threshold'] = args.confidence
    if args.iou:
        config['model']['iou_threshold'] = args.iou
    
    # Parse video source
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source
    
    # Setup output directory
    if args.save:
        setup_output_directory(Path(args.output).parent)
    
    # Print configuration
    print("\n" + "="*60)
    print("🎯 REAL-TIME OBJECT DETECTION")
    print("="*60)
    print(f"Model: {config['model']['name']}")
    print(f"Device: {config['model']['device']}")
    print(f"Video Source: {video_source}")
    print(f"Resolution: {args.resize[0]}x{args.resize[1]}")
    print(f"Confidence Threshold: {config['model']['confidence_threshold']}")
    print(f"IoU Threshold: {config['model']['iou_threshold']}")
    if args.skip > 0:
        print(f"Frame Skip: {args.skip}")
    if args.save:
        print(f"Output: {args.output}")
    print("="*60 + "\n")
    
    try:
        # Initialize detector
        detector = ObjectDetector(
            model_name=config['model']['name'],
            confidence_threshold=config['model']['confidence_threshold'],
            iou_threshold=config['model']['iou_threshold'],
            device=config['model']['device']
        )
        
        # Initialize video processor
        processor = VideoProcessor(
            detector=detector,
            input_source=video_source,
            resize_width=args.resize[0],
            resize_height=args.resize[1],
            frame_skip=args.skip,
            show_fps=config['display']['show_fps'],
            show_labels=config['display']['show_labels'],
            show_confidence=config['display']['show_confidence'],
            bbox_thickness=config['display']['bbox_thickness'],
            font_scale=config['display']['font_scale']
        )
        
        # Run detection
        processor.run(
            save_video=args.save,
            output_path=args.output,
            max_frames=args.max_frames
        )
        
        # Save detections if configured
        if config['output']['save_detections']:
            detection_path = Path(config['output']['output_path']) / 'detections.json'
            processor.save_detections(str(detection_path))
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
