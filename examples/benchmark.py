"""
Example: Benchmark Different YOLO Models
Compare performance of different YOLO model variants.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import ObjectDetector
from src.utils import benchmark_model, print_system_info


def main():
    """Benchmark different YOLO models."""
    
    print_system_info()
    
    # Models to benchmark (from fastest to most accurate)
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    device = 'cpu'  # Change to 'cuda' if you have GPU
    
    print("🔬 Benchmarking YOLO Models")
    print("="*60)
    print(f"Device: {device}")
    print(f"Models: {', '.join(models)}")
    print("="*60 + "\n")
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print('='*60)
        
        try:
            # Initialize detector
            detector = ObjectDetector(
                model_name=model_name,
                confidence_threshold=0.5,
                device=device
            )
            
            # Run benchmark
            result = benchmark_model(
                detector,
                num_frames=100,
                resolution=(640, 480)
            )
            
            results[model_name] = result
            
        except Exception as e:
            print(f"❌ Error benchmarking {model_name}: {e}")
            continue
    
    # Print comparison
    print("\n" + "="*60)
    print("📊 BENCHMARK COMPARISON")
    print("="*60)
    print(f"{'Model':<15} {'Avg FPS':<12} {'Avg Time (ms)':<15}")
    print("-"*60)
    
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['average_fps']:<12.2f} {result['average_time_ms']:<15.2f}")
    
    print("="*60)
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("  - yolov8n: Best for real-time applications (20-30+ FPS on CPU)")
    print("  - yolov8s: Good balance of speed and accuracy")
    print("  - yolov8m: Better accuracy, slower inference")
    print("  - Use GPU (CUDA) for 5-10x speedup on all models")
    

if __name__ == "__main__":
    main()
