from ultralytics.utils.benchmarks import benchmark
"""
ultralytics docs/params: https://docs.ultralytics.com/modes/benchmark/#arguments
"""
# Benchmark on GPU
benchmark(model='/home/xdoestech/Desktop/object_detection/runs/detect/train24/weights/best.pt', data='/home/xdoestech/Desktop/object_detection/Traffic-Signs-2-4/data.yaml', imgsz=640, half=False, int8=True, device=0)