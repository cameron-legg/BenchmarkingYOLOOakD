import time
from cpu_benchmarked import cpu_benchmark
from gpu_benchmarked import gpu_benchmark
from oakd_benchmarked import oakd_benchmark
import os

VIDEO_PATH = "benchmarkvid.mp4"
WALL_TIMES_FILE = "benchmark_results/wall_times.txt"
os.makedirs("benchmark_results", exist_ok=True)

def log_wall_time(name: str, start: float, end: float):
    duration = round(end - start, 2)
    with open(WALL_TIMES_FILE, "a") as f:
        f.write(f"{name}: {duration} seconds\n")

# GPU Benchmark
start = time.time()
gpu_benchmark(VIDEO_PATH)
end = time.time()
log_wall_time("gpu_benchmark", start, end)

# OAK-D Benchmarks
for shaves in [6, 8, 16]:
    label = f"oakd_benchmark_{shaves}shaves"
    start = time.time()
    oakd_benchmark(VIDEO_PATH, num_shaves=shaves)
    end = time.time()
    log_wall_time(label, start, end)
    
# CPU Benchmark
start = time.time()
cpu_benchmark(VIDEO_PATH)
end = time.time()
log_wall_time("cpu_benchmark", start, end)
