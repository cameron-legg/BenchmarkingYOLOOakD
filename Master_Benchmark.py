import time
import os
import traceback
from cpu_benchmarked import cpu_benchmark
from gpu_benchmarked import gpu_benchmark
from oakd_benchmarked import oakd_benchmark

VIDEO_PATH = "benchmarkvid-small.mp4"
BENCHMARK_DIR = "benchmark_results"
WALL_TIMES_FILE = os.path.join(BENCHMARK_DIR, "wall_times.txt")
ERROR_LOG_FILE = os.path.join(BENCHMARK_DIR, "error_log.txt")

os.makedirs(BENCHMARK_DIR, exist_ok=True)

def log_wall_time(name: str, start: float, end: float):
    duration = round(end - start, 2)
    with open(WALL_TIMES_FILE, "a") as f:
        f.write(f"{name}: {duration} seconds\n")

def log_error(name: str, error: Exception):
    with open(ERROR_LOG_FILE, "a") as f:
        f.write(f"\n[ERROR] {name} failed:\n")
        f.write(traceback.format_exc())
        f.write("\n")

total_time_start = time.time()

# GPU Benchmark
try:
    start = time.time()
    gpu_benchmark(VIDEO_PATH)
    end = time.time()
    log_wall_time("gpu_benchmark", start, end)
except Exception as e:
    log_error("gpu_benchmark", e)

# OAK-D Benchmarks with cooldown
for inference_threads in [2]:
    for shaves in [2, 10, 14]:
        time.sleep(600)  # Wait 10 minutes between runs
        label = f"oakd_benchmark_{shaves}shaves_{inference_threads}ithreads"
        try:
            start = time.time()
            oakd_benchmark(VIDEO_PATH, num_shaves=shaves, inference_threads=inference_threads)
            end = time.time()
            log_wall_time(label, start, end)
        except Exception as e:
            log_error(label, e)

# CPU Benchmark
try:
    start = time.time()
    cpu_benchmark(VIDEO_PATH)
    end = time.time()
    log_wall_time("cpu_benchmark", start, end)
except Exception as e:
    log_error("cpu_benchmark", e)



total_time_end = time.time()
log_wall_time("total benchmark runtime", total_time_start, total_time_end)