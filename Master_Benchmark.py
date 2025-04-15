from cpu_benchmarked import cpu_benchmark
from gpu_benchmarked import gpu_benchmark
from oakd_benchmarked import oakd_benchmark

# Benchmarks to run
gpu_benchmark("benchmarkvid.mp4")
cpu_benchmark("benchmarkvid.mp4")

oakd_benchmark("benchmarkvid.mp4",num_shaves=6)
oakd_benchmark("benchmarkvid.mp4",num_shaves=8)
oakd_benchmark("benchmarkvid.mp4",num_shaves=16)