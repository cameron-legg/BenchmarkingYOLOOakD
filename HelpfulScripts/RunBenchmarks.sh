echo "Running the Oak-D, CPU, and GPU benchmarks"

echo "GPU"
python3 gpu_benchmarked.py     # Runs the inferencing on the gpu
echo "Oak-D"
python3 oakd_benchmarked.py         # Runs the inferencing on the oak-d camera
echo "CPU"
python3 cpu_benchmarked.py     # Runs the inferencing on the cpu