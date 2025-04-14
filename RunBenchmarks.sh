echo "Running the Oak-D, CPU, and GPU benchmarks"

python3 fromfile_gpu.py     # Runs the inferencing on the gpu
python3 fromfile_cpu.py     # Runs the inferencing on the cpu
python3 fromfile.py         # Runs the inferencing on the oak-d camera