# YOLO Fall Detection on Oak-D Camera
### Cameron Legg and Ben Hempelmann

## Models
Models can be found in pt_models/ and oakd_models/. The oakd_models directory contains all of the different models compiled for different shave numbers.

## Dependencies
Run the following commands to use the Oak-D camera
```bash
sudo wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Benchmarks
You can run each benchmark individually by running these commands:
```bash
python3 cpu_benchmarked.py
python3 gpu_benchmarked.py
python3 oakd_benchmarked.py
python3 oakd_stream_benchmarked.py # run if you want to do a live feed from the camera
```

To run all of the benchmarks at the same time, you will need to run:
```bash
python3 Master_Benchmark.py
```

NOTE: The input video is stored as benchmarkvid.mp4 in videos/in. If you want to change this, you will need to run the corresponding benchmark function with a different file name or change the file name in Master_Benchmark.py

## Results
The results will be saved / are located in the benchmark_results directory. There will be CSV files with each benchmark run, along with scripts that can be used to visualize the data. If you don't want to include some benchmarks in the graphs, simply move the CSV files to another folder.
Inferenced videos will be saved to videos/out.

