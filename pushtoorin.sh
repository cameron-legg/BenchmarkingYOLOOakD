

tar -cf archive.tar oakd_models/fall_detection_6shaves/falldetectionmodel_openvino_2022.1_6shave.blob \
    oakd_models/fall_detection_6shaves/falldetectionmodel.json \
    oakd_models/fall_detection_8shaves/best_openvino_2022.1_8shave.blob \
    oakd_models/fall_detection_8shaves/best.json \
    oakd_models/fall_detection_16shaves/best_openvino_2022.1_16shave.blob \
    oakd_models/fall_detection_16shaves/best.json \
    pt_models/fall_detection/weights/best.pt \
    videos/in/benchmarkvid.mp4 \
    fromfile_cpu.py \
    fromfile.py \
    fromstream.py \
    fromfile_gpu.py \
    RunBenchmarks.sh \
    benchmark_utils.py \
    cpu_benchmarked.py \
    gpu_benchmarked.py \
    oakd_benchmarked.py


scp archive.tar clegg@orin:~/project

rm archive.tar