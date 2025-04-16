

tar -cf archive.tar ../oakd_models/fall_detection_6shaves/falldetectionmodel_openvino_2022.1_6shave.blob \
    ../oakd_models/fall_detection_6shaves/falldetectionmodel.json \
    ../oakd_models/fall_detection_8shaves/best_openvino_2022.1_8shave.blob \
    ../oakd_models/fall_detection_8shaves/best.json \
    ../oakd_models/fall_detection_16shaves/best_openvino_2022.1_16shave.blob \
    ../oakd_models/fall_detection_16shaves/best.json \
    ../oakd_models/fall_detection_1shave/best_openvino_2022.1_1shave.blob \
    ../oakd_models/fall_detection_1shave/best.json \
    ../oakd_models/fall_detection_2shaves/best_openvino_2022.1_2shave.blob \
    ../oakd_models/fall_detection_2shaves/best.json \
    ../oakd_models/fall_detection_4shaves/best_openvino_2022.1_4shave.blob \
    ../oakd_models/fall_detection_4shaves/best.json \
    ../oakd_models/fall_detection_10shaves/best_openvino_2022.1_10shave.blob \
    ../oakd_models/fall_detection_10shaves/best.json \
    ../oakd_models/fall_detection_12shaves/best_openvino_2022.1_12shave.blob \
    ../oakd_models/fall_detection_12shaves/best.json \
    ../oakd_models/fall_detection_14shaves/best_openvino_2022.1_14shave.blob \
    ../oakd_models/fall_detection_14shaves/best.json \
    ../pt_models/fall_detection/weights/best.pt \
    ../videos/in/benchmarkvid.mp4 \
    ../videos/out/placeholder \
    RunBenchmarks.sh \
    ../cpu_benchmarked.py \
    ../gpu_benchmarked.py \
    ../oakd_benchmarked.py \
    ../tmp/placeholder \
    ../Master_Benchmark.py


scp archive.tar clegg@orin:~/project

rm archive.tar