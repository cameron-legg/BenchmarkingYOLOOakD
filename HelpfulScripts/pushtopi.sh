

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
    fromstream.py


scp archive.tar cameron@192.168.1.47:~/

rm archive.tar