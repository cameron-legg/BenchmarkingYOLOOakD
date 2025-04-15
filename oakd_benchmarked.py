def oakd_benchmark(video_name, num_shaves=8):
    import depthai as dai
    import json
    import os
    import cv2
    import numpy as np
    import time
    import datetime
    import csv
    from collections import deque
    from jetsontools import Tegrastats, get_powerdraw, parse_tegrastats, filter_data

    # Model selection
    # num_shaves = int(input("Number of shaves: 6,8,16"))
    YOLOV8N_MODEL = ""
    YOLOV8N_CONFIG = ""

    if num_shaves == 6:
        print("RUNNING 6 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_6shaves/falldetectionmodel_openvino_2022.1_6shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_6shaves/falldetectionmodel.json"
    elif num_shaves == 8:
        print("RUNNING 8 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_8shaves/best_openvino_2022.1_8shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_8shaves/best.json"
    else:
        print("RUNNING 16 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_16shaves/best_openvino_2022.1_16shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_16shaves/best.json"

    # File setup
    saved_file_timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    INPUT_VIDEO = f"videos/in/{video_name}"
    OUTPUT_VIDEO = f"videos/out/result_oakd_{saved_file_timestamp}.mp4"
    BENCHMARK_CSV = f"benchmark_results/oakd_{saved_file_timestamp}.csv"
    TEMP_POWER = f"tmp/temp_power_oakd_{saved_file_timestamp}.txt"
    os.makedirs("benchmark_results", exist_ok=True)

    # Constants
    CAMERA_PREVIEW_DIM = (640, 640)
    LABELS = ['Fall Detected', 'Walking', 'Sitting']

    def load_config(config_path):
        with open(config_path) as f:
            return json.load(f)

    def create_image_pipeline(config_path, model_path):
        pipeline = dai.Pipeline()
        model_config = load_config(config_path)
        nnConfig = model_config.get("nn_config", {})
        metadata = nnConfig.get("NN_specific_metadata", {})

        detectionIN = pipeline.create(dai.node.XLinkIn)
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        nnOut = pipeline.create(dai.node.XLinkOut)

        nnOut.setStreamName("nn")
        detectionIN.setStreamName("detection_in")
        detectionNetwork.setConfidenceThreshold(metadata.get("confidence_threshold", {}))
        detectionNetwork.setNumClasses(metadata.get("classes", {}))
        detectionNetwork.setCoordinateSize(metadata.get("coordinates", {}))
        detectionNetwork.setAnchors(metadata.get("anchors", {}))
        detectionNetwork.setAnchorMasks(metadata.get("anchor_masks", {}))
        detectionNetwork.setIouThreshold(metadata.get("iou_threshold", {}))
        detectionNetwork.setBlobPath(model_path)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)

        detectionIN.out.link(detectionNetwork.input)
        detectionNetwork.out.link(nnOut.input)

        return pipeline

    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        resized = cv2.resize(arr, shape)
        return resized.transpose(2, 0, 1)

    def frame_norm(frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    def annotate_frame(frame, detections, fps):
        color = (0, 0, 255)
        for detection in detections:
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, LABELS[detection.label], (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    # Init video & pipeline
    pipeline = create_image_pipeline(YOLOV8N_CONFIG, YOLOV8N_MODEL)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps_cap, (frame_width, frame_height))

    # Benchmarking buffers
    fps_queue = deque(maxlen=300)
    timestamps = []

    with open(BENCHMARK_CSV, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "fps", "VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC", "VDD_TOTAL"])

        with Tegrastats(TEMP_POWER, interval=5):
            with dai.Device(pipeline) as device:
                print(f"[INFO] Connected to OAK-D device: {device.getDeviceInfo()}")

                detectionIN = device.getInputQueue("detection_in")
                detectionNN = device.getOutputQueue("nn")

                start_time = time.time()
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame_count >= int(fps_cap * 60 * 3):
                        break

                    t_start = time.time()
                    frame_count += 1

                    nn_data = dai.NNData()
                    nn_data.setLayer("input", to_planar(frame, CAMERA_PREVIEW_DIM))
                    detectionIN.send(nn_data)

                    inDet = detectionNN.get()
                    detections = inDet.detections if inDet else []

                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    fps_queue.append(current_fps)
                    timestamps.append((t_start, time.time()))

                    frame = annotate_frame(frame, detections, current_fps)
                    out.write(frame)

                    if frame_count % 300 == 0:
                        parsed = parse_tegrastats(TEMP_POWER)
                        filtered, _ = filter_data(parsed, timestamps[-300:])
                        power_data = get_powerdraw(filtered)

                        avg_fps = sum(fps_queue) / len(fps_queue)
                        writer.writerow([
                            frame_count,
                            avg_fps,
                            power_data["VDD_IN"].mean,
                            power_data["VDD_CPU_GPU_CV"].mean,
                            power_data["VDD_SOC"].mean,
                            power_data["VDD_TOTAL"].mean,
                        ])

    # Cleanup
    cap.release()
    out.release()
    print(f"[INFO] Saved video to {OUTPUT_VIDEO}")
    print(f"[INFO] Benchmark CSV saved to {BENCHMARK_CSV}")
