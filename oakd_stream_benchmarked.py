def oakd_benchmark_live(num_shaves=8, inference_threads=2):
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

    YOLOV8N_MODEL = ""
    YOLOV8N_CONFIG = ""

    if num_shaves == 6:
        print("RUNNING 6 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_6shaves/falldetectionmodel_openvino_2022.1_6shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_6shaves/falldetectionmodel.json"
    elif num_shaves == 1:
        print("RUNNING 1 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_1shave/best_openvino_2022.1_1shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_1shave/best.json"
    elif num_shaves == 2:
        print("RUNNING 2 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_2shaves/best_openvino_2022.1_2shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_2shaves/best.json"
    elif num_shaves == 4:
        print("RUNNING 4 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_4shaves/best_openvino_2022.1_4shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_4shaves/best.json"
    elif num_shaves == 8:
        print("RUNNING 8 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_8shaves/best_openvino_2022.1_8shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_8shaves/best.json"
    elif num_shaves == 10:
        print("RUNNING 10 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_10shaves/best_openvino_2022.1_10shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_10shaves/best.json"
    elif num_shaves == 12:
        print("RUNNING 12 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_12shaves/best_openvino_2022.1_12shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_12shaves/best.json"
    elif num_shaves == 14:
        print("RUNNING 14 SHAVES MODEL")
        YOLOV8N_MODEL = "oakd_models/fall_detection_14shaves/best_openvino_2022.1_14shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_14shaves/best.json"
    else:
        print("RUNNING 16 SHAVES MODEL")
        num_shaves = 16
        YOLOV8N_MODEL = "oakd_models/fall_detection_16shaves/best_openvino_2022.1_16shave.blob"
        YOLOV8N_CONFIG = "oakd_models/fall_detection_16shaves/best.json"

    saved_file_timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    OUTPUT_VIDEO = f"videos/out/result_oakd_{num_shaves}_shaves_{inference_threads}_threads_{saved_file_timestamp}.mp4"
    BENCHMARK_CSV = f"benchmark_results/oakd_{num_shaves}_shaves_{inference_threads}_threads_{saved_file_timestamp}.csv"
    TEMP_POWER = f"tmp/temp_power_oakd_{num_shaves}_shaves_{inference_threads}_threads_{saved_file_timestamp}.txt"
    os.makedirs("benchmark_results", exist_ok=True)

    CAMERA_PREVIEW_DIM = (640, 640)
    LABELS = ['Fall Detected', 'Walking', 'Sitting']

    def load_config(config_path):
        with open(config_path) as f:
            return json.load(f)

    def create_pipeline(config_path, model_path):
        pipeline = dai.Pipeline()
        model_config = load_config(config_path)
        nnConfig = model_config.get("nn_config", {})
        metadata = nnConfig.get("NN_specific_metadata", {})

        cam_rgb = pipeline.create(dai.node.ColorCamera)
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        xout_video = pipeline.create(dai.node.XLinkOut)
        xout_nn = pipeline.create(dai.node.XLinkOut)

        cam_rgb.setPreviewSize(*CAMERA_PREVIEW_DIM)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)

        xout_video.setStreamName("video")
        xout_nn.setStreamName("nn")

        detectionNetwork.setBlobPath(model_path)
        detectionNetwork.setConfidenceThreshold(metadata.get("confidence_threshold", 0.5))
        detectionNetwork.setNumClasses(metadata.get("classes", 3))
        detectionNetwork.setCoordinateSize(metadata.get("coordinates", 4))
        detectionNetwork.setAnchors(metadata.get("anchors", []))
        detectionNetwork.setAnchorMasks(metadata.get("anchor_masks", {}))
        detectionNetwork.setIouThreshold(metadata.get("iou_threshold", 0.5))
        detectionNetwork.setNumInferenceThreads(inference_threads)

        cam_rgb.preview.link(detectionNetwork.input)
        cam_rgb.preview.link(xout_video.input)
        detectionNetwork.out.link(xout_nn.input)

        return pipeline

    def annotate_frame(frame, detections, fps):
        color = (0, 0, 255)
        for detection in detections:
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, LABELS[detection.label], (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def frame_norm(frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    pipeline = create_pipeline(YOLOV8N_CONFIG, YOLOV8N_MODEL)
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30, CAMERA_PREVIEW_DIM)

    fps_queue = deque(maxlen=300)
    timestamps = []

    with open(BENCHMARK_CSV, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "fps", "VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC", "VDD_TOTAL"])

        with Tegrastats(TEMP_POWER, interval=5):
            with dai.Device(pipeline) as device:
                video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
                detectionNN = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

                start_time = time.time()
                frame_count = 0

                while True:
                    in_frame = video.get()
                    in_det = detectionNN.get()

                    frame = in_frame.getCvFrame()
                    detections = in_det.detections

                    t_start = time.time()
                    frame_count += 1

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

    out.release()
    print(f"[INFO] Saved video to {OUTPUT_VIDEO}")
    print(f"[INFO] Benchmark CSV saved to {BENCHMARK_CSV}")


if __name__ == "__main__":
    oakd_benchmark_live("benchmarkvid.mp4",14,2)