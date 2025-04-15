import depthai as dai
import json
import os
import cv2
import numpy as np
import time
from pathlib import Path
from benchmark_utils import save_unified_benchmark
from jetsontools import Tegrastats, parse_tegrastats, get_powerdraw

ENABLE_BENCHMARK = True

# Model selection
num_shaves = 8
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

INPUT_VIDEO = "videos/in/benchmarkvid.mp4"
OUTPUT_VIDEO = "videos/out/result_oakd.mp4"
BENCHMARK_CSV = "benchmark.csv"
DETECTIONS_JSON = "results.json"
CAMERA_PREVIEW_DIM = (640, 640)
LABELS = ['Fall Detected', 'Walking', 'Sitting']
BENCHMARK_INTERVAL = 300

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)

def create_image_pipeline(config_path, model_path):
    pipeline = dai.Pipeline()
    config = load_config(config_path)
    nn_config = config.get("nn_config", {}).get("NN_specific_metadata", {})

    detectionIN = pipeline.create(dai.node.XLinkIn)
    detectionNN = pipeline.create(dai.node.YoloDetectionNetwork)
    nnOut = pipeline.create(dai.node.XLinkOut)

    nnOut.setStreamName("nn")
    detectionIN.setStreamName("detection_in")
    detectionNN.setBlobPath(model_path)
    detectionNN.setConfidenceThreshold(nn_config.get("confidence_threshold", 0.5))
    detectionNN.setNumClasses(nn_config.get("classes", 1))
    detectionNN.setCoordinateSize(nn_config.get("coordinates", 4))
    detectionNN.setAnchors(nn_config.get("anchors", []))
    detectionNN.setAnchorMasks(nn_config.get("anchor_masks", {}))
    detectionNN.setIouThreshold(nn_config.get("iou_threshold", 0.5))
    detectionNN.setNumInferenceThreads(2)
    detectionNN.input.setBlocking(False)

    detectionIN.out.link(detectionNN.input)
    detectionNN.out.link(nnOut.input)
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
    for det in detections:
        bbox = frame_norm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
        cv2.putText(frame, LABELS[det.label], (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.putText(frame, f"{int(det.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Setup
pipeline = create_image_pipeline(YOLOV8N_CONFIG, YOLOV8N_MODEL)
cap = cv2.VideoCapture(INPUT_VIDEO)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_cap = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps_cap, (frame_width, frame_height))

benchmark_data = []
detection_results = []
interval_data = {'fps': [], 'confidence': []}
header_written = [False]
tegrastats_path = Path("output_stats_oakd.txt")

with dai.Device(pipeline) as device, Tegrastats(tegrastats_path, interval=5):
    print(f"[INFO] Connected to OAK device: {device.getDeviceInfo().name}")

    detectionIN = device.getInputQueue("detection_in")
    detectionNN = device.getOutputQueue("nn")

    start_time = time.time()
    frame_count = 0
    total_inference_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= int(fps_cap * 60 * 3):
            break

        frame_count += 1

        nn_data = dai.NNData()
        nn_data.setLayer("input", to_planar(frame, CAMERA_PREVIEW_DIM))
        detectionIN.send(nn_data)

        inference_start = time.time()
        inDet = detectionNN.get()
        inference_time = time.time() - inference_start
        total_inference_time += inference_time

        detections = inDet.detections if inDet else []
        current_fps = frame_count / (time.time() - start_time)

        frame = annotate_frame(frame, detections, current_fps)
        out.write(frame)

        interval_data['fps'].append(current_fps)
        interval_data['confidence'].extend([float(det.confidence) for det in detections])

        if frame_count % BENCHMARK_INTERVAL == 0:
            output = parse_tegrastats(tegrastats_path)
            if not output:
                print(f"[WARNING] No tegrastats data available at frame {frame_count}. Skipping benchmark.")
            else:
                power_data = get_powerdraw(output)
                save_unified_benchmark("oakd", interval_data, tegrastats_path, (frame_count, frame_count), header_written)
                interval_data = {'fps': [], 'confidence': []}
                header_written[0] = True

        if ENABLE_BENCHMARK:
            benchmark_data.append([frame_count, round(current_fps, 2), round(inference_time, 4)])
            for det in detections:
                bbox = frame_norm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
                detection_results.append({
                    "frame": frame_count,
                    "label": LABELS[det.label],
                    "confidence": float(det.confidence),
                    "bbox": bbox.tolist()
                })

# Save optional logs
cap.release()
out.release()

if ENABLE_BENCHMARK:
    with open(BENCHMARK_CSV, "w") as f:
        f.write("frame,fps,inference_time\n")
        for row in benchmark_data:
            f.write(",".join(map(str, row)) + "\n")

    with open(DETECTIONS_JSON, "w") as f:
        json.dump(detection_results, f, indent=2)

    avg_fps = frame_count / (time.time() - start_time)
    avg_inference = total_inference_time / frame_count
    print(f"[INFO] Benchmark CSV saved to {BENCHMARK_CSV}")
    print(f"[INFO] Detections saved to {DETECTIONS_JSON}")
    print(f"[INFO] Average FPS: {avg_fps:.2f}, Average Inference Time: {avg_inference:.4f} seconds")

print(f"[INFO] Processed video {INPUT_VIDEO} and saved to {OUTPUT_VIDEO}")
