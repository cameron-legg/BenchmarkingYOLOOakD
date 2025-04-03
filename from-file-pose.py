import depthai as dai
import json
import os
import cv2
import numpy as np
import time

ENABLE_BENCHMARK = False  # Toggle to enable/disable benchmarking and result recording

YOLOV8N_MODEL = "oakd_models/pose-fall/yolov8s-pose_openvino_2022.1_6shave.blob"
YOLOV8N_CONFIG = "oakd_models/pose-fall/yolov8s-pose.json"

INPUT_VIDEO = "videos/in/smallest.mp4"
OUTPUT_VIDEO = "videos/out/result-pose.mp4"
BENCHMARK_CSV = "benchmark.csv"
DETECTIONS_JSON = "results.json"

CAMERA_PREVIEW_DIM = (640, 384)
LABELS = ['person']

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)

def create_image_pipeline(config_path, model_path):
    pipeline = dai.Pipeline()
    model_config = load_config(config_path)
    nnConfig = model_config.get("nn_config", {})
    metadata = nnConfig.get("NN_specific_metadata", {})
    classes = metadata.get("classes", {})
    coordinates = metadata.get("coordinates", {})
    anchors = metadata.get("anchors", {})
    anchorMasks = metadata.get("anchor_masks", {})
    iouThreshold = metadata.get("iou_threshold", {})
    confidenceThreshold = metadata.get("confidence_threshold", {})
    detectionIN = pipeline.create(dai.node.XLinkIn)
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    nnOut = pipeline.create(dai.node.XLinkOut)

    nnOut.setStreamName("nn")
    detectionIN.setStreamName("detection_in")
    detectionNetwork.setConfidenceThreshold(confidenceThreshold)
    detectionNetwork.setNumClasses(classes)
    detectionNetwork.setCoordinateSize(coordinates)
    detectionNetwork.setAnchors(anchors)
    detectionNetwork.setAnchorMasks(anchorMasks)
    detectionNetwork.setIouThreshold(iouThreshold)
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
    # color = (0, 0, 255)
    # for detection in detections:
    #     bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
    #     cv2.putText(frame, LABELS[detection.label], (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
    #     cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
    #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for i, box in enumerate(detections):
        x, y, w, h = box.xywh[0]  # Get bounding box details
        fall_detected = w / h > 1.4  # Condition for fall detection

        # Convert coordinates to integers
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Draw bounding box
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)

        # Display "Fall Detected" above the bounding box
        text = "Fall Detected" if fall_detected else "Standing"
        color = (0, 0, 255) if fall_detected else (255, 255, 255)  # Red if fall detected, white otherwise
        cv2.putText(frame, text, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

pipeline = create_image_pipeline(YOLOV8N_CONFIG, YOLOV8N_MODEL)
output_dir = os.path.dirname(OUTPUT_VIDEO)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(INPUT_VIDEO)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_cap = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps_cap, (frame_width, frame_height))

benchmark_data = []
detection_results = []

with dai.Device(pipeline) as device:
    info = device.getDeviceInfo()
    print("[INFO] Connected to OAK device:")
    print(f"       Name: {info.name}")
    print(f"       MX ID: {info.getMxId()}")
    print(f"       State: {info.state}")
    print(f"       USB speed: {device.getUsbSpeed()}")

    detectionIN = device.getInputQueue("detection_in")
    detectionNN = device.getOutputQueue("nn")

    start_time = time.time()
    frame_count = 0
    total_inference_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ⏱️ Stop after 3 minutes
        if frame_count >= int(fps_cap * 60 * 3):
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

        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        frame = annotate_frame(frame, detections, current_fps)
        out.write(frame)

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
