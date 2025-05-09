import depthai as dai
import json
import os
import cv2
import numpy as np
import time
import datetime

num_shaves = int(input("Number of shaves: 6,8,16"))
YOLOV8N_MODEL = ""
YOLOV8N_CONFIG = ""
if(num_shaves == 6):
    # 6 shaves
    print("RUNNING 6 SHAVES MODEL")
    YOLOV8N_MODEL = "oakd_models/fall_detection_6shaves/falldetectionmodel_openvino_2022.1_6shave.blob"
    YOLOV8N_CONFIG = "oakd_models/fall_detection_6shaves/falldetectionmodel.json"
elif(num_shaves == 8):
    # 8 shaves
    print("RUNNING 8 SHAVES MODEL")
    YOLOV8N_MODEL = "oakd_models/fall_detection_8shaves/best_openvino_2022.1_8shave.blob"
    YOLOV8N_CONFIG = "oakd_models/fall_detection_8shaves/best.json"
else:
    # 16 shaves
    print("RUNNING 16 SHAVES MODEL")
    YOLOV8N_MODEL = "oakd_models/fall_detection_16shaves/best_openvino_2022.1_16shave.blob"
    YOLOV8N_CONFIG = "oakd_models/fall_detection_16shaves/best.json"

saved_file_timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
INPUT_VIDEO = "videos/in/benchmarkvid.mp4"
OUTPUT_VIDEO = f"videos/out/result_oakd_{saved_file_timestamp}.mp4"

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
    color = (0, 0, 255)
    for detection in detections:
        bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, LABELS[detection.label], (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
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

cap.release()
out.release()


print(f"[INFO] Processed video {INPUT_VIDEO} and saved to {OUTPUT_VIDEO}")
