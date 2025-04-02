import depthai as dai
import json
import cv2
import numpy as np
import time
import os

# Paths to model and config
YOLOV8N_MODEL = "yolonewfish/best_openvino_2022.1_6shave.blob"
YOLOV8N_CONFIG = "yolonewfish/best.json"

def get_next_video_filename(directory="videos/stream", base_name="result", extension=".mp4"):
    # Check for existing files in the directory
    files = os.listdir(directory)
    
    # Filter files that start with 'result' and have the .mp4 extension
    result_files = [f for f in files if f.startswith(base_name) and f.endswith(extension)]
    
    # Extract the numbers from the filenames (if any)
    numbers = []
    for file in result_files:
        # Extract the number part after 'result'
        num_part = file[len(base_name):-len(extension)]  # Strip 'result' and '.mp4'
        
        # If there is a number, add it to the list
        if num_part.isdigit():
            numbers.append(int(num_part))
    
    # Find the next available number (starting from 0 if no files found)
    next_number = max(numbers, default=-1) + 1
    
    # Return the new filename with the next number
    return os.path.join(directory, f"{base_name}{next_number if next_number > 0 else ''}{extension}")


OUTPUT_VIDEO = get_next_video_filename()
CAMERA_PREVIEW_DIM = (640, 640)

#LABELS = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
LABELS = ['AngelFish', 'BlueTang', 'ButterflyFish', 'ClownFish', 'GoldFish', 'Gourami', 'MorishIdol', 'PlatyFish', 'RibbonedSweetlips', 'ThreeStripedDamselfish', 'YellowCichlid', 'YellowTang', 'ZebraFish']


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

    # Camera node
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(*CAMERA_PREVIEW_DIM)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setFps(30)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)


    # Detection node
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    detectionNetwork.setConfidenceThreshold(confidenceThreshold)
    detectionNetwork.setNumClasses(classes)
    detectionNetwork.setCoordinateSize(coordinates)
    detectionNetwork.setAnchors(anchors)
    detectionNetwork.setAnchorMasks(anchorMasks)
    detectionNetwork.setIouThreshold(iouThreshold)
    detectionNetwork.setBlobPath(model_path)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Link camera to detector
    cam.preview.link(detectionNetwork.input)

    # Output streams
    xout_preview = pipeline.create(dai.node.XLinkOut)
    xout_preview.setStreamName("preview")
    cam.preview.link(xout_preview.input)

    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")
    detectionNetwork.out.link(nnOut.input)

    return pipeline

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

# Ensure output directory exists
output_dir = os.path.dirname(OUTPUT_VIDEO)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Create pipeline and start device
pipeline = create_image_pipeline(YOLOV8N_CONFIG, YOLOV8N_MODEL)

with dai.Device(pipeline) as device:
    info = device.getDeviceInfo()
    print("[INFO] Connected to OAK device:")
    print(f"       Name: {info.name}")
    print(f"       MX ID: {info.getMxId()}")
    print(f"       State: {info.state}")
    print(f"       USB speed: {device.getUsbSpeed()}")

    previewQueue = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
    detectionQueue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)


    # Get resolution for writer
    frame_example = previewQueue.get().getCvFrame()
    frame_height, frame_width = frame_example.shape[:2]
    fps = 30  # consistent with camera
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    start_time = time.time()

    while True:
        inFrame = previewQueue.get()
        frame = inFrame.getCvFrame()

        inDet = detectionQueue.get()
        detections = inDet.detections if inDet is not None else []

        frame_count += 1
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        annotated = annotate_frame(frame, detections, current_fps)
        out.write(annotated)
        if frame_count % 2 == 0:
            cv2.imshow("Preview", annotated)


        if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()

print(f"[INFO] Video saved to {OUTPUT_VIDEO}")
