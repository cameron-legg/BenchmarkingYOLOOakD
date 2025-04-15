import cv2
import numpy as np
import time
import json
import os
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

ENABLE_BENCHMARK = False

MODEL_PATH = "pt_models/fall_detection/compiled_model_edgetpu.tflite"
LABELS = ['Fall Detected', 'Walking', 'Sitting']
INPUT_VIDEO = "videos/in/smallest.mp4"
OUTPUT_VIDEO = "videos/out/result_coral.mp4"
BENCHMARK_CSV = "benchmark_coral.csv"
DETECTIONS_JSON = "results_coral.json"
CAMERA_PREVIEW_DIM = (640, 640)

def load_model(model_path):
    interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    return interpreter

def preprocess_frame(frame, input_size):
    resized = cv2.resize(frame, input_size)
    input_tensor = np.expand_dims(resized, axis=0)
    input_tensor = input_tensor.astype(np.uint8)
    return input_tensor

def annotate_frame(frame, results, fps):
    for det in results:
        bbox = det['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.putText(frame, det['label'], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"{int(det['confidence'] * 100)}%", (bbox[0], bbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Load interpreter
interpreter = load_model(MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(INPUT_VIDEO)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_cap = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps_cap, (frame_width, frame_height))

benchmark_data = []
detection_results = []
frame_count = 0
start_time = time.time()
total_inference_time = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count >= int(fps_cap * 60 * 3):  # stop after 3 minutes
        break

    frame_count += 1
    input_tensor = preprocess_frame(frame, CAMERA_PREVIEW_DIM)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    inference_start = time.time()
    interpreter.invoke()
    inference_time = time.time() - inference_start
    total_inference_time += inference_time

    # Extract detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    frame_results = []
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            bbox = [int(xmin * frame.shape[1]), int(ymin * frame.shape[0]),
                    int(xmax * frame.shape[1]), int(ymax * frame.shape[0])]
            label = LABELS[int(classes[i])]
            frame_results.append({
                "frame": frame_count,
                "label": label,
                "confidence": float(scores[i]),
                "bbox": bbox
            })
            detection_results.append(frame_results[-1])

    elapsed_time = time.time() - start_time
    current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    frame = annotate_frame(frame, frame_results, current_fps)
    out.write(frame)

    if ENABLE_BENCHMARK:
        benchmark_data.append([frame_count, round(current_fps, 2), round(inference_time, 4)])

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
    print(f"[INFO] Benchmark saved to {BENCHMARK_CSV}")
    print(f"[INFO] Detections saved to {DETECTIONS_JSON}")
    print(f"[INFO] Avg FPS: {avg_fps:.2f}, Avg Inference Time: {avg_inference:.4f} seconds")

print(f"[INFO] Processed video {INPUT_VIDEO} and saved to {OUTPUT_VIDEO}")
