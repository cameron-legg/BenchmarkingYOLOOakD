from ultralytics import YOLO
import cv2
import time
from pathlib import Path
from benchmark_utils import save_unified_benchmark
from jetsontools import Tegrastats, parse_tegrastats, get_powerdraw

MODEL_PATH = "pt_models/fall_detection/weights/best.pt"
INPUT_VIDEO = "videos/in/benchmarkvid.mp4"
OUTPUT_VIDEO = "videos/out/result_cpu.mp4"

# Load model on CPU
model = YOLO(MODEL_PATH)
model.to('cpu')

# Setup video capture and output
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

# Benchmark config
frame_count = 0
interval_data = {'fps': [], 'confidence': []}
header_written = [False]
tegrastats_path = Path("output_stats_cpu.txt")
BENCHMARK_INTERVAL = 300

start_time = time.time()

with Tegrastats(tegrastats_path, interval=5):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device='cpu')[0]
        frame_count += 1

        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        interval_data['fps'].append(current_fps)
        interval_data['confidence'].extend([float(r.conf) for r in results.boxes])

        # Benchmark every 300 frames
        if frame_count % BENCHMARK_INTERVAL == 0:
            output = parse_tegrastats(tegrastats_path)

            if not output:
                print(f"[WARNING] No tegrastats data available at frame {frame_count}. Skipping benchmark.")
            else:
                power_data = get_powerdraw(output)
                save_unified_benchmark("cpu", interval_data, tegrastats_path, (frame_count, frame_count), header_written)
                interval_data = {'fps': [], 'confidence': []}
                header_written[0] = True

        # Annotate and save frame
        for r in results.boxes:
            cls = int(r.cls)
            conf = float(r.conf)
            bbox = r.xyxy[0].cpu().numpy().astype(int)
            label = model.names[cls]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {int(conf * 100)}%", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

# Cleanup
cap.release()
out.release()
print(f"[INFO] Processed video {INPUT_VIDEO} and saved to {OUTPUT_VIDEO}")
