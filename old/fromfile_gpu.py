from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "pt_models/fall_detection/weights/best.pt"
INPUT_VIDEO = "videos/in/benchmarkvid.mp4"
OUTPUT_VIDEO = "videos/out/result_gpu.mp4"

# Force model to use GPU
model = YOLO(MODEL_PATH)
model.to('cuda')  # Move model to GPU

cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device='cuda')[0]  # Ensure inference is on GPU
    frame_count += 1

    for r in results.boxes:
        cls = int(r.cls)
        conf = float(r.conf)
        bbox = r.xyxy[0].cpu().numpy().astype(int)  # Still move to CPU for drawing
        label = model.names[cls]

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.putText(frame, f"{label} {int(conf*100)}%", (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    elapsed_time = time.time() - start_time
    current_fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print(f"[INFO] Processed video {INPUT_VIDEO} and saved to {OUTPUT_VIDEO}")
