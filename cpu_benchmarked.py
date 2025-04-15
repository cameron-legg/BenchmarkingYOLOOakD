
def cpu_benchmark(video_name):
    from ultralytics import YOLO
    import cv2
    import time
    import datetime
    import csv
    import os
    from jetsontools import Tegrastats, get_powerdraw, parse_tegrastats, filter_data

    MODEL_PATH = "pt_models/fall_detection/weights/best.pt"
    INPUT_VIDEO = f"videos/in/{video_name}"
    saved_file_timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    OUTPUT_VIDEO = f"videos/out/result_cpu_{saved_file_timestamp}.mp4"
    BENCHMARK_CSV = f"benchmark_results/cpu_{saved_file_timestamp}.csv"

    # Setup directories
    os.makedirs("benchmark_results", exist_ok=True)

    model = YOLO(MODEL_PATH)
    model.to('cpu')
    cap = cv2.VideoCapture(INPUT_VIDEO)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

    frame_count = 0
    timestamps = []
    fps_list = []
    start_time = time.time()

    # Open CSV for writing
    with open(BENCHMARK_CSV, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "fps", "VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC", "VDD_TOTAL"])

        with Tegrastats(f"tmp/temp_power_cpu_{saved_file_timestamp}.txt", interval=5):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                interval_start = time.time()
                results = model(frame, device='cpu')[0]
                interval_end = time.time()

                frame_count += 1
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                fps_list.append(current_fps)
                timestamps.append((interval_start, interval_end))

                for r in results.boxes:
                    cls = int(r.cls)
                    conf = float(r.conf)
                    bbox = r.xyxy[0].cpu().numpy().astype(int)
                    label = model.names[cls]

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {int(conf*100)}%", (bbox[0], bbox[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)

                if frame_count % 300 == 0:
                    parsed = parse_tegrastats(f"tmp/temp_power_cpu_{saved_file_timestamp}.txt")

                    filtered, _ = filter_data(parsed, timestamps[-300:])
                    power_data = get_powerdraw(filtered)

                    writer.writerow([
                        frame_count,
                        sum(fps_list[-300:]) / 300,
                        power_data["VDD_IN"].mean,
                        power_data["VDD_CPU_GPU_CV"].mean,
                        power_data["VDD_SOC"].mean,
                        power_data["VDD_TOTAL"].mean,
                    ])

    cap.release()
    out.release()
    print(f"[INFO] Processed video {INPUT_VIDEO} and saved to {OUTPUT_VIDEO}")
    print(f"[INFO] Benchmarks saved to {BENCHMARK_CSV}")
