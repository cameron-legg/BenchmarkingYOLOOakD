import csv
from statistics import mean
from jetsontools import get_powerdraw, parse_tegrastats, filter_data
from pathlib import Path

def save_unified_benchmark(device_name, interval_data, tegrastats_path, tag, header_written):
    outdir = Path(f"benchmark_results")
    outdir.mkdir(parents=True, exist_ok=True)
    filepath = outdir / f"{device_name}_benchmark.csv"

    with open(tegrastats_path, "r") as f:
        lines = f.readlines()
    output = parse_tegrastats(tegrastats_path)

    # Determine whether we're using a timestamp window or a frame tag
    is_frame_mode = isinstance(tag[0], int) and tag[0] == tag[1]

    if is_frame_mode:
        filtered = output  # No filtering
        time_label = f"frame_{tag[0]}"
    else:
        filtered, _ = filter_data(output, [tag])
        time_label = tag[0]

    # ⚠️ Guard against missing data
    if not filtered or len(filtered) == 0:
        print(f"[WARNING] Skipping benchmark at {time_label} — no tegrastats data available yet.")
        return

    power_data = get_powerdraw(filtered)

    row = {
        "time": time_label,
        "fps": mean(interval_data["fps"]) if interval_data["fps"] else 0,
        "confidence": mean(interval_data["confidence"]) if interval_data["confidence"] else 0,
        "VDD_IN": power_data.get("VDD_IN", type("M", (), {"mean": 0})).mean,
        "VDD_CPU_GPU_CV": power_data.get("VDD_CPU_GPU_CV", type("M", (), {"mean": 0})).mean,
        "VDD_SOC": power_data.get("VDD_SOC", type("M", (), {"mean": 0})).mean,
        "VDD_TOTAL": power_data.get("VDD_TOTAL", type("M", (), {"mean": 0})).mean
    }

    write_header = not filepath.exists() or not header_written[0]
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
            header_written[0] = True
        writer.writerow(row)
