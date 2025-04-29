import os
import pandas as pd

directory = "."
average_fps = {}

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        if os.path.getsize(filepath) == 0:
            continue
        try:
            df = pd.read_csv(filepath)
            avg_fps = df["fps"].mean()
            average_fps[filename] = avg_fps
        except Exception:
            continue

# Sort by average FPS descending
sorted_fps = sorted(average_fps.items(), key=lambda x: x[1], reverse=True)

# Print results
for filename, avg in sorted_fps:
    print(f"{filename}: {avg:.2f} FPS")
