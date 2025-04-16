import os
import pandas as pd
import matplotlib.pyplot as plt

directory = "."  # Change if needed
average_fps = {}

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        if os.path.getsize(filepath) == 0:
            continue
        try:
            df = pd.read_csv(filepath)
            avg_fps = df["fps"].mean()
            average_fps[filename.replace(".csv", "")] = avg_fps
        except Exception:
            continue

# Sort by FPS
sorted_fps = dict(sorted(average_fps.items(), key=lambda item: item[1], reverse=True))

# Plot
plt.figure(figsize=(14, 6))
plt.bar(sorted_fps.keys(), sorted_fps.values())
plt.xticks(rotation=90, fontsize=8)
plt.ylabel("Average FPS")
plt.title("Average FPS per Benchmark Run")
plt.tight_layout()
plt.show()
