import os
import pandas as pd
import matplotlib.pyplot as plt

directory = "."  # Update as needed
average_fps = {}
average_power = {}

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        if os.path.getsize(filepath) == 0:
            continue
        try:
            df = pd.read_csv(filepath)
            base_name = filename.replace(".csv", "")
            average_fps[base_name] = df["fps"].mean()
            average_power[base_name] = df["VDD_TOTAL"].mean()
        except Exception:
            continue

# Prepare scatter plot data
labels = list(set(average_fps.keys()) & set(average_power.keys()))
x = [average_fps[label] for label in labels]
y = [average_power[label] for label in labels]

plt.figure(figsize=(10, 6))
plt.scatter(x, y)

# Annotate each point with the filename (small font)
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]), fontsize=6, alpha=0.7)

plt.xlabel("Average FPS")
plt.ylabel("Average Power Draw (VDD_TOTAL in mW)")
plt.title("Power Consumption vs FPS per Benchmark Run")
plt.grid(True)
plt.tight_layout()
plt.show()
