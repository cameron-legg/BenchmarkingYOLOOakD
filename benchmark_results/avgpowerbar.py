import os
import pandas as pd
import matplotlib.pyplot as plt

directory = "."  # Update this path if needed
average_power = {}

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        if os.path.getsize(filepath) == 0:
            continue
        try:
            df = pd.read_csv(filepath)
            avg_power = df["VDD_TOTAL"].mean()
            average_power[filename.replace(".csv", "")] = avg_power
        except Exception:
            continue

# Sort by power consumption
sorted_power = dict(sorted(average_power.items(), key=lambda item: item[1], reverse=True))

# Plot
plt.figure(figsize=(14, 6))
plt.bar(sorted_power.keys(), sorted_power.values())
plt.xticks(rotation=90, fontsize=8)
plt.ylabel("Average VDD_TOTAL (mW)")
plt.title("Average Power Consumption per Benchmark Run")
plt.tight_layout()
plt.show()
