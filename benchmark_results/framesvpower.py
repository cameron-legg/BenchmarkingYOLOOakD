import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg' if you prefer


directory = "."  # Change to your CSV directory if needed

plt.figure(figsize=(12, 6))

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(directory, filename))
            plt.plot(df["frame"], df["VDD_TOTAL"], label=filename.replace(".csv", ""))
        except Exception as e:
            print(f"Skipping {filename}: {e}")

plt.title("Frames vs VDD_TOTAL")
plt.xlabel("Frame")
plt.ylabel("VDD_TOTAL (mW)")
plt.legend(fontsize="x-small", loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()
