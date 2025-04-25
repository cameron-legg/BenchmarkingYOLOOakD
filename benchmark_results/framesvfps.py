import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg' if you prefer


directory = "."  # Update if needed
plt.figure(figsize=(12, 6))

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        if os.path.getsize(filepath) == 0:
            print(f"Skipping {filename}: File is empty")
            continue
        try:
            df = pd.read_csv(filepath)
            plt.plot(df["frame"], df["fps"], label=filename.replace(".csv", "").rsplit('_',2)[0])
        except Exception as e:
            print(f"Skipping {filename}: {e}")

plt.title("Frames vs FPS", fontsize=25)
plt.xlabel("Frame", fontsize=25)
plt.ylabel("FPS", fontsize=25)
plt.legend(fontsize="x-large", loc="upper right")
plt.grid(True)
plt.tight_layout()
# plt.savefig("frames_vs_fps.png")  # Save instead of showing
plt.show()  # Comment this out for non-interactive environments
