import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


directory = "." 

plt.figure(figsize=(12, 6))

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(directory, filename))
            plt.plot(df["frame"][2:], df["VDD_TOTAL"][2:], label=filename.replace(".csv", "").rsplit('_',2)[0])
        except Exception as e:
            print(f"Skipping {filename}: {e}")

plt.title("Frames vs VDD_TOTAL", fontsize=25)
plt.xlabel("Frame", fontsize=25)
plt.ylabel("VDD_TOTAL (mW)", fontsize=25)
plt.legend(fontsize="x-large", loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()
