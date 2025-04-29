import os
import pandas as pd

directory = "."
average_vdd_total = {}

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        if os.path.getsize(filepath) == 0:
            continue
        try:
            df = pd.read_csv(filepath)
            avg_vdd_total = df["VDD_TOTAL"].mean()
            average_vdd_total[filename] = avg_vdd_total
        except Exception:
            continue

# Sort by average VDD_TOTAL ascending 
sorted_vdd = sorted(average_vdd_total.items(), key=lambda x: x[1])

# Print results
for filename, avg in sorted_vdd:
    print(f"{filename}: {avg:.2f} mW")
