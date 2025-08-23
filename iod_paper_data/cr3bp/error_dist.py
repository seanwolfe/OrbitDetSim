import pandas as pd
import matplotlib.pyplot as plt

# === User settings ===
csv_file = "sgd/perf_comp/CR3BP_AllOrbits_GROUND_SGD_META_MASTER.csv"     # path to your CSV
column_name = "POS_RMSE"    # name of the column you want to plot
num_bins = 300                 # number of histogram bins

# === Read CSV ===
df = pd.read_csv(csv_file)

# === Plot histogram ===
plt.figure(figsize=(5,6))
plt.hist(df[column_name].dropna() * 1.5e8, bins=num_bins, edgecolor='black', alpha=0.7)
plt.xlabel("Position RMSE (km)")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()
