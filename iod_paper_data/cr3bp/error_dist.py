import pandas as pd
import matplotlib.pyplot as plt

# === User settings ===
csv_file = "bh_plus_range/perfcomp/Error_files_horilyapperf/CR3BP_Horizontal Lyapunov Orbits_GROUND_CONSTRAINED_BASIN_HOPPINGmeta_data.csv"     # path to your CSV
column_name = "POS_RMSE"    # name of the column you want to plot
num_bins = 100                 # number of histogram bins

# === Read CSV ===
df = pd.read_csv(csv_file)

# === Plot histogram ===
plt.figure(figsize=(5,6))
plt.hist(df[column_name].dropna() * 1.5e8, bins=num_bins, edgecolor='black', alpha=0.7)
plt.xlabel("Position RMSE (km)")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()
