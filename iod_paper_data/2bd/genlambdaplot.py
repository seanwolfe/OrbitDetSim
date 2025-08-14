import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_name = "sgd/2BD_GEO_GROUND_SGD_META_MASTER.csv"
data = pd.read_csv(file_name)

# Ensure PHYSICS_WEIGHT is numeric
data['PHYSICS_WEIGHT'] = pd.to_numeric(data['PHYSICS_WEIGHT'], errors='coerce')
physics_weights = sorted(data['PHYSICS_WEIGHT'].unique())

# --- Position RMSE ---
fig, ax = plt.subplots(figsize=(6, 4))
pos_data = [data.loc[data['PHYSICS_WEIGHT'] == w, 'POS_RMSE'] for w in physics_weights]
ax.boxplot(pos_data, positions=physics_weights, widths=[0.5*w for w in physics_weights], showfliers=False)
ax.set_xscale('log')
ax.set_xlabel("Physics Weight")
ax.set_ylabel("Position RMSE (km)")
ax.grid(True, which='both', linestyle='--', alpha=0.5)

# Add padding on x-axis
padding = 0.4  # 10%
ax.set_xlim(physics_weights[0] * (1 - padding), physics_weights[-1] * (1 + padding))
plt.tight_layout()

# --- Velocity RMSE ---
fig2, ax2 = plt.subplots(figsize=(6, 4))
vel_data = [data.loc[data['PHYSICS_WEIGHT'] == w, 'VEL_RMSE'] for w in physics_weights]

ax2.boxplot(vel_data, positions=physics_weights, widths=[0.3*w for w in physics_weights], showfliers=False)
ax2.set_xscale('log')
ax2.set_xlabel("Physics Weight")
ax2.set_ylabel("Velocity RMSE (km/s)")
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

# Add padding
ax2.set_xlim(physics_weights[0] * (1 - padding), physics_weights[-1] * (1 + padding))
plt.tight_layout()
plt.show()
