import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

#################################################
# MAE heatmap for regressor, position
################################################

# Load the CSV file
file_path = ''  # Replace with your file path if different
data = pd.read_csv(file_path)

# Define bin edges for both V (magnitude) and Omega (sky motion)
lambda_f_bins = np.linspace(data['PHYSICS_WEIGHT'].min(), data['PHYSICS_WEIGHT'].max(), len(data['PHYSICS_WEIGHT']))  # 10 bins for V
lambda_rho_bins = np.linspace(data['LAMBDA_DIST'].min(), data['LAMBDA_DIST'].max(), len(data['LAMBDA_DIST'])) # 10 bins for Omega

# Digitize the data to assign it to bins
data['PHYSICS_WEIGHT_BIN'] = np.digitize(data['PHYSICS_WEIGHT'], lambda_f_bins[1:-1], right=True)
data['LAMBDA_DIST_BIN'] = np.digitize(data['LAMBDA_DIST'], lambda_rho_bins[1:-1], right=True)


# Group by the bins and calculate accuracy for each bin
binned_data = data.groupby(['PHYSICS_WEIGHT_BIN', 'LAMBDA_DIST_BIN']).agg(
    pos_mae=('position_MAE', 'mean')
).reset_index()
binned_data['pos_mae'] = binned_data['pos_mae'] * 224

# Pivot the data to create a matrix for the heatmap
heatmap_matrix = binned_data.pivot(index='v_bin', columns='omega_bin', values='pos_mae')

# Replace bin indices with bin midpoints for better readability
# heatmap_matrix.index = [f"{(v_bins[i-1] + v_bins[i]) / 2:.2f}" for i in heatmap_matrix.index]
# heatmap_matrix.columns = [f"{(omega_bins[i-1] + omega_bins[i]) / 2:.2f}" for i in heatmap_matrix.columns]

norm = mcolors.Normalize(vmax=8)

# Create the heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(heatmap_matrix, annot=True, cmap='Purples_r', norm=norm, cbar_kws={'label': 'Position MAE [pixels]'},
                 linewidths=0.5, fmt='.1f', linecolor='white')

ax.set_xticks(np.arange(len(omega_bins)))  # Shift to match heatmap grid
ax.set_yticks(np.arange(len(v_bins)))
ax.set_xticklabels([f"{edge:.0f}" for edge in omega_bins])  # Format tick labels
ax.set_yticklabels([f"{edge:.1f}" for edge in v_bins])

# Customize the plot
ax.set_xlabel('Sky motion (arcsec/h) [Omega]')
ax.set_ylabel('Magnitude [V]')
ax.invert_yaxis()  # Match the typical order of magnitudes

# Display the plot
plt.tight_layout()
plt.show()