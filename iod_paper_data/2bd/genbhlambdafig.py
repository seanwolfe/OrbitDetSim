import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

#################################################
# MAE heatmap for regressor, position
################################################

# Load the CSV file
file_path = '2BD_GEO_GROUND_CONSTRAINED_BASIN_HOPPINGmeta_MASTER.csv'  # Replace with your file path if different
data = pd.read_csv(file_path)

# Define bin edges for both V (magnitude) and Omega (sky motion)
lambda_f_bins = [1, 10, 100, 1000]
lambda_rho_bins = [1e-11, 1e-9, 1e-7, 1e-5]

# Digitize the data to assign it to bins
data['PHYSICS_WEIGHT_BIN'] = np.digitize(data['PHYSICS_WEIGHT'], lambda_f_bins, right=True)
data['LAMBDA_DIST_BIN'] = np.digitize(data['LAMBDA_DIST'], lambda_rho_bins, right=True)


# Group by the bins and calculate accuracy for each bin
binned_data = data.groupby(['PHYSICS_WEIGHT_BIN', 'LAMBDA_DIST_BIN']).agg(
    vel_rmse=('VEL_RMSE', 'mean')
).reset_index()

# Pivot the data to create a matrix for the heatmap
heatmap_matrix = binned_data.pivot(index='PHYSICS_WEIGHT_BIN', columns='LAMBDA_DIST_BIN', values='vel_rmse')

# Replace bin indices with bin midpoints for better readability
# heatmap_matrix.index = [f"{(v_bins[i-1] + v_bins[i]) / 2:.2f}" for i in heatmap_matrix.index]
# heatmap_matrix.columns = [f"{(omega_bins[i-1] + omega_bins[i]) / 2:.2f}" for i in heatmap_matrix.columns]

norm = mcolors.Normalize(vmax=0.01)

# Create the heatmap
plt.figure()
ax = sns.heatmap(heatmap_matrix, annot=False, cmap='Purples_r', norm=norm, cbar_kws={'label': 'Velocity RMSE [km/s]'},
                 linewidths=0.5, fmt='.1f', linecolor='white')

ax.set_xticks(np.arange(len(lambda_f_bins)))  # Shift to match heatmap grid
ax.set_yticks(np.arange(len(lambda_rho_bins)))
ax.set_xticklabels([f"{edge:.0e}" for edge in lambda_f_bins])
ax.set_yticklabels([f"{edge:.0e}" for edge in lambda_rho_bins])

# Customize the plot
ax.set_xlabel('Physics Weight')
ax.set_ylabel('Range Weight')
ax.invert_yaxis()  # Match the typical order of magnitudes

# Display the plot
plt.tight_layout()
plt.show()