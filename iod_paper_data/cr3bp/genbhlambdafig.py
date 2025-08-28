import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---- Load data ----
file_path = 'CR3BP_Horizontal Lyapunov Orbits_GROUND_SGDmeta_data.csv'
data = pd.read_csv(file_path)
data['POS_RMSE'] *= 149597871
# data = data[data['LAMBDA_DIST'] == 1e-3]

# ---- Bin centers (NOT edges) ----
lambda_f_centers  = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])  # physics weight (x)
lambda_rho_centers = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])            # range weight (y)

def edges_from_log_centers(centers: np.ndarray) -> np.ndarray:
    """
    Build log-scale bin edges from positive, monotonically increasing bin centers.
    Uses geometric midpoints; extrapolates first/last edge.
    """
    c = np.asarray(centers, dtype=float)
    if np.any(c <= 0):
        raise ValueError("All centers must be > 0 for log edges.")
    if not np.all(np.diff(c) > 0):
        raise ValueError("Centers must be strictly increasing.")
    logc = np.log10(c)
    mids = 0.5 * (logc[:-1] + logc[1:])
    edges_log = np.empty(len(c) + 1)
    edges_log[1:-1] = mids
    # extrapolate first/last edge symmetrically in log space
    edges_log[0]  = logc[0]  - (mids[0]   - logc[0])
    edges_log[-1] = logc[-1] + (logc[-1] - mids[-1])
    return 10 ** edges_log

# Build edges from centers
lambda_f_edges   = edges_from_log_centers(lambda_f_centers)
lambda_rho_edges = edges_from_log_centers(lambda_rho_centers)

# ---- Digitize using edges (include extremes without creating extra bins) ----
# Clip the top edge slightly so values equal to the max edge don't fall into an overflow bin.
def digitize_to_bins(values, edges):
    vals = np.asarray(values, dtype=float)
    low  = edges[0]
    high = np.nextafter(edges[-1], -np.inf)  # just below top edge
    vals = np.clip(vals, low, high)
    # np.digitize returns bin indices in 1..len(edges)-1
    return np.digitize(vals, edges, right=False)

data['PHYSICS_WEIGHT_BIN'] = digitize_to_bins(data['PHYSICS_WEIGHT'].to_numpy(), lambda_f_edges)
# data['LAMBDA_DIST_BIN']    = digitize_to_bins(data['LAMBDA_DIST'].to_numpy(),   lambda_rho_edges)
data['WEIGHT_SCALE_FACTOR_BIN']    = digitize_to_bins(data['WEIGHT_SCALE_FACTOR'].to_numpy(),   lambda_rho_edges)

# ---- Aggregate mean POS_RMSE per (λρ-bin, λf-bin) ----
# binned = (data
#           .groupby(['LAMBDA_DIST_BIN', 'PHYSICS_WEIGHT_BIN'], as_index=False)
#           .agg(pos_rmse=('POS_RMSE', 'mean')))
binned = (data
          .groupby(['WEIGHT_SCALE_FACTOR_BIN', 'PHYSICS_WEIGHT_BIN'], as_index=False)
          .agg(pos_rmse=('POS_RMSE', 'mean')))

# Ensure a full grid: rows = λρ bins (y), cols = λf bins (x)
row_index = pd.Index(range(1, len(lambda_rho_centers) + 1), name='WEIGHT_SCALE_FACTOR_BIN')
# row_index = pd.Index(range(1, len(lambda_rho_centers) + 1), name='LAMBDA_DIST_BIN')
col_index = pd.Index(range(1, len(lambda_f_centers) + 1),   name='PHYSICS_WEIGHT_BIN')

# heat = (binned
#         .pivot(index='LAMBDA_DIST_BIN', columns='PHYSICS_WEIGHT_BIN', values='pos_rmse')
#         .reindex(index=row_index, columns=col_index))
heat = (binned
        .pivot(index='WEIGHT_SCALE_FACTOR_BIN', columns='PHYSICS_WEIGHT_BIN', values='pos_rmse')
        .reindex(index=row_index, columns=col_index))

# Normalize color scale
norm = mcolors.Normalize(vmax=0.002 * 149597871)

# ---- Plot ----
plt.figure(figsize=(9, 5))
ax = sns.heatmap(
    heat,
    cmap='Purples_r',
    norm=norm,
    cbar_kws={'label': 'Position RMSE [km]'},
    linewidths=0.5,
    linecolor='white',
    annot=False
)

# Place ticks at cell centers; label with the BIN CENTERS
ax.set_xticks(np.arange(len(col_index)) + 0.5)
ax.set_yticks(np.arange(len(row_index)) + 0.5)
ax.set_xticklabels([f"{c:.0e}" for c in lambda_f_centers])
ax.set_yticklabels([f"{c:.0e}" for c in lambda_rho_centers])

ax.set_xlabel('Physics Weight')
# ax.set_ylabel('Range Weight')
ax.set_ylabel('Weight Scale Factor')
ax.invert_yaxis()  # optional, to show smaller λ_ρ at top

plt.tight_layout()
plt.show()
