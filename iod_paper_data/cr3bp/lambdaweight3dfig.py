import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
file_path = 'CR3BP_Horizontal Lyapunov Orbits_GROUND_CONSTRAINED_BASIN_HOPPINGmeta_data.csv'
df = pd.read_csv(file_path)
df['POS_RMSE'] *= 149597871  # convert to km

# Create a string for the hyperparameter combination
def format_combo(row):
    x = np.log10(row['PHYSICS_WEIGHT'])
    y = np.log10(row['WEIGHT_SCALE_FACTOR'])
    z = np.log10(row['LAMBDA_DIST'])
    return r'$\left(10^{{{:.0f}}}, 10^{{{:.0f}}}, 10^{{{:.0f}}}\right)$'.format(x, y, z)

df['param_combo'] = df.apply(format_combo, axis=1)

# Compute mean POS_RMSE per hyperparameter combination
mean_rmse = df.groupby('param_combo')['POS_RMSE'].mean().reset_index()
mean_rmse = mean_rmse.rename(columns={'POS_RMSE': 'mean_rmse'})

# Select top n combinations with lowest mean RMSE
n = 20
top_combos = mean_rmse.nsmallest(n, 'mean_rmse')['param_combo']

# Compute mean RMSE of the top n candidates
mean_of_top_n = mean_rmse.nsmallest(n, 'mean_rmse')['mean_rmse'].mean()
print(f"Mean RMSE of the top {n} candidates: {mean_of_top_n:.2f} km")

# Filter the original df to only include these top combinations
top_df = df[df['param_combo'].isin(top_combos)]



# Plot whisker/box plot
plt.figure()
# Example boxplot
ax = sns.boxplot(
    x='POS_RMSE',
    y='param_combo',
    data=top_df,
    orient='h',
    showmeans=True,
    meanprops={
        "marker":"^",         # triangle
        "markerfacecolor":"orange",
        "markeredgecolor":"orange"
    },
    medianprops={
        "color":"orange",
        "linewidth":1
    },
    boxprops={
        "facecolor":"none",   # no fill
        "edgecolor":"black"
    }
)
plt.xlabel('Position RMSE [km]')
plt.ylabel(r'Hyperparameter Combination $(\lambda_f, c_{wb}, \lambda_{\rho})$')
plt.show()
