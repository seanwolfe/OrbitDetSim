import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load data ===
file_path = "CR3BP__GROUND_METAMASTER.csv"   # <-- change this
df = pd.read_csv(file_path)

# === Extract Jacobi constant ===
# Take last part of FILE_USED (after last /) and convert to float
df['JACOBI_CONSTANT'] = df['FILE_USED'].apply(lambda x: float(os.path.basename(str(x))))
df['POS_RMSE'] *= 149597871
df['VEL_RMSE'] *= 29.8

# === Define orbit types ===
orbit_types = ["Horizontal Lyapunov Orbits", "Vertical Lyapunov Orbits", "Halo Orbits"]

# === Make line plots ===
for orbit in orbit_types:
    plt.figure()
    subset = df[df['ORBIT_TYPE'] == orbit]
    sns.lineplot(
        data=subset,
        x="JACOBI_CONSTANT",
        y="VEL_RMSE",
        hue="METHOD",
        marker="o"
    )
    plt.title(f"{orbit}")
    plt.xlabel("Jacobi Constant (non-dim)")
    plt.ylabel("Velocity RMSE [km/s]")
    plt.legend(title="Method")
    plt.tight_layout()

# === Whisker/box plot for all orbit+method combinations ===
plt.figure()
sns.boxplot(
    data=df,
    x="ORBIT_TYPE",
    y="VEL_RMSE",
    hue="METHOD"
)
plt.xlabel("Orbit Type")
plt.ylabel("Velocity RMSE [km/s]")
plt.legend(title="Method")
plt.tight_layout()
plt.show()
