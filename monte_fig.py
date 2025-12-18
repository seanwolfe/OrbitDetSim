import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# CSV file
# ==========================
csv_path = "coverage_results_all_6.csv"   # <-- change this

# ==========================
# Column groups for each method
# ==========================
proposed_cols = [
    "Proposed_0_covered",
    "Proposed_1_covered",
    "Proposed_2_covered",
    "Proposed_3plus_covered",
]

mean_cols = [
    "Mean_0_covered",
    "Mean_1_covered",
    "Mean_2_covered",
    "Mean_3plus_covered",
]

categories = ["0 covered", "1 covered", "2 covered", "3+ covered"]

# ==========================
# Load data
# ==========================
df = pd.read_csv(csv_path)

proposed_fracs = [df[col].mean() for col in proposed_cols]
mean_fracs     = [df[col].mean() for col in mean_cols]

# ==========================
# Plot grouped bar chart
# ==========================
x = np.arange(len(categories))
width = 0.25
gap = 0.06    # gap between Proposed and Mean bars

fig, ax = plt.subplots(figsize=(7, 4))

# Colors (edges fully opaque, face very light)
proposed_color = "#1f77b4"
mean_color     = "#ff7f0e"

ax.bar(
    x - (width/2 + gap/2),
    proposed_fracs,
    width,
    label="Proposed Coordination Algorithm",
    edgecolor=proposed_color,
    facecolor="#a4c8f5",
    alpha=0.85,
    linewidth=1.8
)

ax.bar(
    x + (width/2 + gap/2),
    mean_fracs,
    width,
    label="Pointing to Mean",
    edgecolor=mean_color,
    facecolor="#f5cba4",
    alpha=0.65,
    linewidth=1.8
)

ax.set_xticks(x)
ax.set_xticklabels(categories)

ax.set_ylabel("Fraction of Trials", fontsize=11)
ax.set_xlabel("Coverage of Object at True Position", fontsize=11)

ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)

ax.legend()

plt.tight_layout()
plt.show()
