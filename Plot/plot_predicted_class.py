import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load the data
path = "/mnt/data/patient-level_slicecounts.csv"
df = pd.read_csv(path)

# Prepare long-format slice-level data
rows = []
for _, r in df.iterrows():
    subject = r["subject"]
    true_label = r["true_label(0:MDDonly 1:MDDRBD)"]
    
    # Add rows for slices predicted as MDD-only (0)
    for _ in range(int(r["pred_mddonly"])):
        rows.append({
            "subject": subject,
            "true_label": true_label,
            "pred_label": 0
        })
    
    # Add rows for slices predicted as MDD+RBD (1)
    for _ in range(int(r["pred_mddrbd"])):
        rows.append({
            "subject": subject,
            "true_label": true_label,
            "pred_label": 1
        })

long_df = pd.DataFrame(rows)

# Map subjects to y positions
subjects = df["subject"].unique()
subject_to_y = {s: i for i, s in enumerate(subjects)}
long_df["y"] = long_df["subject"].map(subject_to_y)

# Map predicted labels to x positions
label_to_x = {0: 0, 1: 1}
long_df["x"] = long_df["pred_label"].map(label_to_x)

# Determine correctness and colors
long_df["correct"] = long_df["true_label"] == long_df["pred_label"]
color_map = {True: "green", False: "red"}
colors = long_df["correct"].map(color_map)

# Add small horizontal jitter to x so overlapping dots spread a bit
np.random.seed(0)  # for reproducibility
jitter = (np.random.rand(len(long_df)) - 0.5) * 0.15
x_jittered = long_df["x"] + jitter

# Figure size scales with number of subjects
fig_height = max(4, len(subjects) * 0.2)
fig, ax = plt.subplots(figsize=(7, fig_height))

# Scatter plot
ax.scatter(x_jittered, long_df["y"], c=colors, s=20, alpha=0.8, edgecolors="none")

# Axis formatting
ax.set_xticks([0, 1])
ax.set_xticklabels(["MDD-only", "MDD+RBD"])
ax.set_yticks(range(len(subjects)))
ax.set_yticklabels(subjects)

ax.set_xlabel("Predicted class")
ax.set_ylabel("Subject")
ax.set_title("Slice-level predictions per subject\nGreen = correct, Red = incorrect")

# Custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Correct',
           markerfacecolor='green', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Incorrect',
           markerfacecolor='red', markersize=8)
]
ax.legend(handles=legend_elements, title="Prediction")

plt.tight_layout()
plt.show()
