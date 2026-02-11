import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from pathlib import Path

from data_file_paths import RDM_OUTPUT_DIR, RDM_PLOTS_DIR

# Paths
input_dir = Path(RDM_OUTPUT_DIR)  # folder with long-form CSVs
output_dir = Path(RDM_PLOTS_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# Process each long-form CSV
for file in input_dir.glob("*.csv"):
    df = pd.read_csv(file)

    # Extract subject name (should be same throughout file)
    feature_name = df["subject"].iloc[0]

    # Build RDM square matrix
    labels = sorted(set(df["label1"]) | set(df["label2"]))
    rdm = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for _, row in df.iterrows():
        l1, l2 = row["label1"], row["label2"]
        d = row["dissimilarity"]
        rdm.loc[l1, l2] = d
        rdm.loc[l2, l1] = d
        rdm.loc[l1, l1] = 0
        rdm.loc[l2, l2] = 0

    # Handle missing entries
    rdm_values = rdm.fillna(0).to_numpy()

    # Rank-transform and scale to [0, 1]
    ranked = rankdata(rdm_values, method="average").reshape(rdm_values.shape)
    scaled = ranked / np.max(ranked)

    # Plot
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        scaled,
        ax=ax,
        cmap="terrain",
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        vmin=0,
        vmax=1
    )
    ax.set_title(feature_name.replace("_", " "), fontsize=10)

    # Save in multiple formats
    for ext in ["png", "svg", "eps"]:
        fig.savefig(output_dir / f"acoust_rdm_{feature_name}.{ext}", dpi=300, format=ext)

    plt.close(fig)
    print(f"Saved: acoust_rdm_{feature_name}")