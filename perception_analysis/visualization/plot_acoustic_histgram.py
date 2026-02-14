import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from perception_analysis.data_file_paths import RSA_BARPANELS_PATH, RSA_RESULTS_BY_CATEGORY_PATH

# ─── SETTINGS ────────────────────────────────────────────────────────────────

# mapping from raw variable names to display labels
BASE_LABELS = {
    "AR_mean":     r"AR",
    "pF0_mean":    r"$f_0$ M",
    "pF0_cv":      r"$f_0$ CV",
    "pF1_mean":    r"$F_1$ M",
    "pF1_cv":      r"$F_1$ CV",
    "pF2_mean":    r"$F_2$ M",
    "pF2_cv":      r"$F_2$ CV",
    "pF3_mean":    r"$F_3$ M",
    "pF3_cv":      r"$F_3$ CV",
    "pF4_mean":    r"$F_4$ M",
    "pF4_cv":      r"$F_4$ CV",
    "FD_mean":     r"FD M",
    "FD_cv":       r"FD CV",
    "H1H2c_mean":  r"$H_1^*-H_2^*$ M",
    "H1H2c_cv":    r"$H_1^*-H_2^*$ CV",
    "H2H4c_mean":  r"$H_2^*-H_4^*$ M",
    "H2H4c_cv":    r"$H_2^*-H_4^*$ CV",
    "H42Kc_mean":  r"$H_4^*-H_{2kHz}^*$ M",
    "H42Kc_cv":    r"$H_4^*-H_{2kHz}^*$ CV",
    "H2KH5Kc_mean":r"$H_{2kHz}^*-H_{5kHz}$ M",
    "H2KH5Kc_cv":  r"$H_{2kHz}^*-H_{5kHz}$ CV",
    "energy_mean": r"Energy M",
    "energy_cv":   r"Energy CV",
    "CPP_mean":    r"CPP M",
    "CPP_cv":      r"CPP CV",
    "SHR_mean":    r"SHR M",
    "SHR_cv":      r"SHR CV",
    "HNR_mean":    r"HNR M",
    "HNR_cv":      r"HNR CV"
}

# define groups
group1_vars = {"AR_mean"}
group2_vars = {"pF0_mean", "pF0_cv"}
group3_vars = {"pF1_mean", "pF1_cv", "pF2_mean", "pF2_cv",
               "pF3_mean", "pF3_cv", "pF4_mean", "pF4_cv",
               "FD_mean", "FD_cv"}
group4_vars = {"H1H2c_mean", "H1H2c_cv", "H2H4c_mean", "H2H4c_cv",
               "H42Kc_mean", "H42Kc_cv", "H2KH5Kc_mean", "H2KH5Kc_cv"}
group5_vars = {"energy_mean", "energy_cv", "CPP_mean", "CPP_cv",
               "SHR_mean", "SHR_cv", "HNR_mean", "HNR_cv"}

def get_group(var_name):
    if var_name in group1_vars:
        return 1
    if var_name in group2_vars:
        return 2
    if var_name in group3_vars:
        return 3
    if var_name in group4_vars:
        return 4
    if var_name in group5_vars:
        return 5
    return None

# use first five colors from Set3
cmap = plt.get_cmap("Set3")
GROUP_COLORS = {
    1: cmap(3),
    2: cmap(5),
    3: cmap(2),
    4: cmap(0),
    5: cmap(4)
}

# categories in plotting order and their titles
CATEGORIES = ["all", "can", "eng", "mixed"]
CATEGORY_TITLES = {
    "all":   "All stimuli",
    "can":   "Cantonese-only stimuli",
    "eng":   "English-only stimuli",
    "mixed": "Mixed-language stimuli"
}

# variable order (use BASE_LABELS keys order)
VAR_ORDER = list(BASE_LABELS.keys())

# ─── LOAD RSA SUMMARY ─────────────────────────────────────────────────────────

df = pd.read_csv(RSA_RESULTS_BY_CATEGORY_PATH)

# ─── SETUP FIGURE WITH 4 PANELS ───────────────────────────────────────────────

fig, axes = plt.subplots(
    nrows=4, ncols=1,
    figsize=(12, 10.5),
    sharex=True
)

# Reduce vertical space between panels
plt.subplots_adjust(hspace=0.45)

# Prepare x‐tick positions and number labels
num_vars = len(VAR_ORDER)
x_ticks = np.arange(num_vars)
x_number_labels = [str(i+1) for i in range(num_vars)]

# ─── PLOT EACH CATEGORY ───────────────────────────────────────────────────────

for idx, (ax, category) in enumerate(zip(axes, CATEGORIES)):
    # Filter for this category
    cat_df = df[df["category"] == category].copy()
    # Reindex to ensure VAR_ORDER sequence
    cat_df = cat_df.set_index("feature").reindex(VAR_ORDER).reset_index()
    
    # Extract data
    means = cat_df["mean_rho"].values
    sems = cat_df["sem_rho"].values
    p_fdr = cat_df["p_fdr"].values
    
    # Determine bar colors: significant => group color; else white
    colors = []
    for feat, p in zip(VAR_ORDER, p_fdr):
        grp = get_group(feat)
        if (not np.isnan(p)) and (p < 0.05) and (grp is not None):
            colors.append(GROUP_COLORS[grp])
        else:
            colors.append("white")
    
    # Plot bars (no outline)
    ax.bar(
        x_ticks, means,
        color=colors,
        edgecolor="none",
        width=0.8
    )
    
    # Plot error bars: black if significant, white if not
    for xi, mean, sem, color in zip(x_ticks, means, sems, colors):
        err_color = "black" if color != "white" else "white"
        ax.errorbar(
            xi, mean,
            yerr=sem,
            fmt='none',
            ecolor=err_color,
            elinewidth=1,
            capsize=4
        )
    
    # Title and axes
    ax.set_title(CATEGORY_TITLES[category], fontsize=18, pad=6)
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0, .1, .2, .3, .4, .5])
    ax.yaxis.set_tick_params(labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylabel("ρ", fontsize=16)
    
    # Set x‐limits slightly beyond bar positions
    ax.set_xlim(-0.5, num_vars - 0.5)
    
    # Ensure x‐ticks with numbers 1–29 appear on all panels, right above x-axis
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_number_labels, rotation=0, fontsize=14)
    ax.tick_params(axis='x', which='both', labelbottom=True, pad=5)

# ─── ADD NAME LABELS ON BOTTOM PANEL ─────────────────────────────────────────

bottom_ax = axes[-1]
# Keep numbers above x-axis
bottom_ax.set_xlim(-0.9, num_vars - 0.1)
bottom_ax.set_xticks(x_ticks)
bottom_ax.set_xticklabels(x_number_labels, rotation=0, fontsize=14)
bottom_ax.tick_params(axis='x', which='both', labelbottom=True, pad=5)

# Add variable names below numbers (horizontal, one line)
for xi, var in enumerate(VAR_ORDER):
    bottom_ax.text(
        xi+0.06, -0.22,                # further below x-axis
        BASE_LABELS[var],         # display label
        rotation=90,
        ha='center', va='top',
        fontsize=14,
        transform=bottom_ax.get_xaxis_transform()
    )

# ─── SAVE FIGURE ──────────────────────────────────────────────────────────────
out_base = os.path.splitext(RSA_BARPANELS_PATH)[0]  # remove .png extension
plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
plt.savefig(out_base + ".tiff", dpi=300, bbox_inches="tight")
plt.savefig(out_base + ".pdf", bbox_inches="tight")
plt.savefig(out_base + ".svg", bbox_inches="tight")
plt.savefig(out_base + ".eps", format="eps", bbox_inches="tight")
plt.show()