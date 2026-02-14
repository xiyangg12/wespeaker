from pathlib import Path
"""
This file is used to save the paths of the data files used in the project.
It helps to keep the code organized and makes it easier to update the paths if needed.
"""

# Paths to csv data files
FILE_PATH = Path("/Users/lixiyang/Desktop/git/wespeaker/perception_analysis/files")

PERCEPT_DATASET_PATH = FILE_PATH / "finetuned_model_similarities.csv"
SAME_SPKEAR_PAIR_ACCURACY_PATH = FILE_PATH / "same_identity_means.csv"
RDM_OUTPUT_DIR = FILE_PATH / "rdms_models_finetuned"
HUMAN_RDM_PATH = FILE_PATH / "rdms_human"
ACOUSTIC_RDM_PATH = FILE_PATH / "rdms_acoustic"
RSA_RESULTS_PATH = FILE_PATH / "rsa_summary_acoustic_model.csv"
RSA_RESULTS_BY_CATEGORY_PATH = FILE_PATH / "rsa_summary_by_category_acoustic_model_finetuned.csv"

# Paths to plots
PLOTS_PATH = Path("/Users/lixiyang/Desktop/git/wespeaker/perception_analysis/plots")

RDM_PLOTS_DIR = PLOTS_PATH / "rdms"
RSA_BARPANELS_PATH = PLOTS_PATH / "rsa_barpanels_acoustic_model_finetuned.png"
