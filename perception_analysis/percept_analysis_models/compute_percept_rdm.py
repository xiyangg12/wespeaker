import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_file_paths import PERCEPT_DATASET_PATH, RDM_OUTPUT_DIR

def is_model_name(col_name):
    return col_name.startswith("vox") or col_name.startswith("cn")

# Create output directory if it doesn't exist
os.makedirs(RDM_OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(PERCEPT_DATASET_PATH)

# Identify all subject columns (they start with 'subj')
subject_cols = [col for col in df.columns if is_model_name(col)]

# Initialize a scaler for 0–1 normalization
scaler = MinMaxScaler(feature_range=(0, 1))

# ─── PROCESS EACH SUBJECT ────────────────────────────────────────────────────

for subj in subject_cols:
    # Extract trial_id, stim1, stim2, and this subject's ratings
    subj_df = df[["trial_id", "stim1", "stim2", subj]].copy()
    
    # Remove the ".wav" extension from stim1 and stim2, and rename columns
    subj_df["label1"] = subj_df["stim1"].str.replace(r"\.wav$", "", regex=True)
    subj_df["label2"] = subj_df["stim2"].str.replace(r"\.wav$", "", regex=True)
    subj_df = subj_df.drop(columns=["stim1", "stim2"])
    
    # Normalize the ratings to [0, 1] for this subject
    ratings = subj_df[subj].values.reshape(-1, 1)
    ratings = 1 - ratings  # Invert ratings to convert similarity to dissimilarity
    normalized = scaler.fit_transform(ratings).flatten()
    # The matrics we store are similarity matrices, 
    # so we need to convert the ratings to dissimilarity by inverting them (1 - normalized_rating)
    subj_df[subj] = normalized  # Convert similarity to dissimilarity
    
    # Rename the subject column to "dissimilarity"
    subj_df.rename(columns={subj: "dissimilarity"}, inplace=True)
    
    # Add a column "subject" with the original subject column name
    subj_df["subject"] = subj
    
    # Reorder columns: trial_id, subject, label1, label2, dissimilarity
    subj_df = subj_df[["trial_id", "subject", "label1", "label2", "dissimilarity"]]
    
    # Save to CSV named after the subject (e.g., "subjC001.csv")
    out_path = os.path.join(RDM_OUTPUT_DIR, f"{subj}.csv")
    subj_df.to_csv(out_path, index=False)