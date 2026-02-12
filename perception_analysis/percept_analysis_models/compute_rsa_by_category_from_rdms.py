import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from collections import defaultdict
from data_file_paths import HUMAN_RDM_PATH, RDM_OUTPUT_DIR, RSA_RESULTS_BY_CATEGORY_PATH, RSA_RESULTS_PATH, ACOUSTIC_RDM_PATH
from statsmodels.stats.multitest import fdrcorrection


def load_percept_rdms(percept_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all perceptual RDM CSVs. Returns a dict: subject -> DataFrame
    Each DataFrame has columns: l1, l2, dissimilarity (lowercased labels)
    """
    percept_paths = glob.glob(os.path.join(percept_dir, "*.csv"))
    percept_rdms = {}
    for path in percept_paths:
        df = pd.read_csv(path)
        # Rename and lowercase labels
        df = df.rename(columns={"label1": "l1", "label2": "l2", "dissimilarity": "dissimilarity"})
        df["l1"] = df["l1"].str.lower()
        df["l2"] = df["l2"].str.lower()
        if "subject" in df.columns:
            data = df["subject"]
        elif "feature" in df.columns:
            data = df["feature"]
        else:
            raise ValueError(f"CSV at {path} must contain either 'subject' or 'feature' column.")
        subj_name = data.iloc[0]
        percept_rdms[subj_name] = df[["l1", "l2", "dissimilarity"]].copy()
    return percept_rdms

def filter_by_category(df_merge, category):
    """
    Given a merged DataFrame with columns ["l1","l2","dissimilarity_percept","dissimilarity_acoust"],
    return a subset according to:
      - "all": no filtering
      - "can": both l1 and l2 contain "can"
      - "eng": both l1 and l2 contain "eng"
      - "mixed": one contains "can" and the other contains "eng"
    """
    if category == "all":
        return df_merge
    elif category == "can":
        return df_merge[df_merge["l1"].str.contains("can") & df_merge["l2"].str.contains("can")]
    elif category == "eng":
        return df_merge[df_merge["l1"].str.contains("eng") & df_merge["l2"].str.contains("eng")]
    elif category == "mixed":
        return df_merge[
            ((df_merge["l1"].str.contains("can") & df_merge["l2"].str.contains("eng")) |
             (df_merge["l1"].str.contains("eng") & df_merge["l2"].str.contains("can")))
        ]
    else:
        return pd.DataFrame(columns=df_merge.columns)

def compute_spearman_rhos(percept_rdms_1: dict[str, pd.DataFrame], percept_rdms_2: dict[str, pd.DataFrame], suffix_1: str, suffix_2: str) -> dict[str, list[float]]:
    categories = ["all", "can", "eng", "mixed"]
    # Initialize a dict to collect rhos: (feature, category) -> list of rhos for each subject
    category_rhos = defaultdict(list)

    for subj_1 in percept_rdms_1:
        subj_df_1 = percept_rdms_1[subj_1]
        for subj_2 in percept_rdms_2:
            subj_df_2 = percept_rdms_2[subj_2]
            # Merge on (l1, l2), left join to keep all perceptual pairs
            merged = pd.merge(
                subj_df_1, subj_df_2,
                on=["l1", "l2"],
                how="left",
                suffixes=(suffix_1, suffix_2)
            )
            # Fill missing human dissimilarities with 0
            merged[f"dissimilarity{suffix_2}"] = merged[f"dissimilarity{suffix_2}"].fillna(0.0)
            
            # For each category, filter and compute Spearman rho (or 0 if no data)
            for cat in categories:
                subset = filter_by_category(merged, cat)
                if subset.empty:
                    rho = 0.0
                else:
                    rho, _ = spearmanr(
                        subset[f"dissimilarity{suffix_1}"],
                        subset[f"dissimilarity{suffix_2}"]
                    )
                    if np.isnan(rho):
                        rho = 0.0
                category_rhos[(subj_2, cat)].append(rho)
    return category_rhos

def compute_rsa(category_rhos: dict[tuple[str, str], list[float]]) -> list[dict]:
    results = []
    for (feature, category), rhos in category_rhos.items():
        rhos_arr = np.array(rhos, dtype=float)
        
        # Compute mean, standard deviation, and standard error of the mean
        mean_rho = np.mean(rhos_arr)
        sd_rho = np.std(rhos_arr, ddof=1) if len(rhos_arr) > 1 else 0.0
        sem_rho = sd_rho / np.sqrt(len(rhos_arr)) if len(rhos_arr) > 1 else 0.0
        
        # One-sided Wilcoxon signed-rank test (alternative: median > 0)
        try:
            w_stat, p_val = wilcoxon(rhos_arr, alternative="greater")
        except TypeError:
            # Fall back for older scipy versions
            w_stat, p_two = wilcoxon(rhos_arr)
            p_val = (p_two / 2) if np.median(rhos_arr) > 0 else (1 - p_two / 2)
        
        results.append({
            "feature": feature,
            "category": category,
            "mean_rho": mean_rho,
            "sd_rho": sd_rho,
            "sem_rho": sem_rho,
            "wilcoxon_stat": w_stat,
            "p_raw": p_val
        })
    return results

def fdr_correction(results: list[dict[str, float]]) -> list[dict]:
    p_vals = [r["p_raw"] for r in results]
    p_vals_clean = [pv if not np.isnan(pv) else 1.0 for pv in p_vals]
    _, p_fdr = fdrcorrection(p_vals_clean, alpha=0.05, method="indep")
    for r, fdr_p in zip(results, p_fdr):
        r["p_fdr"] = fdr_p
    return results

def save_to_csv(results: list[dict[str, float]]) -> None:
    results_df = pd.DataFrame(results)[[
        "feature", "category", "mean_rho", "sd_rho", "sem_rho",
        "wilcoxon_stat", "p_raw", "p_fdr"
    ]]
    results_df.to_csv(RSA_RESULTS_BY_CATEGORY_PATH, index=False)

def main():
    # acoustic data
    suffix_2 = "_acoustic"
    percept_rdms_2  = load_percept_rdms(ACOUSTIC_RDM_PATH)

    # Model data
    suffix_1 = "_model"
    percept_rdms_1 = load_percept_rdms(RDM_OUTPUT_DIR)

    # # Human data
    # suffix_1 = "_human"
    # percept_rdms_1  = load_percept_rdms(HUMAN_RDM_PATH)

    # Compute Spearman rhos
    rhos = compute_spearman_rhos(percept_rdms_1, percept_rdms_2, suffix_1, suffix_2)
    # Compute RSA results
    rsa_results = compute_rsa(rhos)
    # Apply FDR correction
    rsa_results_fdr = fdr_correction(rsa_results)
    # Save results to CSV
    save_to_csv(rsa_results_fdr)

if __name__ == "__main__":   
    main()





