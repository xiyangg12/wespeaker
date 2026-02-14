"""
Script to modify columns in CSV files with specific transformation rules.
"""
import os
import pandas as pd
import re


def extract_model_name(col_name):
    """
    Extract model name from columns with pattern: /home/xiyali/git/wespeaker/exp/{model_name}/models
    Returns the model_name if pattern matches, otherwise returns the original column name.
    """
    pattern = r"/home/xiyali/git/wespeaker/exp/(.+?)/models"
    match = re.search(pattern, col_name)
    if match:
        return match.group(1)
    return col_name


def modify_csv_columns(input_csv_path, output_csv_path=None):
    """
    Modify CSV columns according to the following rules:
    1. Extract model_name from columns matching the pattern: /home/xiyali/git/wespeaker/exp/{model_name}/models
    
    Args:
        input_csv_path: Path to the input CSV file
        output_csv_path: Path to the output CSV file (if None, will overwrite input)
    """
    if output_csv_path is None:
        output_csv_path = input_csv_path
    
    # Read the CSV
    df = pd.read_csv(input_csv_path)
    print(f"📖 Read CSV with {len(df)} rows and {len(df.columns)} columns")
    print(f"Original columns: {list(df.columns)}")
    
    # Create a mapping of old column names to new ones
    column_mapping = {}
    for col in df.columns:
        new_col = extract_model_name(col)
        if new_col != col:
            column_mapping[col] = new_col
            print(f"  ✓ Renamed: {col} → {new_col}")
    
    # TODO: Add your custom column rename rules here
    # Example:
    # column_mapping['old_column_name'] = 'new_column_name'
    
    # Apply the renaming
    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)
        print(f"\n✅ Applied {len(column_mapping)} column transformations")
    else:
        print("\n⚠️  No column transformations applied")
    
    print(f"Modified columns: {list(df.columns)}")
    
    # Save the modified CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\n💾 Saved modified CSV to: {output_csv_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    csv_path = "/Users/lixiyang/Desktop/git/wespeaker/perception_analysis/files/finetuned_model_similarities.csv"
    
    if os.path.exists(csv_path):
        modify_csv_columns(csv_path)
    else:
        print(f"❌ File not found: {csv_path}")
