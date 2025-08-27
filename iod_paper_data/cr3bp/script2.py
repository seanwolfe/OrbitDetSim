import os
import glob
import pandas as pd

def merge_csvs(input_folder, output_file):
    # Find all csv files in the folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    # Read and concatenate
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    # Save to a single CSV
    merged_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_folder = "meta_files"   # <-- change this
    output_file = "merged.csv"             # <-- change this if needed
    merge_csvs(input_folder, output_file)
    print(f"Merged CSV saved to {output_file}")