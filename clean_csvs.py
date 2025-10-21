import os
import pandas as pd
import numpy as np

# Folder containing your CSVs
data_folder = 'data'

# Specify which columns to remove from which files
columns_to_remove_by_file = {
    'f1_results_2014.csv': ['Race_BroadcastName'],
    'f1_results_2015.csv': ['Race_BroadcastName'],
    'f1_results_2016.csv': ['Race_BroadcastName'],
    'f1_results_2017.csv': ['Race_BroadcastName'],
    
    # Add more files and columns as needed
}

for filename in os.listdir(data_folder):
    if filename in columns_to_remove_by_file:
        file_path = os.path.join(data_folder, filename)
        print(f"Processing {filename}...")
        df = pd.read_csv(file_path)

        # Remove specified columns for this file
        cols_to_remove = columns_to_remove_by_file[filename]
        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns], errors='ignore')

        # Remove rows where all values are empty or NaN
        df = df.dropna(how='all')

        # Replace all remaining NaN/empty cells with an empty string
        df = df.replace({np.nan: ''})

        # Save back to the same file
        df.to_csv(file_path, index=False)
        print(f"  Saved cleaned {filename}")

print("Selected files processed.")