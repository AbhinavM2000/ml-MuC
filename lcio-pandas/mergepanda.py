import os
import pandas as pd

# Get the current directory
directory = os.getcwd()

# List all .abnv files in the directory
abnv_files = [file for file in os.listdir(directory) if file.endswith('.abnv')]

# Initialize an empty list to store dataframes
dfs = []

# Iterate over each .abnv file
for file in abnv_files:
    # Load the .abnv file into a pandas dataframe
    df = pd.read_csv(os.path.join(directory, file))
    # Append the dataframe to the list
    dfs.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Export the combined dataframe to master_hard.abnv
output_file = 'master_BIB.abnv'
combined_df.to_csv(output_file, index=False)

print(f"Combined dataframe exported to {output_file}")
