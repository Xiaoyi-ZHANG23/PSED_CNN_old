import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

target_col = 'Kr_cm3_per_cm3_value'
output_dir = '/projects/p32082/PSED_CNN_old/data'
os.makedirs(output_dir, exist_ok=True)

df_iso = pd.read_csv('/projects/p32082/PSED_CNN_old/data/10bar_col_property.csv')
df_iso['database'] = df_iso['project'].apply(lambda x: 'qmof' if x.startswith('qmof') else 'ToBaCCo')
# Remove rows with NaN in the target
df_iso = df_iso[~df_iso[target_col].isna()]

# Save the combined CSV for later reference:
df_iso.to_csv(f'{output_dir}/all.csv', index=False)

base_dir = "/projects/p32082/PSED_CNN_old/data/New_Parsed/"
subdir   = "ASCI_Grids"
file_name = "energy_grid.txt"
pos_cap = 0
grid = 41

normal_files = []    # MOF IDs with a normal sized file (68921 lines)
abnormal_files = []
absent_files   = []
empty_files    = []

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        asci_grids_path = os.path.join(folder_path, subdir)
        if os.path.exists(asci_grids_path):
            file_path = os.path.join(asci_grids_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                if len(lines) == 68921:
                    normal_files.append(folder)
                else:
                    # e.g. empty or abnormal
                    if len(lines) == 0:
                        empty_files.append(folder)
                    abnormal_files.append(folder)
            else:
                absent_files.append(folder)

print(f"Normal files:   {len(normal_files)}")
print(f"Abnormal files: {len(abnormal_files)}")
print(f"Absent files:   {len(absent_files)}")
print(f"Empty files:    {len(empty_files)}")

# Filter normal_files to those that also appear in df_iso['project']
filtered_mofs = [m for m in normal_files if m in df_iso['project'].values]
print(f"Filtered MOFs in df_iso: {len(filtered_mofs)}")

# Split into train/test
train_files, test_files = train_test_split(filtered_mofs, test_size=0.1, random_state=42)
print(f"Training files: {len(train_files)}")
print(f"Testing files:  {len(test_files)}")

dataset_mof = {'train': train_files, 'test': test_files}

# Save metadata (names) + grid info into clean.json for each split
for subset in ['train', 'test']:
    subset_dir = os.path.join(output_dir, subset)
    os.makedirs(subset_dir, exist_ok=True)

    subset_data = {
        'name': dataset_mof[subset],  # MOF IDs
        'grid': grid,
        'size': len(dataset_mof[subset])
    }
    with open(f'{subset_dir}/clean.json', 'w') as f:
        json.dump(subset_data, f)
    print(f"Saved {subset} metadata to {subset_dir}/clean.json")

# Build the numeric 3D arrays and save them to clean.npy
for subset in ['train', 'test']:
    subset_dir = os.path.join(output_dir, subset)
    array_list = []
    for mof_file in dataset_mof[subset]:
        file_path = os.path.join(base_dir, mof_file, subdir, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Convert lines to float, replacing '?' with pos_cap=0
        data_array = np.array(lines, dtype=object)
        clean_array = np.where(data_array == '?\n', pos_cap, data_array).astype(float)
        # Clip values above pos_cap=0
        cap_array = np.clip(clean_array, None, pos_cap)
        # Reshape (68921 -> 41 x 41 x 41)
        reshaped_array = cap_array.reshape((grid, grid, grid))
        # Should be no NaNs if your checks pass
        assert np.isnan(reshaped_array).sum() == 0
        array_list.append(reshaped_array)

    dataset_array = np.stack(array_list, axis=0)  # shape: (N, 41, 41, 41)
    np.save(f"{subset_dir}/clean.npy", dataset_array)
    print(f"Saved {subset} data array to {subset_dir}/clean.npy")
