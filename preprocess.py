import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# df_iso1 = pd.read_csv('/data/yll6162/mof_cnn/PSED_data/extracted_cm3_per_cm3_values_part1.csv')
# df_iso2 = pd.read_csv('/data/yll6162/mof_cnn/PSED_data/extracted_cm3_per_cm3_values_part2.csv')
# df_iso3 = pd.read_csv('/data/yll6162/mof_cnn/PSED_data/extracted_cm3_per_cm3_values_part3.csv')
target_col = 'Kr_cm3_per_cm3_value'
output_dir = '/projects/p32082/PSED_CNN_old/data'
# os.makedirs(output_dir, exist_ok=True)
# df_iso1 = pd.read_csv('/data/yll6162/mof_cnn/PSED_data/converted_10bar_5k.csv')
# df_iso2 = pd.read_csv('/data/yll6162/mof_cnn/PSED_data/converted_10bar_qmof.csv')
# df_iso3 = 
df_iso = pd.read_csv('/projects/p32082/PSED_CNN_old/data/10bar_col_with_property.csv')
df_iso['database'] = df_iso['project'].apply(lambda x: 'qmof' if x.startswith('qmof') else 'ToBaCCo')
df_iso = df_iso[~df_iso[target_col].isna()] # exclude NAN values
df_iso.to_csv(f'{output_dir}/all.csv', index=False)
df_iso
import os

# Define the top-level directory
base_dir = "/projects/p32082/PSED_CNN_old/data/New_Parsed/" ## REPLACE WITH YOUR DIRECTORY

# Define the relative subdirectory and file names to check
subdir = "ASCI_Grids"
file_names = ["energy_grid.txt"]
abnormal_files = []
empty_files = []
absent_files = []
normal_files = []
# Loop through all subdirectories under the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    # Ensure it is a directory
    if os.path.isdir(folder_path):
        asci_grids_path = os.path.join(folder_path, subdir)
        
        # Ensure the ASCI_Grids subdirectory exists
        if os.path.exists(asci_grids_path):
            # print(f"\nProcessing ASCI_Grids in folder: {folder}")
            
            # Check for each required file
            for file_name in file_names:
                file_path = os.path.join(asci_grids_path, file_name)
                
                # Verify the file exists
                if os.path.exists(file_path):
                    # print(f"\nReading file: {file_name} in folder {folder}")
                    try:
                        # Open and print each line of the file
                        total_count = 0

                        with open(file_path, 'r') as file:
                            lines = file.readlines()
                            if len(lines) != 68921:
                                if len(lines) == 0:
                                    empty_files.append(folder)

                                # print(f"Error: {file_name} in folder {folder} has {len(lines)} lines")
                                abnormal_files.append(folder)
                                break
                            else:
                                normal_files.append(folder)
                    except Exception as e:
                        print(f"Error reading {file_name} in folder {folder}: {e}")
                else:
                    absent_files.append(folder)
                    print(f"{file_name} does not exist in folder: {folder}")
        else:
            print(f"ASCI_Grids subdirectory does not exist in folder: {folder}")

print(f"Absent energy_grid.txt files: {len(absent_files)}")
print(f"Empty energy_grid.txt files: {len(empty_files)}")
print(f"Abnormal energy_grid.txt files: {len(abnormal_files)}")
print(f"Normal energy_grid.txt files: {len(normal_files)}")
#Filter by isotherm values
filterd_mofs = []
unmatched_mofs = []
for mof_file in normal_files:
    if mof_file in df_iso['project'].values:
        filterd_mofs.append(mof_file)
print(f"Filtered mofs: {len(filterd_mofs)}")
print(f"Unmatched mofs: {len(normal_files) - len(filterd_mofs)}")
from sklearn.model_selection import train_test_split
import json
test_size = 0.1

train_files, test_files = train_test_split(filterd_mofs, test_size=0.1, random_state=42)
print(f"Training files: {len(train_files)}")
print(f"Testing files: {len(test_files)}")

datset_mof = {'train': train_files, 'test': test_files}

grid = 41

for subset in ['train', 'test']:
    dir_path = os.path.join(output_dir, subset)
    os.makedirs(dir_path, exist_ok=True)
    subset_data = {}
    subset_data['name'] = datset_mof[subset]
    subset_data['grid'] = grid
    subset_data['size'] = len(datset_mof[subset])
    with open(f'{dir_path}/clean.json', 'w') as f:
        json.dump(subset_data, f)
    print(f"saved {subset} data to {dir_path}/clean.json")
    
pos_cap = 0
source_dir = "/projects/p32082/PSED_CNN_old/data/New_Parsed/"
grid = 41
data_splits = [train_files, test_files]

for subset in ['train', 'test']:
    array_list = []
    dir_path = os.path.join(output_dir, subset)
    os.makedirs(dir_path, exist_ok=True)
    for mof_file in datset_mof[subset]:
        with open(f"{source_dir}/{mof_file}/ASCI_Grids/energy_grid.txt", "r") as file:
            lines = file.readlines()  # Each line is stored as an element in the list
        # Step 1: Convert list to NumPy array
        data_array = np.array(lines, dtype=object)
        clean_array = np.where(data_array == '?\n', pos_cap, data_array).astype(float)
        cap_array = np.clip(clean_array, None, pos_cap)
        reshaped_array = cap_array.reshape((grid, grid, grid))
        assert np.isnan(reshaped_array).sum()==0
        array_list.append(reshaped_array)

    dataset_array = np.stack(array_list, axis=0)
    np.save(f"{dir_path}/clean.npy", dataset_array)
    print(f"saved {subset} data to {dir_path}/clean.npy")