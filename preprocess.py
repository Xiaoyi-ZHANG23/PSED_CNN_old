import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import train_test_split
import json

# df_iso1 = pd.read_csv('/data/yll6162/mof_cnn/PSED_data/extracted_cm3_per_cm3_values_part1.csv')
# df_iso2 = pd.read_csv('/data/yll6162/mof_cnn/PSED_data/extracted_cm3_per_cm3_values_part2.csv')
# df_iso3 = pd.read_csv('/data/yll6162/mof_cnn/PSED_data/extracted_cm3_per_cm3_values_part3.csv')
pressure = '0.25bar'
pressure_map = {'0.1bar': '0p1bar', '1bar': '1bar', '10bar': '10bar', '0.25bar': '0p25bar', '0.5bar': '0p5bar'}
pressure_str = pressure_map[pressure]
grid_type = '' #empty for original grid, or translated or rotated
# ref_data_split = f'/data/yll6162/mof_cnn/data_mix_{pressure_str}'
ref_data_split = None
target_col1 = 'Xe_cm3_per_cm3_value'
target_col2 = 'Kr_cm3_per_cm3_value'
output_dir = f'/data/yll6162/mof_cnn/data_mix_{pressure_str}_{grid_type}'.rstrip('_')
input_csv = '/data/yll6162/mof_cnn/PSED_data/combined_data_by_pressure_volume.xlsx'
os.makedirs(output_dir, exist_ok=True)
# df_iso1 = pd.read_csv('/data/yll6162/mof_cnn/PSED_data/converted_10bar_5k.csv')
# df_iso2 = pd.read_csv('/data/yll6162/mof_cnn/PSED_data/converted_10bar_qmof.csv')
# df_iso3 = 
df_iso = pd.read_excel(input_csv, sheet_name=pressure)
df_iso['database'] = df_iso['project'].apply(lambda x: 'qmof' if x.startswith('qmof') else 'ToBaCCo')
df_iso = df_iso[~df_iso[target_col1].isna()] 
df_iso = df_iso[~df_iso[target_col2].isna()] # exclude NAN values
df_iso.to_csv(f'{output_dir}/all.csv', index=False)


# Define the top-level directory
base_dir = "/data/yll6162/mof_cnn/PSED_data/New_Parsed_new_new/New_Parsed/" ## REPLACE WITH YOUR DIRECTORY

# Define the relative subdirectory and file names to check
subdir = "ASCI_Grids"
file_names = [f"{grid_type}_energy_grid.txt".lstrip('_')]
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

print(f"Absent {grid_type} energy_grid.txt files: {len(absent_files)}")
print(f"Empty {grid_type} energy_grid.txt files: {len(empty_files)}")
print(f"Abnormal {grid_type} energy_grid.txt files: {len(abnormal_files)}")
print(f"Normal {grid_type} energy_grid.txt files: {len(normal_files)}")
#Filter by isotherm values
filterd_mofs = []
unmatched_mofs = []
for mof_file in normal_files:
    if mof_file in df_iso['project'].values:
        filterd_mofs.append(mof_file)
print(f"Filtered mofs: {len(filterd_mofs)}")
print(f"Unmatched mofs: {len(normal_files) - len(filterd_mofs)}")


datset_mof = {}
if ref_data_split and os.path.exists(ref_data_split):
    print(f"using reference dataset split from {ref_data_split}")
    for split in ['train', 'test', 'val']:
        with open(f'{ref_data_split}/{split}/clean.json', 'r') as f:
            ref_data = json.load(f)
            subset_files = [sample for sample in ref_data['name'] if sample in filterd_mofs]
            datset_mof[split] = subset_files
            print(f"Loaded {split} files: {len(subset_files)} from original reference dataset split {len(ref_data['name'])}")
else:
    print("No reference dataset split found. Splitting data randomly.")
    test_size = 0.1
    val_size = 0.1
    train_files, test_files = train_test_split(filterd_mofs, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size/(1-test_size), random_state=42)
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Testing files: {len(test_files)}")
    datset_mof = {'train': train_files, 'test': test_files, 'val': val_files}

grid = 41

for subset in ['train', 'test', 'val']:
    dir_path = os.path.join(output_dir, subset)
    os.makedirs(dir_path, exist_ok=False)
    subset_data = {}
    subset_data['name'] = datset_mof[subset]
    subset_data['grid'] = grid
    subset_data['size'] = len(datset_mof[subset])
    with open(f'{dir_path}/clean.json', 'w') as f:
        json.dump(subset_data, f)
    print(f"saved {subset} data to {dir_path}/clean.json")
    
pos_cap = 0

grid = 41


for subset in ['train', 'test', 'val']:
    array_list = []
    dir_path = os.path.join(output_dir, subset)
    grid_file = f"{grid_type}_energy_grid.txt".lstrip('_')
    for mof_file in datset_mof[subset]:
        with open(f"{base_dir}/{mof_file}/ASCI_Grids/{grid_file}", "r") as file:
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