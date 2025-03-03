
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import json
import shutil  # <-- needed if you want to remove directories

pressure = '0.1bar'
pressure_map = {'0.1bar': '0p1bar', '1bar': '1bar', '10bar': '10bar', 
                '0.25bar': '0p25bar', '0.5bar': '0p5bar'}
pressure_str = pressure_map[pressure]

# Choose your three grid types
grid_types = ['center', 'rotated', 'translated']
SEED = 42

# If you had a reference split path, set it here; otherwise None
ref_data_split = 'sample_split_3/'

target_col1 = 'Xe_cm3_per_cm3_value'
target_col2 = 'Kr_cm3_per_cm3_value'

# Decide your output directory
if len(grid_types) > 1:
    output_dir = f'/data/yll6162/mof_cnn/data_mix_{pressure_str}_{len(grid_types)}_grids'
else:
    output_dir = f'/data/yll6162/mof_cnn/data_mix_{pressure_str}_{grid_types[0]}'.rstrip('_')

# Your input Excel file
input_csv = 'cleaned_data.xlsx'
os.makedirs(output_dir, exist_ok=True)
property_csv = 'sample_split_single/1bar_all.csv'
property_cols = ["PLD", "LCD"]

# The top-level directory containing MOF subfolders
base_dir = "/data/yll6162/mof_cnn/PSED_data/New_Parsed_new_new/New_Parsed/"
subdir = "ASCI_Grids"
csv_only = True # if true, will skip clean.josn and clean.npy generation
# Read the isotherm data from the relevant sheet

def main():
    df_iso = pd.read_excel(input_csv, sheet_name=pressure)
    df_iso['database'] = df_iso['project'].apply(lambda x: 'qmof' if x.startswith('qmof') else 'ToBaCCo')

    # Clean out rows with NaN in target columns
    df_iso = df_iso[~df_iso[target_col1].isna() & ~df_iso[target_col2].isna()]

    # Calculate your Xe selectivity
    df_iso['Xe_selectivity'] = (df_iso['Xe_cm3_per_cm3_value']/0.2) / (df_iso['Kr_cm3_per_cm3_value']/0.8)
    df_iso.loc[(df_iso['Xe_cm3_per_cm3_value'] == 0) & (df_iso['Kr_cm3_per_cm3_value'] == 0), 
            'Xe_selectivity'] = 0



    df_property = pd.read_csv(property_csv)
    df_iso = df_iso.merge(df_property[["project"] + property_cols], on='project', how='left')

    if ref_data_split and os.path.exists(ref_data_split):
        df_iso_all3 = df_iso.copy()
        for grid_type in grid_types:
            df_iso_all3[grid_type] = True 
    else:
        # 1) For each grid_type, check which MOFs have valid grid files
        for grid_type in grid_types:
            df_iso[grid_type] = False  # add a boolean column, default False
            grid_type_str = grid_type if grid_type != 'center' else ''
            energy_file = f"{grid_type_str}_energy_grid.txt".lstrip('_')

            abnormal_files = []
            empty_files = []
            absent_files = []
            normal_files = []

            # Loop over each MOF folder in base_dir
            for folder in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder)
                if os.path.isdir(folder_path):
                    asci_grids_path = os.path.join(folder_path, subdir)
                    if os.path.exists(asci_grids_path):
                        file_path = os.path.join(asci_grids_path, energy_file)
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, 'r') as f:
                                    lines = f.readlines()
                                    if len(lines) != 68921:
                                        if len(lines) == 0:
                                            empty_files.append(folder)
                                        abnormal_files.append(folder)
                                    else:
                                        normal_files.append(folder)
                                        # Mark this MOF as True for this grid_type
                                        df_iso.loc[df_iso['project'] == folder, grid_type] = True
                            except Exception as e:
                                print(f"Error reading {energy_file} in folder {folder}: {e}")
                        else:
                            absent_files.append(folder)
                            print(f"{energy_file} does not exist in folder: {folder}")
                    else:
                        print(f"ASCI_Grids subdirectory does not exist in folder: {folder}")

            print(f"\n=== Summary for {grid_type} ===")
            print(f"Absent {grid_type} energy_grid.txt files: {len(absent_files)}")
            print(f"Empty {grid_type} energy_grid.txt files: {len(empty_files)}")
            print(f"Abnormal {grid_type} energy_grid.txt files: {len(abnormal_files)}")
            print(f"Normal {grid_type} energy_grid.txt files: {len(normal_files)}")

        # 2) Filter to MOFs that have *all three* grids = True
        df_iso_all3 = df_iso[df_iso['center'] & df_iso['rotated'] & df_iso['translated']]

    # We'll do the "melt" only on df_iso_all3
    old_cols = df_iso_all3.columns.tolist()
    # remove the grid_types from old_cols, otherwise they'd also appear in the melt "id_vars"
    # but you can keep them if you prefer, as it won't break anything. Typically we'd do:
    cols_no_grids = [c for c in old_cols if c not in grid_types]

    df_melted = df_iso_all3.melt(
        id_vars=cols_no_grids,
        value_vars=grid_types,
        var_name='grid_type',
        value_name='has_grid'
    )

    # 3) Keep only rows where has_grid==True
    df_filtered = df_melted[df_melted['has_grid']]
    df_filtered['sample'] = df_filtered['project'] + "--" + df_filtered['grid_type']
    df_iso_final = df_filtered.drop(columns=['has_grid'])

    # 4) Save to CSV (renamed to "all_3grids.csv" to reflect the filtering)
    df_iso_final.to_csv(f'{output_dir}/all_{len(grid_types)}grids.csv', index=False)
    print(f"Filtered to MOFs with all {len(grid_types)} grids. Final shape: {df_iso_final.shape}")
    if csv_only:
        return None

    # 5) Prepare for Splitting
    # filterd_mofs are now each row's "project". Note that multiple rows
    # can have the same "project" if they differ in grid_type.

    filterd_mofs = df_iso_final['project'].tolist()


    grid = 41
    pos_cap = 0
    subset_data = {}
    if ref_data_split and os.path.exists(ref_data_split):
        total_size = 0
        samples = []
        for subset in ['train', 'test', 'val']:
            dir_path = os.path.join(ref_data_split, subset)
            with open(f'{dir_path}/clean.json', 'r') as f:
                subset_data[subset] = json.load(f)
                total_size += subset_data[subset]['size']
                samples.extend(subset_data[subset]['name'])
            # Write JSON
            dir_path = os.path.join(output_dir, subset)
            with open(f'{dir_path}/clean.json', 'w') as f:
                json.dump(subset_data[subset], f)
            print(f"{subset}: {subset_data[subset]['size']} samples")
            print(f"saved {subset} data to {dir_path}/clean.json")
        df_iso_final = df_iso_final[df_iso_final['sample'].isin(samples)]
        assert total_size == df_iso_final.shape[0]
    else:
        print("\nNo reference dataset split found. Splitting data randomly.")
        test_size = 0.1
        val_size = 0.1

        # If you want to keep all grid types of each MOF in the same split:
        # Step 1: get unique MOFs
        unique_mofs = list(set(filterd_mofs))

        # Step 2: train_test_split at MOF level
        train_mofs, test_mofs = train_test_split(unique_mofs, test_size=test_size, random_state=SEED)
        train_mofs, val_mofs = train_test_split(train_mofs, test_size=val_size/(1 - test_size), random_state=SEED)

        print(f"Training MOFs: {len(train_mofs)}")
        print(f"Validation MOFs: {len(val_mofs)}")
        print(f"Testing MOFs: {len(test_mofs)}")

        datset_mof = {'train': train_mofs, 'test': test_mofs, 'val': val_mofs}

        # 6) Build the JSON splits and the .npy arrays
        # You have multiple grid types, so the "else" block from your code
        # (handling multiple grid types) is relevant.

        for subset in ['train', 'test', 'val']:
            subset_data[subset] = {}
            dir_path = os.path.join(output_dir, subset)

            # # Optionally remove if you want a fresh directory:
            # if os.path.exists(dir_path):
            #     shutil.rmtree(dir_path)

            os.makedirs(dir_path, exist_ok=True)
            subset_data[subset]['name'] = []

            # For each MOF in that split, gather the 3 grid types from df_iso_final
            for mof in datset_mof[subset]:
                # For each grid type, check if it's in df_iso_final
                for gtype in grid_types:
                    # (MOF, gtype) must exist in df_iso_final
                    if ((df_iso_final['project'] == mof) & (df_iso_final['grid_type'] == gtype)).any():
                        subset_data[subset]['name'].append(f"{mof}--{gtype}")

            subset_data[subset]['grid'] = grid
            subset_data[subset]['size'] = len(subset_data[subset]['name'])



    # 7) Create the clean.npy for each subset
    for subset in ['train', 'test', 'val']:
        array_list = []
        dir_path = os.path.join(output_dir, subset)

        for sample_file in subset_data[subset]['name']:
            mof_file, grid_type = sample_file.split('--')
            grid_type_str = grid_type if grid_type != 'center' else ''
            grid_file = f"{grid_type_str}_energy_grid.txt".lstrip('_')

            with open(os.path.join(base_dir, mof_file, "ASCI_Grids", grid_file), "r") as file:
                lines = file.readlines()

            # Convert lines -> floats, handle '?' placeholders, reshape
            data_array = np.array(lines, dtype=object)
            clean_array = np.where(data_array == '?\n', pos_cap, data_array).astype(float)
            cap_array = np.clip(clean_array, None, pos_cap)
            reshaped_array = cap_array.reshape((grid, grid, grid))
            assert np.isnan(reshaped_array).sum() == 0

            array_list.append(reshaped_array)

        dataset_array = np.stack(array_list, axis=0)
        np.save(f"{dir_path}/clean.npy", dataset_array)
        print(f"saved {subset} data of shape {dataset_array.shape} to {dir_path}/clean.npy")
    return None

if __name__ == "__main__":
    main()