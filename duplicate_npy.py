import numpy as np

# File paths
npy_file_path = "/projects/p32082/PSED_CNN_old/split/data_mix_2pressure_3_grids/val/clean.npy"
modified_npy_path = "/projects/p32082/PSED_CNN_old/split/data_mix_2pressure_3_grids/val/clean_modified.npy"

# Load the original .npy file
data = np.load(npy_file_path)  # Assuming shape is (N, 41, 41, 41)

# Duplicate and append
modified_data = np.concatenate([data, data], axis=0)  # Stack them along the first axis

# Save the modified .npy file
np.save(modified_npy_path, modified_data)

print(f"Modified NPY saved as {modified_npy_path}. Original shape: {data.shape}, New shape: {modified_data.shape}")