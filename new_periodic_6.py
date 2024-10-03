import os
import pandas as pd
import numpy as np
from scipy.ndimage import rotate

# Paths to the CSV files
train_csv = 'train_new_periodic.csv'
test_csv = 'test_new_periodic.csv'

# Paths to the source and destination directories
source_dir = '/projects/b1013/from_kunhuan_to_xiaoyi/all_npy'
train_dest_dir = '/projects/b1013/from_kunhuan_to_xiaoyi/train'
test_dest_dir = '/projects/b1013/from_kunhuan_to_xiaoyi/test'

# Create destination directories if they do not exist
os.makedirs(train_dest_dir, exist_ok=True)
os.makedirs(test_dest_dir, exist_ok=True)

# Function to set all values greater than 0 to 0
def clamp_values(grid):
    return np.where(grid > 0, 0, grid)

# Step 1: Adaptive Tiling and Cropping
def adaptive_tile_and_crop(grid, target_crop_size=(80, 80, 80)):
    tile_factors = []
    for target_dim, dim_size in zip(target_crop_size, grid.shape):
        min_tiles = (target_dim + dim_size - 1) // dim_size
        if min_tiles % 2 == 0:
            min_tiles += 1
        tile_factors.append(max(3, min_tiles))
    tiled_grid = np.tile(grid, tile_factors)
    start_indices = [(tiled_grid.shape[i] - target_crop_size[i]) // 2 for i in range(3)]
    cropped_grid = tiled_grid[
        start_indices[0]:start_indices[0] + target_crop_size[0],
        start_indices[1]:start_indices[1] + target_crop_size[1],
        start_indices[2]:start_indices[2] + target_crop_size[2]
    ]
    return cropped_grid

# Step 2: Apply Translation and Rotation
def apply_translation_and_rotation(grid):
    # Calculate maximum translation based on half the grid size in each dimension
    max_translation = [dim_size // 2 for dim_size in grid.shape]
    
    # Generate random translation values within the allowed range
    tx = np.random.uniform(-max_translation[0], max_translation[0])
    ty = np.random.uniform(-max_translation[1], max_translation[1])
    tz = np.random.uniform(-max_translation[2], max_translation[2])
    
    # Apply the translation by rolling the grid
    translated_grid = np.roll(grid, shift=(int(tx), int(ty), int(tz)), axis=(0, 1, 2))
    angle_range = 180
    # Choose a random angle from 0, 90, 180, 270 degrees
    angle_x = np.random.uniform(-angle_range, angle_range)
    angle_y = np.random.uniform(-angle_range, angle_range)
    angle_z = np.random.uniform(-angle_range, angle_range)
    
    # Apply the rotations
    rotated_grid = rotate(translated_grid, angle_x, axes=(1, 2), reshape=False)
    rotated_grid = rotate(rotated_grid, angle_y, axes=(0, 2), reshape=False)
    rotated_grid = rotate(rotated_grid, angle_z, axes=(0, 1), reshape=False)

    # Ensure the rotated grid is cropped to the original grid size
    min_shape = np.minimum(grid.shape, rotated_grid.shape)
    final_rotated_grid = rotated_grid[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    return final_rotated_grid

# Step 3: Pick a 40x40x40 Grid from the Center
def pick_center_grid(grid, target_size=(40, 40, 40)):
    x_start = (grid.shape[0] - target_size[0]) // 2
    y_start = (grid.shape[1] - target_size[1]) // 2
    z_start = (grid.shape[2] - target_size[2]) // 2
    center_grid = grid[x_start:x_start+target_size[0], 
                       y_start:y_start+target_size[1], 
                       z_start:z_start+target_size[2]]
    return center_grid


# Function to filter and process files based on dimensions and transformations
def process_train_files(csv_file, dest_dir, output_csv):
    data = pd.read_csv(csv_file)
    project_names = data['project']
    valid_npy_arrays = []
    updated_data = pd.DataFrame(columns=data.columns)
    
    for project in project_names:
        matching_files = [f for f in os.listdir(source_dir) if f.startswith(project) and f.endswith('.npy')]
        
        for file in matching_files:
            npy_path = os.path.join(source_dir, file)
            npy_array = np.load(npy_path)
            
            if npy_array.shape[0] > 40 or npy_array.shape[1] > 40 or npy_array.shape[2] > 40:
                print(f"Disregarding {file} due to large dimensions: {npy_array.shape}")
                data = data[data['project'] != project]
            else:
                # Set all values greater than 0 to 0
                npy_array = clamp_values(npy_array)

                # Apply the grid processing steps
                cropped_grid = adaptive_tile_and_crop(npy_array)                
                # First version: with translation and rotation
                transformed_grid = apply_translation_and_rotation(cropped_grid)
                final_grid_transformed = pick_center_grid(transformed_grid)
                
                # Second version: without translation and rotation
                final_grid_untransformed = pick_center_grid(cropped_grid)
                
                # Stack the two grids along a new axis to keep them together
                combined_grid = np.stack((final_grid_transformed, final_grid_untransformed), axis=0)
                
                # Save the processed grids to the destination directory
                np.save(os.path.join(dest_dir, file), combined_grid)
                valid_npy_arrays.append(combined_grid)

                # Modify the 'project' column for non-transformed data
                matching_row = data[data['project'] == project]
                matching_row_non_transformed = matching_row.copy()
                matching_row_non_transformed['project'] += "-2"
                
                # Add the modified rows to the updated CSV DataFrame
                updated_data = pd.concat([updated_data, matching_row, matching_row_non_transformed], ignore_index=True)
                print(f"Processed {file} with dimensions: {npy_array.shape}")
    
    # Save the updated CSV with each project duplicated and non-transformed rows modified
    updated_data.to_csv(output_csv, index=False)
    combined_npy = np.vstack(valid_npy_arrays)
    return combined_npy


# Function to process test files without translation and rotation
def process_test_files(csv_file, dest_dir, output_csv):
    data = pd.read_csv(csv_file)
    project_names = data['project']
    valid_npy_arrays = []
    updated_data = pd.DataFrame(columns=data.columns)
    
    for project in project_names:
        matching_files = [f for f in os.listdir(source_dir) if f.startswith(project) and f.endswith('.npy')]
        
        for file in matching_files:
            npy_path = os.path.join(source_dir, file)
            npy_array = np.load(npy_path)
            
            if npy_array.shape[0] > 40 or npy_array.shape[1] > 40 or npy_array.shape[2] > 40:
                print(f"Disregarding {file} due to large dimensions: {npy_array.shape}")
                data = data[data['project'] != project]
            else:
                # Set all values greater than 0 to 0
                npy_array = clamp_values(npy_array)

                # Apply the grid processing steps
                cropped_grid = adaptive_tile_and_crop(npy_array)                                
                # Second version: without translation and rotation
                final_grid_untransformed = pick_center_grid(cropped_grid)
                
                # Add a new axis to the grid to maintain shape consistency
                combined_grid = np.expand_dims(final_grid_untransformed, axis=0)
                
                # Save the processed grids to the destination directory
                np.save(os.path.join(dest_dir, file), combined_grid)
                valid_npy_arrays.append(combined_grid)

                # Modify the 'project' column for non-transformed data
                matching_row = data[data['project'] == project]
                matching_row_non_transformed = matching_row.copy()
                matching_row_non_transformed['project'] += "-2"
                
                # Add the modified rows to the updated CSV DataFrame
                updated_data = pd.concat([updated_data, matching_row_non_transformed], ignore_index=True)
    
    updated_data.to_csv(output_csv, index=False)
    combined_npy = np.vstack(valid_npy_arrays)
    return combined_npy


# Process and combine train and test files
train_npy = process_train_files(train_csv, train_dest_dir, 'train_done_periodic6.csv')
np.save('train_done_periodic6.npy', train_npy)

test_npy = process_test_files(test_csv, test_dest_dir, 'test_done_periodic6.csv')
np.save('test_done_periodic6.npy', test_npy)

print("Files processed and saved to train_done_periodic.npy, test_done_periodic.npy, train_done_periodic.csv, and test_done_periodic.csv")
