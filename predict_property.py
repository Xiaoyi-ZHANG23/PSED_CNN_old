import os
import json
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

# 1) Import your custom classes from model_property.py
from model_property import (
    RetNet,
    CustomDataset,
    load_data  # or load_data_id if you need IDs
)

##############################################
# A) Define your paths and parameters
##############################################
MODEL_CHECKPOINT_PATH = "/projects/p32082/PSED_CNN_old/all_save/Mixall_pressures_1_grids_all_Kr_cm3_per_cm3_value_no_64_2025-02-27_15-43_64_0.0001_0.0001_60_0.5_Adam__2_pressure_3split_84_property_state_dict.pt"  
# ^ Adjust this to the .pt file you saved after training

LOAD_DIR    = "/scratch/ekn8665/data_temp"     # Directory with train/, val/, test/
CSV_PATH    = "/scratch/ekn8665/data_temp/all_combined.csv"
TEST_DIR    = os.path.join(LOAD_DIR, "test")
OUTPUT_DIR  = "/projects/p32082/PSED_CNN_old/all_save"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL  = "Kr_cm3_per_cm3_value"  # The column you predict
INDEX_COL   = "sample"                # The index in the CSV
BATCH_SIZE  = 64                      # Same or any batch size you prefer
PROPERTY = "pressure"

###################################################
# B) Utility: get sample names from clean_modified.json
###################################################
def get_sample_names(dir_path):
    """
    Reads clean_modified.json in dir_path and returns
    the list of sample names in the same order as X_test.
    """
    json_path = os.path.join(dir_path, "clean_modified.json")
    with open(json_path, 'r') as f:
        info = json.load(f)
    return info["name"]


def main():
    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) Load test data
    X_test, y_test = load_data(TEST_DIR, CSV_PATH, TARGET_COL, INDEX_COL)

    # 3) Load sample names
    test_names = get_sample_names(TEST_DIR)
    assert len(test_names) == len(X_test), (
        f"Mismatch: {len(test_names)} names vs {len(X_test)} test samples!"
    )

    # 4) Reshape for your 3D CNN if needed
    X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2], X_test.shape[3])

    # 5) Load the entire CSV so we can grab other columns (like 'pressure')
    df_all = pd.read_csv(CSV_PATH, index_col=INDEX_COL)

    # Let's define a helper to fetch the pld (or pressure) for test samples
    def get_pld_list(dir_path):
        json_path = os.path.join(dir_path, "clean_modified.json")
        with open(json_path, "r") as f:
            info = json.load(f)
        names = info["name"]
        return df_all.loc[names, PROPERTY].values.astype(np.float32)

    pld_test = get_pld_list(TEST_DIR)

    # 6) You may also want to store the pressure in the final CSV
    #    We'll grab the same "pressure" values but as float or string
    property_test = df_all.loc[test_names, PROPERTY].values  # shape: (N,)

    # 7) Normalization
    #    In your training code, you used separate stats for each dataset.
    test_mean = X_test.mean()
    test_std  = X_test.std()

    transform_test = transforms.Normalize(mean=test_mean, std=test_std)

    # 8) Create test dataset
    test_dataset = CustomDataset(
        X=X_test,
        y=y_test,
        pld=pld_test,
        transform_X=transform_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    # 9) Define model and load checkpoint
    model = RetNet(pld_dim=1).to(device)
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 10) Inference loop
    all_preds = []
    all_truth = []
    for Xb, yb, pldb in test_loader:
        Xb, pldb = Xb.to(device), pldb.to(device)
        with torch.no_grad():
            preds = model(Xb, pldb)  # or model.forward(...) or model.predict(...)
        all_preds.append(preds.cpu().numpy())
        all_truth.append(yb.numpy())

    y_pred = np.concatenate(all_preds, axis=0).ravel()
    y_true = np.concatenate(all_truth, axis=0).ravel()

    # 11) Build the detailed CSV
    #     We add an extra column for 'pressure'
    df_pred = pd.DataFrame({
        "sample_name":     test_names,
        "variable":        TARGET_COL,
        PROPERTY:        property_test,  # the extra column
        "true_value":      y_true,
        "predicted_value": y_pred
    })

    # 12) Save to CSV
    output_csv_path = os.path.join(OUTPUT_DIR, "predictions_Kr_all_pressures.csv")
    df_pred.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to: {output_csv_path}")


if __name__ == "__main__":
    main()

