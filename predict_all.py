import os
import json
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

# Import the same modules/classes from your model_property_all.py
# (Adjust the import name if your file is called something else.)
from model_property_all import (
    RetNet,
    CustomDataset,
    load_data
)


################################################################################
# 1) Define Paths & Hyperparameters
################################################################################

# Path to your saved checkpoint (the .pt file created in training.py)
MODEL_CHECKPOINT_PATH = "/projects/p32082/PSED_CNN_old/model/Mix1bar_1_grids_all_Xe_cm3_per_cm3_value_..._3features_state_dict.pt"
# ^ Adjust to the actual file name you saved during training.

# Paths for your test data
LOAD_DIR   = "/projects/p32082/PSED_CNN_old/split/data_mix_1bar_3_grids"  # same as training
CSV_PATH   = os.path.join(LOAD_DIR, "all.csv")
TEST_DIR   = os.path.join(LOAD_DIR, "test")

OUTPUT_DIR = "/projects/p32082/PSED_CNN_old/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The target column and index column used in training
TARGET_COL = "Xe_cm3_per_cm3_value"
INDEX_COL  = "sample"

# Batch size for inference
BATCH_SIZE = 64  # or any other convenient size

################################################################################
# 2) Utility Function: Get sample names from `clean.json`
################################################################################
def get_sample_names(dir_path):
    """
    Reads `clean.json` in dir_path and returns the list of sample names
    in the same order as the data loaded by load_data.
    """
    json_path = os.path.join(dir_path, "clean.json")
    with open(json_path, 'r') as f:
        info = json.load(f)
    return info["name"]


################################################################################
# 3) Main Inference Procedure
################################################################################
def main():
    # --------------------------------------------------------------------------
    # A) Select device
    # --------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # B) Load test data (X_test, y_test)
    # --------------------------------------------------------------------------
    X_test, y_test = load_data(TEST_DIR, CSV_PATH, TARGET_COL, INDEX_COL)
    print(f"[INFO] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # --------------------------------------------------------------------------
    # C) Get sample names to align predictions with test data
    # --------------------------------------------------------------------------
    test_names = get_sample_names(TEST_DIR)
    assert len(test_names) == len(X_test), (
        f"Mismatch: {len(test_names)} names vs {len(X_test)} samples!"
    )

    # --------------------------------------------------------------------------
    # D) Load the extra features (PLD, LCD, density_g_cm3)
    #    from the same CSV, in the same order
    # --------------------------------------------------------------------------
    # We'll replicate the logic from training, but just for test data
    df_all = pd.read_csv(CSV_PATH, index_col=INDEX_COL)

    def get_extras_list(dir_path):
        json_path = os.path.join(dir_path, "clean.json")
        with open(json_path, 'r') as f:
            info = json.load(f)
        names = info['name']
        # Adjust these column names if different in your CSV
        return df_all.loc[names, ["PLD", "LCD", "density_g_cm3"]].values.astype(np.float32)

    extras_test = get_extras_list(TEST_DIR)
    assert len(extras_test) == len(X_test), (
        f"Mismatch in extras_test vs X_test: {extras_test.shape[0]} vs {len(X_test)}"
    )
    print(f"[INFO] extras_test shape: {extras_test.shape} (should be Nx3)")

    # --------------------------------------------------------------------------
    # E) Reshape X_test for 3D CNN and define normalization
    # --------------------------------------------------------------------------
    X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2], X_test.shape[3])

    # If in training you used separate stats for test data, replicate that here.
    # Otherwise, you can load training stats and apply them. 
    # For simplicity, let's compute from X_test:
    test_mean = X_test.mean()
    test_std  = X_test.std()

    transform_test = transforms.Normalize(mean=test_mean, std=test_std)

    # --------------------------------------------------------------------------
    # F) Create test dataset & dataloader
    # --------------------------------------------------------------------------
    test_dataset = CustomDataset(
        X=X_test,
        y=y_test,
        extras=extras_test,
        transform_X=transform_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    # --------------------------------------------------------------------------
    # G) Define model architecture and load checkpoint
    #    Make sure extra_dim matches the number of extra features.
    # --------------------------------------------------------------------------
    model = RetNet(extra_dim=3).to(device)
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[INFO] Model loaded successfully.")

    # --------------------------------------------------------------------------
    # H) Run inference on test set
    # --------------------------------------------------------------------------
    all_preds = []
    all_truth = []
    for Xb, yb, extras_b in test_loader:
        Xb, extras_b = Xb.to(device), extras_b.to(device)
        with torch.no_grad():
            # If your code used model.predict(...) in training, do so here.
            # Otherwise, just do model(Xb, extras_b).
            y_pred_b = model.forward(Xb, extras_b)
        all_preds.append(y_pred_b.cpu().numpy())
        all_truth.append(yb.numpy())

    y_pred = np.concatenate(all_preds).ravel()
    y_true = np.concatenate(all_truth).ravel()

    # --------------------------------------------------------------------------
    # I) Build a DataFrame with all the columns you want
    # --------------------------------------------------------------------------
    # extras_test has shape (N, 3), so:
    #   extras_test[:, 0] = PLD
    #   extras_test[:, 1] = LCD
    #   extras_test[:, 2] = density_g_cm3
    df_pred = pd.DataFrame({
        "sample_name":     test_names,
        "variable":        TARGET_COL,
        "PLD":             extras_test[:, 0],
        "LCD":             extras_test[:, 1],
        "density_g_cm3":   extras_test[:, 2],
        "true_value":      y_true,
        "predicted_value": y_pred
    })

    # --------------------------------------------------------------------------
    # J) Save to CSV
    # --------------------------------------------------------------------------
    output_path = os.path.join(OUTPUT_DIR, "predictions_detailed_extras.csv")
    df_pred.to_csv(output_path, index=False)
    print(f"[INFO] Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
