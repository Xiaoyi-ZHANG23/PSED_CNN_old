import os
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms

from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error
from torcheval.metrics.functional import r2_score

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# -----------------------------------------------------------------
# Import from model_property.py
#    RetNet         -> Should now accept 'extra_dim' in constructor
#    LearningMethod -> Expects (X, y, extras) in training loop
#    CustomDataset  -> Takes X, y, extras
#    init_weights, load_data
# -----------------------------------------------------------------
from model_property_all import (
    RetNet,
    LearningMethod,
    CustomDataset,
    init_weights,
    load_data
)

# -----------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------
SEED = 1
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------
# Hyperparameters & Setup
# -----------------------------------------------------------------
hyper_params = {
    'batch_size':   64,
    'num_epochs':   1000,
    'eval_freq':    40,
    'learning_rate':1e-4,
    'weight_decay': 1e-4,
    'step_size':    60,
    'gamma':        0.5,
    'optimizer':    'Adam',
    'patience':     100
}

# Example: we want to predict Xe adsorption
# target_col  = 'Xe_cm3_per_cm3_value'
target_col  = 'Kr_cm3_per_cm3_value'
index_col   = 'sample'
pressure = '0.1bar'
pressure_map = {'0.1bar': '0p1bar', '1bar': '1bar', '10bar': '10bar', '0.25bar': '0p25bar', '0.5bar': '0p5bar'}
pressure_str = pressure_map[pressure]
# Directory structure
csv_dir = f"/data/yll6162/mof_cnn/data_mix_{pressure_str}_3_grids"  # direcotry for target value and extra features
grid_dir = "/data/yll6162/mof_cnn/data_mix_1bar_3_grids" # directory for grid data and data splits, which is not necessary the same as csv_dir
model_name  = (
    f"My3DCNN_extras_{target_col}_"
    f"{hyper_params['batch_size']}_"
    f"{hyper_params['learning_rate']}_"
    f"{hyper_params['weight_decay']}_"
    f"{hyper_params['step_size']}_"
    f"{hyper_params['gamma']}_"
    f"{hyper_params['optimizer']}"
)

model_save_dir = "/data/yll6162/mof_cnn/model"
output_dir     = "pred"
os.makedirs(output_dir, exist_ok=True)
num_grids = 3
grid_type = 'all'
property_cols = ["PLD", "LCD"]
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
model_name = f'Mix{pressure_str}_{num_grids}_grids_{grid_type}_{target_col}_no_64_{timestamp}'
model_name = f"{model_name}_{hyper_params['batch_size']}_{hyper_params['learning_rate']}_{hyper_params['weight_decay']}_{hyper_params['step_size']}_{hyper_params['gamma']}_{hyper_params['optimizer']}"

def setup_logger(log_dir="./log", log_filename=None):
    os.makedirs(log_dir, exist_ok=True)
    if log_filename is None:
        log_filename = f"{model_name}_{timestamp}.log"
    log_file = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger_ = logging.getLogger(__name__)
    return logger_

logger = setup_logger()
logger.info(f"Hyperparameters: {hyper_params}")

# -----------------------------------------------------------------
# Device
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# -----------------------------------------------------------------
# 1) Load train, val, test data (X, y) from load_dir
# -----------------------------------------------------------------
train_dir = os.path.join(grid_dir, "train")
val_dir   = os.path.join(grid_dir, "val")
test_dir  = os.path.join(grid_dir, "test")
csv_path  = os.path.join(csv_dir, f"all_{num_grids}grids.csv")

X_train, y_train = load_data(train_dir, csv_path, target_col, index_col)
X_val,   y_val   = load_data(val_dir,   csv_path, target_col, index_col)
X_test,  y_test  = load_data(test_dir,  csv_path, target_col, index_col)

logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logger.info(f"X_val shape:   {X_val.shape},   y_val shape:   {y_val.shape}")
logger.info(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")

# -----------------------------------------------------------------
# 2) Load PLD, LCD, density (or any extra features) from CSV 
#    for train, val, test sets
# -----------------------------------------------------------------
df_all = pd.read_csv(csv_path, index_col=index_col)

def get_extras_list(dir_path):
    """
    Reads 'clean.json' in dir_path, returns a float array of extras 
    from df_all. For example, [PLD, LCD, density].
    """
    json_path = os.path.join(dir_path, "clean.json")
    with open(json_path, 'r') as f:
        info = json.load(f)
    names = info['name']  # list of sample names

    # Example: read 3 columns from df_all
    # Adjust column names to match your CSV
    return df_all.loc[names, ["PLD", "LCD"]].values.astype(np.float32)

extras_train = get_extras_list(train_dir)  # shape (N_train, 3)
extras_val   = get_extras_list(val_dir)    # shape (N_val, 3)
extras_test  = get_extras_list(test_dir)   # shape (N_test, 3)

logger.info(f"extras_train shape: {extras_train.shape}")
logger.info(f"extras_val   shape: {extras_val.shape}")
logger.info(f"extras_test  shape: {extras_test.shape}")

# Basic check: no NaNs
assert not np.isnan(extras_train).any(), "NaNs in extras_train!"
assert not np.isnan(extras_val).any(),   "NaNs in extras_val!"
assert not np.isnan(extras_test).any(),  "NaNs in extras_test!"

# -----------------------------------------------------------------
# 3) Check that X has correct shape for 3D CNN, e.g. (N, D, H, W).
#    Then reshape to (N, 1, D, H, W).
# -----------------------------------------------------------------
X_train = X_train.reshape(-1, 1, X_train.shape[1], X_train.shape[2], X_train.shape[3])
X_val   = X_val.reshape(-1, 1, X_val.shape[1],   X_val.shape[2],   X_val.shape[3])
X_test  = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2],  X_test.shape[3])

# -----------------------------------------------------------------
# 4) Normalization transforms (optional)
# -----------------------------------------------------------------
standardization_train = transforms.Normalize(mean=X_train.mean(), std=X_train.std())
standardization_val   = transforms.Normalize(mean=X_val.mean(),   std=X_val.std())
standardization_test  = transforms.Normalize(mean=X_test.mean(),  std=X_test.std())

# -----------------------------------------------------------------
# 5) Build Datasets & Dataloaders
# -----------------------------------------------------------------

train_dataset = CustomDataset(
    X=X_train,
    y=y_train,
    extras=extras_train,
    transform_X=standardization_train
)
val_dataset = CustomDataset(
    X=X_val,
    y=y_val,
    extras=extras_val,
    transform_X=standardization_val
)
test_dataset = CustomDataset(
    X=X_test,
    y=y_test,
    extras=extras_test,
    transform_X=standardization_test
)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=hyper_params['batch_size'],
    shuffle=True,
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=2 * hyper_params['batch_size'],
    shuffle=False,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=2 * hyper_params['batch_size'],
    shuffle=False,
    pin_memory=True
)

# -----------------------------------------------------------------
# 6) Instantiate RetNet with extra_dim=3
# -----------------------------------------------------------------
net = RetNet(extra_dim=len(property_cols)).to(device)
criterion = nn.L1Loss().to(device)

optimizer = optim.Adam(
    net.parameters(),
    lr=hyper_params['learning_rate'],
    weight_decay=hyper_params['weight_decay']
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=hyper_params['step_size'],
    gamma=hyper_params['gamma'],
    verbose=True
)

# -----------------------------------------------------------------
# 7) Weight Initialization
# -----------------------------------------------------------------
# from model_property import init_weights
net.apply(lambda m: init_weights(m, a=0.01))

# Optionally initialize final bias
with torch.no_grad():
    net.fc2[-1].bias.fill_(y_train.mean())
logger.info(f"Initialized final bias to: {y_train.mean():.4f}")

# -----------------------------------------------------------------
# 8) Wrap in LearningMethod & Train
# -----------------------------------------------------------------
# from model_property import LearningMethod
model = LearningMethod(
    network=net,
    optimizer=optimizer,
    criterion=criterion,
    logger=logger
)

model.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_loss_freq=hyper_params['eval_freq'],
    epochs=hyper_params['num_epochs'],
    scheduler=scheduler,
    metric=r2_score,
    device=device,
    verbose=True,
    patience=hyper_params['patience']
)

# -----------------------------------------------------------------
# 9) Evaluate on Test
# -----------------------------------------------------------------
model.net.eval()
predictions = []
true_values = []

for xb, yb, extras_b in test_loader:
    xb, extras_b = xb.to(device), extras_b.to(device)
    with torch.no_grad():
        y_pred_b = model.predict(xb, extras_b)
    predictions.append(y_pred_b.cpu())
    true_values.append(yb)

y_pred = torch.cat(predictions).numpy().ravel()
y_true = torch.cat(true_values).numpy().ravel()

# -----------------------------------------------------------------
# 10) Metrics
# -----------------------------------------------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error
r2  = r2_score(torch.tensor(y_pred), torch.tensor(y_true)).item()
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

logger.info(f"Test R^2:  {r2:.6f}")
logger.info(f"Test MSE:  {mse:.6f}")
logger.info(f"Test MAE:  {mae:.6f}")

# -----------------------------------------------------------------
# 11) Save Model
# -----------------------------------------------------------------
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(
    model_save_dir, 
    f"{model_name}__3features_state_dict.pt"
)

torch.save(
    {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': hyper_params['num_epochs']
    },
    model_save_path
)
logger.info(f"Model and optimizer state dict saved to {model_save_path}")

# -----------------------------------------------------------------
# 12) Save Predictions (CSV)
# -----------------------------------------------------------------
predictions_save_path = os.path.join(output_dir, f"predictions_{model_name}_3features.csv")
df_predictions = pd.DataFrame({
    "true_value": y_true,
    "predicted_value": y_pred
})
df_predictions.to_csv(predictions_save_path, index=False)
logger.info(f"Predictions saved to {predictions_save_path}")

# -----------------------------------------------------------------
# 13) Parity Plot
# -----------------------------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Parity Plot")
plt.grid(True)
parity_plot_path = os.path.join(output_dir, f'parity_{model_name}_{timestamp}_3features.png')
plt.savefig(parity_plot_path)
logger.info(f"Saved parity plot to {parity_plot_path}")
plt.close()
