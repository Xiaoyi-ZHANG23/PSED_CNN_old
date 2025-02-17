import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error
from torcheval.metrics.functional import r2_score

# -----------------------------------------------------------------
# Import from model_property.py
#   RetNet   -> Must accept 'pld_dim' in its constructor
#   CustomDataset -> Must handle (X, y, pld)
#   init_weights, load_data, LearningMethod
# -----------------------------------------------------------------
from model_property import (
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
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------
# Hyperparameters & Setup
# -----------------------------------------------------------------
hyper_params = {
    'batch_size':   64,
    'num_epochs':   300,
    'eval_freq':    40,
    'learning_rate':1e-4,
    'weight_decay': 1e-4,
    'step_size':    60,
    'gamma':        0.5,
    'optimizer':    'Adam',
    'patience':     100
}

# Example: we want to predict Kr adsorption
target_col  = 'Kr_cm3_per_cm3_value'
index_col   = 'project'

# This directory should have subfolders: train/, val/, test/, and a CSV file 'all.csv'
load_dir    = "/projects/p32082/PSED_CNN_old/split/data_mix_1bar"  # adjust as needed
model_name  = (
    f"My3DCNN_PLD_{target_col}_"
    f"{hyper_params['batch_size']}_"
    f"{hyper_params['learning_rate']}_"
    f"{hyper_params['weight_decay']}_"
    f"{hyper_params['step_size']}_"
    f"{hyper_params['gamma']}_"
    f"{hyper_params['optimizer']}"
)

model_save_dir = "/projects/p32082/PSED_CNN_old/model"
output_dir     = "/projects/p32082/PSED_CNN_old/output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------
# Logger
# -----------------------------------------------------------------
def setup_logger(log_dir="./log", log_filename=None):
    os.makedirs(log_dir, exist_ok=True)
    if log_filename is None:
        from datetime import datetime
        log_filename = f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
    log_file = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logger()
logger.info(f"Hyperparameters: {hyper_params}")

# -----------------------------------------------------------------
# Device
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# -----------------------------------------------------------------
# 1) Load train, val, test data directly from load_dir
# -----------------------------------------------------------------
# Make sure your load_data can handle (dir_batch, path_to_csv, target_name, index_col)
train_dir = os.path.join(load_dir, "train")
val_dir   = os.path.join(load_dir, "val")
test_dir  = os.path.join(load_dir, "test")
csv_path  = os.path.join(load_dir, "all.csv")

X_train, y_train = load_data(train_dir, csv_path, target_col, index_col)
X_val,   y_val   = load_data(val_dir,   csv_path, target_col, index_col)
X_test,  y_test  = load_data(test_dir,  csv_path, target_col, index_col)

logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logger.info(f"X_val shape:   {X_val.shape},   y_val shape:   {y_val.shape}")
logger.info(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")

# -----------------------------------------------------------------
# 2) Load PLD from the same CSV for each set
#    We'll use clean.json from each subfolder to determine sample order.
# -----------------------------------------------------------------
df_all = pd.read_csv(csv_path, index_col=index_col)

def get_pld_list(dir_path):
    """Reads clean.json in dir_path, returns a float array of PLD from df_all."""
    json_path = os.path.join(dir_path, "clean.json")
    with open(json_path, 'r') as f:
        info = json.load(f)
    names = info['name']  # list of MOF names
    return df_all.loc[names, "PLD"].values.astype(np.float32)

pld_train = get_pld_list(train_dir)
pld_val   = get_pld_list(val_dir)
pld_test  = get_pld_list(test_dir)

logger.info(f"pld_train shape: {pld_train.shape}")
logger.info(f"pld_val   shape: {pld_val.shape}")
logger.info(f"pld_test  shape: {pld_test.shape}")

# -----------------------------------------------------------------
# 3) Basic checks (NaN, etc.)
# -----------------------------------------------------------------
assert not np.isnan(X_train).any(), "NaNs in X_train!"
assert not np.isnan(y_train).any(), "NaNs in y_train!"
assert not np.isnan(X_val).any(),   "NaNs in X_val!"
assert not np.isnan(y_val).any(),   "NaNs in y_val!"
assert not np.isnan(X_test).any(),  "NaNs in X_test!"
assert not np.isnan(y_test).any(),  "NaNs in y_test!"

# -----------------------------------------------------------------
# 4) Reshape for 3D CNN: (N, 41, 41, 41) -> (N, 1, 41, 41, 41)
#    Adjust if your data size is different
# -----------------------------------------------------------------
X_train = X_train.reshape(-1, 1, X_train.shape[1], X_train.shape[2], X_train.shape[3])
X_val   = X_val.reshape(-1, 1, X_val.shape[1],   X_val.shape[2],   X_val.shape[3])
X_test  = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2], X_test.shape[3])

# -----------------------------------------------------------------
# 5) Separate Normalization (like in the first code)
#    If you prefer a single normalization (train stats only),
#    you can adapt accordingly.
# -----------------------------------------------------------------
standardization_train = transforms.Normalize(mean=X_train.mean(), std=X_train.std())
standardization_val   = transforms.Normalize(mean=X_val.mean(),   std=X_val.std())
standardization_test  = transforms.Normalize(mean=X_test.mean(),  std=X_test.std())

# -----------------------------------------------------------------
# 6) Build Datasets & Dataloaders
# -----------------------------------------------------------------
train_dataset = CustomDataset(
    X=X_train,
    y=y_train,
    pld=pld_train,
    transform_X=standardization_train  # use train stats
)
val_dataset = CustomDataset(
    X=X_val,
    y=y_val,
    pld=pld_val,
    transform_X=standardization_val
)
test_dataset = CustomDataset(
    X=X_test,
    y=y_test,
    pld=pld_test,
    transform_X=standardization_test
)

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
# 7) Model, Loss, Optimizer, Scheduler
#    We'll pass 'pld_dim=1' since we have 1 PLD feature.
# -----------------------------------------------------------------
net = RetNet(pld_dim=1).to(device)
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
# 8) Weight Initialization
# -----------------------------------------------------------------
net.apply(lambda m: init_weights(m, a=0.01))
with torch.no_grad():
    # Initialize final bias as mean of y_train
    net.fc2[-1].bias.fill_(y_train.mean())
logger.info(f"Initialized final bias to: {y_train.mean():.4f}")

# -----------------------------------------------------------------
# 9) Wrap in LearningMethod & Train
# -----------------------------------------------------------------
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
# 10) Evaluate on Test
# -----------------------------------------------------------------
model.net.eval()
predictions = []
true_values = []

for xb, yb, pld_b in test_loader:
    xb, pld_b = xb.to(device), pld_b.to(device)
    with torch.no_grad():
        y_pred_b = model.predict(xb, pld_b)
    predictions.append(y_pred_b.cpu())
    true_values.append(yb)

y_pred = torch.cat(predictions).numpy().ravel()
y_true = torch.cat(true_values).numpy().ravel()

# -----------------------------------------------------------------
# 11) Metrics
# -----------------------------------------------------------------
r2  = r2_score(torch.tensor(y_pred), torch.tensor(y_true)).item()
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

logger.info(f"Test R^2:  {r2:.6f}")
logger.info(f"Test MSE:  {mse:.6f}")
logger.info(f"Test MAE:  {mae:.6f}")

# -----------------------------------------------------------------
# 12) Save Model
# -----------------------------------------------------------------
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, f"{model_name}_property_state_dict.pt")

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
# 13) Save Predictions (CSV)
# -----------------------------------------------------------------
predictions_save_path = os.path.join(output_dir, f"predictions_{model_name}.csv")
df_predictions = pd.DataFrame({
    "true_value": y_true,
    "predicted_value": y_pred
})
df_predictions.to_csv(predictions_save_path, index=False)
logger.info(f"Predictions saved to {predictions_save_path}")

# -----------------------------------------------------------------
# 14) Parity Plot
# -----------------------------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Parity Plot")
plt.grid(True)

parity_plot_path = os.path.join(output_dir, f"parity_{model_name}_PLD.png")
plt.savefig(parity_plot_path)
logger.info(f"Saved parity plot to {parity_plot_path}")
plt.close()
