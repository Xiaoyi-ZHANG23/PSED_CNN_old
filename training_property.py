import os
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torcheval.metrics.functional import r2_score
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------------------------------------------
# Import everything from your model_property.py
# -----------------------------------------------------------------
from model_property import (
    RetNet,
    LearningMethod,
    CustomDataset,
    init_weights,
    load_data  # this loads (X, y) from clean.json/clean.npy + CSV
)

# -----------------------------------------------------------------
# Set random seed
# -----------------------------------------------------------------
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------
# Hyperparameters & Paths
# -----------------------------------------------------------------
hyper_params = {'batch_size':64, 'num_epochs':300, 'eval_freq':40,
                'learning_rate':1e-4, 'weight_decay':1e-4, 'step_size':60, 'gamma':0.5,
                 'optimizer':'Adam', 'patience':100}


# We'll train a 3D CNN for Kr adsorption
target_col = 'Kr_cm3_per_cm3_value'  
index_col  = 'project'
model_name = f"My3DCNN_{target_col}"

base_dir       = "/projects/p32082/PSED_CNN_old/data"
train_dir      = os.path.join(base_dir, "train")
test_dir       = os.path.join(base_dir, "test")
csv_path       = os.path.join(base_dir, "all.csv")
model_save_dir = "/projects/p32082/PSED_CNN_old/model"
plot_save_dir  = "/projects/p32082/PSED_CNN_old/image"
os.makedirs(plot_save_dir, exist_ok=True)

# -----------------------------------------------------------------
# Logger Setup
# -----------------------------------------------------------------
def setup_logger(log_dir="./log", log_filename=None):
    os.makedirs(log_dir, exist_ok=True)
    if log_filename is None:
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
# 1) Load raw (X, y) from training set
# -----------------------------------------------------------------
X_train_all, y_train_all = load_data(
    dir_batch=train_dir,
    path_to_csv=csv_path,
    target_name=target_col,
    index_col=index_col
)
# load_data returns X of shape (N, 41, 41, 41), y of shape (N,)

# -----------------------------------------------------------------
# 2) Load raw (X, y) from test set
# -----------------------------------------------------------------
X_test, y_test = load_data(
    dir_batch=test_dir,
    path_to_csv=csv_path,
    target_name=target_col,
    index_col=index_col
)
# X_test shape (M, 41, 41, 41), y_test shape (M,)

# -----------------------------------------------------------------
# (Optional) If you have an additional PLD column, 
# you can load it similarly from the CSV. 
# But your 'load_data' function doesn't load PLD yet.
# Let's do that manually by reading the same CSV:
# -----------------------------------------------------------------
import pandas as pd
df_all = pd.read_csv(csv_path, index_col=index_col)

# Get the MOF names from train_dir's 'clean.json'
import json
with open(os.path.join(train_dir, 'clean.json'), 'r') as f:
    train_info = json.load(f)
train_names = train_info['name']
with open(os.path.join(test_dir, 'clean.json'), 'r') as f:
    test_info = json.load(f)
test_names = test_info['name']

pld_train_all = df_all.loc[train_names, 'PLD'].values.astype(np.float32)
pld_test      = df_all.loc[test_names,  'PLD'].values.astype(np.float32)

# Now we have X_train_all, y_train_all, pld_train_all
# and X_test, y_test, pld_test

logger.info(f"Train shapes: X={X_train_all.shape}, y={y_train_all.shape}, pld={pld_train_all.shape}")
logger.info(f"Test shapes:  X={X_test.shape},      y={y_test.shape},      pld={pld_test.shape}")

# -----------------------------------------------------------------
# 3) Train-Val Split
# -----------------------------------------------------------------
from sklearn.model_selection import train_test_split

train_size = 0.9
val_size   = 1 - train_size

X_train, X_val, y_train, y_val, pld_train, pld_val = train_test_split(
    X_train_all, y_train_all, pld_train_all,
    test_size=val_size,
    random_state=SEED,
)

logger.info(f"Final TRAIN: {X_train.shape}, {y_train.shape}, {pld_train.shape}")
logger.info(f"VAL:         {X_val.shape}, {y_val.shape}, {pld_val.shape}")

# -----------------------------------------------------------------
# 4) (N, 41, 41, 41) -> (N, 1, 41, 41, 41) for single 3D channel
# -----------------------------------------------------------------
X_train = X_train.reshape(-1, 1, 41, 41, 41)
X_val   = X_val.reshape(-1, 1, 41, 41, 41)
X_test  = X_test.reshape(-1, 1, 41, 41, 41)

# -----------------------------------------------------------------
# 5) Data Transforms (e.g. normalization)
# -----------------------------------------------------------------
train_mean = X_train.mean()
train_std  = X_train.std()
normalize_transform = transforms.Normalize(mean=train_mean, std=train_std)

# -----------------------------------------------------------------
# 6) Build Datasets & Dataloaders
# -----------------------------------------------------------------
train_dataset = CustomDataset(X_train, y_train, pld_train, transform_X=normalize_transform)
val_dataset   = CustomDataset(X_val,   y_val,   pld_val,   transform_X=normalize_transform)
test_dataset  = CustomDataset(X_test,  y_test,  pld_test,  transform_X=normalize_transform)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=2*hyper_params['batch_size'], shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=2*hyper_params['batch_size'], shuffle=False, pin_memory=True)

# -----------------------------------------------------------------
# 7) Model, Loss, Optimizer
# -----------------------------------------------------------------
net = RetNet(pld_dim=1).to(device)  # 'pld_dim=1' since we have 1 pld feature
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
    net.fc2[-1].bias.fill_(y_train.mean())  # final bias

logger.info("Model initialized.")

# -----------------------------------------------------------------
# 9) Create the LearningMethod object
# -----------------------------------------------------------------
model = LearningMethod(network=net, optimizer=optimizer, criterion=criterion, logger=logger)

# -----------------------------------------------------------------
# 10) Train using the EXACT 'train' method in model_property.py
# -----------------------------------------------------------------
model.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_loss_freq=hyper_params['eval_freq'],           # Evaluate after each epoch
    epochs=hyper_params['num_epochs'],
    scheduler=scheduler,
    metric=r2_score,
    device=device,
    verbose=True,
    patience=hyper_params['patience']
)

# -----------------------------------------------------------------
# 11) Evaluate on Test
# -----------------------------------------------------------------
model.net.eval()
test_preds = []
test_true  = []

for xb, yb, pld_b in test_loader:
    xb, yb, pld_b = xb.to(device), yb, pld_b.to(device)
    with torch.no_grad():
        pred_b = model.predict(xb, pld_b)
    test_preds.append(pred_b.cpu())
    test_true.append(yb)

y_pred = torch.cat(test_preds).numpy().ravel()
y_true = torch.cat(test_true).numpy().ravel()

# -----------------------------------------------------------------
# 12) Metrics
# -----------------------------------------------------------------
from torcheval.metrics.functional import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

r2  = r2_score(torch.tensor(y_pred), torch.tensor(y_true)).item()
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

logger.info(f"\nTest R^2:  {r2:.4f}")
logger.info(f"Test MSE:  {mse:.4f}")
logger.info(f"Test MAE:  {mae:.4f}")

# -----------------------------------------------------------------
# 13) Save Model
# -----------------------------------------------------------------
os.makedirs(model_save_dir, exist_ok=True)
save_path = os.path.join(model_save_dir, f"{model_name}_property_state_dict.pt")

torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': hyper_params['num_epochs']
}, save_path)

logger.info(f"Model saved to {save_path}")

# -----------------------------------------------------------------
# 14) (Optional) Parity Plot
# -----------------------------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Parity Plot")
plt.grid(True)

os.makedirs(plot_save_dir, exist_ok=True)
plot_path = os.path.join(plot_save_dir, f"parity_property_{model_name}.png")
plt.savefig(plot_path)
logger.info(f"Saved parity plot to {plot_path}")
plt.close()
