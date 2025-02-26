import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from itertools import cycle, combinations
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score


class RetNet(nn.Module):
    def __init__(self, extra_dim=1):
        """
        extra_dim: number of features in your 'extras' vector (PLD, LCD, density, etc.)
        """
        super().__init__()

        # -----------------------------
        #   1) Convolutional Blocks
        # -----------------------------
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=12, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=12),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=24, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=24),
            nn.LeakyReLU(),
        )
        # Matches channels for residual
        self.adjust_channels = nn.Conv3d(
            in_channels=12, out_channels=24, kernel_size=1, 
            padding=0, padding_mode='circular', bias=False
        )
        self.max1 = nn.MaxPool3d(kernel_size=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=32, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=32),
            nn.LeakyReLU(),
        )
        self.max2 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=64),
            nn.LeakyReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=120, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=120),
            nn.LeakyReLU(),
        )

        # -----------------------------
        #   2) Flatten + FC Layers
        # -----------------------------
        self.fc1 = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(0.3),
            nn.Linear(120000, 1176),
            nn.BatchNorm1d(num_features=1176),
            nn.LeakyReLU(),

            nn.Linear(1176, 84),
            nn.BatchNorm1d(num_features=84),
            nn.LeakyReLU(),
        )

        # 
        # The final fully-connected block must handle 
        # 84 + extra_dim input features
        #
        self.fc2 = nn.Sequential(
            nn.Linear(84 + extra_dim, 20),
            nn.BatchNorm1d(num_features=20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x, extras):
        """
        x      : 3D image input (batch_size, 1, D, H, W)
        extras : additional features (batch_size, extra_dim)
        """
        # -- conv1, conv2 with residual
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x1 = self.adjust_channels(x1)
        x2 += x1  # Residual connection

        # -- maxpool + conv3
        x3 = self.max1(x2)
        x3 = self.conv3(x3)

        # -- maxpool + conv4, conv5
        x4 = self.max2(x3)
        x4 = self.conv4(x4)
        x4 = self.conv5(x4)

        # -- Flatten  # or use self.fc1 which includes Flatten
        x4 = self.fc1(x4)

        # 
        # Concatenate all your extra features
        # e.g., if extras has shape (batch_size, extra_dim)
        #
        x4 = torch.cat([x4, extras], dim=1)

        # Final dense layers
        x4 = self.fc2(x4)
        return x4

    
class LearningMethod:
    def __init__(self, network, optimizer, criterion, logger=None):
        self.net = network 
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger

    def train(
            self, train_loader, val_loader,
            val_loss_freq=15, epochs=1, scheduler=None,
            metric=r2_score, device=None, tb_writer=None,
            verbose=True, patience=None
            ):
        
        self.scheduler = scheduler
        self.val_loss_freq = val_loss_freq
        self.train_hist = []
        self.train_metric = []
        self.val_hist = []
        self.val_metric = []
        self.writer = tb_writer
        self.train_batch_size = train_loader.batch_size
        self.val_batch_size = val_loader.batch_size
        self.epochs = epochs
        self.early_stopping_patience = patience
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0

        cycled_val_loader = cycle(val_loader)

        # ...
        best_val_loss = float('inf')
        best_model_state = None
        no_improve_epochs = 0

        for e in range(epochs):
            epoch_train_loss = 0.0
            num_train_samples = 0
            if verbose and self.logger is not None:
                self.logger.info(f'\nEpoch: {e}')

            # Training phase
            for X_train, y_train, extras_train in train_loader:
                self.net.train()
                X_train, y_train, extras_train = (
                    X_train.to(device),
                    y_train.to(device),
                    extras_train.to(device),
                )

                self.optimizer.zero_grad()
                y_train_hat = self.net(X_train, extras_train)
                train_loss = self.criterion(input=y_train_hat.ravel(), target=y_train)

                train_loss.backward()
                self.optimizer.step()
                epoch_train_loss += train_loss.item() * X_train.size(0)
                num_train_samples += X_train.size(0)

            # Validation phase
            epoch_train_loss_avg = epoch_train_loss / num_train_samples
            self.train_hist.append(epoch_train_loss_avg)
            total_val_loss = 0.0
            total_samples = 0
            self.net.eval()
            val_metrics = []

            with torch.no_grad():
                for i, (X_val, y_val, extras_val) in enumerate(val_loader):
                    X_val, y_val, extras_val = (
                        X_val.to(device),
                        y_val.to(device),
                        extras_val.to(device),
                    )
                    y_val_hat = self.net(X_val, extras_val)
                    batch_val_loss = self.criterion(input=y_val_hat.ravel(), target=y_val).item()

                    total_val_loss += batch_val_loss * X_val.size(0)
                    total_samples += X_val.size(0)

                    # Calculate metrics
                    batch_metric = metric(input=y_val_hat.ravel(), target=y_val)
                    val_metrics.append(batch_metric)

            # Compute average validation loss and metric
            epoch_val_loss = total_val_loss / total_samples
            avg_val_metric = torch.mean(torch.tensor(val_metrics)).item()

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = self.net.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            self.logger.info(
                f'End of Epoch {e}: val_loss = {epoch_val_loss:.3f}, '
                f'val_metric = {avg_val_metric:.3f}, '
                f'best_val_loss = {best_val_loss:.3f}'
            )

            if no_improve_epochs >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {e+1} epochs. "
                    f"No improvement for {self.early_stopping_patience} consecutive epochs."
                )
                self.net.load_state_dict(best_model_state)
                break

            if scheduler:
                self.scheduler.step()

        self.logger.info('\nTraining finished!')

    @torch.no_grad()
    def predict(self, X, extras):
        self.net.eval()
        y_pred = self.net(X, extras)
        return y_pred


class CustomDataset(Dataset):
    def __init__(self, X, y, extras, transform_X=None, transform_y=None):
        """
        X      : shape (N, 1, D, H, W) for 3D input
        y      : shape (N,)
        extras : shape (N, extra_dim) 
        """
        self.transform_X = transform_X
        self.transform_y = transform_y

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.extras = extras.astype(np.float32)  # shape: (N, extra_dim)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample_x      = torch.tensor(self.X[idx], dtype=torch.float32)
        sample_y      = torch.tensor(self.y[idx], dtype=torch.float32)
        sample_extras = torch.tensor(self.extras[idx], dtype=torch.float32)

        if self.transform_X:
            sample_x = self.transform_X(sample_x)
        if self.transform_y:
            sample_y = self.transform_y(sample_y)

        return sample_x, sample_y, sample_extras



@torch.no_grad()
def init_weights(m, initialization='normal', **kwargs):
    if initialization == 'normal':
        if type(m) == nn.Linear:
            m.weight = nn.init.kaiming_normal_(m.weight, **kwargs)
    elif initialization == 'uniform':
        if type(m) == nn.Linear:
            m.weight = nn.init.kaiming_uniform_(m.weight, **kwargs)


def load_data(dir_batch, path_to_csv, target_name, index_col, size=None):
    with open(f'{dir_batch}/clean.json', 'r') as fhand:
        names = json.load(fhand)['name']
    print(dir_batch, len(names))
    df = pd.read_csv(path_to_csv, index_col=index_col)

    y = df.loc[names, target_name].values
    X = np.load(f'{dir_batch}/clean.npy', mmap_mode='r')

    if size is None:
        return X, y
    else:
        return X[:size], y[:size]
    
def load_data_id(dir_batch, path_to_csv, target_name, index_col, size=None):
    with open(f'{dir_batch}/clean.json', 'r') as fhand:
        names = json.load(fhand)['name']
    print(dir_batch, len(names))
    df = pd.read_csv(path_to_csv, index_col=index_col)

    y = df.loc[names, target_name].values
    X = np.load(f'{dir_batch}/clean.npy', mmap_mode='r')
    ids = names
    if size is None:
        return X, y, ids
    else:
        return X[:size], y[:size], ids[:size]