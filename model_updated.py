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
    def __init__(self):
        super().__init__()
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
        
        # Adding a 1x1 convolution to match the number of channels
        self.adjust_channels = nn.Conv3d(in_channels=12, out_channels=24, kernel_size=1, padding=0, padding_mode='circular', bias=False)
        
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

        # Placeholder for dynamically calculating the input size for the first linear layer
        self.fc1 = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(0.3),
            nn.Linear(120000, 1176), 
            nn.BatchNorm1d(num_features=1176),
            nn.LeakyReLU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1176, 84),
            nn.BatchNorm1d(num_features=84),
            nn.LeakyReLU(),
            nn.Linear(84, 20),
            nn.BatchNorm1d(num_features=20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        x = x
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        # Adjust x1 to have the same number of channels as x2
        x1 = self.adjust_channels(x1)
        
        x2 += x1  # Residual connection
        
        x3 = self.max1(x2)
        x3 = self.conv3(x3)
        x4 = self.max2(x3)
        x4 = self.conv4(x4)
        x4 = self.conv5(x4)
        
        # Flatten x4
        x4 = x4.flatten(1)

        
        x4 = self.fc1(x4)
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

        # Training and validation phase.
        counter = 0
    
        best_val_loss = float('inf')  # Initialize best validation loss.
        best_model_state = None       # To store the best model's state.
        no_improve_epochs = 0         # Counter for epochs without improvement.

        for e in range(epochs):

            if verbose:
                self.logger.info(f'\nEpoch: {e}')

            # Training phase.
            for X_train, y_train in train_loader:
                self.net.train() # Set to training mode.
                X_train = X_train.to(device)
                y_train = y_train.to(device)# Convert labels to Double
                # Keep track of the iteration number.
                counter += 1

                X_train, y_train = X_train.to(device), y_train.to(device)

                # Initialize zero gradients.
                self.optimizer.zero_grad()

                # Calculate train loss.
                y_train_hat = self.net(X_train)
                train_loss = self.criterion(input=y_train_hat.ravel(), target=y_train)

                # Update the parameters.
                train_loss.backward()
                self.optimizer.step()

                # Validation phase.
                if (counter % val_loss_freq == 0):
                    self.net.eval() # Set to inference mode.

                    X_val, y_val = next(cycled_val_loader)
                    X_val, y_val = X_val.to(device), y_val.to(device)

                    # Account for correct training metric calculation.
                    yth = self.predict(X_train)

                    # Calculate validation loss.
                    y_val_hat = self.predict(X_val)
                    val_loss = self.criterion(input=y_val_hat.ravel(), target=y_val)

                    train_metric = metric(input=yth.ravel(), target=y_train)
                    val_metric = metric(input=y_val_hat.ravel(), target=y_val)

                    self.logger.info(
                        f'{f"Iteration {counter}":<20} -> '
                        f'{f"train_loss = {train_loss.item():.3f}":<22} '
                        f'{f"val_loss = {val_loss.item():.3f}":<22} '
                        f'{f"train_metric = {train_metric:.3f}":<22} '
                        f'{f"val_metric = {val_metric:.3f}":>22}'
                    )

            # End of epoch validation for early stopping.
            total_val_loss = 0.0
            total_samples = 0
            self.net.eval()  # Set to inference mode.
            with torch.no_grad():
                for i, (X_val, y_val) in enumerate(val_loader):
                    # print(i)
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    y_val_hat = self.net(X_val)
                    batch_val_loss = self.criterion(input=y_val_hat.ravel(), target=y_val).item()
                    total_val_loss += batch_val_loss * X_val.size(0)
                    total_samples += X_val.size(0)

            epoch_val_loss = total_val_loss / total_samples  # Average validation loss.

            # Early stopping logic.
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = self.net.state_dict()  # Save best model.
                no_improve_epochs = 0  # Reset counter.
            else:
                no_improve_epochs += 1  # Increment no improvement counter.

            self.logger.info(
                f'End of Epoch {e}: val_loss = {epoch_val_loss:.3f}, '
                f'best_val_loss = {best_val_loss:.3f}'
            )

            if no_improve_epochs >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {e+1} epochs. "
                    f"No improvement for {self.early_stopping_patience} consecutive epochs."
                )
                self.net.load_state_dict(best_model_state)  # Restore best model.
                break

            # Learning rate scheduler.
            if scheduler:
                self.scheduler.step()
        
        self.logger.info('\nTraining finished!')

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        X = X 
        y_pred = self.net(X)

        return y_pred

class CustomDataset(Dataset):
    def __init__(self, X, y, transform_X=None, transform_y=None):
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample_x = torch.tensor(self.X[idx], dtype=torch.float32)
        sample_y = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.transform_X:
            sample_x = self.transform_X(sample_x)
        if self.transform_y:
            sample_y = self.transform_y(sample_y)

        return sample_x, sample_y

            

class Rotate90:
    def __init__(self):
        self.planes = list(combinations([1, 2, 3], 2))
        self.n_choices = len(self.planes)

    def __call__(self, sample):
        plane = self.planes[np.random.choice(self.n_choices)]
        direction = np.random.choice([-1, 1])

        return torch.rot90(sample, k=direction, dims=plane)


class Flip:
    def __call__(self, sample):
        axis = np.random.choice([1, 2, 3])

        return torch.flip(sample, [axis])


class Reflect:
    def __init__(self):
        self.planes = list(combinations([1, 2, 3], 2))
        self.n_choices = len(self.planes)

    def __call__(self, sample):
        plane = self.planes[np.random.choice(self.n_choices)]

        return torch.transpose(sample, *plane)


class Roll:
    def __call__(self, sample):
        axis = np.random.choice([1, 2, 3])
        shift = np.random.choice([1, 2, 4, 6, 10])
        direction = np.random.choice([-1, 1])

        return torch.roll(sample, shifts=shift * direction, dims=axis)


class Identity:
    def __call__(self, sample):

        return sample


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