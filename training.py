import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torcheval.metrics.functional import r2_score
from sklearn.metrics import mean_squared_error
from model import CustomDataset, Flip, Rotate90, Reflect, Identity
from model import load_data, LearningMethod, RetNet, init_weights
import matplotlib.pyplot as plt

# For reproducible results.
# See also -> https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

# Requires installation with GPU support.
# See also -> https://pytorch.org/get-started/locally/
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'MixGasRetNet'
data_dir = '/data/yll6162/mof_cnn/data_mix_13k'
target = 'Xe_mol_per_kg_value'

# Load training data.
X_train, y_train = load_data(
    f'{data_dir}/train',
    f'{data_dir}/all.csv',
    target,
    'project',
)

# Load validation data.
X_val, y_val = load_data(
    f'{data_dir}/test',
    f'{data_dir}/all.csv',
    target,
    'project',
    size=1000
)
print(X_train.shape)
print(y_train.shape)
# Transformations for standardization + data augmentation.
standardization = transforms.Normalize(X_train.mean(), X_train.std())

augmentation = transforms.Compose([
    standardization,
    transforms.RandomChoice([Reflect(), Identity()]),
    # transforms.RandomChoice([Rotate90(), Flip(), Reflect(), Identity()]),
])

# Adding a channel dimension required for CNN.
X_train, X_val = [X.reshape(X.shape[0], 1, *X.shape[1:]) for X in [X_train, X_val]]

# Create the dataloaders.
train_loader = DataLoader(
    CustomDataset(X=X_train, y=y_train, transform_X=standardization),
    batch_size=64, shuffle=True, pin_memory=True,
)

val_loader = DataLoader(
    CustomDataset(X=X_val, y=y_val, transform_X=standardization),
    batch_size=256, pin_memory=True,
)

# Define the architecture, loss and optimizer.
net = RetNet().double().to(device)
criterion = nn.L1Loss().to(device) 
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

# Define the learning rate scheduler.
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=50,
    gamma=0.5, verbose=True
)

# Initialize weights.
net.apply(lambda m: init_weights(m, a=0.01))

# Initialize bias of the last layer with E[y_train].
torch.nn.init.constant_(net.fc2[-1].bias, y_train.mean())

model = LearningMethod(net, optimizer, criterion)
print(net)


# Use Tensorboard. Needs to be fixed!
# See also -> https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
#writer = SummaryWriter(log_dir='experiments/')

model.train(
    train_loader=train_loader, val_loader=val_loader,
    metric=r2_score, epochs=50, scheduler=scheduler,
    device=device, verbose=True, #tb_writer=writer,
)

# Calculate R^2 and MSE on the whole validation set.
predictions = []
for x, _ in val_loader:
    y_pred = model.predict(x.to(device))
    predictions.append(y_pred.cpu())

y_pred = torch.cat(predictions).numpy()
y_true = y_val.reshape(len(y_val), -1)

# Calculate R^2 and MSE
r2 = r2_score(torch.tensor(y_pred), torch.tensor(y_true)).item()
mse = mean_squared_error(y_true, y_pred)

print(f'R^2: {r2}')
print(f'MSE: {mse}')

# Plot Parity Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Parity Plot')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig(f'{model_name}_parity_plot12-4.png')

plt.show()

# Save the trained model.
# See also -> https://pytorch.org/tutorials/beginner/saving_loading_models.html
#torch.save(model, f'{model_name}.pt')
model_save_path = f'{model_name}_state_dict4.pt'
optimizer_save_path = f'{model_name}_optimizer_state_dict4.pt'

torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 50,  # or whatever the current epoch is
    'loss': criterion,  # optionally save the loss function
}, model_save_path)

print(f'Model and optimizer state dicts saved to {model_save_path} and {optimizer_save_path}.')