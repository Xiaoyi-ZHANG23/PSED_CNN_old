import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torcheval.metrics.functional import r2_score
from sklearn.metrics import mean_squared_error
from model_updated import CustomDataset, Flip, Rotate90, Reflect, Identity
from model_updated import load_data, LearningMethod, RetNet, init_weights
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import logging

# For reproducible results.
# See also -> https://pytorch.org/docs/stable/notes/randomness.html

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# eval_freq: evaluate val loos and check early stopping every k batches
hyper_params = {'batch_size':48, 'num_epochs':300, 'eval_freq':20,
                'learning_rate':1e-4, 'weight_decay':1e-4, 'step_size':60, 'gamma':0.5,
                 'optimizer':'Adam', 'patience':100}


# load_dir = '/data/yll6162/data/'
load_dir = '/data/yll6162/mof_cnn/data_mix_13k'
# target_col = 'Xe_cm3_per_cm3_value'
target_col = 'Kr_cm3_per_cm3_value'
model_name = f'Mix_{target_col}'
model_name = f"{model_name}_{hyper_params['batch_size']}_{hyper_params['learning_rate']}_{hyper_params['weight_decay']}_{hyper_params['step_size']}_{hyper_params['gamma']}_{hyper_params['optimizer']}"

def setup_logger(log_dir="./log", log_filename=None):
    os.makedirs(log_dir, exist_ok=True)
    if log_filename is None:
        log_filename = f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
    log_file = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logger()
logger.info(hyper_params)
# Requires installation with GPU support.
# See also -> https://pytorch.org/get-started/locally/
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load training data.
X_train, y_train = load_data(
    f'{load_dir}/train',
    f'{load_dir}/all.csv',
    target_col,
    'project',
)



# Load test data.
X_test, y_test = load_data(
    f'{load_dir}/test',
    f'{load_dir}/all.csv',
    target_col,
    'project',
)
assert np.isnan(X_train).sum()==0
assert np.isnan(y_train).sum()==0
assert np.isnan(X_test).sum()==0
assert np.isnan(y_test).sum()==0

# Transformations for standardization + data augmentation.
standardization = transforms.Normalize(X_train.mean(), X_train.std())

augmentation = transforms.Compose([
    standardization,
    transforms.RandomChoice([Reflect(), Identity()]),
    # transforms.RandomChoice([Rotate90(), Flip(), Reflect(), Identity()]),
])

# Adding a channel dimension required for CNN.
X_train, X_test = [X.reshape(X.shape[0], 1, *X.shape[1:]) for X in [X_train, X_test]]

# Split the training data into training and validation sets.
train_size = 8/9
val_size = 1 - train_size
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size = val_size,
    random_state=SEED
)
logger.info(X_train.shape)
logger.info(y_train.shape)

logger.info(X_val.shape)
logger.info(y_val.shape)

logger.info(X_test.shape)
logger.info(y_test.shape)

print(X_train[0])
# Create the dataloaders.
train_loader = DataLoader(
    CustomDataset(X=X_train, y=y_train, transform_X=standardization),
    batch_size=hyper_params['batch_size'], shuffle=True, pin_memory=True,
)

val_loader = DataLoader(
    CustomDataset(X=X_val, y=y_val, transform_X=standardization),
    batch_size=2 * hyper_params['batch_size'], shuffle=True, pin_memory=True,
)

test_loader = DataLoader(
    CustomDataset(X=X_test, y=y_test, transform_X=standardization),
    batch_size=2 * hyper_params['batch_size'], pin_memory=True,
)

# Define the architecture, loss and optimizer.
net = RetNet().double().to(device)
criterion = nn.L1Loss().to(device) 
optimizer = optim.Adam(net.parameters(), lr=hyper_params['learning_rate'], weight_decay=hyper_params['weight_decay'])

# Define the learning rate scheduler.
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=hyper_params['step_size'],
    gamma=hyper_params['gamma'], verbose=True
)

# Initialize weights.
net.apply(lambda m: init_weights(m, a=0.01))

# Initialize bias of the last layer with E[y_train].
torch.nn.init.constant_(net.fc2[-1].bias, y_train.mean())

model = LearningMethod(net, optimizer, criterion, logger=logger)
logger.info(net)


# Use Tensorboard. Needs to be fixed!
# See also -> https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
#writer = SummaryWriter(log_dir='experiments/')

model.train(
    train_loader=train_loader, val_loader=val_loader, val_loss_freq=hyper_params['eval_freq'],
    metric=r2_score, epochs=hyper_params['num_epochs'], scheduler=scheduler,
    device=device, verbose=True, #tb_writer=writer,
    patience=hyper_params['patience']
)

# Calculate R^2 and MSE on the whole validation set.
predictions = []
for x, _ in test_loader:
    y_pred = model.predict(x.to(device))
    predictions.append(y_pred.cpu())

y_pred = torch.cat(predictions).numpy()
y_true = y_test.reshape(len(y_test), -1)

# Calculate R^2 and MSE
r2 = r2_score(torch.tensor(y_pred), torch.tensor(y_true)).item()
mse = mean_squared_error(y_true, y_pred)

logger.info(f'R^2: {r2}')
logger.info(f'MSE: {mse}')

# Plot Parity Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Parity Plot')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig(f'./pred/parity_{model_name}.png')



# plt.show()

# Save the trained model.
# See also -> https://pytorch.org/tutorials/beginner/saving_loading_models.html
#torch.save(model, f'{model_name}.pt')
model_save_path = f'./model/{model_name}_state_dict4.pt'
optimizer_save_path = f'./model/{model_name}_optimizer_state_dict4.pt'

torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': hyper_params['num_epochs'],  # or whatever the current epoch is
    'loss': criterion,  # optionally save the loss function
}, model_save_path)

logger.info(f'Model and optimizer state dicts saved to {model_save_path} and {optimizer_save_path}.')