import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torcheval.metrics.functional import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model_updated import CustomDataset, Flip, Rotate90, Reflect, Identity
from model_updated import load_data_id, LearningMethod, RetNet, init_weights
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import logging
import pandas as pd

# For reproducible results.
# See also -> https://pytorch.org/docs/stable/notes/randomness.html

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

def model_predict(model_dir, model_name, target_col, id_col):
    # Define the model and loss function
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net = RetNet().to(device)
    criterion = nn.L1Loss().to(device)  # Ensure this matches the saved criterion type

    # Define optimizer (it will be loaded with its state later)
    optimizer = torch.optim.Adam(net.parameters())

    # Load the saved checkpoint
    model_save_path = f'{model_dir}/{model_name}_state_dict4.pt'  
    print(model_save_path)
    checkpoint = torch.load(model_save_path, weights_only=False)

    # Load model and optimizer state dictionaries
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load additional details if needed
    start_epoch = checkpoint['epoch']  # Starting epoch (if you want to resume training)
    criterion = checkpoint['loss']  # Load the loss function (if serialized)

    # Ensure everything is on the correct device
    net.to(device)



    # Optionally, test the model or resume training:
    # Example: Set the model to evaluation mode


    # def setup_logger(log_dir="./log", log_filename=None):
    #     os.makedirs(log_dir, exist_ok=True)
    #     if log_filename is None:
    #         log_filename = f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
    #     log_file = os.path.join(log_dir, log_filename)
    #     logging.basicConfig(
    #         level=logging.INFO,
    #         format="%(asctime)s - %(levelname)s - %(message)s",
    #         handlers=[
    #             logging.FileHandler(log_file),
    #             logging.StreamHandler()
    #         ]
    #     )
    #     logger = logging.getLogger(__name__)
    #     return logger

    # logger = setup_logger()

    # Requires installation with GPU support.
    # See also -> https://pytorch.org/get-started/locally/
    




    # Load test data.
    X_test, y_test, ids_test = load_data_id(
        f'{load_dir}/test',
        f'{load_dir}/all.csv',
        target_col,
        id_col,
    )

    assert np.isnan(X_test).sum()==0
    assert np.isnan(y_test).sum()==0

    # Transformations for standardization + data augmentation.
    # standardization = transforms.Normalize(X_train.mean(), X_train.std())
    standardization = transforms.Normalize(X_test.mean(), X_test.std())


    # Adding a channel dimension required for CNN.
    X_test = X_test.reshape(X_test.shape[0], 1, *X_test.shape[1:])


    print(X_test.shape)
    print(y_test.shape)


    # Create the dataloaders.


    test_loader = DataLoader(
        CustomDataset(X=X_test, y=y_test, transform_X=standardization),
        batch_size=64, pin_memory=True,
    )



    model = LearningMethod(net, None, None, logger=None)
    # print(net)


    # Use Tensorboard. Needs to be fixed!
    # See also -> https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
    #writer = SummaryWriter(log_dir='experiments/')


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
    mae = mean_absolute_error(y_true, y_pred)

    print(f'R^2: {r2}')
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    test_rst = {id_col: ids_test,f'{target_col}': y_true, f'{target_col}_pred': y_pred}
    df_test = pd.DataFrame.from_dict(test_rst)
    df_test.to_csv(f'./pred/{model_name}_pred.csv', index=False)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Parity Plot')
    plt.grid(True)




def derive_pred_multi_grid(df_pred):

    df_pred['mof'] = df_pred['sample'].apply(lambda x: x.split('--')[0])
    df_pred['grid_type'] = df_pred['sample'].apply(lambda x: x.split('--')[1])
    df_pred_mof = df_pred.groupby('mof').agg(
        Xe_cm3_per_cm3_value=('Xe_cm3_per_cm3_value','first'),
        Xe_cm3_per_cm3_value_pred_mean=('Xe_cm3_per_cm3_value_pred','first')
    ).reset_index()


    y_pred = df_pred_mof['Xe_cm3_per_cm3_value_pred_mean'].values
    y_true = df_pred_mof['Xe_cm3_per_cm3_value'].values
    # Calculate R^2 and MSE
    r2 = r2_score(torch.tensor(y_pred), torch.tensor(y_true)).item()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f'R^2: {r2}')
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    # test_rst = {id_col: ids_test,f'{target_col}': y_true, f'{target_col}_pred': y_pred}
    # df_test = pd.DataFrame.from_dict(test_rst)
    # df_test.to_csv(f'./pred/{model_name}_pred.csv', index=False)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Parity Plot')
    plt.grid(True)

# model_name = 'Mix1bar_3_grids_all_Xe_cm3_per_cm3_value_no_64_64_0.0001_0.0001_60_0.5_Adam'
# model_name = 'Mix1bar_Xe_cm3_per_cm3_value_no_64_64_0.0001_0.0001_60_0.5_Adam'
model_name = 'Mix1bar_Kr_cm3_per_cm3_value__no_64_64_0.0001_0.0001_60_0.5_Adam'

model_dir = '/data/yll6162/mof_cnn/model'
num_grids = 1
pressure = '1bar'
# load_dir = f'/data/yll6162/mof_cnn/data_mix_{pressure}_{num_grids}_grids'
load_dir = f'/data/yll6162/mof_cnn/data_mix_{pressure}'
col = 'Kr_cm3_per_cm3_value'

id_col = 'project'
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_pred = model_predict(model_dir, model_name, col, id_col)


if num_grids > 1:
    derive_pred_multi_grid(df_pred)








