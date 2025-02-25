import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import MoleculeNet
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

import pandas as pd
import ast

from src.lib import utils_2
from src.lib.model import BoostingRegressor

# Bayesian Optimization Function
def train_and_evaluate(hidden_dim, num_layers, dropout_rate, lr):
    hidden_dim = int(hidden_dim)
    num_layers = int(num_layers)
    dropout_rate = float(dropout_rate)
    lr = float(lr)
    
    model = BoostingRegressor(input_dim=X_train.shape[1], hidden_dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for batch_X, batch_y in validation_loader:
            y_pred = model(batch_X)
            y_preds.append(y_pred.cpu().numpy())
            y_trues.append(batch_y.cpu().numpy())
    
    y_preds = np.vstack(y_preds)
    y_trues = np.vstack(y_trues)
    #mse = mean_squared_error(y_trues, y_preds)
    r2 = r2_score(y_trues, y_preds)
    #return -mse  # Negative for maximization
    return r2

if __name__ == "__main__":
    config = utils_2.load_config("configs/train_config.yaml")

    # Extract training parameters
    test_size = config["training"]["test_size"]
    validation_size_within_train = config["training"]["validation_size_within_train"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]

    # Extract data paths
    raw_data_dir = config["data"]["raw_data_dir"]
    clustered_data_path = config["data"]["clustered_data_path"]
    bayesian_opt_progress_path = config["data"]["bayesian_opt_progress_path"]

    # Extract Bayesian Optimization parameters
    bayesian_params = config["bayesian_optimization"]
    init_points = config["bayesian_optimization"]["init_points"]
    n_iter = config["bayesian_optimization"]["n_iter"]

    # Extract model parameters
    model_save_path = config["model"]["save_path"]

    print(f"Loading clustered data from : {clustered_data_path}")

    df = pd.read_csv(clustered_data_path)

    # Extract features and targets
    temp_ID_X = [ast.literal_eval(a) for a in df[df['clusters']==0]['fingerprints']]
    fingerprints = np.array([a for a in temp_ID_X])

    targets = np.asarray(df[df['clusters']==0]['targets'], dtype=np.float32)

    X_train, X_test_ID, y_train, y_test_ID  = train_test_split(fingerprints, 
                                                               targets, 
                                                               train_size=validation_size_within_train, 
                                                               random_state=42)

    # Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size_within_train, random_state=42) 

    # Create DataLoader instances
    train_dataset = utils_2.ESOLDataset(X_train, y_train)
    validation_dataset = utils_2.ESOLDataset(X_val, y_val)
    test_dataset = utils_2.ESOLDataset(X_test_ID, y_test_ID)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Bayesian Optimization to find the best model parameters
    # Define Bayesian Optimization Parameters
    pbounds = {
        'hidden_dim': (bayesian_params['hidden_dim']['min'], bayesian_params['hidden_dim']['max']),
        'num_layers': (bayesian_params['num_layers']['min'], bayesian_params['num_layers']['max']),
        'dropout_rate': (bayesian_params['dropout_rate']['min'], bayesian_params['dropout_rate']['max']),
        'lr': (bayesian_params['lr']['min'], bayesian_params['lr']['max']),
    }

    optimizer = BayesianOptimization(
        f=train_and_evaluate,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params = optimizer.max['params']
    print("Best Parameters Found:", best_params)

    # Visualization of Bayesian Optimization Progress
    utils_2.plot_bayesian_optimization_progress(optimizer.res, save_path=bayesian_opt_progress_path)