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
import json

from src.lib import utils_2
from src.lib.model import BoostingRegressor


if __name__ == "__main__":
    config = utils_2.load_config("configs/train_config.yaml")

    # Extract training parameters
    train_size = config["training"]["train_size"]
    validation_size_within_train = config["training"]["validation_size_within_train"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]

    # Extract data paths
    raw_data_dir = config["data"]["raw_data_dir"]
    clustered_data_path = config["data"]["clustered_data_path"]
    bayesian_opt_best_parameters_path = config["data"]["bayesian_opt_best_parameters_path"]
    in_prediction_path = config["data"]["in_prediction_path"]
    ood_prediction_path = config["data"]["ood_prediction_path"]

    # Extract model parameters
    model_save_path = config["model"]["save_path"]
    device = config["model"]["device"]

    print(f"Loading clustered data from : {clustered_data_path}")

    df = pd.read_csv(clustered_data_path)

    # Extract features and targets
    temp_ID_X = [ast.literal_eval(a) for a in df[df['clusters']==0]['fingerprints']]
    fingerprints = np.array([a for a in temp_ID_X])

    targets = np.asarray(df[df['clusters']==0]['targets'], dtype=np.float32)

    X_train, X_test_ID, y_train, y_test_ID  = train_test_split(fingerprints, 
                                                               targets, 
                                                               train_size=train_size, 
                                                               random_state=42)

    # Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size_within_train, random_state=42) 

    print(y_train.shape, y_val.shape, y_test_ID.shape)

    # Create DataLoader instances
    train_dataset = utils_2.ESOLDataset(X_train, y_train)
    validation_dataset = utils_2.ESOLDataset(X_val, y_val)
    test_dataset = utils_2.ESOLDataset(X_test_ID, y_test_ID)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Best parameters
    with open(bayesian_opt_best_parameters_path, 'r') as file:
        best_params = json.load(file)

    # Load model
    loaded_model = BoostingRegressor(input_dim=X_test_ID.shape[1], 
                                    hidden_dim=int(best_params['hidden_dim']), 
                                    num_layers=int(best_params['num_layers']), 
                                    dropout_rate=best_params['dropout_rate']).to(device)
    loaded_model.load_state_dict(torch.load(model_save_path))
    loaded_model.eval()
    print("Model loaded successfully.")

    ################ In Distribution Prediction ################
    r2, y_true_id, y_pred_id = utils_2.compute_r2(loaded_model, test_loader, device)
    print(f"In Distribution Test Data R^2 Score: {r2:.4f}")
    utils_2.plot_results(y_true_id, y_pred_id, title="In Distribution Prediction", save_path=in_prediction_path)
    ################ Out of Distribution Prediction ################
    # Extract features and targets
    temp_OOD_X = [ast.literal_eval(a) for a in df[df['clusters']==-1]['fingerprints']]
    X_test_OOD = np.array([a for a in temp_OOD_X])

    y_test_OOD = np.asarray(df[df['clusters']==-1]['targets'], dtype=np.float32)
    test_OOD_dataset = utils_2.ESOLDataset(X_test_OOD, y_test_OOD)
    test_OOD_loader = DataLoader(test_OOD_dataset, batch_size=batch_size, shuffle=False)

    r2_ood, y_true_ood, y_pred_ood = utils_2.compute_r2(loaded_model, test_OOD_loader, device)
    print(f"Out of Distribution Test Data R^2 Score: {r2_ood:.4f}")
    utils_2.plot_results(y_true_ood, y_pred_ood, title="Out of Distribution Prediction", save_path=ood_prediction_path)