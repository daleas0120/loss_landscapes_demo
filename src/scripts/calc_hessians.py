import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
import ast
import json
import copy
from collections import OrderedDict

import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_interface.model_wrapper import ModelWrapper
from abc import ABC, abstractmethod

from src.lib import botcher_hessian as hess
from src.lib import botcher_utilities

from src.lib import utils_2
from src.lib.model import BoostingRegressor

def force_wts_into_model(og_layer_names, new_model_wts, empty_model, old_model_state_dict):

    new_model_wt_dict = copy.deepcopy(old_model_state_dict)

    for layer, new_param in zip(og_layer_names, new_model_wts):
        if new_param.shape == old_model_state_dict[layer].shape:
            new_model_wt_dict[layer] = new_param
        else:
            print(layer+" incompatible")

    err_layers = empty_model.load_state_dict(new_model_wt_dict, strict=False)
    print(err_layers)

    return empty_model

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

    # Hessian parameters
    model_eig_max_path = config["hessian"]["model_eig_max_path"]
    model_eig_min_path = config["hessian"]["model_eig_min_path"]

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

    ################ Get Hessain Eigenvectors ################
    loss_func = torch.nn.MSELoss()
    func = copy.deepcopy(loaded_model)
    og_params = [i[1] for i in func.named_parameters() if len(i[1].size()) > 1]
    og_layer_names = [i[0] for i in func.named_parameters() if len(i[1].size())>1]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    maxeig, mineig, maxeigvec, mineigvec, num_iter = hess.min_max_hessian_eigs(
        func, X_train_tensor, y_train_tensor, loss_func, all_params=False
        )
    print(maxeig, mineig)

    ################ Format as two new models ################
    max_model_wts = hess.npvec_to_tensorlist(maxeigvec, og_params)
    min_model_wts = hess.npvec_to_tensorlist(mineigvec, og_params)

    model_eig_max = copy.deepcopy(func)
    model_eig_min = copy.deepcopy(func)

    # There will be some incompatible keys due to the batch norm values
    # the original batch norm values will be retained

    model_wt_dict = OrderedDict([i for i in loaded_model.named_parameters()])

    model_eig_max = force_wts_into_model(og_layer_names, max_model_wts, model_eig_max,  model_wt_dict)
    model_eig_min = force_wts_into_model(og_layer_names, min_model_wts, model_eig_min,  model_wt_dict)

    torch.save(model_eig_max.state_dict(), model_eig_max_path)
    torch.save(model_eig_min.state_dict(), model_eig_min_path)