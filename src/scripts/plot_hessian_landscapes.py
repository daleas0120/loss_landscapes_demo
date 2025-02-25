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

# This is the custom model wrapper for the loss landscapes calculation
class Metric(ABC):
    """ A quantity that can be computed given a model or an agent. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model_wrapper: ModelWrapper):
        pass

class Loss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, model, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.model = model
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        outputs = model_wrapper.forward(self.inputs)
        #err = self.loss_fn(self.target[0], outputs) # This is code from Ashley which may cause a bug in my code
        err = self.loss_fn(self.target, outputs)
        return err

def calc_average_loss_array(data_loader):
    loss_landscapes_list = []
    for batch_X, batch_y in data_loader:
        metric = Loss(loss_func, func.eval(), batch_X, batch_y)
        try:
            loss_data_fin = loss_landscapes.planar_interpolation(
                model_start=func.eval(), 
                model_end_one=model_eig_max.eval(),
                model_end_two=model_eig_min.eval(),
                metric=metric, steps=steps, deepcopy_model=True
                )
            loss_landscapes_list.append(loss_data_fin)
        except Exception as e:
            print(e)
            # continue
            
    # Convert list to a NumPy array (3D array)
    stacked_arrays = np.stack(loss_landscapes_list)  # Shape: (num_arrays, rows, cols)

    # Compute the average across the first axis (axis=0)
    average_array = np.mean(stacked_arrays, axis=0)
    return average_array


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

    # Hessian plots paths
    trained_data_loss_landscape_path = config["hessian"]["trained_data_loss_landscape_path"]
    test_id_data_loss_landscape_path = config["hessian"]["test_id_data_loss_landscape_path"]
    test_ood_data_loss_landscape_path = config["hessian"]["test_ood_data_loss_landscape_path"]
    test_id_ood_diff_loss_landscape_path = config["hessian"]["test_id_ood_diff_loss_landscape_path"]

    # Extract model parameters
    model_save_path = config["model"]["save_path"]
    device = config["model"]["device"]

    # Hessian parameters
    model_eig_max_path = config["hessian"]["model_eig_max_path"]
    model_eig_min_path = config["hessian"]["model_eig_min_path"]

    # loss landscape parameters
    distance = config["loss_landscape"]["distance"]
    steps = config["loss_landscape"]["steps"]

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

    ################ Out of Distribution Data ################
    # Extract features and targets
    temp_OOD_X = [ast.literal_eval(a) for a in df[df['clusters']==-1]['fingerprints']]
    X_test_OOD = np.array([a for a in temp_OOD_X])

    y_test_OOD = np.asarray(df[df['clusters']==-1]['targets'], dtype=np.float32)
    test_OOD_dataset = utils_2.ESOLDataset(X_test_OOD, y_test_OOD)
    test_OOD_loader = DataLoader(test_OOD_dataset, batch_size=batch_size, shuffle=False)
    ################################################################

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

    # Load eigenvectors
    model_eig_max = copy.deepcopy(func)
    model_eig_max.load_state_dict(torch.load(model_eig_max_path, weights_only=True))
    model_eig_min = copy.deepcopy(func)
    model_eig_min.load_state_dict(torch.load(model_eig_min_path, weights_only=True))

    # Create 2D directed loss surface
    average_array_trained = calc_average_loss_array(data_loader=train_loader)
    utils_2.plot_loss_contours(average_array_trained, title="Trained Data", vmin=0, vmax=2, scale="linear", 
                               save_path = trained_data_loss_landscape_path)
    
    average_array_test = calc_average_loss_array(data_loader=test_loader)
    utils_2.plot_loss_contours(average_array_test, title="ID Test", vmin=0, vmax=2, scale="linear",
                               save_path = test_id_data_loss_landscape_path)


    average_array_test_OOD = calc_average_loss_array(data_loader=test_OOD_loader)
    utils_2.plot_loss_contours(average_array_test, title="ID Test", vmin=0, vmax=2, scale="linear",
                               save_path = test_ood_data_loss_landscape_path)

    utils_2.plot_loss_contours(abs(average_array_test - average_array_test_OOD), title="Abs. Difference", vmin=0, vmax=2, 
                               scale="linear",
                               save_path = test_id_ood_diff_loss_landscape_path)
