# Preprocess data for ESOl dataset and create cluster labels
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import MoleculeNet
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from src.lib import utils_2


if __name__ == "__main__":
    config = utils_2.load_config("configs/train_config.yaml")

    # Extract training parameters
    test_size = config["training"]["test_size"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]
    optimizer_type = config["training"]["optimizer"]

    # Extract data paths
    raw_data_dir = config["data"]["raw_data_dir"]
    clustered_data_path = config["data"]["clustered_data_path"]
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]

    # Extract clustering parameters
    pca_n_components = config["clustering"]["pca_n_components"]
    dbscan_epsilon = config["clustering"]["dbscan_epsilon"]
    dbscan_min_samples = config["clustering"]["dbscan_min_samples"]

    # Extract model parameters
    model_arch = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]
    model_save_path = config["model"]["save_path"]

    # Print loaded config (for debugging)
    print(f"Training {model_arch} for {epochs} epochs with {optimizer_type} optimizer.")
    print(f"Train data: {train_path}, Validation data: {val_path}")
    print(f"Model will be saved to: {model_save_path}")


    # Load ESOL dataset
    dataset = MoleculeNet(root=raw_data_dir, name="ESOL")

    # Extract features and targets
    smiles_list = [data.smiles for data in dataset]
    fingerprints = np.array([utils_2.mol_to_fp(smiles) for smiles in smiles_list])
    targets = np.array([data.y.item() for data in dataset])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(fingerprints, targets, test_size=test_size, random_state=42)

    # PCA
    pca = PCA(n_components=pca_n_components)
    pca.fit(fingerprints)
    transformed_data = pca.transform(fingerprints)

    # DBSCAN clustering
    epsilon = dbscan_epsilon 
    min_samples = dbscan_min_samples  
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = dbscan.fit_predict(transformed_data)

    utils_2.save_dataframe(clustered_data_path, 
                           smiles=smiles_list, 
                           fingerprints=fingerprints.tolist(), 
                           targets=targets, 
                           clusters=clusters)