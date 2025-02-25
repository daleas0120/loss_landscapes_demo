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
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import r2_score


# Function to compute Morgan fingerprints using MorganGenerator
def mol_to_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)

# Load YAML config
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

import pandas as pd

def save_dataframe(df_path, **columns):
    """
    Create a DataFrame from given column data and save it as a CSV.

    Parameters:
    - df_path (str): Path to save the CSV file.
    - **columns: Column names as keys and lists/arrays as values.

    Example usage:
    save_dataframe("data.csv", smiles=smiles_list, fingerprints=fingerprints.tolist(), targets=targets, clusters=clusters)
    """
    df = pd.DataFrame(columns)

    df.to_csv(df_path, index=False)
    print(f"Data saved to {df_path}")

# Create a PyTorch Dataset class
class ESOLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def plot_dbscan_clustering_results(transformed_data, clusters, figsize=(7, 5), title="DBSCAN Clustering Results", 
                                   xlabel="Feature 1", ylabel="Feature 2", cmap='plasma', save_path=None):
    """
    Visualizes the results of DBSCAN clustering with a scatter plot.

    Parameters:
        transformed_data (array-like): The transformed data points (2D).
        clusters (array-like): The cluster labels for each data point.
        figsize (tuple): Size of the figure (default is (7, 5)).
        title (str): Title of the plot (default is "DBSCAN Clustering Results").
        xlabel (str): Label for the x-axis (default is "Feature 1").
        ylabel (str): Label for the y-axis (default is "Feature 2").
        cmap (str): Colormap for the scatter plot (default is 'plasma').
    """
    plt.figure(figsize=figsize)
    scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=clusters, cmap=cmap)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def save_json_object(data, file_path):
    import json
    with open(file_path, 'w') as file:
        json.dump(data, file) 
    print(f"JSON data saved to '{file_path}'")

def plot_bayesian_optimization_progress(optimizer_results, xlabel="Iteration", ylabel="R^2", title="Bayesian Optimization Progress", 
                                        show_grid=True, figsize=(10, 5), save_path=None):
    """
    Visualizes the progress of Bayesian optimization based on the optimization results.
    
    Parameters:
        optimizer_results (list of dicts): List containing the optimization results. Each dict should contain a 'target' key.
        xlabel (str): Label for the x-axis (default is "Iteration").
        ylabel (str): Label for the y-axis (default is "R^2").
        title (str): Title of the plot (default is "Bayesian Optimization Progress").
        show_grid (bool): Whether to show grid on the plot (default is True).
        figsize (tuple): Size of the plot figure (default is (10, 5)).
        save_path (str or None): Path where the plot should be saved. If None, the plot is not saved (default is None).
    """
    iteration_values = [res['target'] for res in optimizer_results]
    
    plt.figure(figsize=figsize)
    plt.plot(iteration_values, marker='o', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if show_grid:
        plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_results(y_true, y_pred, title="Results", save_path=None):
    # Create the scatter plot
    plt.figure(figsize=(4,4))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Add a diagonal line representing perfect predictions
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # Set labels and title
    plt.xlabel('Actual ESOL Values')
    plt.ylabel('Predicted ESOL Values')
    plt.title(title)

    # Adjust layout and display the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def compute_r2(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    y_true, y_pred = [], []

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move to device (CPU/GPU)
            outputs = model(inputs)  # Get model predictions
            y_true.extend(targets.cpu().numpy())  # Store ground truth
            y_pred.extend(outputs.cpu().numpy())  # Store predictions
    
    # Compute R^2 score
    return r2_score(y_true, y_pred), y_true, y_pred

def plot_loss_contours(average_array, title, vmin=None, vmax=None, scale="linear", save_path=None):
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(1, 1)

    log_data = np.log10(average_array)

    if vmin is None:
        vmin = np.min(log_data)
    if vmax is None:
        vmax = np.max(log_data)

    levels = np.linspace(vmin, vmax, 50)

    # Choose scaling method
    if scale == "log":
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:  # Default to linear scale
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Contour plot with chosen normalization
    contour = ax.contourf(log_data, levels=levels, cmap='jet', norm=norm, extend='both')

    plt.title(title)
    ax.axis('square')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')

    # Add colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label(f"Log10 Loss ({scale} scale)")

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
