# Preprocess data for ESOl dataset and create cluster labels
from torch_geometric.datasets import MoleculeNet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from src.lib import utils_2


if __name__ == "__main__":
    config = utils_2.load_config("configs/train_config.yaml")

    # Extract training parameters
    test_size_for_clustering = config["training"]["test_size_for_clustering"]


    # Extract data paths
    raw_data_dir = config["data"]["raw_data_dir"]
    clustered_data_path = config["data"]["clustered_data_path"]
    dbscan_clustering_plot_path = config["data"]["dbscan_clustering_plot_path"]

    # Extract clustering parameters
    pca_n_components = config["clustering"]["pca_n_components"]
    dbscan_epsilon = config["clustering"]["dbscan_epsilon"]
    dbscan_min_samples = config["clustering"]["dbscan_min_samples"]

    # Extract model parameters
    model_save_path = config["model"]["save_path"]

    print(f"Model will be saved to: {model_save_path}")

    # Load ESOL dataset
    dataset = MoleculeNet(root=raw_data_dir, name="ESOL")

    # Extract features and targets
    smiles_list = [data.smiles for data in dataset]
    fingerprints = np.array([utils_2.mol_to_fp(smiles) for smiles in smiles_list])
    targets = np.array([data.y.item() for data in dataset])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(fingerprints, targets, test_size=test_size_for_clustering, random_state=42)

    # PCA
    pca = PCA(n_components=pca_n_components)
    pca.fit(fingerprints)
    transformed_data = pca.transform(fingerprints)

    # DBSCAN clustering
    epsilon = dbscan_epsilon 
    min_samples = dbscan_min_samples  
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = dbscan.fit_predict(transformed_data)

    utils_2.plot_dbscan_clustering_results(transformed_data, clusters, save_path=dbscan_clustering_plot_path)

    utils_2.save_dataframe(clustered_data_path, 
                           smiles=smiles_list, 
                           fingerprints=fingerprints.tolist(), 
                           targets=targets, 
                           clusters=clusters)