import pathlib
import torch 
import numpy as np
from sklearn.datasets import fetch_openml 

def load_mnist(device): 
    "Loads mnist dataset into Tensors on given device"
    script_directory = pathlib.Path(__file__).parent
    folder_path = pathlib.Path(script_directory) / "datasets"
    feature_path = folder_path / "mnist_features.csv"
    label_path = folder_path / "mnist_labels.csv"

    if feature_path.exists() and label_path.exists():
        X=torch.load(feature_path).to(device)
        y=torch.load(label_path).to(device)
    else: 
        mnist = fetch_openml('mnist_784', version=1)
        X = torch.from_numpy(mnist["data"].to_numpy().astype(np.float32)).to(device)
        y = torch.from_numpy(mnist["target"].to_numpy().astype(int)).to(device)

        torch.save(X, feature_path)
        torch.save(y, label_path)
    return X, y

def load_cifar(device): 
    "Loads mnist dataset into Tensors on given device"
    script_directory = pathlib.Path(__file__).parent
    folder_path = pathlib.Path(script_directory) / "datasets"
    feature_path = folder_path / "cifar_features.csv"
    label_path = folder_path / "cifar_labels.csv"

    if feature_path.exists() and label_path.exists():
        X=torch.load(feature_path).to(device)
        y=torch.load(label_path).to(device)
    else: 
        mnist = fetch_openml('CIFAR_10', version=1)
        X = torch.from_numpy(mnist["data"].to_numpy().astype(np.float32)).to(device)
        y = torch.from_numpy(mnist["target"].to_numpy().astype(int)).to(device)

        torch.save(X, feature_path)
        torch.save(y, label_path)
    return X, y