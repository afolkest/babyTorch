import pathlib
import torch 
import numpy as np
import pandas as pd 
from sklearn.datasets import fetch_openml 

def load_mnist(device): 
    "Loads mnist (and downloads if necessary) dataset into Tensors on given device"
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
    "Loads CIFAR (and downloads if necessary) dataset into Tensors on given device"
    script_directory = pathlib.Path(__file__).parent
    folder_path = pathlib.Path(script_directory) / "datasets"
    feature_path = folder_path / "cifar_features.csv"
    label_path = folder_path / "cifar_labels.csv"

    if feature_path.exists() and label_path.exists():
        X=torch.load(feature_path).to(device)
        y=torch.load(label_path).to(device)
    else: 
        cifar = fetch_openml('CIFAR_10', version=1)
        X = torch.from_numpy(cifar["data"].to_numpy().astype(np.float32)).to(device)
        y = torch.from_numpy(cifar["target"].to_numpy().astype(int)).to(device)

        torch.save(X, feature_path)
        torch.save(y, label_path)
    return X, y

def load_bach(device="cpu"): 
    "Loads the Bach chorale dataset"
    script_directory = pathlib.Path(__file__).parent
    data_dir = script_directory / "datasets" / "jsb_chorales" 
    train_files = [i for i in data_dir.glob('train/*.*')]
    valid_files = [i for i in data_dir.glob('valid/*.*')]
    test_files = [i for i in data_dir.glob('test/*.*')] 

    test, val, train = [], [], []
    for file in train_files: 
        train.append(torch.tensor(pd.read_csv(file).values, dtype=torch.torch.uint8))
    for file in valid_files: 
        val.append(torch.tensor(pd.read_csv(file).values, dtype=torch.torch.uint8))
    for file in test_files: 
        test.append(torch.tensor(pd.read_csv(file).values, dtype=torch.torch.uint8))
    return train, val, test
