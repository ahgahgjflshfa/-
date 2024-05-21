import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def view_data(DATA_FOLDER: str="../data"):
    """
    Visualize temperature data from multiple CSV files stored in the specified data folder.

    Args:
        DATA_FOLDER (optional): The folder containing the CSV files to visualize. Defaults to "./data".

    Returns:
        None
    """

    dfs = []

    file_path = os.path.join(DATA_FOLDER, "data.csv")

    # skip first two rows and uses only 2nd column
    df = pd.read_csv(file_path, usecols=[5])    # usecols=[2] if choose to view download directory

    # add download to dataframe
    dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)

    df_combined['Temperature'] = pd.to_numeric(df_combined['Temperature'], errors='coerce')

    # df_combined = df_combined.drop('SeaPres', axis=1)

    df_combined.plot(xlabel='Time', ylabel='Temperature (°C)', figsize=(15, 8))

    print(len(df_combined))

    plt.show()

def plot_loss_curves(results: dict[str, list[float]]):
    """

    Args:
        results:

    Returns:

    """

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # accuracy = results["train_acc"]
    # test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    # plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # # Plot accuracy
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, accuracy, label="train_accuracy")
    # plt.plot(epochs, test_accuracy, label="test_accuracy")
    # plt.title("Accuracy")
    # plt.xlabel("Epochs")
    # plt.legend()

    plt.show()
def plot_model_performance(model: nn.Module,
                           dataset: Dataset):
    """

    Args:
        model:
        dataset: Testing dataset.

    Returns:

    """
    model.eval()

    predictions = []
    with torch.no_grad():
        for X, y in dataset:
            p = model(X).item()
            predictions.append(p)

    plot_len = len(predictions)
    print(predictions)