import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import plotly.graph_objs as go
from plotly.subplots import make_subplots

def view_data(DATA_FOLDER: str="./data"):
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

def plot_loss_curves(results):
    """
    Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "test_loss": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(9, 9))

    # Plot loss
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()

def plot_model_performance(
    model: nn.Module,
    dataset: Dataset
):
    """
    Plot the performance of the model by comparing actual temperatures with predicted temperatures.

    Args:
        model: Trained model for temperature prediction.
        dataset: Testing dataset containing input features and actual temperatures.

    Returns:
        None
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():  # Disable gradient calculation during inference
        for X, y in dataset:
            # Add batch dimension to X
            X = X.to("cuda").unsqueeze(0)

            # Forward pass
            pred = model(X)

            # Append prediction to the list
            predictions.extend(pred.squeeze().cpu().numpy())  # Convert to numpy array

    # Get the actual temperatures from the test dataset
    actual_temperatures = []

    for _, y in dataset:
        actual_temperatures.extend(y.squeeze().cpu().numpy())  # Convert to numpy array

    actual_temperatures = np.array(actual_temperatures)
    predictions = np.array(predictions).flatten()  # Flatten the predictions array

    fig = make_subplots()

    fig.add_trace(go.Scatter(
        x=list(range(len(actual_temperatures))),
        y=actual_temperatures,
        mode='lines',
        name='Actual Temperatures'
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(predictions))),
        y=predictions,
        mode='lines',
        name='Predicted Temperatures'
    ))

    fig.update_layout(
        title='Actual vs. Predicted Temperatures',
        xaxis_title='Time',
        yaxis_title='Temperature',
        legend=dict(x=0, y=1, traceorder='normal'),
        width=800,
        height=600
    )

    return fig