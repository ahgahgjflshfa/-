import torch
import numpy as np

def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the accuracy of predicted temperatures as a percentage.

    Args:
    predictions: Predicted temperature values (Tensor)
    labels: True temperature values (Tensor)

    Returns:
    accuracy: Prediction accuracy as a percentage
    """
    # Convert predictions and labels to numpy arrays
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # Calculate absolute error
    abs_error = np.abs(predictions - labels)

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(abs_error)

    # Calculate accuracy as a percentage
    accuracy = 100 - mae  # Assuming higher accuracy corresponds to lower MAE

    return accuracy

