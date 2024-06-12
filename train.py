import os
import torch
from torch import nn
from tqdm import trange
from models.model import LSTMModel
from dataset import WeatherDataset
from torch.utils.data import DataLoader

import neptune
from neptune.types import File
from dotenv import load_dotenv

import matplotlib.pyplot as plt
from utils.plot_functions import plot_model_performance

import torch.profiler

def train_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu"
):
    """
    Performs a single training step on the given model using the given training data and hyperparameters.
    Args:
        model: The model to train.
        data_loader: The DataLoader object used for training.
        loss_fn: The loss function used for training.
        optimizer: The optimizer used for training.
        device (optional): The device to use for training and inference. Default is "cpu".

    Returns:

    """
    train_loss = 0

    # Put model into training mode
    model.train()

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('../log/lstm'),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        for batch, (X, y) in enumerate(data_loader):
            # prof.step() # Notify profiler of the start of the step

            # Put data on target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logits = model(X)

            # Calculate loss and accuracy (per batch)
            loss = loss_fn(y_logits.unsqueeze(2), y)
            train_loss += loss.item()   # Accumulate train loss

            run["train/batch/loss"].append(loss)

            # Optimizer zero grad
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

    # Calculate average loss and accuracy
    train_loss /= len(data_loader)

    return train_loss

def test_step(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: str = "cpu"
):
    """
    Performs a single testing step on the given model using the given testing data and hyperparameters.

    Args:
        model: The model to test.
        data_loader: The DataLoader object used for testing.
        loss_fn: The loss function used for testing.
        device (optional): The device to use for training and inference. Default is "cpu".

    Returns:

    """
    test_loss = 0
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logits = model(X)

            # Calculate loss and accuracy (per batch)
            loss = loss_fn(y_logits.unsqueeze(2), y)
            test_loss += loss.item()

            run['test/batch/loss'].append(loss)

        # Calculate average loss
        test_loss /= len(data_loader)

    return test_loss

def train(
    model: nn.Module,
    epochs: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str = "cpu"
) -> dict:
    """
    Trains the specified neural network model using the given training data and hyperparameters.

    Args:
        model: The neural network model to be trained.
        epochs (optional): The number of epochs (iterations over the entire training dataset) to train the model.
                           Default is 200.
        train_dataloader: The DataLoader object used for training.
        test_dataloader: The DataLoader object used for testing.
        loss_fn: The loss function used for training.
        optimizer: The optimizer used for training.
        device (optional): The device to use for training and inference. Default is "cpu".

    Returns:
        A dictionary containing all train loss, train accuracy, test loss and test accuracy while training the model.
    """
    results = {"train_loss": [], "test_loss": []}

    for epoch in trange(epochs):
        # Train step
        train_loss = train_step(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        # Test step
        test_loss = test_step(
            model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # scheduler.step(test_loss)

        if epoch % 10 == 0:
            ckp = model.state_dict()
            torch.save(ckp, f"../models/model.pth")
            print("Checkpoint saved.")

        # if epoch % 10 == 0:
        #     # progress_bar.update(1)
        #     print(f"\nEpoch: {epoch} | Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    return results

if __name__ == "__main__":

    load_dotenv()

    # Neptune Setup
    run = neptune.init_run(
        project="ahgahgjflshfa/SchoolProject",
        api_token=os.getenv("TOKEN"),
        dependencies="../requirements.txt"
    )

    params = {
        'sequence': 20,
        'bs': 64,
        'lr': 3e-4,
        'weight_decay': 1.5e-3,
        'input_sz': 22,
        'hid_sz1': 128,
        'hid_sz2': 512,
        'hid_sz3': 256,
        'n_layers': 1,
        'output_sz': 24
    }
    run["parameters"] = params

    run["train/datas"].track_files('../data/train')

    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = WeatherDataset('../data/train', params['sequence'])
    test_dataset = WeatherDataset('../data/test', params['sequence'])

    training_dataloader = DataLoader(train_dataset, batch_size=params['bs'], shuffle=True)
    testing_dataloader = DataLoader(test_dataset, batch_size=params['bs'], shuffle=False)

    model = LSTMModel(
        input_size=params['input_sz'],
        hidden_size1=params['hid_sz1'],
        hidden_size2=params['hid_sz2'],
        hidden_size3=params['hid_sz3'],
        num_layers=params['n_layers'],
        output_size=params['output_sz']
    ).to(device)


    # Prepare optimizer and loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    # Check if there is a saved model
    save_path = "../models/model.pth"

    try:
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print(f"Model parameters loaded from {save_path}")
    except RuntimeError as e:
        print(f"Saved model state dict miss match current model.")

    EPOCHS = 100

    results = train(
        model=model,
        epochs=EPOCHS,
        train_dataloader=training_dataloader,
        test_dataloader=testing_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    run["model/file"].upload(save_path)

    # Plot performance
    fig1 = plot_model_performance(model, train_dataset)
    run["model/train-performance"].upload(fig1)

    fig2 = plot_model_performance(model, test_dataset)
    run["model/test-performance"].upload(fig2)

    run.stop()

    # Save model
    torch.save(model.state_dict(), save_path)

    print(f"Model parameters saved to {save_path}")
