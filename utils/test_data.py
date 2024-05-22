import torch
from dataset import WeatherDataset

train_dataset = WeatherDataset("data/train", 30)
test_dataset = WeatherDataset("data/test", 30)

nan_found = False
for i in range(len(train_dataset)):
    X, y = train_dataset[i]
    if torch.isnan(X).any() or torch.isnan(y).any():
        print(f"NaN found in sample {i}")
        print(f"X: {X}")
        print(f"y: {y}")
        nan_found = True
if not nan_found:
    print("No NaN values found in the dataset.")

nan_found = False
for i in range(len(test_dataset)):
    X, y = test_dataset[i]
    if torch.isnan(X).any() or torch.isnan(y).any():
        print(f"NaN found in sample {i}")
        print(f"X: {X}")
        print(f"y: {y}")
        nan_found = True
if not nan_found:
    print("No NaN values found in the dataset.")