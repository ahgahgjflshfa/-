import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class WeatherDataset(Dataset):
    def __init__(self, root: str | Path, seq_length):
        self.path = root
        self.data = self.load_data()
        self.sequence_length = seq_length

    def load_data(self):
        # Load data from csv file
        df = pd.read_csv(self.path)

        # Preprocess data
        df['Month'] = df['Date'].str.split('-', expand=True)[1]
        df = pd.get_dummies(df, columns=['PrecpType', 'Month'])
        df = df.drop(columns=['Date'])
        data = df.values

        # Standardize data
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)

        # Transform data to torch.tensor
        data = torch.from_numpy(data.astype(np.float32))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start_idx = max(idx - self.sequence_length, 0)

        X = self.data[start_idx:idx]

        # Prevent index out of range error
        if len(X) < self.sequence_length:
            padding_size = self.sequence_length - len(X)
            padding = torch.zeros(padding_size, len(self.data[0]))
            X = torch.cat([padding, X], dim=0)

        y = self.data[idx][4].unsqueeze(0)

        return X, y