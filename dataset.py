import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class WeatherDataset(Dataset):
    def __init__(self, root: str | Path, seq_length):
        self.paths = list(Path(root).glob("./*.csv"))
        self.sequence_length = seq_length
        self.datas = self.load_datas()
        self.X_scaler = self.prepare_scaler()

    def prepare_scaler(self) -> StandardScaler:
        scaler = StandardScaler()
        scaler.fit(self.datas)
        return scaler


    def load_data(self, index: int):
        # Load data from csv file
        path = self.paths[index]
        df = pd.read_csv(path)
        data = df.values

        # Transform data to torch.tensor
        data = torch.from_numpy(data.astype(np.float32))

        return data

    def load_datas(self):
        datas = []

        for i in range(len(self.paths)):
            datas.append(self.load_data(i))

        datas = torch.cat(datas, dim=0)

        return datas

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start_idx = max(idx - self.sequence_length, 0)

        # Check if idx is less than sequence_length
        if idx < self.sequence_length:
            # Ignore this index and move to the next one
            idx = self.sequence_length

        X = self.datas[start_idx * 24: idx * 24]

        # Prevent index out of range error
        if len(X) == 0:
            X = torch.zeros(self.sequence_length * 24, 22)

        elif len(X) < self.sequence_length * 24:
            padding_size = self.sequence_length * 24 - len(X)
            padding = torch.zeros(padding_size, 22) # Create a zero tensor with the size of the (padding, in_feature).
            X = torch.cat([padding, X], dim=0)

        X = torch.from_numpy(self.X_scaler.transform(X).astype(np.float32))

        y = self.datas[idx * 24 : idx * 24 + 24][:, 1].unsqueeze(1)

        return X, y

    def __iter__(self):
        self._index = self.sequence_length  # Index is the day index not paths' index
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration

        X, y = self.__getitem__(self._index)
        self._index += 1

        return X, y