import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size1,
                 hidden_size2,
                 hidden_size3,
                 num_layers,
                 output_size):
        super(LSTMModel, self).__init__()

        # Define the input size, hidden size, and number of layers
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.num_layers = num_layers

        # Define the LSTM layers with fixed sizes
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size1,
                             num_layers=num_layers,
                             batch_first=True)  # First LSTM layer
        self.lstm2 = nn.LSTM(input_size=hidden_size1,
                             hidden_size=hidden_size2,
                             num_layers=num_layers,
                             batch_first=True) # Second LSTM layer
        self.lstm3 = nn.LSTM(input_size=hidden_size2,
                             hidden_size=hidden_size3,
                             num_layers=num_layers,
                             batch_first=True) # Third LSTM layer

        # Define the fully connected output layer (Dense layer)
        self.fc = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        # Initialize hidden and cell states for each layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(x.device)
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(x.device)
        c1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(x.device)
        h2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size3).to(x.device)
        c2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size3).to(x.device)

        # Forward propagate LSTM
        x1, _ = self.lstm1(x, (h0.detach(), c0.detach()))
        x2, _ = self.lstm2(x1, (h1.detach(), c1.detach()))
        out, _ = self.lstm3(x2, (h2.detach(), c2.detach()))

        # Pass the output of the last LSTM layer to the fully connected layer
        out = self.fc(out[:, -1, :])
        return out