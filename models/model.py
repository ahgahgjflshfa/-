import torch
from torch import nn

class WeatherPredictModel(nn.Module):
    def __init__(self, input_size, hidden_unit, num_layers, output_size):
        super().__init__()
        self.hidden_unit = hidden_unit
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_unit,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_unit, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_unit).to(device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_unit).to(device=x.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out