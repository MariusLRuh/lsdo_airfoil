import torch 
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_1 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_3 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_4 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_5 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm_1(x, (h0, c0))
        out, _ = self.lstm_2(out, (h0, c0))
        out, _ = self.lstm_3(out, (h0, c0))
        out, _ = self.lstm_4(out, (h0, c0))
        out, _ = self.lstm_5(out, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out