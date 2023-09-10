import torch
from torch import nn

class Singh_gruTorch(nn.Module):

    def __init__(self, output_size):
        super(Singh_gruTorch, self).__init__()
        self.gru = nn.GRU(input_size=12, hidden_size=256, num_layers=2, dropout=0.4)
        self.drop = nn.Dropout(0.4)
        self.linear = nn.Linear(250*256, 5)
        self.batch_norm = nn.BatchNorm1d(256)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        ls = self.gru(x)
        ls = self.drop(ls[0])
        ls = torch.transpose(ls, 1, 2)
        ls = self.batch_norm(ls)
        ls = torch.reshape(ls, (-1, 250*256))
        ls = self.linear(ls)
        return ls