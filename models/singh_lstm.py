import torch
from torch import nn

from util.layer import AdaptiveConcatPoolRNN


class Singh_lstmTorch(nn.Module):

    def __init__(self, output_size):
        super(Singh_lstmTorch, self).__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=256, num_layers=2, dropout=0.4)
        self.linear = nn.Linear(250 * 256, 5)
        self.batch_norm = nn.BatchNorm1d(256)
        self.head = nn.Sequential(
            AdaptiveConcatPoolRNN(False),
            nn.BatchNorm1d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(in_features=768, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=128, out_features=5, bias=True)
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 0, 1)
        x = self.lstm(x)
        x = torch.transpose(x[0], 0, 1)
        x = torch.transpose(x, 1, 2)
        return self.head(x)
