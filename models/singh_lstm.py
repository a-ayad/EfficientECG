import torch
from torch import nn


class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional

    def forward(self, x):
        # input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)

        if (self.bidirectional is False):
            t3 = x[:, :, -1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:, :channels, -1], x[:, channels:, 0]], 1)
        out = torch.cat([t1.squeeze(-1), t2.squeeze(-1), t3], 1)  # output shape bs, 3*ch
        return out

class Singh_lstmTorch(nn.Module):

    def __init__(self, output_size):
        super(Singh_lstmTorch, self).__init__()
        # self.model = nn.Sequential(

        #        nn.LSTM(64, 256), #64
        #        nn.Dropout(0.2),
        #        nn.LSTM(256, 100), #256
        #        nn.Dropout(0.2),
        #        nn.LSTM(100, 100), # 100
        #   nn.Dropout(0.2),
        #        nn.Linear(100,5)

        # )
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

    def forward(self, input):
        input = torch.transpose(input, 1 ,2)
        input = torch.transpose(input, 0, 1)
        input = self.lstm(input)
        input = torch.transpose(input[0], 0, 1)
        input = torch.transpose(input, 1, 2)
        return self.head(input)
