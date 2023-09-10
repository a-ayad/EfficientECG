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


class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
