from torch import nn
from util.layer import AdaptiveConcatPool1d


class WangFCNTorch(nn.Module):
    def __init__(self, channel_width, channels, output_size, batch_size):
        super(WangFCNTorch, self).__init__()
        self.batch_size = batch_size
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=128, kernel_size=8, padding=(3,)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=(2,)),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=(1,)),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )
        self.head = nn.Sequential(
            AdaptiveConcatPool1d(),
            nn.Flatten(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.stack(x)
        #re = torch.reshape(x, (-1, 128))
        return self.head(x)

