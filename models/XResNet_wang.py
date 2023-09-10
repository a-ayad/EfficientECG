from torch import nn
from util.layer import AdaptiveConcatPool1d

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, size, stride, process_downsample=False):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=size, kernel_size=5,stride=stride, padding=(2,))
        self.batch_norm1 = nn.BatchNorm1d(size)
        self.act_fun1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=size, out_channels=size, kernel_size=3,padding=(1,))
        self.batch_norm2 = nn.BatchNorm1d(size)
        self.downsample = None
        if process_downsample:
            self.downsample = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=(2,))
            self.downnorm = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.act_fun1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        if self.downsample is not None:
            skip = self.downsample(skip)
            skip = self.downnorm(skip)
        x += skip
        return self.act_fun1(x)

class Output(nn.Module):
    def __init__(self, output_size):
        super(Output,self).__init__()
        self.head = nn.Sequential(
            AdaptiveConcatPool1d(),
            nn.Flatten(),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.25),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.5),
            nn.Linear(128, output_size)

        )

    def forward(self, x):
        return self.head(x)


class XResNet(nn.Module):

    def __init__(self, in_channels, output_size):
        super(XResNet, self).__init__()
        self.conv0 = nn.Conv1d(12, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        self.batch_norm0 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = nn.ReLU()
        self.block1 = ResnetBlock(128, 128, stride=(1,))
        self.block2 = ResnetBlock(128, 128, stride=(2,), process_downsample=True)
        self.block3 = ResnetBlock(128, 128, stride=(2,), process_downsample=True)
        self.output_net = Output(output_size)

    def forward(self, x):
        x = self.conv0(x)
        x = self.batch_norm0(x)
        x = self.relu0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output_net(x)
        return x


