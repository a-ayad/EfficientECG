import torch
from torch import nn

##### From Fast.ai
class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class ResnetBlock(nn.Module):

    def __init__(self,in_channels, size, stride, downsample=False):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=size, kernel_size=5,stride=stride, padding=(2,))
        self.batch_norm1 = nn.BatchNorm1d(size)
        self.act_fun1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=size, out_channels=size, kernel_size=3,padding=(1,))
        self.batch_norm2 = nn.BatchNorm1d(size)
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=(2,))
            self.downnorm = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #self.downsample = downsample


    def forward(self, input):
        skip = input
        input = self.conv1(input)
        input = self.batch_norm1(input)
        input = self.act_fun1(input)
        input = self.conv2(input)
        input = self.batch_norm2(input)
        if self.downsample is not None:
            skip = self.downsample(skip)
            skip = self.downnorm(skip)
        input += skip
        return self.act_fun1(input)
        #return torch.add(input, skip)

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

    def forward(self, input):
        return self.head(input)


class XResNet(nn.Module):

    def __init__(self, in_channels, output_size):
        super(XResNet, self).__init__()
        self.conv0 = nn.Conv1d(12, 128, kernel_size=(7,), stride=(1,), padding=(3,), bias=False)
        self.batch_norm0 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = nn.ReLU()
        self.block1 = ResnetBlock(128, 128, stride=(1,))
        self.block2 = ResnetBlock(128, 128,stride=(2,),downsample=True)
        self.block3 = ResnetBlock(128, 128,stride=(2,),downsample=True)
        self.output_net = Output(output_size)

    def forward(self, input):
        input = self.conv0(input)
        input = self.batch_norm0(input)
        input = self.relu0(input)

        input = self.block1(input)




        input = self.block2(input)
        ### Use Pooling Layer here




        input = self.block3(input)


        input = self.output_net(input)
        return input


