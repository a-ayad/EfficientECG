# *---------------------------------------------------------------------------------------*
# * This code is adapted from Thorir Mar Infolgsson from the paper                        *
# * ECG-TCN: Wearable Cardiac Arrhythmia Detection with a Temporal Convolutional Network  *
# * DOI:  10.1109/AICAS51828.2021.9458520                                                 *
# * https://github.com/pulp-platform/ecg-tcn/blob/main/utils/model_maker.py               *
# *---------------------------------------------------------------------------------------*

import nemo
from torch import nn as nn

class Small_TCN_5(nn.Module):
    def __init__(self):
        super(Small_TCN_5, self).__init__()
        n_inputs = 12
        # Hyperparameters for TCN
        Kt = 19
        pt = 0.3
        Ft = 11
        classes = 5

        self.pad0 = nn.ConstantPad1d(padding=(Kt - 1, 0), value=0)
        self.conv0 = nn.Conv1d(in_channels=n_inputs, out_channels=n_inputs + 1, kernel_size=19, bias=False)
        self.act0 = nn.ReLU()
        self.batchnorm0 = nn.BatchNorm1d(num_features=n_inputs + 1)

        # First block
        dilation = 1
        self.upsample = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=1, bias=False)
        self.upsamplerelu = nn.ReLU()
        self.upsamplebn = nn.BatchNorm1d(num_features=Ft)
        self.pad1 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv1 = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=Kt, dilation=1, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=Ft)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=pt)
        self.pad2 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv2 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=1, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features=Ft)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=pt)
        self.add1 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd1 = nn.ReLU()

        # Second block
        dilation = 2
        self.pad3 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv3 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_features=Ft)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=pt)
        self.pad4 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv4 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(num_features=Ft)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=pt)
        self.add2 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd2 = nn.ReLU()

        # Third block
        dilation = 4
        self.pad5 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv5 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm5 = nn.BatchNorm1d(num_features=Ft)
        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=pt)
        self.pad6 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv6 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm6 = nn.BatchNorm1d(num_features=Ft)
        self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=pt)
        self.add3 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd3 = nn.ReLU()

        # fourth block
        dilation = 8
        self.pad7 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv7 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm7 = nn.BatchNorm1d(num_features=Ft)
        self.act7 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=pt)
        self.pad8 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv8 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm8 = nn.BatchNorm1d(num_features=Ft)
        self.act8 = nn.ReLU()
        self.dropout8 = nn.Dropout(p=pt)
        self.add4 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd4 = nn.ReLU()

        # fifth block
        dilation = 16
        self.pad9 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv9 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm9 = nn.BatchNorm1d(num_features=Ft)
        self.act9 = nn.ReLU()
        self.dropout9 = nn.Dropout(p=pt)
        self.pad10 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv10 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm10 = nn.BatchNorm1d(num_features=Ft)
        self.act10 = nn.ReLU()
        self.dropout10 = nn.Dropout(p=pt)
        self.add5 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd5 = nn.ReLU()

        # Last layer
        self.linear = nn.Linear(in_features=Ft * 250, out_features=classes, bias=False)

    def forward(self, x):
        # Now we propagate through the network correctly
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)

        # TCN
        # First block
        res = self.pad1(x)
        res = self.conv1(res)
        res = self.batchnorm1(res)
        res = self.act1(res)
        res = self.dropout1(res)
        res = self.pad2(res)
        res = self.conv2(res)
        res = self.batchnorm2(res)
        res = self.act2(res)
        res = self.dropout2(res)

        x = self.upsample(x)
        x = self.upsamplebn(x)
        x = self.upsamplerelu(x)

        x = self.add1(x, res)
        x = self.reluadd1(x)

        # Second block
        res = self.pad3(x)
        # res = self.pad3(res)
        res = self.conv3(res)
        res = self.batchnorm3(res)
        res = self.act3(res)
        res = self.dropout3(res)
        res = self.pad4(res)
        res = self.conv4(res)
        res = self.batchnorm4(res)
        res = self.act4(res)
        res = self.dropout4(res)
        x = self.add2(x, res)
        x = self.reluadd2(x)

        # Third block
        res = self.pad5(x)
        # res = self.pad5(res)
        res = self.conv5(res)
        res = self.batchnorm5(res)
        res = self.act5(res)
        res = self.dropout5(res)
        res = self.pad6(res)
        res = self.conv6(res)
        res = self.batchnorm6(res)
        res = self.act6(res)
        res = self.dropout6(res)
        x = self.add3(x, res)
        x = self.reluadd3(x)


        # Linear layer to classify
        x = x.flatten(1)
        o = self.linear(x)
        return o  # Return directly without softmax

class Small_TCN_4(nn.Module):
    def __init__(self, classes):
        super(Small_TCN_4, self).__init__()
        n_inputs = 12
        # Hyperparameters for TCN
        Kt = 9
        pt = 0.3
        Ft = 11

        self.pad0 = nn.ConstantPad1d(padding=(Kt - 1, 0), value=0)
        self.conv0 = nn.Conv1d(in_channels=n_inputs, out_channels=n_inputs + 1, kernel_size=9, bias=False)
        self.act0 = nn.ReLU()
        self.batchnorm0 = nn.BatchNorm1d(num_features=n_inputs + 1)

        # First block
        dilation = 1
        self.upsample = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=1, bias=False)
        self.upsamplerelu = nn.ReLU()
        self.upsamplebn = nn.BatchNorm1d(num_features=Ft)
        self.pad1 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv1 = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=Kt, dilation=1, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=Ft)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=pt)
        self.pad2 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv2 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=1, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features=Ft)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=pt)
        self.add1 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd1 = nn.ReLU()

        # Second block
        dilation = 2
        self.pad3 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv3 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_features=Ft)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=pt)
        self.pad4 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv4 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(num_features=Ft)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=pt)
        self.add2 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd2 = nn.ReLU()

        # Third block
        dilation = 4
        self.pad5 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv5 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm5 = nn.BatchNorm1d(num_features=Ft)
        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=pt)
        self.pad6 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv6 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm6 = nn.BatchNorm1d(num_features=Ft)
        self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=pt)
        self.add3 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd3 = nn.ReLU()

        # fourth block
        dilation = 8
        self.pad7 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv7 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm7 = nn.BatchNorm1d(num_features=Ft)
        self.act7 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=pt)
        self.pad8 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv8 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm8 = nn.BatchNorm1d(num_features=Ft)
        self.act8 = nn.ReLU()
        self.dropout8 = nn.Dropout(p=pt)
        self.add4 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd4 = nn.ReLU()

        # fifth block
        dilation = 16
        self.pad9 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv9 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm9 = nn.BatchNorm1d(num_features=Ft)
        self.act9 = nn.ReLU()
        self.dropout9 = nn.Dropout(p=pt)
        self.pad10 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv10 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm10 = nn.BatchNorm1d(num_features=Ft)
        self.act10 = nn.ReLU()
        self.dropout10 = nn.Dropout(p=pt)
        self.add5 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd5 = nn.ReLU()

        # Last layer
        self.linear = nn.Linear(in_features=Ft * 250, out_features=classes, bias=False)

    def forward(self, x):
        # Now we propagate through the network correctly
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)

        # TCN
        # First block
        res = self.pad1(x)
        res = self.conv1(res)
        res = self.batchnorm1(res)
        res = self.act1(res)
        res = self.dropout1(res)
        res = self.pad2(res)
        res = self.conv2(res)
        res = self.batchnorm2(res)
        res = self.act2(res)
        res = self.dropout2(res)

        x = self.upsample(x)
        x = self.upsamplebn(x)
        x = self.upsamplerelu(x)

        x = self.add1(x, res)
        x = self.reluadd1(x)

        # Second block
        res = self.pad3(x)
        # res = self.pad3(res)
        res = self.conv3(res)
        res = self.batchnorm3(res)
        res = self.act3(res)
        res = self.dropout3(res)
        res = self.pad4(res)
        res = self.conv4(res)
        res = self.batchnorm4(res)
        res = self.act4(res)
        res = self.dropout4(res)
        x = self.add2(x, res)
        x = self.reluadd2(x)

        # Third block
        res = self.pad5(x)
        # res = self.pad5(res)
        res = self.conv5(res)
        res = self.batchnorm5(res)
        res = self.act5(res)
        res = self.dropout5(res)
        res = self.pad6(res)
        res = self.conv6(res)
        res = self.batchnorm6(res)
        res = self.act6(res)
        res = self.dropout6(res)
        x = self.add3(x, res)
        x = self.reluadd3(x)

        # Fourth block
        res = self.pad7(x)
        # res = self.pad5(res)
        res = self.conv7(res)
        res = self.batchnorm7(res)
        res = self.act7(res)
        res = self.dropout7(res)
        res = self.pad8(res)
        res = self.conv8(res)
        res = self.batchnorm8(res)
        res = self.act8(res)
        res = self.dropout8(res)
        x = self.add4(x, res)
        x = self.reluadd4(x)

        # Linear layer to classify
        x = x.flatten(1)
        o = self.linear(x)
        return o  # Return directly without softmax


class Small_TCN_3(nn.Module):
    def __init__(self):
        super(Small_TCN_3, self).__init__()
        n_inputs = 12
        # Hyperparameters for TCN
        Kt = 11
        pt = 0.3
        Ft = 11
        classes = 5

        self.pad0 = nn.ConstantPad1d(padding=(Kt - 1, 0), value=0)
        self.conv0 = nn.Conv1d(in_channels=n_inputs, out_channels=n_inputs + 1, kernel_size=11, bias=False)
        self.act0 = nn.ReLU()
        self.batchnorm0 = nn.BatchNorm1d(num_features=n_inputs + 1)

        # First block
        dilation = 1
        self.upsample = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=1, bias=False)
        self.upsamplerelu = nn.ReLU()
        self.upsamplebn = nn.BatchNorm1d(num_features=Ft)
        self.pad1 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv1 = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=Kt, dilation=1, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=Ft)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=pt)
        self.pad2 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv2 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=1, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features=Ft)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=pt)
        self.add1 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd1 = nn.ReLU()

        # Second block
        dilation = 2
        self.pad3 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv3 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_features=Ft)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=pt)
        self.pad4 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv4 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(num_features=Ft)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=pt)
        self.add2 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd2 = nn.ReLU()

        # Third block
        dilation = 4
        self.pad5 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv5 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm5 = nn.BatchNorm1d(num_features=Ft)
        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=pt)
        self.pad6 = nn.ConstantPad1d(padding=((Kt - 1) * dilation, 0), value=0)
        self.conv6 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm6 = nn.BatchNorm1d(num_features=Ft)
        self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=pt)
        self.add3 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd3 = nn.ReLU()

        # Last layer
        self.linear = nn.Linear(in_features=Ft * 250, out_features=classes, bias=False)

    def forward(self, x):
        # Now we propagate through the network correctly
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)

        # TCN
        # First block
        res = self.pad1(x)
        res = self.conv1(res)
        res = self.batchnorm1(res)
        res = self.act1(res)
        res = self.dropout1(res)
        res = self.pad2(res)
        res = self.conv2(res)
        res = self.batchnorm2(res)
        res = self.act2(res)
        res = self.dropout2(res)

        x = self.upsample(x)
        x = self.upsamplebn(x)
        x = self.upsamplerelu(x)

        x = self.add1(x, res)
        x = self.reluadd1(x)

        # Second block
        res = self.pad3(x)
        # res = self.pad3(res)
        res = self.conv3(res)
        res = self.batchnorm3(res)
        res = self.act3(res)
        res = self.dropout3(res)
        res = self.pad4(res)
        res = self.conv4(res)
        res = self.batchnorm4(res)
        res = self.act4(res)
        res = self.dropout4(res)
        x = self.add2(x, res)
        x = self.reluadd2(x)

        # Second block
        res = self.pad5(x)
        # res = self.pad5(res)
        res = self.conv5(res)
        res = self.batchnorm5(res)
        res = self.act5(res)
        res = self.dropout5(res)
        res = self.pad6(res)
        res = self.conv6(res)
        res = self.batchnorm6(res)
        res = self.act6(res)
        res = self.dropout6(res)
        x = self.add3(x, res)
        x = self.reluadd3(x)

        # Linear layer to classify
        x = x.flatten(1)
        o = self.linear(x)
        return o  # Return directly without softmax
