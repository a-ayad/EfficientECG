from time import time

import torch.jit

from torch.utils.data import DataLoader
from tqdm import tqdm


import wandb
from models import wangFCN as wangFCNtorch, singh_lstm as singh_lstmTorch, singh_gru as singh_gruTorch, \
    XResNet as xResNetTorch, TCNTorch as TCNTorch, SmallTCN as SmallTCN
from util import datahandler as dh

from torchmetrics.classification import Accuracy, F1Score, AUROC



def test_model(config):
    print("Testing routing started!")
    model = None
    specification = dh.load_specification(config)
    if config.architecture == 'wangFCN':
        model = wangFCNtorch.WangFCNTorch(specification['channel_width'],
                                          specification['num_channels'],
                                          specification['num_classes'],
                                          config.batch_size)
    if config.architecture == 'singh_LSTM':
        model = singh_lstmTorch.Singh_lstmTorch(5)
    if config.architecture == 'singh_GRU':
        model = singh_gruTorch.Singh_gruTorch(5)
    if config.architecture == 'xResNet':
        model = xResNetTorch.XResNet(specification['num_channels'],
                                     specification['num_classes'])
    if config.architecture == 'TCN':
        model = TCNTorch.Small_TCN_5()
    if config.architecture == 'TCN4':
        model = TCNTorch.Small_TCN_4(specification['num_classes'])
    if config.architecture == 'SmallTCN':
        model = SmallTCN.Small_TCN()
    if model is None:
        exit(0)

    test_dataset = dh.EcgTorchDataset(config,
                                      specification,
                                      'test')
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

    if config.log:
        wandb.init(project='ECG_Testing_2Classes', name=config.experiment)

    model_filepath = ('train/' + config.experiment + '/' + config.architecture) + '/torch_model'
    model.load_state_dict(torch.load(model_filepath))
    model.eval()
    test_accuracy = Accuracy("binary")
    test_f1 = F1Score("binary")
    test_auc = AUROC("binary",num_classes=specification['num_classes'])



    pbar = tqdm(test_dataloader, total=len(test_dataloader))
    i = 0
    delta_time = 0

    for (inputs, labels) in pbar:
        vec = torch.zeros((config.batch_size,12,1000))

        vec[0:0] = inputs[0:0]
        vec[0:5] = inputs[0:5]
        inputs = vec
        sample_window =  specification['channel_width']
        pbar.set_description('Testing ')
        i = i + 1
        start_time = time()
        output = torch.zeros(specification['num_classes'])
        start = 0
        stop = sample_window

        for y in range(0, int(specification['channel_width_test'] / specification['channel_width']) * 2 - 1):
            output = output + torch.sigmoid(model(inputs[:, :, start:stop]))
            start, stop = start + int(sample_window/2), stop + int(sample_window/2)
        delta_time += time() - start_time
        output = output / (int(specification['channel_width_test'] / specification['channel_width']) * 2 - 1)

        acc = test_accuracy(output.detach(), labels.int()).numpy()
        f1 = test_f1(output.detach(), labels.int()).numpy()
        auc = test_auc(output.detach(), labels.int()).numpy()
        pbar.set_postfix({'acc': acc, 'AUC': auc, 'F1': f1})
    pbar.clear()
    pbar.close()
    input = inputs[0:1, :, 0:250]

    total_test_acc = test_accuracy.compute()
    total_test_f1 = test_f1.compute()
    total_test_auc = test_auc.compute()
    if config.log:
        wandb.log({'acc': total_test_acc,
                   'f1': total_test_f1,
                   'AUC': total_test_auc,
                   'Time': delta_time / i})
    print('Acc: {}, F1: {}, AUC: {}'
          .format(total_test_acc, total_test_f1, total_test_auc))



