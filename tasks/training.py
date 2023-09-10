import torch.nn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from models import FCN_wang as wangFCNtorch, LSTM_singh as singh_lstmTorch, GRU_singh as singh_gruTorch, \
    XResNet_wang as xResNetTorch, TCN as TCNTorch
from util import datahandler as dh


def train(config):
    print("Training routing started!")
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
    if config.architecture == 'TCN5':
        model = TCNTorch.Small_TCN_5()
    if config.architecture == 'TCN4':
        model = TCNTorch.Small_TCN_4(specification['num_classes'])
    if config.architecture == 'TCN3':
        model = TCNTorch.Small_TCN_3()
    if model is None:
        exit(0)
    if config.log:
        wandb.init(project='ECG_Training', name=config.experiment)
        wandb.watch(model, log_freq=100)
    print(model)
    train_dataset = dh.EcgTorchDataset(config,
                                       specification,
                                       'train')
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)

    val_dataset = dh.EcgTorchDataset(config,
                                     specification,
                                     'val')
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)

    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_vloss = 100000

    for epoch in range(config.epochs):
        # Training
        model.train(True)

        running_loss = 0
        i = 0
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for data in pbar:
            pbar.set_description('EPOCH {}'.format(epoch + 1))
            i = i + 1
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / i})
        pbar.clear()
        pbar.close()
        avg_loss = running_loss / len(train_dataloader)
        model.train(False)
        # Validation
        running_vloss = 0

        pbar = tqdm(val_dataloader, total=len(val_dataloader))
        i = 0
        for (inputs, labels) in pbar:
            i = i + 1
            pbar.set_description('EPOCH {} validation'.format(epoch + 1))
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_vloss += loss.item()
            pbar.set_postfix({'vloss': running_vloss / i})
        pbar.clear()
        pbar.close()
        avg_vloss = running_vloss / len(val_dataloader)
        if config.log:
            wandb.log({'loss': avg_loss,
                       'val_loss': avg_vloss})
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(),
                       ('train/' + config.experiment + '/' + config.architecture) + '/torch_model')