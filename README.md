# SSL_SL_ECG
This is the repository containing the code for this paper "Efficient and Private ECG Classification System Using Split and Semi-Supervised learning"



### - TCN5-Split-ECG Classification: https://github.com/a-ayad/Split_ECG_Classification
### - Semi-supervised Learning algorithms: https://github.com/TorchSSL/TorchSSL
### - Time-series Augmentation: https://github.com/uchidalab/time_series_augmentation
### - Performance calculations: https://icl.utk.edu/papi/


This is a reposiitory that includese all the code related to the experiments developed by B.Sc Völker, Benedikt <benedikt.voelker@rwth-aachen.de> for his master thesis

# Usage of run.py

--experiment: the name of the current experiment. \
default: default_name

--dataset: the dataset to use. \
default: PTB-XL

## Preprocessing

--preprocess: flag for preproccessing data 

## Training

--train: flag for training the model. \
model will be stored under /train/{experiment}/checkpoints/

--architecture: Architecture to train model \
NEEDS to be set! Options are: wangFCN

--batch_size: Batch size for training. Will be used also for testing in training process. \
default: 2

--epochs: Number of epochs to train. \
default: 1

--learning_rate: initial Learing Rate \
default: 0.001

--optimizer: Optimizer to change learning rate \
default: Adam

--overwrite: Flag for ignore exsisting checkpoints \
Will overwrite old models with the same experiment name

## Testing

--test: Set this flag to test the trained model.
The model in /train/{experiment}/checkpoints/ is used
