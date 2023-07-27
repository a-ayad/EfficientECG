# Main Script for our ECG Classification
import argparse
import os
import sys

import datasets.PTBXL as ptbxl
# from tasks.benchmark_remote import rpc_benchmark
from tasks.testing import test_model # export_model
from tasks.training import train
# from tasks.feature_importance import  get_feature_importance


def create_config():
    parser = argparse.ArgumentParser(description="Test suite for our "
                                                 "implementation of "
                                                 "different ECG "
                                                 "Classification Algorithms")
    parser.add_argument('--experiment', type=str, metavar='exp',
                        default='default_name', help='Name of Experiment')
    train_argument = parser.add_argument('--train', action='store_true',
                                         default=False,
                                         help='Set this flag for training')
    parser.add_argument('--test', action='store_true',
                        default=False, help='Set this flag for testing')
    parser.add_argument('--export_tvm', action='store_true',
                        default=False, help='Set this flag to export Model ')
    parser.add_argument('--feature_importance', action='store_true',
                        default=False, help='Set this flag to get Feature Importance ')
    parser.add_argument('--run_remotely', action='store_true',
                        default=False, help='Set this flag to export Model ')
    parser.add_argument('--preprocess', action='store_true',
                        default=False, help='Set this flag to preprocess data')
    parser.add_argument('--debug', action='store_true',
                        default=False, help='Set this flag to preprocess data')
    parser.add_argument('--batch_size', type=int, metavar='Batch_Size',
                        default=2, help="Batch size for training")
    parser.add_argument('--epochs', type=int, metavar='epochs',
                        default=1, help='Number of Epochs for training')
    parser.add_argument('--architecture', type=str, metavar='Arch',
                        choices=['wangFCN', 'xResNet', 'TCN',
                                 'singh_LSTM', 'singh_GRU',
                                 'SmallTCN', 'TCN4'])
    parser.add_argument('--dataset', type=str, metavar='dataset',
                        choices=['PTB-XL'], default='PTB-XL')
    parser.add_argument('--ip', type=str, metavar='dataset')
    parser.add_argument('--learning_rate', type=float, metavar='LR',
                        default=0.001, help='Initial Learning Rate')
    parser.add_argument('--optimizer', type=str, metavar='Opt',
                        default='Adam', choices=['Adam'])
    parser.add_argument('--target', type=str, metavar='Opt',
                        choices=['host', 'jetson'])
    parser.add_argument('--overwrite', action='store_true',
                        default=False, help='Ignore existing checkpoints')
    parser.add_argument('--log', action='store_true',
                        default=False, help='Log to W and b')
    try:
        config = parser.parse_args()
        if sum([config.train, config.test,
                config.preprocess, config.export_tvm,
                config.run_remotely, config.feature_importance]) != 1:
            raise argparse.ArgumentError(argument=train_argument,
                                         message="Only one Flag must be set.")
        if (config.train and config.architecture is None):
            raise argparse.ArgumentError(argument=train_argument,
                                         message="Architecuture "
                                                 "needs to be specified "
                                                 "for Training")
        return config

    except argparse.ArgumentError as err:
        print("An error while parsing your arguments happened", err)


def init(config):
    try:
        if not config.preprocess:
            if not os.path.exists(('data/preprocess/' +
                                   config.experiment + '/')
                                  + config.dataset):
                raise Exception('Please download the Dataset,'
                                ' place it under train/'
                                + config.dataset + ' and preprocess '
                                                   'data first')
        if config.preprocess:
            if len(os.listdir('data/' + config.dataset + '/')) == 0:
                raise Exception('Please download the Dataset'
                                ' and place it under /data/'
                                + config.dataset)
    except Exception as err:
        print(err)
        sys.exit(1)
    if config.train:
        if not os.path.exists('train'):
            os.makedirs('train')
        if not os.path.exists('train/' + config.experiment):
            os.makedirs('train/' + config.experiment)
        if not os.path.exists('train/' + config.experiment + '/' + config.architecture):
            os.makedirs('train/' + config.experiment + '/' + config.architecture)
            if not os.path.exists('train/' + config.experiment + '/' + config.architecture +
                                '/checkpoints'):
                os.makedirs('train/' + config.experiment + '/' + config.architecture + '/checkpoints')
    if config.test or config.export_tvm:
        try:
            if not os.path.exists('train/' + config.experiment):
                raise Exception('Experiment not under /train/' +
                                config.experiment)
        except Exception as err:
            print(err)
            sys.exit(1)

    if config.preprocess:
        experiment_and_dataset = (config.experiment + '/') + config.dataset
        if not os.path.exists('data/preprocess/' + config.experiment):
            os.makedirs('data/preprocess/' + config.experiment)
        if not os.path.exists('data/preprocess/' + experiment_and_dataset):
            os.makedirs('data/preprocess/' + experiment_and_dataset)
        if not os.path.exists('data/preprocess/'
                              + experiment_and_dataset + '/train'):
            os.makedirs('data/preprocess/' + experiment_and_dataset + '/train')
        if not os.path.exists('data/preprocess/'
                              + experiment_and_dataset + '/val'):
            os.makedirs('data/preprocess/' + experiment_and_dataset + '/val')
        if not os.path.exists('data/preprocess/'
                              + experiment_and_dataset + '/test'):
            os.makedirs('data/preprocess/' + experiment_and_dataset + '/test')


def main():
    print("Test Script for Efficient ECG "
          "Heartbeat Classification")
    config = create_config()
    init(config)
    if config.preprocess:
        preprocess(config)
    if config.train:
        train(config)
    if config.test:
        test_model(config)
    if config.export_tvm:
        export_model(config)
    if config.run_remotely:
        rpc_benchmark(config)
    if config.feature_importance:
        get_feature_importance(config)


def preprocess(config):
    if config.dataset == 'PTB-XL':
        print("Preproceesing PTB-XL")
        ptbxl.process(config)


if __name__ == '__main__':
    main()
