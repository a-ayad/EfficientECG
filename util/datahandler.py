# Data handling utilities
from typing import Iterator

import numpy
import tensorflow as tf
import numpy as np
import json
import os
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co


def write_record_samples(samples: list, filename: str) -> None:
    """
        Takes a list of samples, serializes them and writes them to a file
    :param samples: List of samples
    :param filename: Filename with file location
    """
    with tf.io.TFRecordWriter(filename) as writer:
        for sample in samples:
            serialized = sample.serialize()
            writer.write(serialized)


def read_record(filename):
    """
        Read s specified dataset from disk
    :rtype: object
    """
    raw_dataset = tf.data.TFRecordDataset(filename)
    sampleList = []
    for raw_record in raw_dataset.take(-1):
        features = {}
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        for key, feature in example.features.feature.items():
            features[key] = getattr(feature, feature
                                    .WhichOneof('kind')).value[0]
        sampleList += [Sample(features)]
    return sampleList


def read_record_dataset(filenames):
    """
    Takes a List of Filenames and reads them into Dataset
    :param filenames: List of Filenmaes
    :return: tf.data.Dataset
    """
    return tf.data.TFRecordDataset(filenames=filenames)


class Sample:

    def __init__(self, features):
        """
        Object to hold a Sample. Superclass to load Sample from file
        :rtype: object
        """
        self.features = features

    def serialize(self):
        """
        Serializes the self.features attribute to a tfrecord compatible string
        :return:
        """
        return tf.train.Example(features=tf.train
                                .Features(feature=self.features)) \
            .SerializeToString()


def byte_feature(feature: bytes or str) -> tf.train.Feature:
    """
        Casts a Byte or String to a tf.train.Feature Byte
    :param feature: String or Byte
    :return: tf.train.Feature Byte
    """
    if isinstance(feature, type(tf.constant(0))):
        feature = feature.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))


def int_feature(feature: bool or int) -> tf.train.Feature:
    """
        Casts a Bool or Int to a tf.train.Feature Int64
    :param feature: Bool or Int
    :return: tf.train.Feature Int64
    """
    if type(feature) == int:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[feature]))
    if type(feature) == np.ndarray:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=feature))


def float_feature(feature: float or np.ndarray) -> tf.train.Feature:
    """
        Casts a float to a tf.train.Feature Float
        :param feature: Float
        :return: tf.train.Feature Float
        """
    if type(feature) == float:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[feature]))
    if type(feature) == np.ndarray:
        return tf.train.Feature(float_list=tf.train.FloatList(value=feature))


def write_specification(specification, config):
    """
    Writes specifiaction for a dataset
    :param specification:
    :param config:
    """
    with open((('data/preprocess/' + config.experiment) + '/' + config.dataset)
              + '/data.json', 'w') as fp:
        json.dump(specification, fp)


def load_specification(config):
    """
    Loads a specification for the dataset
    :param config:
    :return:
    """
    if not config.preprocess:
        filename = (('data/preprocess/' + config.experiment) + '/'
                    + config.dataset) \
                   + '/data.json'
    else:
        filename = ('data/' + config.dataset) + '/data.json'
    f = open(filename)
    return json.load(f)


class dataset_generator:
    def __init__(self, config, specification, split):
        """
        Creates a Generator that is iterable to load dataset
        :param config:
        :param split:
        """
        self.specification = specification
        if split == 'train':
            files = os.listdir((('data/preprocess/' + config.experiment)
                                + '/' + config.dataset)
                               + '/train')
            files = [((('data/preprocess/' + config.experiment)
                       + '/' + config.dataset) + '/train/')
                     + s for s in files]
            self.size = self.specification['num_samples_train']
        if split == 'val':
            files = os.listdir((('data/preprocess/' + config.experiment)
                                + '/' + config.dataset)
                               + '/val')
            files = [((('data/preprocess/' + config.experiment)
                       + '/' + config.dataset) + '/val/')
                     + s for s in files]
            self.size = self.specification['num_samples_val']
        if split == 'test':
            files = os.listdir((('data/preprocess/' + config.experiment)
                                + '/' + config.dataset)
                               + '/test')
            files = [((('data/preprocess/' + config.experiment)
                       + '/' + config.dataset) + '/test/')
                     + s for s in files]
            self.size = self.specification['num_samples_test']
        self.dataset = read_record_dataset(files)
        self.iter_obj = iter(self.dataset)
        self.batch_size = config.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        samples = []
        classes = []
        for i in range(self.batch_size):
            try:
                raw_record = next(self.iter_obj)
                sample = tf.io.parse_single_example(
                    # Data
                    raw_record,

                    # Schema
                    {"CLASS": tf.io.FixedLenFeature(
                        [self.specification['num_classes']], dtype=tf.int64),
                        "SMPL_RATE": tf.io.FixedLenFeature([], dtype=tf.int64),
                        "ECG": tf.io.FixedLenFeature(
                            [self.specification['num_channels'] *
                             self.specification['channel_width']],
                            dtype=tf.float32)}
                )
                ecg = sample['ECG']
                samples += [tf.reshape(ecg,
                                       (self.specification['channel_width'],
                                        self.specification['num_channels']))]
                classes += [sample['CLASS']]
            except StopIteration:
                self.iter_obj = iter(self.dataset)
        return tf.stack(samples), tf.stack(classes)


class EcgTorchDataset(IterableDataset):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, config, specification, split):
        super(EcgTorchDataset).__init__()
        self.specification = specification
        if split == 'train':
            files = os.listdir((('data/preprocess/' + config.experiment)
                                + '/' + config.dataset)
                               + '/train')
            files = [((('data/preprocess/' + config.experiment)
                       + '/' + config.dataset) + '/train/')
                     + s for s in files]
            self.size = self.specification['num_samples_train']
            self.channel_width = self.specification['channel_width']
        if split == 'val':
            files = os.listdir((('data/preprocess/' + config.experiment)
                                + '/' + config.dataset)
                               + '/val')
            files = [((('data/preprocess/' + config.experiment)
                       + '/' + config.dataset) + '/val/')
                     + s for s in files]
            self.size = self.specification['num_samples_val']
            self.channel_width = self.specification['channel_width']
        if split == 'test':
            files = os.listdir((('data/preprocess/' + config.experiment)
                                + '/' + config.dataset)
                               + '/test')
            files = [((('data/preprocess/' + config.experiment)
                       + '/' + config.dataset) + '/test/')
                     + s for s in files]
            self.size = self.specification['num_samples_test']
            self.channel_width = self.specification['channel_width_test']
        self.dataset = read_record_dataset(files)
        self.iter_obj = iter(self.dataset)
        self.batch_size = config.batch_size

    def __iter__(self) -> Iterator[T_co]:
        return self

    def __len__(self):
        return self.size

    def __next__(self):
        try:
            raw_record = next(self.iter_obj)
            sample = tf.io.parse_single_example(
                # Data
                raw_record,
                # Schema
                {"CLASS": tf.io.FixedLenFeature(
                    [self.specification['num_classes']], dtype=tf.int64),
                    "SMPL_RATE": tf.io.FixedLenFeature([], dtype=tf.int64),
                    "ECG": tf.io.FixedLenFeature(
                        [self.specification['num_channels'] *
                         self.channel_width],
                        dtype=tf.float32)}
            )
            ecg = sample['ECG']

            ecg = tf.reshape(ecg,
                             (self.channel_width,
                              self.specification['num_channels']))
            sample_class = sample['CLASS']
        except StopIteration:
            self.iter_obj = iter(self.dataset)
            raise StopIteration

        return torch.from_numpy(ecg.numpy().transpose()), torch.from_numpy(
            sample_class.numpy().astype(numpy.float32))
