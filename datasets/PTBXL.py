import math

import numpy as np

import util.datahandler as dh
import json
import csv
import wfdb
import wfdb.processing as wfdb_process
from tqdm import tqdm
import os
import tensorflow as tf
import numpy as np
import scipy.stats as stats
import random


def load_json():
    """
    Opens data.json
    :return: Serialized JSON
    """
    f = open('data/PTB-XL/data.json')
    return json.load(f)


class PTBXLSample(dh.Sample):
    def __init__(self, ecg_features, class_as_number, sampling_rate):
        """
        Subclass from datahandler.Sample to hold a PTBXL Sample.
        :param ecg_features: ECG features as dict. This can hold >= 1 leads.
        :param class_as_number: Class of Sample
        :param sampling_rate: Sampling Rate of Sample
        """
        features = {
            "ECG": dh.float_feature(ecg_features),
            "CLASS": dh.int_feature(class_as_number),
            "SMPL_RATE": dh.int_feature(sampling_rate)
        }
        # features.update(ecg_features)
        super().__init__(features)


def get_samplerate(specification):
    """
    Retrieve Samplerate from specification
    :param specification: Dict for specification
    :return: sample rate
    """
    return specification['sample_rate']


def resample(data, specification):
    rate = get_samplerate(specification)
    y = np.array([np.array(wfdb_process.resample_sig(data[:, x], 500, rate)[0])
                  for x in range(data.shape[1])])
    return np.transpose(y)


def normalize(data, specification):
    if specification['normalization'] == 'z-score':
        return stats.zscore(data, axis=0)
    else:
        return data


def generate_split(specification):
    """
    Split data into train, test and validation subsets. Based on Specification
    :param specification: Specification dict that states the split ratio
    :return: train, test and val splits as list of integers
    """
    if specification['random_split'] == "True":
        permutation = np.random.permutation(np.arange(21837))
    else:
        permutation = np.arange(21837)
    for splits in specification['split']:
        split = specification['split'][splits]
        if splits == 'train':
            train_split = split
        if splits == 'test':
            test_split = split
        if splits == 'val':
            val_split = split
    if train_split + test_split + val_split != 100:
        raise Exception('Inconsistent split data in data.json')
    no_samples_train = math.floor(train_split / 100 * 21837)
    no_samples_val = math.floor(val_split / 100 * 21837)
    return permutation[0:no_samples_train], \
        permutation[no_samples_train + 1:no_samples_train + no_samples_val], \
        permutation[no_samples_train + no_samples_val::]


def get_class(specification, annotations):
    """
    Get the class of annotaions according to specification
    :param specification: Specification as dict
    :param annotations: Data read from the database file as string
    :return: Class Number and Class Name, otherwise None,None
    """
    annotations = json.loads(annotations.replace("'", "\""))
    class_vec = np.zeros(len(specification['classes']), dtype=np.int64)
    class_names = []
    for classes_from_spec in specification['classes']:
        class_name = next(iter(specification['classes'][classes_from_spec]))
        for sub_class in \
                specification['classes'][classes_from_spec][class_name]:
            if sub_class in annotations:
                class_vec[int(classes_from_spec)] = 1
                class_names += [class_name]
    return class_vec, class_names


def get_ecg_data(specification, filename):
    """
    Retrieve ECG Leads, according to specifictaion from file
    :param specification: Dict, that specifies which Leads to get
    :param filename: Filename of ECG_data
    :return:
    """
    filepath = "data/PTB-XL/"
    channels = []
    for ecg_channel in specification['ecg_channels']:
        channel_name = specification['ecg_channels'][ecg_channel]
        if channel_name == 'I':
            channels += [0]
        if channel_name == 'II':
            channels += [1]
        if channel_name == 'III':
            channels += [2]
        if channel_name == 'AVR':
            channels += [3]
        if channel_name == 'AVl':
            channels += [4]
        if channel_name == 'AVF':
            channels += [5]
        if channel_name == 'V1':
            channels += [6]
        if channel_name == 'V2':
            channels += [7]
        if channel_name == 'V3':
            channels += [8]
        if channel_name == 'V4':
            channels += [9]
        if channel_name == 'V5':
            channels += [10]
        if channel_name == 'V6':
            channels += [11]
    record = wfdb.rdrecord(filepath + filename,
                           channels=channels)
    return record.p_signal, record.sig_name


def process(config):
    """
    Main Routine to Process PTB_XL
    """
    specification = load_json()
    train_split, val_split, test_split = generate_split(specification)
    train_samples, val_samples, test_samples = [], [], []
    no_train_files, no_val_files, no_test_files = 0, 0, 0
    rows = []
    channel_width = 0
    num_mulitsamples_train = int(specification['multisamples'])
    sample_lenght_frac_train = specification['sample_length']
    num_mulitsamples_test = int(specification['multisamples_test'])
    sample_lenght_frac_test = specification['sample_length_test']
    feature_counter = {}
    with open('data/PTB-XL/ptbxl_database.csv', newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        next(reader, None)
        for row in reader:
            rows += [row]

    for i in tqdm(np.arange(21837)):
        # row[0]: ecg_id , row[1]: patient_id , row[11]: scp_codes
        # row[27]: filename_hr
        row = rows[i]
        class_vec, class_names = get_class(specification, row[11])
        if not class_names:
            continue
        ecg_data, leads = get_ecg_data(specification, row[27])
        sample_rate = get_samplerate(specification)
        ecg_data = resample(ecg_data, specification)
        channel_width = ecg_data.shape[0]
        ecg_data = normalize(ecg_data, specification)
        sample_length_train = int(channel_width * sample_lenght_frac_train)
        sample_length_test = int(channel_width * sample_lenght_frac_test)
        if np.any(np.isnan(ecg_data)):
            continue
        if int(row[0]) in train_split:
            for j in range(0, num_mulitsamples_train):
                start = random.randint(0, channel_width - sample_length_train)
                data = ecg_data[start:(start + sample_length_train)]
                train_samples += [PTBXLSample(data.reshape(-1),
                                              class_vec,
                                              sample_rate)]
        if int(row[0]) in val_split:
            for j in range(0, num_mulitsamples_train):
                start = random.randint(0, channel_width - sample_length_train)
                data = ecg_data[start:(start + sample_length_train)]
                val_samples += [PTBXLSample(data.reshape(-1),
                                            class_vec,
                                            sample_rate)]
        if int(row[0]) in test_split:
            for j in range(0, num_mulitsamples_test):
                start = random.randint(0, channel_width - sample_length_test)
                data = ecg_data[start:(start + sample_length_test)]
                test_samples += [PTBXLSample(data.reshape(-1),
                                             class_vec,
                                             sample_rate)]
        if len(train_samples) % (1000 * num_mulitsamples_train) == 0 and\
                not len(train_samples) == 0:
            filename = (("data/preprocess/" + config.experiment)
                        + "/PTB-XL/train/train_" +
                        str(no_train_files)
                        + ".tfrecord")
            dh.write_record_samples(train_samples, filename)
            train_samples = []
            no_train_files += 1
        if len(val_samples) % (1000 * num_mulitsamples_train) == 0 and\
                not len(val_samples) == 0:
            filename = (("data/preprocess/" + config.experiment)
                        + "/PTB-XL/val/val_" +
                        str(no_val_files)
                        + ".tfrecord")
            dh.write_record_samples(val_samples, filename)
            val_samples = []
            no_val_files += 1
        if len(test_samples) % (1000 * num_mulitsamples_test) == 0 and\
                not len(test_samples) == 0:
            filename = (("data/preprocess/" + config.experiment)
                        + "/PTB-XL/test/test_" +
                        str(no_test_files)
                        + ".tfrecord")
            dh.write_record_samples(test_samples, filename)
            test_samples = []
            no_test_files += 1
    filename = (("data/preprocess/" + config.experiment)
                + "/PTB-XL/train/train_"
                + str(no_train_files)
                + ".tfrecord")
    dh.write_record_samples(train_samples, filename)
    filename = (("data/preprocess/" + config.experiment)
                + "/PTB-XL/val/val_"
                + str(no_val_files)
                + ".tfrecord")
    dh.write_record_samples(val_samples, filename)
    filename = (("data/preprocess/" + config.experiment)
                + "/PTB-XL/test/test_"
                + str(no_test_files)
                + ".tfrecord")
    dh.write_record_samples(test_samples, filename)
    specification['num_samples_train'] = \
        no_train_files * (1000 * num_mulitsamples_train) + len(train_samples)
    specification['num_samples_test'] = \
        no_test_files * (1000 * num_mulitsamples_test) + len(test_samples)
    specification['num_samples_val'] = \
        no_val_files * (1000 * num_mulitsamples_train) + len(val_samples)
    specification['channel_width'] = int(channel_width *
                                         specification['sample_length'])
    specification['channel_width_test'] = int(channel_width *
                                              specification['sample_length_test'])
    dh.write_specification(specification, config)



