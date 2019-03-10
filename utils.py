# Dataset Utils
from random import shuffle
import numpy as np
import h5py
from bitarray import bitarray
import os
import errno

CATEGORIES = ['Pseudo-Random', 'Random']


def open_bin_file(path):
    a = bitarray()
    with open(path, 'rb') as fh:
        a.fromfile(fh)
        return a


def generate_dataset(random_raw_path, pseudorandom_raw_path, out_path, num_bits, dataset_len):
    print("Generating Dataset...")
    dataset_len_per_type = int(dataset_len/2)
    random_arr = open_bin_file(random_raw_path)
    pseudorandom_arr = open_bin_file(pseudorandom_raw_path)

    if len(random_arr) < num_bits * dataset_len_per_type:
        raise Exception("The random source file is not long enough")
    if len(pseudorandom_arr) < num_bits * dataset_len_per_type:
        raise Exception("The pseudo-random source file is not long enough")

    random_arr = random_arr[:int(num_bits * dataset_len_per_type)]
    pseudorandom_arr = pseudorandom_arr[:int(num_bits * dataset_len_per_type)]

    # cast to numpy array
    random_arr = np.array(random_arr)
    pseudorandom_arr = np.array(pseudorandom_arr)

    # reshape to dataset format
    random_arr = random_arr.reshape(dataset_len_per_type, num_bits)
    pseudorandom_arr = pseudorandom_arr.reshape(dataset_len_per_type, num_bits)

    training_data = []

    # add target value
    for data in pseudorandom_arr:
        training_data.append(
            [data.tolist(), CATEGORIES.index('Pseudo-Random')])

    for data in random_arr:
        training_data.append([data.tolist(), CATEGORIES.index('Random')])

    # shuffle the training data
    shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    if not os.path.exists(os.path.dirname(out_path)):
        try:
            os.makedirs(os.path.dirname(out_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    dset = h5py.File(name=out_path, mode='w')
    print('Saving X (Data)...')
    dset['data'] = X
    print('Saving y (Targets)...')
    dset['targets'] = y
    dset.close()
    print(f'Done. Saved at {out_path}')
    return out_path


def generate_mixed_dataset(random_raw_path, pseudorandom_raw_paths, out_path, num_bits, dataset_len):
    print("Generating Mixed Dataset...")
    random_dataset_len = int(dataset_len/2)
    pseudo_random_dataset_len = int(
        (dataset_len - random_dataset_len)/len(pseudorandom_raw_paths))
    random_arr = open_bin_file(random_raw_path)
    random_arr = random_arr[:int(num_bits * random_dataset_len)]
    random_arr = np.array(random_arr)
    random_arr = random_arr.reshape(random_dataset_len, num_bits)

    pseudorandom_arrs = []
    for pseudorandom_raw_path in pseudorandom_raw_paths:
        pseudorandom_arr = open_bin_file(pseudorandom_raw_path)
        pseudorandom_arr = pseudorandom_arr[:int(
            num_bits * pseudo_random_dataset_len)]
        pseudorandom_arr = np.array(pseudorandom_arr)
        pseudorandom_arr = pseudorandom_arr.reshape(
            pseudo_random_dataset_len, num_bits)
        pseudorandom_arrs.append(pseudorandom_arr)

    training_data = []

    print(f'random len: {len(random_arr)}')
    for data in random_arr:
        training_data.append([data.tolist(), CATEGORIES.index('Random')])

    for dataset in pseudorandom_arrs:
        print(f'pseudo random len: {len(dataset)}')
        for data in dataset:
            training_data.append(
                [data.tolist(), CATEGORIES.index('Pseudo-Random')])

    shuffle(training_data)

    print(f'dataset len: {len(training_data)}')

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    dset = h5py.File(name=out_path, mode='w')
    print('Saving X (Data)...')
    dset['data'] = X
    print('Saving y (Targets)...')
    dset['targets'] = y
    dset.close()
    print(f'Done. Saved at {out_path}')


def load_dataset(path, validation_split):
    print(f'Loading Dataset: {path} ...')

    dset = h5py.File(name=path, mode='r')
    data = dset['data']
    targets = dset['targets']

    size_test_data = int(len(data) * validation_split)
    x_test = data[:size_test_data]
    y_test = targets[:size_test_data]
    x_train = data[size_test_data:]
    y_train = targets[size_test_data:]

    dset.close()

    return (x_train, y_train), (x_test, y_test)
