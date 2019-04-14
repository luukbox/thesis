import ann_models
from utils import generate_dataset, load_dataset
from generate_source_files import save_sequence_as_binary
from sp800_22_tests import test_sequence
from pyfsr import LFSR, NLFSR, FSRFunction
import numpy as np
from decimal import Decimal, ROUND_UP
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from tqdm import tqdm


def round_float(flt):
    '''
    round_float(flt)
    round(flt) and {:.4f}.format(flt) don't always perform as expected
    '''
    return Decimal(str(flt)).quantize(Decimal('.0001'), rounding=ROUND_UP)


def gen_dset_train_model(sequence_path, dset_name, dataset_len, input_size):
    train_data, test_data = load_dataset(
        generate_dataset(
            random_raw_path=f'./binary_sequences/r.bin',
            pseudorandom_raw_path=sequence_path,
            out_path=f'./datasets/{dset_name}.h5',
            num_bits=input_size,
            dataset_len=dataset_len), 0.3)
    (x_train, y_train) = train_data

    # generate network types
    # (model, _) = ann_models.get_fully_connected_model(
    #     input_shape=x_train.shape[1:], data_name=dset_name)

    (model, _) = ann_models.get_convolutional_model(
        input_shape=x_train.shape[1:], data_name=dset_name)

    # compile the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    (x_test, y_test) = test_data
    # train the model
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=test_data,
        epochs=3,
        batch_size=32,
        callbacks=[TensorBoard(log_dir=f'tensorboard_logs/{dset_name}')],
    )
    return model.evaluate(x_test, y_test, 100)


def log_test_run(model_evaluation, nist_results, fsr_name):
    loss = round_float(model_evaluation[0])
    acc = round_float(model_evaluation[1])
    f = open("results.csv", "a+")
    nistresultstr = ""
    success_count = 0
    if not nist_results is None:
        for nist_result in nist_results:
            (_, __, success) = nist_result
            if success:
                success_count += 1
                nistresultstr += ",1"
            else:
                nistresultstr += ",0"
    logstr = f'\n{fsr_name},{loss},{acc},{success_count}{nistresultstr}'
    f.write(logstr)
    f.close()


def run_test_round(sequence, dataset_len, input_size, sequence_name):
    if sequence is None:
        sequence_path = f'./binary_sequences/{sequence_name}.bin'
        evaluation = gen_dset_train_model(
            sequence_path, sequence_name, dataset_len, input_size)
        # nist_results = test_sequence(sequence_path)
        log_test_run(evaluation, None, sequence_name)
    else:
        if len(sequence) < int((dataset_len / 2) * input_size):
            raise Exception(
                f'The sequence you provided is too small. Expected length: {int((dataset_len/2)*input_size)} actual length: {len(sequence)}')
        sequence_path = f'./binary_sequences/{sequence_name}.bin'
        save_sequence_as_binary(
            sequence, sequence_path)
        print("Sequence saved at:", sequence_path)
        evaluation = gen_dset_train_model(
            sequence_path, sequence_name, dataset_len, input_size)
        nist_results = test_sequence(sequence_path)
        log_test_run(evaluation, nist_results, sequence_name)


if __name__ == '__main__':

    # define the length of one training data block (the input size of the ANN)
    input_size = 28**2
    # input_size = 256
    # the length of the dataset
    # dataset_len * input_size can't be larger than the sum of the binary sequences length
    dataset_len = 8000000 / input_size
    from advanced_ciphers.grain import simple_grain_sequence_v1, simple_grain_sequence_v2, simple_grain_sequence_v3, simple_grain_sequence_v4, grain_sequence_no_mask, grain_sequence
    from advanced_ciphers.a51 import simple_a51_sequence_v1, simple_a51_sequence_v2, simple_a51_sequence_v3, simple_a51_sequence_v4, a51_sequence
    from advanced_ciphers.e0 import e0_sequence, simple_e0_sequence_v1
    from advanced_ciphers.rc4 import rc4_sequence

    run_test_round(rc4_sequence(10000000),
                   dataset_len, input_size, "rc4")
    run_test_round(simple_e0_sequence_v1(10000000),
                   dataset_len, input_size, "simple_e0_v1")
    run_test_round(e0_sequence(10000000),
                   dataset_len, input_size, "e0")
    run_test_round(simple_a51_sequence_v1(10000000),
                   dataset_len, input_size, "simple_a51_v1")
    run_test_round(simple_a51_sequence_v2(10000000),
                   dataset_len, input_size, "simple_a51_v2")
    run_test_round(simple_a51_sequence_v3(10000000),
                   dataset_len, input_size, "simple_a51_v3")
    run_test_round(simple_a51_sequence_v4(10000000),
                   dataset_len,  input_size, "simple_a51_v4")
    run_test_round(a51_sequence(10000000),
                   dataset_len, input_size, "a51")
    run_test_round(simple_grain_sequence_v1(10000000),
                   dataset_len,  input_size, "simple_grain_v1")
    run_test_round(simple_grain_sequence_v2(10000000),
                   dataset_len,  input_size, "simple_grain_v2")
    run_test_round(simple_grain_sequence_v3(10000000),
                   dataset_len,  input_size, "simple_grain_v3")
    run_test_round(simple_grain_sequence_v4(7000000),
                   dataset_len,  input_size, "simple_grain_v4")
    run_test_round(grain_sequence_no_mask(10000000),
                   dataset_len,  input_size, "grain_sequence_no_mask")
    run_test_round(grain_sequence(10000000),
                   dataset_len, input_size, "grain")
