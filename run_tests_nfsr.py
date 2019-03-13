from decimal import Decimal, ROUND_UP
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from pyfsr import NLFSR, FSRFunction
from sp800_22_tests import test_sequence
from generate_source_files import generate_pr_sequence
from utils import generate_dataset, load_dataset
import ann_models


def round_float(flt):
    '''
    round_float(flt)
    round(flt) and {:.4f}.format(flt) don't always perform as expected
    '''
    return Decimal(str(flt)).quantize(Decimal('.0001'), rounding=ROUND_UP)


def gen_dset_train_model(sequence_path, dset_name):
    train_data, validation_data = load_dataset(
        generate_dataset(
            random_raw_path=f'./binary_sequences/r.bin',
            pseudorandom_raw_path=sequence_path,
            out_path=f'./datasets/{dset_name}.h5',
            num_bits=input_size,
            dataset_len=dataset_len), 0.25)
    (x_train, y_train) = train_data
    (x_test, y_test) = validation_data
    # generate network types
    (model, _) = ann_models.get_fully_connected_model(
        input_shape=x_train.shape[1:], data_name=dset_name)

    # compile the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    # train the model
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=validation_data,
        epochs=10,
        batch_size=256,
        callbacks=[TensorBoard(log_dir=f'tensorboard_logs/{dset_name}')],
    )
    return model.evaluate(x_test, y_test, 100)


def log_test_run(model_evaluation, nist_results, fsr_name):
    loss = round_float(model_evaluation[0])
    acc = round_float(model_evaluation[1])
    f = open("results.csv", "a+")
    nistresultstr = ""
    success_count = 0
    if nist_results != 'NULL':
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


def run_test_round(fsr):
    sequence_path = generate_pr_sequence(fsr, 2000000)
    # sequence_path = f'./binary_sequences/{str(fsr)}.bin'
    evaluation = gen_dset_train_model(sequence_path, str(fsr))
    nist_results = test_sequence(sequence_path)
    log_test_run(evaluation, nist_results, str(fsr))


if __name__ == '__main__':

    nfsrs = [
        NLFSR(initstate="random", size=16, infunc=FSRFunction(
            [0, 7, 8, 10, 13, "+", "+", "+", "+"])),
        NLFSR(initstate="random", size=16, infunc=FSRFunction(
            [0, 7, "+", 8, 10, "+", 13, "+", "+"])),
        NLFSR(initstate="random", size=20, infunc=FSRFunction(
            [0, 7, 8, 10, 13, "+", "+", "+", "+"])),
        NLFSR(initstate="random", size=20, infunc=FSRFunction(
            [0, 7, "+", 8, 10, "+", 13, "+", "+"])),
        NLFSR(initstate="random", size=20, infunc=FSRFunction(
            [0, 7, "*", 8, 10, "*", 0, "+", "+"])),
        # source: http://www5.rz.rub.de:8032/imperia/md/content/wolf/szmidt_asp.pdf p. 22
        NLFSR(initstate="random", size=25, infunc=FSRFunction(
            [0, 8, 9, 10, 11, 19, 20, 21, 23, "+", "+", "+", "+",
             "+", "+", "+", "+", 6, 21, "*", "+", 10, 14, "*", "+",
             12, 20, "*", "+", 19, 20, "*", "+", 4, 18, 21, "*", "*", "+",
             11, 18, 22, "*", "*", "+", 1, 5, 7, 23, "*", "*", "*", "+"]))
    ]

    # define the length of one training data block (the input size of the ANN)
    input_size = 64

    # the length of the dataset
    # dataset_len * input_size can't be larger than the sum of the binary sequences length
    dataset_len = 4000000 / input_size

    for nfsr in nfsrs:
        run_test_round(nfsr)
