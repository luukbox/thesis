from decimal import Decimal, ROUND_UP
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from pyfsr import LFSR, FSRFunction
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
            out_path=f'./datasets/{lfsr}.h5',
            num_bits=input_size,
            dataset_len=dataset_len), 0.25)
    (x_train, y_train) = train_data
    (x_test, y_test) = validation_data
    # generate network types
    (model, _) = ann_models.get_fully_connected_model(
        input_shape=x_train.shape[1:], data_name=str(lfsr))

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
        callbacks=[TensorBoard(log_dir=f'tensorboard_logs/{str(lfsr)}')],
    )
    return model.evaluate(x_test, y_test, 100)


def log_test_run(model_evaluation, nist_results):
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
    logstr = f'\n{str(lfsr)},{loss},{acc},{success_count}{nistresultstr}'
    f.write(logstr)
    f.close()


def run_test_round(fsr):
    sequence_path = generate_pr_sequence(lfsr, 2000000)
    sequence_path = f'./binary_sequences/{str(lfsr)}.bin'
    evaluation = gen_dset_train_model(sequence_path, str(lfsr))
    nist_results = test_sequence(sequence_path)
    log_test_run(evaluation, 'NULL')


if __name__ == '__main__':
    # the primitive polys we want to test
    primitive_polys = [
        [15, 14],
        [16, 15, 13, 4],
        [17, 14],
        [18, 11],
        [19, 18, 17, 14],
        [20, 17],
        [21, 19],
        [22, 21],
        [23, 18],
        [24, 23, 22, 17]
    ]

    # what happens when different out functions are applied to polys with interesting results?
    outfunc_polys = [
        [23, 18],
        [24, 23, 22, 17]
    ]

    # the outfunctions we want to apply to the polys in question
    outfuncs = [
        FSRFunction([22, 17, "+", 9, "+", 0, "+"]),
        FSRFunction([17, 9, "*", 0, "+"]),
        FSRFunction([22, 11, 9, 3, 0, "+", "+", "+", "+"]),
        FSRFunction([22, 0, "+"]),
        FSRFunction([22, 0, "*"])
    ]

    # define the length of one training data block (the input size of the ANN)
    input_size = 64

    # the length of the dataset
    # dataset_len * input_size can't be larger than the sum of the binary sequences length
    dataset_len = 4000000 / input_size

    for poly in primitive_polys:
        for feedback in ("external", "internal"):
            if poly in outfunc_polys:
                for outfunc in outfuncs:
                    lfsr = LFSR(poly=poly, initstate="random", feedback=feedback,
                                initcycles=2**9, outfunc=outfunc)
                    run_test_round(lfsr)
            else:
                lfsr = LFSR(poly=poly, initstate="random", feedback=feedback,
                            initcycles=2**9)
                run_test_round(lfsr)
