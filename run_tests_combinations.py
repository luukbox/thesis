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


def run_test_round(sequence):
    sequence_name = 'grain'
    sequence_path = f'./binary_sequences/{sequence_name}.bin'
    save_sequence_as_binary(
        sequence, sequence_path)
    print("Sequence saved at:", sequence_path)
    evaluation = gen_dset_train_model(sequence_path, sequence_name)
    nist_results = test_sequence(sequence_path)
    log_test_run(evaluation, nist_results, sequence_name)


if __name__ == '__main__':

    # define the length of one training data block (the input size of the ANN)
    input_size = 64

    # the length of the dataset
    # dataset_len * input_size can't be larger than the sum of the binary sequences length
    dataset_len = 4000000 / input_size

    lfsr = LFSR(poly=[80, 67, 57, 42, 29, 18], initstate="random")
    gx = FSRFunction([
        16, 19, 27, 34, 42, 46, 51, 58, 64, 70, 79, "+", "+", "+", "+", "+", "+", "+", "+", "+", "+",
        16, 19, "*", "+", 42, 46, "*", "+", 64, 70, "*", "+", 19, 27, 34, "*", "*", "+", 46, 51, 58, "*", "*", "+", 16, 34, 51, 70, "*", "*", "*", "+",
        19, 27, 42, 46, "*", "*", "*", "+", 16, 19, 58, 64, "*", "*", "*", "+", 16, 19, 27, 34, 42, "*", "*", "*", "*", "+", 46, 51, 58, 64, 70, "*", "*", "*", "*", "+",
        27, 34, 42, 46, 51, 58, "*", "*", "*", "*", "*", "+"
    ])
    nfsr = NLFSR(initstate="random", size=80, infunc=gx)
    hx = FSRFunction([1, 4, "+", 0, 3, "*", "+", 2, 3, "*", "+", 3, 4, "*", "+", 0, 1, 2, "*", "*", "+",
                      0, 2, 3, "*", "*", "+", 0, 2, 4, "*", "*", "+", 1, 2, 4, "*", "*", "+", 2, 3, 4, "*", "*", "+"])

    sequence_len = 2000000
    sequence = np.zeros(sequence_len)
    for i in tqdm(range(sequence_len), ascii=True, desc="Generating Grain Sequence"):
        lfsr_outbit = lfsr.shift()
        nfsr.shift()
        nfsr.state[0] = np.logical_xor(lfsr_outbit, nfsr.state[0])
        outtaps = [lfsr.state[3], lfsr.state[25],
                   lfsr.state[46], lfsr.state[64], nfsr.state[63]]
        cipherbit = hx.solve(outtaps)
        sequence[i] = cipherbit

    sequence_str = "".join(str(s) for s in sequence.astype(int))
    run_test_round(sequence_str)
