from pyfsr import LFSR
from generate_source_files import generate_pr_sequence
from utils import generate_dataset, load_dataset
import ann_models
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from sp800_22_tests import test_sequence
import numpy as np

# define the polys we want to test
polys = [
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

# defines the length of the training data (and thus the input size of the ANN)
input_size = 64

# the length of the dataset
# dataset_len * input_size can't be larger than the sum of the binary sequences length
dataset_len = 3000000 / input_size

for poly in polys:
    lfsr = LFSR(poly=poly, initstate="random", initcycles=2**9)
    bin_path = generate_pr_sequence(lfsr, 1500000)

    dset_path = generate_dataset(
        random_raw_path=f'./binary_sequences/r.bin',
        pseudorandom_raw_path=bin_path,
        out_path=f'./datasets/{lfsr}.h5',
        num_bits=input_size,
        dataset_len=dataset_len)

    train_data, validation_data = load_dataset(dset_path, 0.25)
    (x_train, y_train) = train_data
    # generate network types
    (model, model_name) = ann_models.get_fully_connected_model(
        input_shape=x_train.shape[1:], data_name=str(lfsr))

    # compile the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    # train the model
    results = model.fit(
        x=x_train,
        y=y_train,
        validation_data=validation_data,
        epochs=10,
        batch_size=256,
        callbacks=[TensorBoard(log_dir=f'tensorboard_logs/{str(lfsr)}')],
    )

    (x_test, y_test) = validation_data
    evaluation = model.evaluate(x_test, y_test, 100)
    test_loss = round(evaluation[0], 4)
    test_acc = round(evaluation[1], 4)
    print("LOSS:", test_loss)
    print("ACC:", test_acc)

    nist_results = test_sequence(bin_path)

    success_count = 0
    for nist_result in nist_results:
        (test_name, p_val, success) = nist_result
        if success:
            success_count += 1
    f = open("results.csv", "a+")
    f.write(
        f'{str(lfsr)};{test_loss};{test_acc};{success_count}/{len(nist_results)}\n')
    f.close()
