from utils import generate_dataset, load_dataset
import h5py

OUT_DIR = "./datasets/"

NAME = "lfsr_poly_11-9"

random_raw_path = './binary_sequences/r_960000.bin'
pseudorandom_raw_path = "./binary_sequences/pr_lfsr_960000_poly_11-9_20190228-162006.bin"

# defines the length of the training data (and thus the input size of the ANN)
input_size = 22

# the length of the dataset
# dataset_len * input_size can't be larger than the sum of the binary sequences length
dataset_len = 1000000 / input_size

generate_dataset(
    random_raw_path=random_raw_path,
    pseudorandom_raw_path=pseudorandom_raw_path,
    out_path=f'{OUT_DIR}{NAME}.h5',
    num_bits=input_size,
    dataset_len=dataset_len)


dset = h5py.File(f'{OUT_DIR}{NAME}.h5', 'r')

print(list(dset.keys()))

data = dset["data"]
targets = dset["targets"]

print(len(data))
print(len(targets))

print(data[0])
print(targets[0])

dset.close()
