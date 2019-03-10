from utils import generate_dataset, load_dataset
import h5py

OUT_DIR = "./datasets/"

random_name = 'r'
pseudorandom_name = "lfsr_(18-11)_ext_out(17-14-9-6-0-xor-xor-xor-xor)"

# defines the length of the training data (and thus the input size of the ANN)
input_size = 64

# the length of the dataset
# dataset_len * input_size can't be larger than the sum of the binary sequences length
dataset_len = 2000000 / input_size

generate_dataset(
    random_raw_path=f'./binary_sequences/{random_name}.bin',
    pseudorandom_raw_path=f'./binary_sequences/{pseudorandom_name}.bin',
    out_path=f'{OUT_DIR}{pseudorandom_name}.h5',
    num_bits=input_size,
    dataset_len=dataset_len)


dset = h5py.File(f'{OUT_DIR}{pseudorandom_name}.h5', 'r')

print(list(dset.keys()))

data = dset["data"]
targets = dset["targets"]

print(len(data))
print(len(targets))

print(data[0])
print(targets[0])

dset.close()
