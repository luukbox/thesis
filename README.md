# Cryptanalysis of Keystream Generators using Machine Learning Techniques

## Requirements

- [python 3.6](https://www.python.org/downloads/release/python-367/) (the version has to be exactly 3.6.\*, since tensorflow doesn't support 3.7 yet)
- [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) (very useful but not necessary)

## Installation

```bash
$ git clone https://github.com/luukbox/thesis-implementation.git
$ cd thesis-implementation
# if you want to work with a virtual environment run the next 2 commands
# otherwise skip to pip install
$ virtualenv -p /path/to/the/python/bin venv
$ source venv/bin/activate # activate the virtual environment
$ python --version # should be 3.6.* at this point !!!!
$ pip install -r requirements.txt # install the dependencies
```

## Generate the binary sequences

1. uncomment `generate_r_sequence()` in `generate_source_files.py`
2. run `python generate_source_files.py` in the console
3. revert step 1.
4. uncomment `generate_pr_sequence(num_bits=960000, poly=[11, 9])` and change the poly if you like to
5. step 2.

## Generate a dataset

1. paste the paths of the binary sequences into `generate_dset.py` (`random_raw_path="", pseudorandom_raw_path=""`), change the `NAME` and other parameters e.g. the input_size
2. run `python generate_dset.py` in the console

## Train a model

1. paste the dataset_path into `train.py`
2. run `python train.py` in the console
