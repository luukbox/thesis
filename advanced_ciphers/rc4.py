import random
import string
import sys
from tqdm import tqdm
import math


def init_state(key):
    keylength = len(key)
    state = [i for i in range(256)]
    j = 0
    for i in range(256):
        j = (j + state[i] + key[i % keylength]) % 256
        state[i], state[j] = state[j], state[i]
    return state


def rc4_generator(key):
    key = convert_key(key)
    state = init_state(key)
    i = 0
    j = 0
    while True:
        i = (i + 1) % 256
        j = (j + state[i]) % 256
        state[i], state[j] = state[j], state[i]
        K = state[(state[i] + state[j]) % 256]
        yield K


def random_char(x):
    return ''.join(random.choice(string.ascii_letters) for _ in range(x))


def convert_key(s):
    return [ord(c) for c in s]


def int_to_bits(n):
    return [n >> i & 1 for i in range(7, -1, -1)]


def rc4_sequence(length):
    key = random_char(10)
    print("KEY:", key)
    keystream_gen = rc4_generator(key)
    seq = ""
    # RC4 generates 1 byte at a time. The length of the output in bits is equal to the length parameter,
    # the range parameter is the number of iterations that are needed to produce #length bits.
    for _ in tqdm(range(math.ceil(length / 8)), ascii=True, desc="Generating RC4 Sequence"):
        seq += "".join(str(bit) for bit in int_to_bits(next(keystream_gen)))
    return seq


if __name__ == '__main__':
    sequence = rc4_sequence(4000000)
    print("RC4 SEQ LEN:", len(sequence))
