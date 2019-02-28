from tqdm import tqdm
import urllib.request
import re
import os
import errno
from datetime import datetime
from bitstring import BitArray
from bitarray import bitarray
import numpy as np
import math
from pyfsr import LFSR

DIR = "./binary_sequences/"


def generate_pr_sequence(num_bits, poly):
    sequence_len = num_bits
    poly_str = '-'.join(str(x) for x in poly)
    out_path = f'{DIR}pr_lfsr_{sequence_len}_poly_{poly_str}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.bin'
    print("OUT PATH: ", out_path)
    r = LFSR(poly=poly, initstate='random', initcycles=2**9)
    sequence = ''.join(str(s) for s in r.sequence(sequence_len))
    save_sequence_as_binary(sequence, out_path)


# fetches the max possible amount from random.org
def generate_r_sequence():
    sequence_len = 960000  # max quota (in bits) for one day
    iterations = math.floor(sequence_len / (80000))

    num_bits = sequence_len
    formatted = ""

    for i in tqdm(range(int(iterations)), ascii=True, desc="Generating Random Dataset:"):
        random_org_request = urllib.request.urlopen(
            'https://www.random.org/cgi-bin/randbyte?nbytes=10000&format=b')
        response = random_org_request.read().decode("utf8")

        if response == "You have used your quota of random bits for today.  See the quota page for details.":
            print("Quota used. Done.")
            break

        random_org_request.close()

        for r in response:
            if r in ["0", "1"]:
                formatted += r
                num_bits -= 1
                if num_bits == 0:
                    break

    out_path = f'{DIR}r_{len(formatted)}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.bin'
    print("OUT PATH: ", out_path)
    save_sequence_as_binary(formatted, out_path)


def save_sequence_as_binary(sequence, path):
    a = bitarray(sequence)
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(path, 'wb') as f:
        a.tofile(f)


# generate_pr_sequence(num_bits=960000, poly=[11, 9])
# generate_r_sequence()
