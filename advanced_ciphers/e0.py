import numpy as np
from pyfsr import LFSR, NLFSR, FSRFunction, logical_xor
from tqdm import tqdm
'''
source: https://www.esat.kuleuven.be/cosic/publications/article-22.pdf
'''


def T2(x):
    as_bin = [int(b) for b in bin(x)[2:]]
    if len(as_bin) == 1:
        as_bin.append(0)
    return [as_bin[1], logical_xor(as_bin[0], as_bin[1], True)]


def simple_e0_sequence_v1(length):
    l1 = LFSR([11, 4], "random")
    l2 = LFSR([9, 5], "random")
    l3 = LFSR([6, 5], "random")
    l4 = LFSR([15, 14], "random")

    ct = [1, 0]
    ct1 = [1, 0]
    st1 = 1

    # generate the actual sequence
    sequence = np.ones(length) * -1

    for i in tqdm(range(length), ascii=True, desc="Generating Simple E0 Sequence v1"):
        ct = ct[0]+2*ct[1]
        l1.shift()
        l2.shift()
        l3.shift()
        l4.shift()
        yt = l1.outbit + l2.outbit + l3.outbit + l4.outbit
        st1 = int((yt + ct) / 2)
        ct_1 = ct
        ct = ct1
        ct1 = np.logical_xor([int(b) for b in bin(st1)[2:]], ct)
        ct1 = np.logical_xor(ct1, T2(ct_1))
        cipherbit = logical_xor(l1.outbit, l2.outbit, True)
        cipherbit = logical_xor(cipherbit, l3.outbit, True)
        cipherbit = logical_xor(cipherbit, l4.outbit, True)
        cipherbit = np.logical_xor(cipherbit, ct1)
        sequence[i] = cipherbit[0]

    return "".join(str(s) for s in sequence.astype(int))


def e0_sequence(length):
    l1 = LFSR([25, 20, 12, 8], "random")
    l2 = LFSR([31, 24, 16, 12], "random")
    l3 = LFSR([33, 28, 24, 4], "random")
    l4 = LFSR([39, 36, 28, 4], "random")

    ct = [1, 0]
    ct1 = [1, 0]
    st1 = 1

    # generate the actual sequence
    sequence = np.ones(length) * -1

    for i in tqdm(range(length), ascii=True, desc="Generating E0 Sequence"):
        ct = ct[0]+2*ct[1]
        l1.shift()
        l2.shift()
        l3.shift()
        l4.shift()
        yt = l1.outbit + l2.outbit + l3.outbit + l4.outbit
        st1 = int((yt + ct) / 2)
        ct_1 = ct
        ct = ct1
        ct1 = np.logical_xor([int(b) for b in bin(st1)[2:]], ct)
        ct1 = np.logical_xor(ct1, T2(ct_1))
        cipherbit = logical_xor(l1.outbit, l2.outbit, True)
        cipherbit = logical_xor(cipherbit, l3.outbit, True)
        cipherbit = logical_xor(cipherbit, l4.outbit, True)
        cipherbit = np.logical_xor(cipherbit, ct1)
        sequence[i] = cipherbit[0]

    return "".join(str(s) for s in sequence.astype(int))


if __name__ == "__main__":
    print(e0_sequence(1000))
