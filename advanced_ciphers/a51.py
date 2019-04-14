'''
    File name: a51.py
    Author: Lukas Müller
    Python Version: 3.6
'''

import numpy as np
from pyfsr import LFSR, logical_xor
from tqdm import tqdm


def simple_a51_sequence_v1(length):
    r1 = LFSR(poly=[18, 11], initstate="random", initcycles=1000)
    r2 = LFSR(poly=[15, 14],  initstate="random", initcycles=1000)
    r3 = LFSR(poly=[17, 14], initstate="random", initcycles=1000)
    sequence = np.ones(length) * -1

    for i in tqdm(range(length), ascii=True, desc="Generating Simpified A51 Sequence v1"):
        (b1, b2, b3) = (r1.state[8], r2.state[10], r3.state[10])
        vote = np.argmax(np.bincount([b1, b2, b3]))
        if b1 == vote:
            r1.shift()
        if b2 == vote:
            r2.shift()
        if b3 == vote:
            r3.shift()
        sequence[i] = logical_xor(
            logical_xor(r1.outbit, r2.outbit, True), r3.outbit, True)

    return "".join(str(s) for s in sequence.astype(int))


def simple_a51_sequence_v2(length):
    r1 = LFSR(poly=[7, 6], initstate="random", initcycles=1000)
    r2 = LFSR(poly=[6, 5],  initstate="random", initcycles=1000)
    r3 = LFSR(poly=[9, 5], initstate="random", initcycles=1000)
    sequence = np.ones(length) * -1

    for i in tqdm(range(length), ascii=True, desc="Generating Simpified A51 Sequence v2"):
        (b1, b2, b3) = (r1.state[5], r2.state[3], r3.state[7])
        vote = np.argmax(np.bincount([b1, b2, b3]))
        if b1 == vote:
            r1.shift()
        if b2 == vote:
            r2.shift()
        if b3 == vote:
            r3.shift()
        sequence[i] = logical_xor(
            logical_xor(r1.outbit, r2.outbit, True), r3.outbit, True)

    return "".join(str(s) for s in sequence.astype(int))


def simple_a51_sequence_v3(length):
    r1 = LFSR(poly=[7, 6], initstate="random", initcycles=1000)
    r2 = LFSR(poly=[15, 14],  initstate="random", initcycles=1000)
    r3 = LFSR(poly=[9, 5], initstate="random", initcycles=1000)
    sequence = np.ones(length) * -1

    for i in tqdm(range(length), ascii=True, desc="Generating Simpified A51 Sequence v3"):
        (b1, b2, b3) = (r1.state[5], r2.state[3], r3.state[7])
        vote = np.argmax(np.bincount([b1, b2, b3]))
        if b1 == vote:
            r1.shift()
        if b2 == vote:
            r2.shift()
        if b3 == vote:
            r3.shift()
        sequence[i] = logical_xor(
            logical_xor(r1.outbit, r2.outbit, True), r3.outbit, True)

    return "".join(str(s) for s in sequence.astype(int))


def simple_a51_sequence_v4(length):
    r1 = LFSR(poly=[7, 6], initstate="random", initcycles=1000)
    r2 = LFSR(poly=[15, 14],  initstate="random", initcycles=1000)
    r3 = LFSR(poly=[17, 14], initstate="random", initcycles=1000)
    sequence = np.ones(length) * -1

    for i in tqdm(range(length), ascii=True, desc="Generating Simpified A51 Sequence v4"):
        (b1, b2, b3) = (r1.state[5], r2.state[3], r3.state[7])
        vote = np.argmax(np.bincount([b1, b2, b3]))
        if b1 == vote:
            r1.shift()
        if b2 == vote:
            r2.shift()
        if b3 == vote:
            r3.shift()
        sequence[i] = logical_xor(
            logical_xor(r1.outbit, r2.outbit, True), r3.outbit, True)

    return "".join(str(s) for s in sequence.astype(int))


def a51_sequence(length):
    r1 = LFSR(poly=[19, 18, 17, 14], initstate="random", initcycles=1000)
    r2 = LFSR(poly=[23, 22, 21, 8],  initstate="random", initcycles=1000)
    r3 = LFSR(poly=[22, 21], initstate="random", initcycles=1000)

    sequence = np.ones(length) * -1

    for i in tqdm(range(length), ascii=True, desc="Generating Real A51 Sequence"):
        (b1, b2, b3) = (r1.state[8], r2.state[10], r3.state[10])
        vote = np.argmax(np.bincount([b1, b2, b3]))
        if b1 == vote:
            r1.shift()
        if b2 == vote:
            r2.shift()
        if b3 == vote:
            r3.shift()
        sequence[i] = logical_xor(
            logical_xor(r1.outbit, r2.outbit, True), r3.outbit, True)
    return "".join(str(s) for s in sequence.astype(int))


if __name__ == "__main__":
    print(a51_sequence(1000))
