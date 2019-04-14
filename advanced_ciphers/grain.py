import numpy as np
from pyfsr import LFSR, NLFSR, FSRFunction, logical_xor
from tqdm import tqdm


def simple_grain_sequence_v1(length):
    lfsr = LFSR(poly=[18, 11], initstate="random", initcycles=1000)
    gx = FSRFunction([
        0, 12, "+", 11, 15, "*", "+"
    ])
    nfsr = NLFSR(initstate="random", size=18, infunc=gx, initcycles=1000)
    hx = FSRFunction([0, 2, "+", 1, 2, "*", "+"])

    sequence = np.zeros(length)
    for i in tqdm(range(length), ascii=True, desc="Generating Simplified Grain Sequence V1"):
        lfsr_outbit = lfsr.shift()
        nfsr.shift()
        nfsr.state[0] = logical_xor(lfsr_outbit, nfsr.state[0], True)
        outtaps = [lfsr.state[3], lfsr.state[14],
                   nfsr.state[7]]
        sequence[i] = logical_xor(nfsr.state[14], hx.solve(outtaps), True)

    return "".join(str(s) for s in sequence.astype(int))


def simple_grain_sequence_v2(length):
    lfsr = LFSR(poly=[15, 14], initstate="random", initcycles=1000)
    gx = FSRFunction([
        2, 12, "*", 11, 14, "*", "+", 0, "+"
    ])
    nfsr = NLFSR(initstate="random", size=15, infunc=gx, initcycles=1000)
    hx = FSRFunction([0, 2, "*", 0, 1, "+", "+"])

    sequence = np.zeros(length)
    for i in tqdm(range(length), ascii=True, desc="Generating Simplified Grain Sequence V2"):
        lfsr_outbit = lfsr.shift()
        nfsr.shift()
        nfsr.state[0] = logical_xor(lfsr_outbit, nfsr.state[0], True)
        outtaps = [lfsr.state[2], lfsr.state[11],
                   nfsr.state[7]]
        # this time we won't mask the output of the filter function
        sequence[i] = hx.solve(outtaps)

    return "".join(str(s) for s in sequence.astype(int))


def simple_grain_sequence_v3(length):
    lfsr = LFSR(poly=[15, 14], initstate="random", initcycles=1000)
    gx = FSRFunction([
        2, 12, "*", 11, 14, "*", "+", 0, "+"
    ])
    nfsr = NLFSR(initstate="random", size=15, infunc=gx, initcycles=1000)
    hx = FSRFunction([0, 2, "*", 0, 1, "+", "+"])

    sequence = np.zeros(length)
    for i in tqdm(range(length), ascii=True, desc="Generating Simplified Grain Sequence V3"):
        lfsr_outbit = lfsr.shift()
        nfsr.shift()
        nfsr.state[0] = logical_xor(lfsr_outbit, nfsr.state[0], True)
        outtaps = [lfsr.state[2], lfsr.state[11],
                   nfsr.state[7]]

        # identical to v2 but this time we mask the outbit
        sequence[i] = logical_xor(nfsr.state[9], hx.solve(outtaps), True)

    return "".join(str(s) for s in sequence.astype(int))


def simple_grain_sequence_v4(length):
    lfsr = LFSR(poly=[16, 9, 8, 4, 3, 2], initstate="random", initcycles=1000)
    gx = FSRFunction([
        0, 3, 5, "+", "+", 2, 7, "*", "+", 1, 4, 9, "*", "*", "+"
    ])
    nfsr = NLFSR(initstate="random", size=10, infunc=gx, initcycles=1000)
    hx = FSRFunction([0, 1, 2, "+", "+", 1, 2, "*", "+", 0, 2, "*", "+"])

    sequence = np.zeros(length)
    for i in tqdm(range(length), ascii=True, desc="Generating Simplified Grain Sequence V4"):
        lfsr_outbit = lfsr.shift()
        nfsr.shift()
        nfsr.state[0] = logical_xor(lfsr_outbit, nfsr.state[0], True)
        outtaps = [lfsr.state[7], lfsr.state[14],
                   nfsr.state[5]]
        # no masking
        sequence[i] = hx.solve(outtaps)

    return "".join(str(s) for s in sequence.astype(int))


def grain_sequence_no_mask(length):
    lfsr = LFSR(poly=[80, 67, 57, 42, 29, 18],
                initstate="random", initcycles=1000)
    gx = FSRFunction([
        16, 19, 27, 34, 42, 46, 51, 58, 64, 70, 79, "+", "+", "+", "+", "+", "+", "+", "+", "+", "+",
        16, 19, "*", "+", 42, 46, "*", "+", 64, 70, "*", "+", 19, 27, 34, "*", "*", "+", 46, 51, 58, "*", "*", "+", 16, 34, 51, 70, "*", "*", "*", "+",
        19, 27, 42, 46, "*", "*", "*", "+", 16, 19, 58, 64, "*", "*", "*", "+", 16, 19, 27, 34, 42, "*", "*", "*", "*", "+", 46, 51, 58, 64, 70, "*", "*", "*", "*", "+",
        27, 34, 42, 46, 51, 58, "*", "*", "*", "*", "*", "+"
    ])
    nfsr = NLFSR(initstate="random", size=80, infunc=gx, initcycles=1000)
    hx = FSRFunction([1, 4, "+", 0, 3, "*", "+", 2, 3, "*", "+", 3, 4, "*", "+", 0, 1, 2, "*", "*", "+",
                      0, 2, 3, "*", "*", "+", 0, 2, 4, "*", "*", "+", 1, 2, 4, "*", "*", "+", 2, 3, 4, "*", "*", "+"])

    sequence = np.zeros(length)
    for i in tqdm(range(length), ascii=True, desc="Generating Grain Sequence without masking the filter function"):
        lfsr.shift()
        nfsr.shift()
        nfsr.state[0] = logical_xor(lfsr.outbit, nfsr.state[0], True)
        # for bi we'll choose 63, so bi+63 would result in (63+63) % 80 = 46
        outtaps = [lfsr.state[3], lfsr.state[25],
                   lfsr.state[46], lfsr.state[64], nfsr.state[46]]
        # identical to grain but without the masking of the output of the filter function
        sequence[i] = hx.solve(outtaps)

    return "".join(str(s) for s in sequence.astype(int))


def grain_sequence(length):
    lfsr = LFSR(poly=[80, 67, 57, 42, 29, 18],
                initstate="random", initcycles=1000)
    gx = FSRFunction([
        16, 19, 27, 34, 42, 46, 51, 58, 64, 70, 79, "+", "+", "+", "+", "+", "+", "+", "+", "+", "+",
        16, 19, "*", "+", 42, 46, "*", "+", 64, 70, "*", "+", 19, 27, 34, "*", "*", "+", 46, 51, 58, "*", "*", "+", 16, 34, 51, 70, "*", "*", "*", "+",
        19, 27, 42, 46, "*", "*", "*", "+", 16, 19, 58, 64, "*", "*", "*", "+", 16, 19, 27, 34, 42, "*", "*", "*", "*", "+", 46, 51, 58, 64, 70, "*", "*", "*", "*", "+",
        27, 34, 42, 46, 51, 58, "*", "*", "*", "*", "*", "+"
    ])
    nfsr = NLFSR(initstate="random", size=80, infunc=gx, initcycles=1000)
    hx = FSRFunction([1, 4, "+", 0, 3, "*", "+", 2, 3, "*", "+", 3, 4, "*", "+", 0, 1, 2, "*", "*", "+",
                      0, 2, 3, "*", "*", "+", 0, 2, 4, "*", "*", "+", 1, 2, 4, "*", "*", "+", 2, 3, 4, "*", "*", "+"])

    sequence = np.zeros(length)
    for i in tqdm(range(length), ascii=True, desc="Generating Real Grain Sequence"):
        lfsr.shift()
        nfsr.shift()
        nfsr.state[0] = logical_xor(lfsr.outbit, nfsr.state[0], True)
        # for bi we'll choose 63, so bi+63 would result in (63+63) % 80 = 46
        outtaps = [lfsr.state[3], lfsr.state[25],
                   lfsr.state[46], lfsr.state[64], nfsr.state[46]]
        # mask the output of the filter function with bi
        sequence[i] = logical_xor(nfsr.state[63], hx.solve(outtaps), True)

    return "".join(str(s) for s in sequence.astype(int))


if __name__ == "__main__":
    print(grain_sequence(1000))
