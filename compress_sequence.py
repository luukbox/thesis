from utils import open_bin_file
import numpy as np
from sklearn.preprocessing import minmax_scale
import sys


def calc_run_num(run):
    # print(run)
    run_num = len(run)
    if run[0] == 0:
        run_num = run_num * -1
    return run_num


def norm_seq(raw):
    return minmax_scale(np.array(raw).astype(float))


def get_compressed_normalized_seq(path_to_bin, normalize):
    seq = open_bin_file(path_to_bin)
    new_seq = []
    is_zero_run = False
    run = []
    print("Compressing sequence " + path_to_bin + "...")
    for i in range(len(seq)):
        bit = seq[i] * 1
        is_zero = bit == 0
        if len(run) > 0:
            if is_zero_run and is_zero:
                run.append(0)
            elif not is_zero_run and is_zero:
                new_seq.append(calc_run_num(run))
                run = []
                is_zero_run = True
                run.append(0)
            elif is_zero_run and not is_zero:
                new_seq.append(calc_run_num(run))
                run = []
                is_zero_run = False
                run.append(1)
            elif not is_zero_run and not is_zero:
                run.append(1)
        else:
            if is_zero:
                is_zero_run = True
                run.append(0)
            else:
                is_zero_run = False
                run.append(1)
    if normalize:
        print("Normalizing sequence...")
        new_seq = norm_seq(new_seq)
    print("Length Original Sequence:", len(seq))
    print("Length Compressed Sequence:", len(new_seq))
    print("Compression ratio: %.4f:1" % (len(seq) / len(new_seq)))
    return new_seq


if __name__ == "__main__":
    bin_file_path = sys.argv[1]
    orig_seq = open_bin_file(bin_file_path)
    seq = get_compressed_normalized_seq(bin_file_path, False)
    print("First 20 bits of seq:", np.array(orig_seq[:20])*1)
    print("First 10 bits of c_seq:", seq[:1000])
