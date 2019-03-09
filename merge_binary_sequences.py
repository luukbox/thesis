from utils import open_bin_file
import sys

if len(sys.argv) != 4:
    raise Exception(
        "please provide the paths of the sequences as first two arguments, then the name of the generated file.")

seq1 = open_bin_file(sys.argv[1])
seq2 = open_bin_file(sys.argv[2])

name = sys.argv[3]

for bit in seq2:
    seq1.append(bit)


with open(f'./binary_sequences/{name}_{len(seq1)}.bin', 'wb') as f:
    seq1.tofile(f)

print(f'DONE. Merged sequence saved as {name}_{len(seq1)}.bin')
