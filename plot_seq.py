import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from utils import open_bin_file
import ntpath
import sys


SIZE = 500

if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        raise Exception(
            "please provide the path of the sequence you want to show and then optionally the size of the data")

    if len(sys.argv) == 3:
        SIZE = int(sys.argv[2])

    seq = open_bin_file(sys.argv[1])[:SIZE**2]
    data = np.array(seq).reshape(SIZE, SIZE) * 1
    plt.figure(num=ntpath.basename(sys.argv[1]))
    plt.imshow(data, cmap='Greys')
    plt.show()
