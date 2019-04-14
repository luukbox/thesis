import sys
import sp800_22_tests
from utils import open_bin_file
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            "please provide the path of the sequences you want to test.")
    results = sp800_22_tests.test_sequence(sys.argv[1])
    sp800_22_tests.print_summary(results)
