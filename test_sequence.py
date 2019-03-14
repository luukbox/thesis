import sys
from sp800_22_tests import test_sequence, print_summary

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            "please provide the path of the sequences you want to test.")

    results = test_sequence(sys.argv[1])
    print_summary(results)
