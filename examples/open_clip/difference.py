"""
With 2 folder paths provided,
this script will output absolute as well as relative difference of paired files.

Examples:
    python difference.py --a_path=./results_a/ --b_path=./results_b/

"""

import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a_path",
        type=str,
        default=None,
        help="A folder path containing at least one .txt file of model results",
    )
    parser.add_argument(
        "--b_path",
        type=str,
        default=None,
        help="Another folder path containing at least one .txt file of model results",
    )
    args = parser.parse_args()
    return args


def difference(a_file, b_file):
    with open(a_file, "r") as f:
        a = eval(f.read())
    with open(b_file, "r") as f:
        b = eval(f.read())
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if not isinstance(a, np.ndarray):
        a = np.array(a)

    if b.shape != a.shape:
        raise ValueError(
            f"{a} has shape {a.shape} "
            f"while {b} has different shape of {b.shape}."
        )

    # abs diff (mean)
    abs_mean = np.mean(np.abs(a - b))
    # abs diff (max)
    abs_max = np.max(np.abs(a - b))
    # relative diff (mean)
    rel_mean = np.mean(np.abs(a - b) / (np.abs(b) + 1e-6))
    # relative diff (max)
    rel_max = np.max(np.abs(a - b) / (np.abs(b) + 1e-6))

    print(
        f"{os.path.basename(a_file).replace('.txt', ':')}\n"
        f" absolute error(mean):       {abs_mean}\n"
        f" absolute error(max):        {abs_max}\n"
        f" relative error(mean):       {rel_mean}\n"
        f" relative error(max):        {rel_max}\n"
        f" relative error(max / mean): {np.max(np.abs(a - b)) / np.mean(np.abs(b))}\n"
    )


def main():
    args = parse_args()

    a_files = []
    b_files = []
    for root, dirs, files in os.walk(args.a_path):
        for file in files:
            a_files.append(os.path.join(root, file))
    a_files = sorted(a_files)
    for root, dirs, files in os.walk(args.b_path):
        for file in files:
            b_files.append(os.path.join(root, file))
    b_files = sorted(b_files)

    if len(a_files) != len(b_files):
        raise ValueError(f"Files in {args.a_path} are different with those in {args.b_path}.")
    for file in range(len(a_files)):
        if os.path.basename(a_files[file]) != os.path.basename(b_files[file]):
            raise ValueError(f"Files in {args.a_path} are different with those in {args.b_path}.")
        difference(a_files[file], b_files[file])


if __name__ == "__main__":
    main()
