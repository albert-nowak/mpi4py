import argparse
import os
from shutil import rmtree

import numpy as np
from tqdm import trange


SAVE_MATRICES_DIRECTORY = 'matrices/'


def generate_matrices(count: int, size: int) -> None:
    """ Generate `count` square matrices of a size `size`. Save them to files. """

    # Check if save directory exists, if so, delete it and create new one
    if os.path.exists(SAVE_MATRICES_DIRECTORY):
        rmtree(SAVE_MATRICES_DIRECTORY)
    os.makedirs(SAVE_MATRICES_DIRECTORY)

    for i in range(count):
        file_name = os.path.join(SAVE_MATRICES_DIRECTORY, f'matrix_{i+1}')
        
        matrix = np.random.randint(-10, 10, (size, size))
        np.save(file_name, matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate square matrices for MPI program to work on.',
    )
    parser.add_argument(
        '-c',
        '--count',
        type=int,
        help='Number of files to generate.',
    )
    parser.add_argument(
        '-s',
        '--size',
        type=int,
        help='Size of a square matrix.',
    )

    args = parser.parse_args()

    # main logic
    generate_matrices(args.count, args.size)
