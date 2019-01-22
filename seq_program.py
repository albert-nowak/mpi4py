import math
import os
import random
import sys
from datetime import datetime
from functools import reduce
from typing import List, Generator

import numpy as np
from tqdm import trange

DEBUG = True
TEST_RUNNER_MODE = True
MATRICES_DIRECTORY = 'matrices/'


def log(text: str) -> None:
    """ Helper function that will log text with time and date. """

    if not DEBUG:
        return

    current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    print('[LOG]' + current_time + text)


def check_requirements():
    """ Check if directory with matrices exists and contains any file. """
    
    if not os.path.exists(MATRICES_DIRECTORY):
        raise Exception('Directory with matrices does not exist.')

    matrices_list = os.listdir(MATRICES_DIRECTORY)
    if len(matrices_list) <= 0:
        raise Exception('Directory with matrices is empty.')


def main():
    """ Distributed matrices sum calculation program. """

    start_time = datetime.now()

    # Make sure that we're good to go
    check_requirements()

    matrices_list = os.listdir(MATRICES_DIRECTORY)
    matrices = []
    for i in range(len(matrices_list)):
        file_name = os.path.join(MATRICES_DIRECTORY, matrices_list[i])
        matrix = np.load(file_name)
        # Matrix det
        det = np.linalg.det(matrix)
        matrices.append(matrix)
    
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.add(result, matrix)

    result_matrix_det = np.linalg.det(result)

    finish_time = datetime.now()

    if not TEST_RUNNER_MODE:
        print(result)
        print(f'Matrix det: {result_matrix_det}')
        print(f'It took {(finish_time - start_time).total_seconds()}')
    else:
        sys.stdout.write(str((finish_time - start_time).total_seconds()))


if __name__ == "__main__":
    main()
