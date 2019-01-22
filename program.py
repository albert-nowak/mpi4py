import math
import os
import random
import sys
from datetime import datetime
from functools import reduce
from typing import List, Generator

import numpy as np
from mpi4py import MPI

DEBUG = True
TEST_RUNNER_MODE = False
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
    

def get_tasks(tasks: List[str], chunk_size: int) -> Generator[int, int, List[str]]:
    """ Yield successive `chunk_size` chunks from tasks list. """
    
    for i in range(0, len(tasks), chunk_size):
        # For non even lists, return biggest last element
        if 0 > len(tasks[i + chunk_size:i + chunk_size * 2]) < chunk_size:
            yield tasks[i:]
        yield tasks[i:i + chunk_size]


def main():
    """ Distributed matrices sum calculation program. """

    start_time = datetime.now()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    # Make sure that we're good to go
    check_requirements()

    # Master
    if rank == 0:
        matrices_list = os.listdir(MATRICES_DIRECTORY)
        workers_count = size - 1
        chunk_size = round(len(matrices_list) / workers_count)
        
        tasks = get_tasks(matrices_list, chunk_size)  # returns generator
        for worker_id in range(1, size):
            try:
                comm.send(next(tasks), dest=worker_id, tag=worker_id)
            except StopIteration:
                raise Exception('Not enough tasks for workers.')

        workers_results = []
        for worker_id in range(workers_count):
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            workers_results.append(data)
        
        if not len(workers_results):
            raise Exception('Empty result')

        result = workers_results[0]
        for matrix in workers_results[1:]:
            result = np.add(result, matrix)

        result_matrix_det = np.linalg.det(result)

        finish_time = datetime.now()

        if not TEST_RUNNER_MODE:
            print(result)
            print(f'Matrix det: {result_matrix_det}')
            print(f'It took {(finish_time - start_time).total_seconds()}')
        else:
            sys.stdout.write(str((finish_time - start_time).total_seconds()))

    # Worker
    else:
        # Wait for list of tasks to work on
        # Expected data format:
        # ['matrix_2.npy', 'matrix_3.npy', 'matrix_1.npy', 'matrix_4.npy', 'matrix_5.npy']
        matrices_file_names = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status) 
        matrices = []

        for matrix_file_name in matrices_file_names:
            file_name = os.path.join(MATRICES_DIRECTORY, matrix_file_name)
            matrix = np.load(file_name)
            # Matrix det
            det = np.linalg.det(matrix)
            matrices.append(matrix)
        
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.add(result, matrix)
        
        comm.send(result, dest=0, tag=rank)
        

if __name__ == "__main__":
    main()
