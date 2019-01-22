import asyncio
import os
import subprocess
import sys
from collections import namedtuple
from datetime import datetime

from tqdm import tqdm

from generate_matrix import generate_matrices


NUMBER_OF_WORKERS = 4
MEASURE_STEPS = [
    (33, 10),
    (33, 1000),
    (330, 10),
    (330, 1000),
    (3300, 10),
    (3300, 1000),
]


def log(text: str) -> None:
    """ Helper function that will log text with time and date. """

    current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    print('[LOG]' + current_time + text)


async def run_tests():
    """ Test runner for MPI project """

    ProgramRun = namedtuple('Times', ['seq_program', 'distr_program'])
    run_times = []

    for measure_step in MEASURE_STEPS:

        # generate matrices for 2 programs to use as input data
        log(f'Generating generating matrices {measure_step[0]} - {measure_step[1]}')
        generate_matrices(measure_step[0], measure_step[1])
        log('Done generating matrices')

        # run seq version of the algorithm
        # seq_program_process = subprocess.run(["python", "seq_program.py"], stdout=subprocess.PIPE)
        # seq_program_time = float(subprocess.stdout)
        log(f'Seq program start for {measure_step[0]} - {measure_step[1]}.')
        seq_program_process = await asyncio.create_subprocess_exec(
            sys.executable, 'seq_program.py',
            stdout=asyncio.subprocess.PIPE,
        )
        log('Seq program done.')

        # Read one line of output.
        seq_program_time = await seq_program_process.stdout.readline()
        # print(seq_program_time)

        # Wait for the subprocess exit.
        await seq_program_process.wait()

        # run MPI version
        # distr_program_process = subprocess.run(
        #     ['mpiexec', '-n', str(NUMBER_OF_WORKERS + 1), 'python', 'program.py'],
        #     stdout=subprocess.PIPE,
        # )
        # distr_program_time = float(distr_program_process.stdout)
        
        log(f'Distr program start for {measure_step[0]} - {measure_step[1]}.')
        distr_program_process = await asyncio.create_subprocess_shell(
            f'mpiexec -np {NUMBER_OF_WORKERS} python program.py',
            stdout=asyncio.subprocess.PIPE,
        )
        log('Distr program done.')

        # Read one line of output.
        distr_program_time = await distr_program_process.stdout.readline()
        # print(distr_program_time)

        # Wait for the subprocess exit.
        await distr_program_process.wait()
        
        run_times.append(ProgramRun(seq_program_time, distr_program_time))

    print(run_times)


if __name__ == "__main__":
    asyncio.run(run_tests())
