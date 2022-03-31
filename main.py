#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Esteban Alonso González - e.alonsogzl@gmail.com
"""

import modules.internal_fns as ifn
import config as cfg
import numpy as np

if(cfg.parallelization == "multiprocessing" or
   cfg.implementation == "open_loop"):
    import multiprocessing as mp
elif cfg.parallelization == "MPI":
    from mpi4py import MPI
elif cfg.parallelization == "PBS.array":
    import sys
    import os
    import multiprocessing as mp
else:
    pass


def MuSA():
    """
    This is the main function. Here the parallelization scheme and the
    implementation is selected. This function is just a wrapper of the real
    assimilation process, which is encapsulated in the ifn.cell_assimilation
    function.

    Raises
    ------
    'Choose an available implementation'
        An available implementation should be choosen.

    'Choose an available paralelization scheme'
        An available paralelization scheme should be choosen.

    Returns
    -------
    None.

    """

    if cfg.implementation == "point_scale":

        print("Running the assimilation in a single point")

        lon_idx, lat_idx = ifn.nc_idx()

        ifn.cell_assimilation(lon_idx, lat_idx)

    elif(cfg.implementation == "distributed"):

        grid = ifn.expand_grid()

        if cfg.parallelization == "sequential":

            print("Running MuSA: Sequentially")

            for row in range(grid.shape[0]):

                lon_idx = grid[row, 0]
                lat_idx = grid[row, 1]
                ifn.cell_assimilation(lon_idx, lat_idx)

        elif cfg.parallelization == "multiprocessing":

            print("Running MuSA: Distributed (multiprocessing)")

            if isinstance(cfg.nprocess, int):
                nprocess = cfg.nprocess
            else:
                nprocess = mp.cpu_count() - 1

            print("Launching " + str(nprocess) + " processes in "
                  + str(mp.cpu_count()) + " processors")

            pool = mp.Pool(processes=nprocess)
            pool.starmap(ifn.cell_assimilation, zip(grid[:, 0], grid[:, 1]))

        elif cfg.parallelization == "MPI":

            rank = MPI.COMM_WORLD.Get_rank()
            size = MPI.COMM_WORLD.Get_size()

            print("Running MuSA: Distributed (MPI) from rank: " + str(rank))

            for row in range(grid.shape[0]):

                # split the cells between ranks
                if row % size != rank:
                    continue

                lon_idx = grid[row, 0]
                lat_idx = grid[row, 1]
                ifn.cell_assimilation(lon_idx, lat_idx)

        elif cfg.parallelization == "PBS.array":

            pbs_task_id = int(os.getenv("PBS_ARRAY_INDEX"))-1
            pbs_task_number = int(sys.argv[1])
            nprocess = int(sys.argv[2])

            ids = np.arange(0, grid.shape[0])
            ids = ids % pbs_task_number == pbs_task_id

            print("Running MuSA: Distributed (PBS.array) from job: " +
                  str(pbs_task_id) + " in " + str(nprocess) + " cores")

            pool = mp.Pool(processes=nprocess)
            pool.starmap(ifn.cell_assimilation,
                         zip(grid[ids, 0], grid[ids, 1]))

        else:

            raise Exception("Choose an available paralelization scheme")

    elif cfg.implementation == "open_loop":

        grid = ifn.expand_grid()

        print("Running FSM simulation: Distributed (multiprocessing)")

        if isinstance(cfg.nprocess, int):
            nprocess = cfg.nprocess
        else:
            nprocess = mp.cpu_count() - 1

        print("Launching " + str(nprocess) + " processes in " +
              str(mp.cpu_count()) + " processors")

        pool = mp.Pool(processes=nprocess)
        pool.starmap(ifn.open_loop_simulation, zip(grid[:, 0], grid[:, 1]))

    else:
        raise Exception("Choose an available implementation")


def check_platform():

    # TODO: provide full suport for wind32

    if (sys.platform not in ("linux", "darwin")):
        raise Exception(sys.platform + " is not supported by MuSA yet")


if __name__ == "__main__":

    check_platform()
    MuSA()
