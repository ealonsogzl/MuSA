#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import modules.internal_fns as ifn
import modules.spatialMuSA as spM
import config as cfg
if cfg.numerical_model == 'FSM2':
    import modules.fsm_tools as model
elif cfg.numerical_model == 'dIm':
    import modules.dIm_tools as model
elif cfg.numerical_model == 'snow17':
    import modules.snow17_tools as model
else:
    raise Exception('Model not implemented')
import numpy as np
import sys

if (cfg.parallelization == "multiprocessing" or
   cfg.implementation == "open_loop"):
    import multiprocessing as mp
elif cfg.parallelization == "HPC.array":
    import multiprocessing as mp
else:
    pass
from modules.cell_assim import cell_assimilation
from mpi4py import MPI
import logging


def MuSA():

    if cfg.parallelization == "HPC.array":
        pass
    else:
        model.model_compile()

    if cfg.implementation in ["distributed", "Spatial_propagation", "open_loop"]:
        grid = ifn.expand_grid()

    """
    This is the main function. Here the parallelization scheme and the
    implementation is selected. This function is just a wrapper of the real
    assimilation process, which is encapsulated in the cell_assimilation
    function.

    Raises
    ------
    'Choose an available implementation'
        An available implementation should be choosen.

    'Choose an available parallelization scheme'
        An available parallelization scheme should be choosen.

    -------
    None.

    """

    if cfg.implementation == "point_scale":

        print("Running the assimilation in a single point")

        lat_idx, lon_idx = ifn.nc_idx()

        cell_assimilation(lat_idx, lon_idx)

    elif cfg.implementation == "distributed":

        if cfg.parallelization == "sequential":

            print("Running MuSA: Sequentially")

            for row in range(grid.shape[0]):

                lat_idx = grid[row, 0]
                lon_idx = grid[row, 1]

                cell_assimilation(lat_idx, lon_idx)

        elif cfg.parallelization == "multiprocessing":

            print("Running MuSA: Distributed (multiprocessing)")

            if cfg.MPI:
                comm = MPI.COMM_WORLD
                nprocess = comm.Get_size() - 1
            else:
                if isinstance(cfg.nprocess, int):
                    nprocess = cfg.nprocess
                else:
                    nprocess = mp.cpu_count() - 1

            print("Launching " + str(nprocess) + " processes in "
                  + str(mp.cpu_count()) + " processors")

            inputs = [grid[:, 0], grid[:, 1]]
            ifn.safe_pool(cell_assimilation, inputs, nprocess)

        elif cfg.parallelization == "HPC.array":

            HPC_task_number = int(sys.argv[1])
            nprocess = int(sys.argv[2])
            HPC_task_id = int(sys.argv[3])-1

            ids = np.arange(0, grid.shape[0])
            ids = ids % HPC_task_number == HPC_task_id

            print("Running MuSA: Distributed (HPC.array) from job: " +
                  str(HPC_task_id) + " in " + str(nprocess) + " cores")

            # compile FSM
            model.model_compile_HPC(HPC_task_id)

            inputs = [grid[ids, 0], grid[ids, 1]]
            ifn.safe_pool(cell_assimilation, inputs, nprocess)

        else:

            raise Exception("Choose an available paralelization scheme")

    elif cfg.implementation == 'Spatial_propagation':
        if cfg.da_algorithm not in ["ES", "IES"]:
            raise Exception("Spatial_propagation needs ES/IES methods")

        ids = np.arange(0, grid.shape[0])

        if cfg.parallelization == "HPC.array":

            # Restart run
            if cfg.restart_run:
                prev_step, prev_j = ifn.return_step_j('spatiallogfile.txt')
            else:
                prev_step, prev_j = 0, 0

            # Log file for restart
            logging.basicConfig(filename='spatiallogfile.txt',
                                level=logging.INFO,
                                format='%(asctime)s - %(message)s')

            HPC_task_number = int(sys.argv[1])
            nprocess = int(sys.argv[2])
            HPC_task_id = int(sys.argv[3])-1

            ids = ids % HPC_task_number == HPC_task_id

            print("Running MuSA: Distributed (HPC.array) from job: " +
                  str(HPC_task_id) + " in " + str(nprocess) + " cores")

            # compile FSM
            model.model_compile_HPC(HPC_task_id)

            # get timestep of GSC maps
            ini_DA_window = spM.domain_steps()

            # DA_loop
            # create a pool inside each task
            # this enumerate is unnecesary
            for gsc_count, step in enumerate(range(len(ini_DA_window))):

                if cfg.restart_run and step < prev_step:
                    continue

                # create prior Ensembles
                inputs = [list(grid[ids, 0]), list(grid[ids, 1]),
                          [ini_DA_window] * len(ids),
                          [step] * len(ids),
                          [gsc_count] * len(ids)]

                ifn.safe_pool(spM.create_ensemble_cell, inputs, nprocess)

                # Wait untill all ensembles are created
                spM.wait_for_ensembles(step, HPC_task_id)

                for j in range(cfg.max_iterations):  # Run spatial assim

                    if cfg.restart_run and j < prev_j:
                        continue
                    # add info to log
                    logging.info(f'step: {step} - j: {j}')

                    inputs = [list(grid[ids, 0]), list(grid[ids, 1]),
                              [step] * len(ids), [j]*len(ids)]

                    ifn.safe_pool(spM.spatial_assim, inputs, nprocess)

                    # Wait untill all ensembles are updated and remove prior
                    spM.wait_for_ensembles(step, HPC_task_id, j)

            # collect results from HPC_task_id = 0
            if HPC_task_id != 0:
                return None
            else:
                inputs = [grid[:, 0], grid[:, 1]]
                ifn.safe_pool(spM.collect_results, inputs, nprocess)

        elif cfg.parallelization == "multiprocessing":

            # Restart run
            if cfg.restart_run:
                prev_step, prev_j = ifn.return_step_j('spatiallogfile.txt')
            else:
                prev_step, prev_j = 0, 0

            # Log file for restart
            logging.basicConfig(filename='spatiallogfile.txt',
                                level=logging.INFO,
                                format='%(asctime)s - %(message)s')

            if cfg.MPI:
                comm = MPI.COMM_WORLD
                nprocess = comm.Get_size() - 1
            else:
                if isinstance(cfg.nprocess, int):
                    nprocess = cfg.nprocess
                else:
                    nprocess = mp.cpu_count() - 1

            # get timestep of GSC maps
            ini_DA_window = spM.domain_steps()

            # DA loop
            for gsc_count, step in enumerate(range(len(ini_DA_window))):

                if cfg.restart_run and step < prev_step:
                    continue

                # create prior Ensembles
                inputs = [list(grid[ids, 0]), list(grid[ids, 1]),
                          [ini_DA_window] * len(ids),
                          [step] * len(ids),
                          [gsc_count] * len(ids)]

                ifn.safe_pool(spM.create_ensemble_cell, inputs, nprocess)

                # Wait untill all ensembles are created
                spM.wait_for_ensembles(step, 0)

                for j in range(cfg.max_iterations):  # Run spatial assim

                    if cfg.restart_run and j < prev_j:
                        continue
                    # add info to log
                    logging.info(f'step: {step} - j: {j}')

                    inputs = [list(grid[:, 0]), list(grid[:, 1]),
                              [step] * grid.shape[0],
                              [j] * grid.shape[0]]

                    ifn.safe_pool(spM.spatial_assim, inputs, nprocess)

                    # Wait untill all ensembles are updated and remove prior
                    spM.wait_for_ensembles(step, 0, j)

            # collect results
            inputs = [grid[:, 0], grid[:, 1]]
            ifn.safe_pool(spM.collect_results, inputs, nprocess)

    elif cfg.implementation == "open_loop":

        print("Running FSM simulation: Distributed (multiprocessing)")

        if isinstance(cfg.nprocess, int):
            nprocess = cfg.nprocess
        else:
            nprocess = mp.cpu_count() - 1

        print("Launching " + str(nprocess) + " processes in " +
              str(mp.cpu_count()) + " processors")

        inputs = [grid[:, 0], grid[:, 1]]
        ifn.safe_pool(ifn.open_loop_simulation, inputs, nprocess)

    else:
        raise Exception("Choose an available implementation")

    if cfg.write_stat_full and cfg.restart_run:
        raise Exception("write_stat_full and restart_run cannot be activated simultaneously. To be implemented.")

    if cfg.write_stat_daily and cfg.restart_run:
        raise Exception("write_stat_daily and restart_run cannot be activated simultaneously. To be implemented.")

def check_platform():

    # TODO: provide full suport for wind32

    if (sys.platform not in ("linux", "darwin")):
        raise Exception(sys.platform + " is not supported by MuSA yet")


if __name__ == "__main__":

    if cfg.parallelization in ["multiprocessing", "HPC.array"]:
        mp.set_start_method('spawn', force=True)

    check_platform()
    ifn.pre_cheks()

    MuSA()
