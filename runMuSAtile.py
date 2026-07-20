#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script containing the main function of MuSA.
This script is adapated from the original MuSA code,
and is designed to be run in a tiling system, where each tile is processed independently.
The code is able to handle different experiments with different configurations.

Author: Esteban Alonso González - alonsoe@ipe.csic.es

edited by: Lucas Boeykens - lucas.boeykens@ugent.be lucas.boeykens@kuleuven.be
"""
import importlib.util
import os, sys, argparse

config_path = os.environ.get("MUSA_CONFIG", "config.py")
spec = importlib.util.spec_from_file_location("config", config_path)
cfg = importlib.util.module_from_spec(spec)
sys.modules["config"] = cfg
spec.loader.exec_module(cfg)

import modules.transformMuSAoutputs_tools as transformMuSA
import modules.internal_fns as ifn
import modules.spatialMuSA as spM
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
import os

def getArgsFromCfg():
    args = {
        "nc_maks_path": cfg.nc_maks_path,
        "output_path": cfg.output_path,
        "nc_forcing_path": cfg.nc_forcing_path,
        "date_ini": cfg.date_ini,
        "date_end": cfg.date_end,
        "remove_output_cells": cfg.remove_output_cells
    }
    return argparse.Namespace(**args)

def autoGenerateOutputPaths() -> None:
    """
    Function that automatically generates the output paths for the MuSA run based on the current configuration.
    """

    paths_to_create = []
    # Always needed by the open-loop and DA result writers
    if getattr(cfg, "output_path", None):
        paths_to_create.append(cfg.output_path)

    # Used by model-specific temporary/intermediate files
    if getattr(cfg, "intermediate_path", None):
        paths_to_create.append(cfg.intermediate_path)

    # Only needed if ensemble saving is enabled
    if getattr(cfg, "save_ensemble", False) and getattr(cfg, "save_ensemble_path", None):
        paths_to_create.append(cfg.save_ensemble_path)

    # Only needed if real-time restart is enabled
    if getattr(cfg, "real_time_restart", False) and getattr(cfg, "real_time_restart_path", None):
        paths_to_create.append(cfg.real_time_restart_path)

    # Only needed for spatial propagation runs
    if getattr(cfg, "implementation", None) == "Spatial_propagation" and getattr(cfg, "spatial_propagation_storage_path", None):
        paths_to_create.append(cfg.spatial_propagation_storage_path)

    for path in paths_to_create:
        os.makedirs(path, exist_ok=True)



def MuSA():

    if cfg.parallelization == "HPC.array":
        pass
    else:
        model.model_compile()

    if cfg.implementation in ["distributed", "Spatial_propagation",
                              "open_loop"]:
        grid = ifn.expand_grid()

    #generate the output paths if they don't exist
    autoGenerateOutputPaths()

    print(cfg.output_path, cfg.nc_forcing_path)

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
            iteration_sims = list()
            count = 0
            for gsc_count, step in enumerate(range(len(ini_DA_window))):

                if cfg.restart_run and step < prev_step:
                    continue

                # create prior Ensembles
                inputs = [list(grid[ids, 0]), list(grid[ids, 1]),
                          [ini_DA_window] * len(ids),
                          [step] * len(ids),
                          [gsc_count] * len(ids),
                          [iteration_sims[-1] if iteration_sims else None] *
                          len(ids)]

                iteration_sims.append(ifn.safe_pool(spM.create_ensemble_cell,
                                                    inputs,
                                                    nprocess,
                                                    in_mem=cfg.spatial_in_mem))

                # Wait untill all ensembles are created
                if not cfg.spatial_in_mem:
                    spM.wait_for_ensembles(step, 0)

                for j in range(cfg.max_iterations):  # Run spatial assim

                    if cfg.restart_run and j < prev_j:
                        continue
                    # add info to log
                    logging.info(f'step: {step} - j: {j}')

                    inputs = [list(grid[:, 0]), list(grid[:, 1]),
                              [step] * grid.shape[0],
                              [j] * grid.shape[0],
                              [iteration_sims[-1] if iteration_sims else None]
                              * len(ids)]

                    iteration_sims.append(ifn.safe_pool(spM.spatial_assim,
                                                        inputs,
                                                        nprocess,
                                                        in_mem=cfg.spatial_in_mem))

                    # Wait untill all ensembles are updated and remove prior
                    if cfg.spatial_in_mem:
                        if j != cfg.max_iterations:
                            iteration_sims[count] = None
                    else:
                        spM.wait_for_ensembles(step, 0, j)

                    count = count + 1

                count = count + 1  # Not a bug, is to skip the last iter

            iteration_sims = [x for x in iteration_sims if x is not None]

            inputs = [grid[:, 0], grid[:, 1], [iteration_sims] *
                      len(ids)]

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


def check_platform():

    # TODO: provide full suport for wind32

    if (sys.platform not in ("linux", "darwin")):
        raise Exception(sys.platform + " is not supported by MuSA yet")
    

def transform_results():
    """
    Function that transforms the results of the MuSA run into a more user-friendly format.
    This function is called at the end of the MuSA run.

    The zarr store contains still all time steps, whereas for the sites only daily outputs are retrieved.
    """
    #get the arguments from the config file
    args=getArgsFromCfg()

    if args.nc_maks_path is None:
        transformMuSA.saveFinalOutputToZarr(args, removeCells=args.remove_output_cells)
    else:
        transformMuSA.saveFinalOutputSitesOnly(args, removeCells=args.remove_output_cells)


if __name__ == "__main__":

    if cfg.parallelization in ["multiprocessing", "HPC.array"]:
        mp.set_start_method('spawn', force=True)

    check_platform()
    ifn.pre_cheks()

    MuSA()
    
    transform_results()
