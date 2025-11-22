#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates the correlated parameter maps needed for
spatial propagation. It can be considered a necessary pre-processor
in case of enabling spatial propagation.

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import modules.spatialMuSA as spM
import modules.internal_fns as ifn
import sys
import config as cfg
import numpy as np
import os


def main():

    if cfg.implementation != 'Spatial_propagation':
        print("implementation != 'Spatial_propagation',"
              " this script is not necesary")
        return None

    # get DA steps
    ini_DA_window = spM.domain_steps()

    if cfg.parallelization == "multiprocessing":
        if cfg.MPI:
            raise Exception('MPI not suppoted yet for pre_main_spatial.py')
        # Sample prior parameters
        spM.generate_prior_maps_onenode(ini_DA_window)

    elif cfg.parallelization == "HPC.array":

        HPC_task_number = int(sys.argv[1])
        HPC_task_id = int(sys.argv[2])-1

        if HPC_task_number > len(ini_DA_window):
            print('There are {DAW} DAWs and {tasks} tasks were requested,'
                  ' It is recommended to reduce the number of jobs'
                  ' in the array'.format(DAW=str(len(ini_DA_window)),
                                         tasks=str(HPC_task_number)
                                         ))

        ids = np.arange(0, len(ini_DA_window))
        ids = ids % HPC_task_number == HPC_task_id

        spM.generate_prior_maps_onenode(ini_DA_window[ids])

        dist_name = os.path.join(cfg.spatial_propagation_storage_path,
                                 'dist_0.nc')
        ifn.change_chunk_size_nccopy(dist_name)


if __name__ == "__main__":
    print('Running MuSA pre-processor')

    main()
