#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean paths
Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""
import config as cfg
import glob
import os


def clean():

    fsm_path = cfg.fsm_src_path
    save_ensemble_path = cfg.save_ensemble_path
    spatial_propagation_storage_path = cfg.spatial_propagation_storage_path

    bin_name = [os.path.join(fsm_path, "FSM2")]
    Ensembl_files = glob.glob(save_ensemble_path + "*ensbl*.gz")
    spatial_stuff = glob.glob(spatial_propagation_storage_path+'*')

    rm_files = bin_name + Ensembl_files + spatial_stuff

    for f in rm_files:
        if os.path.isfile(f):
            os.remove(f)


if __name__ == "__main__":

    clean()
