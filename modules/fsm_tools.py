#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions to interact with FSM.

Author: Esteban Alonso González - e.alonsogzl@gmail.com
"""
import os
import shutil
import subprocess
import tempfile
import pandas as pd
import config as cfg
import modules.met_tools as met
import secrets

# TODO: homogenize documentation format


def fsm_copy(x_id, y_id):
    """
    Copy FSM code to a new location

    Parameters:
    ----------
    X : int
        x coordinate
    Y : int
        y coordinate
    from_directory : str
        FSM source code directory
    to_directory : str
        Destination. If None, FSM copied to a temporary filesystem

    Returns:
    ----------
    final_directory : str
        Final destination of the FSM code
   """
    from_directory = cfg.fsm_src_path
    to_directory = cfg.tmp_path

    if to_directory is None:
        tmp_dir = tempfile.mkdtemp()
        final_directory = os.path.join(tmp_dir,
                                       (str(x_id) + "_" + str(y_id) + "_FSM"))
    else:
        token = secrets.token_urlsafe(16)  # safe path to run multiple sesions
        final_directory = os.path.join(to_directory,
                                       token,
                                       (str(x_id) + "_" + str(y_id) + "_FSM"))
    if os.path.exists(final_directory):
        shutil.rmtree(final_directory, ignore_errors=True)

    shutil.copytree(from_directory, final_directory)

    return final_directory


def write_nlst(temp_dest, step):

    Dzsnow = cfg.Dzsnow
    Nsmax = len(Dzsnow)

    Dzsnow = [str(element) for element in Dzsnow]
    Dzsnow = ", ".join(Dzsnow)

    # Read in the file
    with open(os.path.join(temp_dest, "nlst_base"), "r") as file:
        filedata = file.read()

    # Replace number of layers and thickness
    filedata = filedata.replace('pyNSMAX', str(Nsmax))
    filedata = filedata.replace('pyDZSNOW', Dzsnow)

    if step == 0:
        filedata = filedata.replace('pyINIT', "\n")
    else:
        filedata = filedata.replace('pyINIT', "start_file = 'out_dump'")

    # Write the file out again
    with open(os.path.join(temp_dest, "nlst"), 'w') as file:
        file.write(filedata)


def fsm_compile(temp_dest):

    # fsm_path = cfg.fsm_src_path
    # TODO: provide full suport for wind32
    with open(os.path.join(temp_dest, "compil_base.sh"), "r") as file:
        filedata = file.read()

    # Canopy options, to be updated if canopy module is enabled
    filedata = filedata.replace('pyCANMOD', str(1))
    filedata = filedata.replace('pyCANRAD', str(1))

    # Fortran compiler
    filedata = filedata.replace('pyFC', cfg.FC)

    # Parameterizations
    filedata = filedata.replace('pyALBEDO', str(cfg.ALBEDO))
    filedata = filedata.replace('pyCONDCT', str(cfg.CONDCT))
    filedata = filedata.replace('pyDENSITY', str(cfg.DENSITY))
    filedata = filedata.replace('pyEXCHNG', str(cfg.EXCHNG))
    filedata = filedata.replace('pyHYDROL', str(cfg.HYDROL))
    filedata = filedata.replace('pySNFRAC', str(cfg.SNFRAC))

    compile_path = os.path.join(temp_dest, "compil.sh")

    # Ensure the compile.sh file is not there
    if (os.path.exists(compile_path)):
        os.remove(compile_path)

    # Write the file out again
    with open(compile_path, "x") as file:
        file.write(filedata)

    # Forze executable permision
    os.chmod(compile_path, 509)

    bash_command = "cd " + temp_dest + " && " + "./compil.sh"
    subprocess.call(bash_command, shell=True)


def fsm_run(fsm_path, step=0):
    """
    Just run FSM in a directory

    Parameters:

    fsm_path : str
        Location of FSM binary

    Returns:

    [None]

   """
    write_nlst(fsm_path, step)

    fsm_exe_dir = os.path.join(fsm_path, "FSM2")
    order = fsm_exe_dir + " < nlst"
    fsm_run_comand = subprocess.call(order, shell=True, cwd=fsm_path)
    if fsm_run_comand != 0:
        raise Exception("FSM failed")


def fsm_read_output(fsm_path, read_dump=True):
    """
    Read FSM outputs and return it in a dataframe

    Parameters:

    fsm_path : str
        Location of FSM outputs

   """
    # HACK: column/index names and number of columns/index are hardcoded here
    # Potential incompatibility in future versions of FSM.
    state_dir = os.path.join(fsm_path, "out_stat.txt")
    state = pd.read_csv(state_dir, header=None, delim_whitespace=True)
    state.columns = ["year", "month", "day", "hour", "snd",
                     "SWE", "Tsrf", "fSCA", "alb"]
    if (state.isnull().values.any()):
        raise Exception('''nan found in FSM2 output: check forcing or
                        change FORTRAN compiler''')

    if read_dump:
        dump_dir = os.path.join(fsm_path, "out_dump")
        dump = pd.read_csv(dump_dir, header=None, delim_whitespace=True,
                           names=list(range(4)))
        dump.index = ["albs", "Dsnw", "Nsnow", "Qcan", "Rgrn", "Slice", "Sliq",
                      "Sveg", "Tcan", "Tsnow", "Tsoil", "Tsrf", "Tveg", "Vsmc"]

    if read_dump:
        return state, dump
    else:
        return state


def fsm_remove(fsm_path):
    """
    Remove the temporal FSM directory
    Parameters
    ----------
    fsm_path : string
        FSM temporal location.

    Returns
    -------
    None.

    """
    if os.path.exists(fsm_path):
        if cfg.tmp_path is not None:  # Remove the random path
            fsm_path = os.path.split(fsm_path)[0]

        shutil.rmtree(fsm_path, ignore_errors=True)


def write_input(fsm_path, fsm_input_data):
    """
    Write the FSM input to the temporal directory

    Parameters
    ----------
    fsm_path : string
        FSM temporal location.

    fsm_input_data : pandas dataframe
        FSM input data.

    Returns
    -------
    None.

    """
    file_name = os.path.join(fsm_path, "input.txt")
    fsm_input_data.to_csv(file_name, header=None, index=None,
                          sep=' ', mode='w')


def stable_forcing(forcing_df):

    temp_forz_def = forcing_df.copy()

    # Negative SW to 0
    temp_forz_def["SW"].values[temp_forz_def["SW"].values < 0] = 0

    # Negative LW to 0
    temp_forz_def["LW"].values[temp_forz_def["LW"].values < 0] = 0

    # Negative Prec to 0
    temp_forz_def["Prec"].values[temp_forz_def["Prec"].values < 0] = 0

    # Negative wind to 0
    temp_forz_def["Ua"].values[temp_forz_def["Ua"].values < 0] = 0

    # Not to allow HR values out of 1-100%
    temp_forz_def["RH"].values[temp_forz_def["RH"].values > 100] = 100
    # 1% of RH is actually almost impossible, increase?
    temp_forz_def["RH"].values[temp_forz_def["RH"].values < 0] = 1

    return temp_forz_def


def fsm_forcing_wrt(forcing_df, temp_dest):

    temp_forz_def = forcing_df.copy()
    temp_forz_def = stable_forcing(forcing_df)

    if cfg.precipitation_phase == "Harder":

        Rf, Sf = met.pp_psychrometric(temp_forz_def["Ta"],
                                      temp_forz_def["RH"],
                                      temp_forz_def["Prec"])

    elif cfg.precipitation_phase == "temp_thld":

        Rf, Sf = met.pp_temp_thld_log(temp_forz_def["Ta"],
                                      temp_forz_def["Prec"])
    else:

        raise Exception("Precipitation phase partitioning not implemented")

    temp_forz_def.insert(6, "Sf", Sf)
    temp_forz_def.insert(7, "Rf", Rf)
    del temp_forz_def["Prec"]

    file_name = os.path.join(temp_dest, "input.txt")

    temp_forz_def.to_csv(file_name, sep="\t", header=False, index=False)


def write_dump(dump, fsm_path):
    """
    Parameters
    ----------
    dump : TYPE
        DESCRIPTION.
    fsm_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    dump_copy = dump.copy()
    file_name = os.path.join(fsm_path, "out_dump")
    dump_copy.iloc[2, 0] = str(int(dump_copy.iloc[2, 0]))
    dump_copy.to_csv(file_name, header=None, index=None, sep=' ', mode='w',
                     na_rep='NaN')


def get_var_state_position(var):

    state_columns = ("year", "month", "day", "hour", "snd",
                     "SWE", "Tsrf", "fSCA", "alb", "SCA")

    return state_columns.index(var)


def get_layers(hs):

    if hs <= 0:
        l1 = 0
        l2 = 0
        l3 = 0
        n = 0

    elif hs <= 0.2:
        l1 = hs
        l2 = 0
        l3 = 0
        n = 1

    elif 0.2 < hs <= 0.5:
        l1 = 0.1
        l2 = hs - 0.1
        l3 = 0
        n = 2

    elif hs > 0.5:
        l1 = 0.1
        l2 = 0.2
        l3 = hs - 0.3
        n = 3

    return int(n), float(l1), float(l2), float(l3)
