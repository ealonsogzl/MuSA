#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions to interact with FSM.

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""
import os
import shutil
import subprocess
import tempfile
import datetime as dt
import pandas as pd
import config as cfg
import constants as cnt
import modules.met_tools as met
import secrets
import time
import copy
import pdcast as pdc
import warnings
import pyarrow as pa
import pyarrow.csv as csv
import numpy as np
import netCDF4 as nc
import modules.internal_fns as ifn
from statsmodels.stats.weightstats import DescrStatsW
if cfg.DAsord:
    from modules.user_optional_fns import snd_ord
# TODO: homogenize documentation format

if cfg.DAsord:
    model_columns = ("year", "month", "day", "hour", "snd",
                     "SWE", "Tsrf", "fSCA", "alb", 'H', 'LE',
                     tuple(cfg.DAord_names))
    # , "Tsnow1", "Tsnow2", "Tsnow3",

else:
    model_columns = ("year", "month", "day", "hour", "snd",
                     "SWE", "Tsrf", "fSCA", "alb", 'H', 'LE')
    # , "Tsnow1", "Tsnow2", "Tsnow3",)
# TODO: create a smarter function that changes the compilation of FSM and
# pd colum names dynamically to reduce/increase the model outputs.


def model_copy(y_id, x_id):
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
                                       (str(y_id) + "_" + str(x_id) + "_FSM"))
    else:
        token = secrets.token_urlsafe(16)  # safe path to run multiple sesions
        final_directory = os.path.join(to_directory,
                                       token,
                                       (str(y_id) + "_" + str(x_id) + "_FSM"))
    if os.path.exists(final_directory):
        shutil.rmtree(final_directory, ignore_errors=True)

    shutil.copytree(from_directory, final_directory)

    return final_directory


def write_nlst(temp_dest, params, step):

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

    # fSCA parameters
    filedata = filedata.replace('pySWEsca', str(params['SWEsca']))
    filedata = filedata.replace('pyTaf', str(params['Taf']))
    filedata = filedata.replace('pyCV', str(params['subgrid_cv']))

    # Vegetation characteristics
    filedata = filedata.replace('pyvegh', str(params['vegh']))
    filedata = filedata.replace('pyVAI', str(params['VAI']))
    filedata = filedata.replace('pyfsky', str(params['fsky']))

    # FSM2 internal parameters
    filedata = filedata.replace('pyalb0', str(params['alb0']))
    filedata = filedata.replace('pyasmn', str(params['asmn']))
    filedata = filedata.replace('pyasmx', str(params['asmx']))
    filedata = filedata.replace('pyeta0', str(params['eta0']))
    filedata = filedata.replace('pyhfsn', str(params['hfsn']))
    filedata = filedata.replace('pykfix', str(params['kfix']))
    filedata = filedata.replace('pyrcld', str(params['rcld']))
    filedata = filedata.replace('pyrfix', str(params['rfix']))
    filedata = filedata.replace('pyrgr0', str(params['rgr0']))
    filedata = filedata.replace('pyrhof', str(params['rhof']))
    filedata = filedata.replace('pyrhow', str(params['rhow']))
    filedata = filedata.replace('pyrmlt', str(params['rmlt']))
    filedata = filedata.replace('pySalb', str(params['Salb']))
    filedata = filedata.replace('pysnda', str(params['snda']))
    filedata = filedata.replace('pyTalb', str(params['Talb']))
    filedata = filedata.replace('pytcld', str(params['tcld']))
    filedata = filedata.replace('pytmlt', str(params['tmlt']))
    filedata = filedata.replace('pytrho', str(params['trho']))
    filedata = filedata.replace('pyWirr', str(params['Wirr']))
    filedata = filedata.replace('pyz0sn', str(params['z0sn']))

    # FSM2 soil parameters
    filedata = filedata.replace('pyfcly', str(params['fcly']))
    filedata = filedata.replace('pyfsnd', str(params['fsnd']))
    filedata = filedata.replace('pygsat', str(params['gsat']))
    filedata = filedata.replace('pyz0sf', str(params['z0sf']))

    # Vegetation parameters
    filedata = filedata.replace('pyacn0', str(params['acn0']))
    filedata = filedata.replace('pyacns', str(params['acns']))
    filedata = filedata.replace('pyavg0', str(params['avg0']))
    filedata = filedata.replace('pyavgs', str(params['avgs']))
    filedata = filedata.replace('pycvai', str(params['cvai']))
    filedata = filedata.replace('pygsnf', str(params['gsnf']))
    filedata = filedata.replace('pyhbas', str(params['hbas']))
    filedata = filedata.replace('pykext', str(params['kext']))
    filedata = filedata.replace('pyleaf', str(params['leaf']))
    filedata = filedata.replace('pysvai', str(params['svai']))
    filedata = filedata.replace('pytunl', str(params['tunl']))
    filedata = filedata.replace('pywcan', str(params['wcan']))

    if step == 0:
        filedata = filedata.replace('pyINIT', "\n")
    else:
        filedata = filedata.replace('pyINIT', "start_file = 'out_dump'")

    # Write the file out again
    with open(os.path.join(temp_dest, "nlst"), 'w') as file:
        file.write(filedata)


def model_compile():

    fsm_path = cfg.fsm_src_path
    # TODO: provide full suport for wind32

    bin_name = os.path.join(fsm_path, "FSM2")

    try:  # remove FSM binary if exists
        os.remove(bin_name)
    except OSError:
        pass

    with open(os.path.join(fsm_path, "compil_base.sh"), "r") as file:
        filedata = file.read()

    # Canopy options, to be updated if canopy module is enabled
    filedata = filedata.replace('pyCANMOD', str(cfg.CANMOD))
    filedata = filedata.replace('pyCANRAD', str(cfg.CANRAD))

    # Fortran optimization
    filedata = filedata.replace('pyOPT', cfg.OPTIMIZATION)

    # Parameterizations
    filedata = filedata.replace('pyALBEDO', str(cfg.ALBEDO))
    filedata = filedata.replace('pyCONDCT', str(cfg.CONDCT))
    filedata = filedata.replace('pyDENSITY', str(cfg.DENSITY))
    filedata = filedata.replace('pyEXCHNG', str(cfg.EXCHNG))
    filedata = filedata.replace('pyHYDROL', str(cfg.HYDROL))
    filedata = filedata.replace('pySGRAIN', str(cfg.SGRAIN))
    filedata = filedata.replace('pySNFRAC', str(cfg.SNFRAC))

    compile_path = os.path.join(fsm_path, "compil.sh")

    # Ensure the compile.sh file is not there
    if (os.path.exists(compile_path)):
        os.remove(compile_path)

    # Write the file out again
    with open(compile_path, "x") as file:
        file.write(filedata)

    # Forze executable permision
    os.chmod(compile_path, 509)

    bash_command = "cd " + fsm_path + " && " + "./compil.sh"
    subprocess.call(bash_command, shell=True)


def model_compile_HPC(HPC_task_id):
    # Compile FSM in the first HPC task
    fsm_path = cfg.fsm_src_path
    file_name = os.path.join(fsm_path, "FSM2")

    if HPC_task_id == 0:
        model_compile()
    else:
        while True:
            if os.path.isfile(file_name):
                break
            else:
                time.sleep(5)


def model_run(fsm_path):
    """
    Just run FSM in a directory

    Parameters:

    fsm_path : str
        Location of FSM binary

    Returns:

    [None]

    """
    fsm_exe_dir = os.path.join(fsm_path, "FSM2")
    order = [fsm_exe_dir]
    fsm_run_command = subprocess.call(
        order, cwd=fsm_path,
        stdin=open(os.path.join(fsm_path, "nlst"), "r"))
    # stdout=subprocess.DEVNULL)
    # https://stackoverflow.com/questions/41171791/how-to-suppress-or-capture-the-output-of-subprocess-run

    if fsm_run_command != 0:
        raise Exception("FSM failed")


def model_read_output(fsm_path, read_dump=True):
    """
    Read FSM outputs and return it in a dataframe

    Parameters:

    fsm_path : str
        Location of FSM outputs

   """
    # HACK: column/index names and number of columns/index are hardcoded here
    # Potential incompatibility in future versions of FSM.
    #  engine="pyarrow", do not waork with spaces, come back to this.
    state_dir = os.path.join(fsm_path, "out_stat.dat")

    dt = np.dtype([('year', 'int32'), ('month', 'int32'), ('day', 'int32'),
                   ('hour', 'float32'), ('snd', 'float32'),
                   ('SWE', 'float32'), ('Tsrf', 'float32'),
                   ('fSCA', 'float32'), ('alb', 'float32'),
                   ('H', 'float32'), ('LE', 'float32')])
    # ('Tsnow1', 'float32'), ('Tsnow2', 'float32'),
    # ('Tsnow3', 'float32')

    data = np.fromfile(state_dir, dtype=dt)
    state = pd.DataFrame(data)

    # Save some memory (downcast is slow and excessive)
    # TODO: directly change the types in the FSM code
    state['year'] = state.year.astype('uint16')
    state['month'] = state.month.astype('uint8')
    state['day'] = state.day.astype('uint8')
    state['hour'] = state.hour.astype('uint8')

    # add optional variables
    if cfg.DAsord:
        state = snd_ord(state)

    if (state.isnull().values.any()):
        error_dir = shutil.copytree(fsm_path,
                                    "./DATA/ERRORS/{cords}".
                                    format(cords=os.path.basename(fsm_path)),
                                    dirs_exist_ok=True)

        raise Exception('NaN found in FSM2 output.\n'
                        'Error dir can be found in :{error_dir}\n'
                        'Checklist:\n'
                        u'\u2022 Check main forcing, its units and internal '
                        'unit conversion in constants.py\n'
                        u'\u2022 Wrong perturbation_strategy?\n'
                        u'\u2022 Check sd_errors/mean_errors in constants.py,'
                        ' be carefull with the creation of glaciers\n'
                        u' If this is all right try some of this:\n'
                        u'\u2022 Change da_algorithm\n'
                        u'\u2022 Change FORTRAN compiler\n'.format(
                            error_dir=error_dir))

    if read_dump:
        dump_dir = os.path.join(fsm_path, "out_dump")
        dump = pd.read_csv(dump_dir, header=None, delim_whitespace=True,
                           names=list(range(4)))
        dump.index = ["albs", "Dsnw", "Nsnow", "Qcan", "Rgrn", "Slice", "Sliq",
                      "Sveg", "Tcan", "Tsnow", "Tsoil", "Tsrf", "Tveg", "Vsmc",
                      "fsnow", "D_a", "D_m", "D_ave"]

    if read_dump:
        return state, dump
    else:
        return state


def model_remove(fsm_path):
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

    # remove small vegetation
    temp_forz_def["vegh"].values[temp_forz_def["vegh"].values <
                                 temp_forz_def["hbas"].values + 1] = 0.0
    temp_forz_def["VAI"].values[temp_forz_def["vegh"].values <
                                temp_forz_def["hbas"].values + 1] = 0.0
                     
    # This attempts to overcome the 'drizzle' generated by lossy compression
    # methods such as 'netcdf-pack', which can destabilize the model.
    # We remove the precipitation below ~0.001mm/h by rounding.
    temp_forz_def.Prec = temp_forz_def.Prec.round(7)

    return temp_forz_def


def model_forcing_wrt(forcing_df, temp_dest, step=0):

    temp_forz_def = forcing_df.copy()
    temp_forz_def = stable_forcing(temp_forz_def)

    if cfg.precipitation_phase == "Harder":

        Rf, Sf = met.pp_psychrometric(temp_forz_def["Ta"].values,
                                      temp_forz_def["RH"].values,
                                      temp_forz_def["Prec"].values)

    elif cfg.precipitation_phase == "temp_thld":

        Rf, Sf = met.pp_temp_thld_log(temp_forz_def["Ta"].values,
                                      temp_forz_def["Prec"].values)

    elif cfg.precipitation_phase == "Liston":

        Rf, Sf = met.linear_liston(temp_forz_def["Ta"].values,
                                   temp_forz_def["Prec"].values)

    else:

        raise Exception("Precipitation phase partitioning not implemented")

    temp_forz_def.insert(6, "Sf", Sf)
    temp_forz_def.insert(7, "Rf", Rf)

    file_name = os.path.join(temp_dest, "input.txt")

    params = {"VAI": temp_forz_def.iloc[0]["VAI"],
              "vegh": temp_forz_def.iloc[0]["vegh"],
              "fsky": temp_forz_def.iloc[0]["fsky"],
              "Taf": temp_forz_def.iloc[0]["Taf"],
              "SWEsca": temp_forz_def.iloc[0]["SWEsca"],
              "subgrid_cv": temp_forz_def.iloc[0]["subgrid_cv"],
              "asmn": temp_forz_def.iloc[0]["asmn"],
              "asmx": temp_forz_def.iloc[0]["asmx"],
              "eta0": temp_forz_def.iloc[0]["eta0"],
              "hfsn": temp_forz_def.iloc[0]["hfsn"],
              "kfix": temp_forz_def.iloc[0]["kfix"],
              "rcld": temp_forz_def.iloc[0]["rcld"],
              "rfix": temp_forz_def.iloc[0]["rfix"],
              "rgr0": temp_forz_def.iloc[0]["rgr0"],
              "rhof": temp_forz_def.iloc[0]["rhof"],
              "rhow": temp_forz_def.iloc[0]["rhow"],
              "rmlt": temp_forz_def.iloc[0]["rmlt"],
              "Salb": temp_forz_def.iloc[0]["Salb"],
              "snda": temp_forz_def.iloc[0]["snda"],
              "Talb": temp_forz_def.iloc[0]["Talb"],
              "tcld": temp_forz_def.iloc[0]["tcld"],
              "tmlt": temp_forz_def.iloc[0]["tmlt"],
              "trho": temp_forz_def.iloc[0]["trho"],
              "Wirr": temp_forz_def.iloc[0]["Wirr"],
              "z0sn": temp_forz_def.iloc[0]["z0sn"],
              "alb0": temp_forz_def.iloc[0]["alb0"],
              "fcly": temp_forz_def.iloc[0]["fcly"],
              "fsnd": temp_forz_def.iloc[0]["fsnd"],
              "gsat": temp_forz_def.iloc[0]["gsat"],
              "z0sf": temp_forz_def.iloc[0]["z0sf"],
              "acn0": temp_forz_def.iloc[0]["acn0"],
              "acns": temp_forz_def.iloc[0]["acns"],
              "avg0": temp_forz_def.iloc[0]["avg0"],
              "avgs": temp_forz_def.iloc[0]["avgs"],
              "cvai": temp_forz_def.iloc[0]["cvai"],
              "gsnf": temp_forz_def.iloc[0]["gsnf"],
              "hbas": temp_forz_def.iloc[0]["hbas"],
              "kext": temp_forz_def.iloc[0]["kext"],
              "leaf": temp_forz_def.iloc[0]["leaf"],
              "svai": temp_forz_def.iloc[0]["svai"],
              "tunl": temp_forz_def.iloc[0]["tunl"],
              "wcan": temp_forz_def.iloc[0]["wcan"]}

    write_nlst(temp_dest, params, step)

    del temp_forz_def["Prec"]  # remove total precipitation, we use the rates

    del temp_forz_def["VAI"]
    del temp_forz_def["vegh"]
    del temp_forz_def["fsky"]
    del temp_forz_def["Taf"]
    del temp_forz_def["SWEsca"]
    del temp_forz_def["subgrid_cv"]
    del temp_forz_def["asmn"]
    del temp_forz_def["asmx"]
    del temp_forz_def["eta0"]
    del temp_forz_def["hfsn"]
    del temp_forz_def["kfix"]
    del temp_forz_def["rcld"]
    del temp_forz_def["rfix"]
    del temp_forz_def["rgr0"]
    del temp_forz_def["rhof"]
    del temp_forz_def["rhow"]
    del temp_forz_def["rmlt"]
    del temp_forz_def["Salb"]
    del temp_forz_def["snda"]
    del temp_forz_def["Talb"]
    del temp_forz_def["tcld"]
    del temp_forz_def["tmlt"]
    del temp_forz_def["trho"]
    del temp_forz_def["Wirr"]
    del temp_forz_def["z0sn"]
    del temp_forz_def["alb0"]
    del temp_forz_def["fcly"]
    del temp_forz_def["fsnd"]
    del temp_forz_def["gsat"]
    del temp_forz_def["z0sf"]
    del temp_forz_def["acn0"]
    del temp_forz_def["acns"]
    del temp_forz_def["avg0"]
    del temp_forz_def["avgs"]
    del temp_forz_def["cvai"]
    del temp_forz_def["gsnf"]
    del temp_forz_def["hbas"]
    del temp_forz_def["kext"]
    del temp_forz_def["leaf"]
    del temp_forz_def["svai"]
    del temp_forz_def["tunl"]
    del temp_forz_def["wcan"]

    # TODO: Explore export to binary
    # https://stackoverflow.com/questions/44074122/reading-in-fortran-binaries-written-with-python
    # temp_forz_def.to_csv(file_name, sep="\t",
    #                     header=False,
    #                     index=False,
    #                     chunksize=1000)

    # write the csv with pyarrow
    temp_forz_def = pa.Table.from_pandas(temp_forz_def)
    csv.write_csv(temp_forz_def,
                  file_name,
                  csv.WriteOptions(include_header=False,
                                   batch_size=8760,
                                   delimiter=' '))


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

    state_columns = model_columns

    return state_columns.index(var)


def storeDA(Result_df, step_results, observations_sbst, error_sbst,
            time_dict, step):

    vars_to_perturbate = cfg.vars_to_perturbate
    var_to_assim = cfg.var_to_assim
    error_names = cfg.obs_error_var_names

    rowIndex = Result_df.index[time_dict["Assimilaiton_steps"][step]:
                               time_dict["Assimilaiton_steps"][step + 1]]

    if len(var_to_assim) > 1:
        for i, var in enumerate(var_to_assim):
            Result_df.loc[rowIndex, var] = observations_sbst[:, i]
            Result_df.loc[rowIndex, error_names[i]] = error_sbst[:, i]
    else:
        var = var_to_assim[0]
        Result_df.loc[rowIndex, var] = observations_sbst
        Result_df.loc[rowIndex, error_names] = error_sbst

    # Add perturbation parameters to Results
    for var_p in vars_to_perturbate:
        Result_df.loc[rowIndex, var_p +
                      "_noise_mean"] = step_results[var_p + "_noise_mean"]
        Result_df.loc[rowIndex, var_p +
                      "_noise_sd"] = step_results[var_p + "_noise_sd"]


def storeOL(OL_FSM, Ensemble, observations_sbst, time_dict, step):

    ol_data = Ensemble.origin_state.copy()

    # remove time ids fomr FSM output
    ol_data.drop(ol_data.columns[[0, 1, 2, 3]], axis=1, inplace=True)
    # TODO: modify directly FSM code to not to output time id's

    # Store colums
    for n, name_col in enumerate(ol_data.columns):
        OL_FSM[name_col] = ol_data.iloc[:, [n]].to_numpy()


def store_sim(updated_Sim, sd_Sim, Ensemble,
              time_dict, step, MCMC=False, save_prior=False):

    if MCMC:
        list_state = copy.deepcopy(Ensemble.state_members_mcmc)
    else:
        list_state = copy.deepcopy(Ensemble.state_membres)
    # remove time ids fomr FSM output
    # TODO: modify directly FSM code to not to output time id's
    for lst in range(len(list_state)):
        data = list_state[lst]
        data.drop(data.columns[[0, 1, 2, 3]], axis=1, inplace=True)

    rowIndex = updated_Sim.index[time_dict["Assimilaiton_steps"][step]:
                                 time_dict["Assimilaiton_steps"][step + 1]]

    # Get updated columns
    if save_prior:
        pesos = np.ones_like(Ensemble.wgth)
    else:
        pesos = Ensemble.wgth

    for n, name_col in enumerate(list(list_state[0].columns)):
        # create matrix of colums
        col_arr = [list_state[x].iloc[:, n].to_numpy()
                   for x in range(len(list_state))]
        col_arr = np.vstack(col_arr)

        d1 = DescrStatsW(col_arr, weights=pesos)
        average_sim = d1.mean
        sd_sim = d1.std

        updated_Sim.loc[rowIndex, name_col] = average_sim
        sd_Sim.loc[rowIndex, name_col] = sd_sim


def init_result(del_t, DA=False):

    if DA:
        # Concatenate
        col_names = ["Date"]

        # Create results dataframe
        Results = pd.DataFrame(np.nan, index=range(len(del_t)),
                               columns=col_names)

        Results["Date"] = [x.strftime('%d/%m/%Y-%H:%S') for x in del_t]
        return Results

    else:
        # Concatenate
        col_names = ["Date", "snd", "SWE", "Tsrf",
                     "fSCA", "alb", "H", "LE"]

        # Create results dataframe
        Results = pd.DataFrame(np.nan, index=range(len(del_t)),
                               columns=col_names)

        Results["Date"] = [x.strftime('%d/%m/%Y-%H:%S') for x in del_t]

        Results["snd"] = [np.nan for x in del_t]
        Results["SWE"] = [np.nan for x in del_t]
        Results["Tsrf"] = [np.nan for x in del_t]
        Results["alb"] = [np.nan for x in del_t]
        Results["fSCA"] = [np.nan for x in del_t]
        Results["H"] = [np.nan for x in del_t]
        Results["LE"] = [np.nan for x in del_t]

        Results = Results.astype({'snd': 'float32',
                                  'SWE': 'float32',
                                  'Tsrf': 'float32',
                                  'fSCA': 'float32',
                                  'alb': 'float32',
                                  'H': 'float32',
                                  'LE': 'float32'})

        return Results


def forcing_table(lat_idx, lon_idx, step=0):

    nc_forcing_path = cfg.nc_forcing_path
    frocing_var_names = cfg.frocing_var_names
    param_var_names = cfg.param_var_names
    date_ini = cfg.date_ini
    date_end = cfg.date_end
    intermediate_path = cfg.intermediate_path

    # Path to intermediate file
    final_directory = os.path.join(intermediate_path,
                                   (str(lat_idx) + "_" +
                                    str(lon_idx) + ".pkl"))

    # try to read the forcing from a dumped file
    if os.path.exists(final_directory) and (cfg.restart_forcing or
                                            (cfg.implementation ==
                                             "Spatial_propagation" and
                                             step != 0)):

        forcing_df = ifn.io_read(final_directory)

    else:

        short_w = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                       frocing_var_names["SW_var_name"],
                                       date_ini, date_end)

        long_wave = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                         frocing_var_names["LW_var_name"],
                                         date_ini, date_end)

        prec = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                    frocing_var_names["Precip_var_name"],
                                    date_ini, date_end)

        temp = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                    frocing_var_names["Temp_var_name"],
                                    date_ini, date_end)

        rel_humidity = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                            frocing_var_names["RH_var_name"],
                                            date_ini, date_end)

        wind = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                    frocing_var_names["Wind_var_name"],
                                    date_ini, date_end)

        if frocing_var_names["Press_var_name"] == "from_DEM":

            with nc.Dataset(cfg.dem_path) as dem:
                topo = dem.variables[cfg.nc_dem_varname][lat_idx, lon_idx]
                sfc_pres = met.pres_from_dem(topo)
                press = np.full_like(wind, sfc_pres)

        else:
            press = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                         frocing_var_names["Press_var_name"],
                                         date_ini, date_end)

        # Search for parameters or use the default settings
        # vegetation parameters
        try:
            vegh = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["vegh_var_name"],
                                        date_ini, date_end)
        except KeyError:
            vegh = np.repeat(cnt.vegh, len(prec))

        try:
            VAI = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                       param_var_names["VAI_var_name"],
                                       date_ini, date_end)
        except KeyError:
            VAI = np.repeat(cnt.VAI, len(prec))

        try:
            fsky = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["fsky_var_name"],
                                        date_ini, date_end)
        except KeyError:
            fsky = np.repeat(cnt.fsky, len(prec))

        # FSM2 internal parameters
        try:
            alb0 = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["alb0_var_name"],
                                        date_ini, date_end)
        except KeyError:
            alb0 = np.repeat(cnt.alb0, len(prec))

        try:
            SWEsca = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                          param_var_names["SWEsca_var_name"],
                                          date_ini, date_end)
        except KeyError:
            SWEsca = np.repeat(cnt.SWEsca, len(prec))

        try:
            Taf = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                       param_var_names["Taf_var_name"],
                                       date_ini, date_end)
        except KeyError:
            Taf = np.repeat(cnt.Taf, len(prec))

        try:
            subgrid_cv = ifn.nc_array_forcing(nc_forcing_path,
                                              lat_idx, lon_idx,
                                              param_var_names["cv_var_name"],
                                              date_ini, date_end)
        except KeyError:
            subgrid_cv = np.repeat(cnt.subgrid_cv, len(prec))

        try:
            asmn = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["asmn_var_name"],
                                        date_ini, date_end)
        except KeyError:
            asmn = np.repeat(cnt.asmn, len(prec))
        try:
            asmx = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["asmx_var_name"],
                                        date_ini, date_end)
        except KeyError:
            asmx = np.repeat(cnt.asmx, len(prec))
        try:
            eta0 = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["eta0_var_name"],
                                        date_ini, date_end)
        except KeyError:
            eta0 = np.repeat(cnt.eta0, len(prec))
        try:
            hfsn = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["hfsn_var_name"],
                                        date_ini, date_end)
        except KeyError:
            hfsn = np.repeat(cnt.hfsn, len(prec))
        try:
            kfix = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["kfix_var_name"],
                                        date_ini, date_end)
        except KeyError:
            kfix = np.repeat(cnt.kfix, len(prec))
        try:
            rcld = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["rcld_var_name"],
                                        date_ini, date_end)
        except KeyError:
            rcld = np.repeat(cnt.rcld, len(prec))
        try:
            rfix = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["rfix_var_name"],
                                        date_ini, date_end)
        except KeyError:
            rfix = np.repeat(cnt.rfix, len(prec))
        try:
            rgr0 = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["rgr0_var_name"],
                                        date_ini, date_end)
        except KeyError:
            rgr0 = np.repeat(cnt.rgr0, len(prec))
        try:
            rhof = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["rhof_var_name"],
                                        date_ini, date_end)
        except KeyError:
            rhof = np.repeat(cnt.rhof, len(prec))
        try:
            rhow = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["rhow_var_name"],
                                        date_ini, date_end)
        except KeyError:
            rhow = np.repeat(cnt.rhow, len(prec))
        try:
            rmlt = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["rmlt_var_name"],
                                        date_ini, date_end)
        except KeyError:
            rmlt = np.repeat(cnt.rmlt, len(prec))
        try:
            Salb = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["Salb_var_name"],
                                        date_ini, date_end)
        except KeyError:
            Salb = np.repeat(cnt.Salb, len(prec))
        try:
            snda = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["snda_var_name"],
                                        date_ini, date_end)
        except KeyError:
            snda = np.repeat(cnt.snda, len(prec))
        try:
            Talb = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["Talb_var_name"],
                                        date_ini, date_end)
        except KeyError:
            Talb = np.repeat(cnt.Talb, len(prec))
        try:
            tcld = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["tcld_var_name"],
                                        date_ini, date_end)
        except KeyError:
            tcld = np.repeat(cnt.tcld, len(prec))
        try:
            tmlt = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["tmlt_var_name"],
                                        date_ini, date_end)
        except KeyError:
            tmlt = np.repeat(cnt.tmlt, len(prec))
        try:
            trho = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["trho_var_name"],
                                        date_ini, date_end)
        except KeyError:
            trho = np.repeat(cnt.trho, len(prec))
        try:
            Wirr = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["Wirr_var_name"],
                                        date_ini, date_end)
        except KeyError:
            Wirr = np.repeat(cnt.Wirr, len(prec))
        try:
            z0sn = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["z0sn_var_name"],
                                        date_ini, date_end)
        except KeyError:
            z0sn = np.repeat(cnt.z0sn, len(prec))

        # FSM2 soil parameters
        try:
            fcly = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["fcly_var_name"],
                                        date_ini, date_end)
        except KeyError:
            fcly = np.repeat(cnt.fcly, len(prec))
        try:
            fsnd = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["fsnd_var_name"],
                                        date_ini, date_end)
        except KeyError:
            fsnd = np.repeat(cnt.fsnd, len(prec))
        try:
            gsat = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["gsat_var_name"],
                                        date_ini, date_end)
        except KeyError:
            gsat = np.repeat(cnt.gsat, len(prec))
        try:
            z0sf = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["z0sf_var_name"],
                                        date_ini, date_end)
        except KeyError:
            z0sf = np.repeat(cnt.z0sf, len(prec))

        # FSM2 vegetation parameters
        try:
            acn0 = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["acn0_var_name"],
                                        date_ini, date_end)
        except KeyError:
            acn0 = np.repeat(cnt.acn0, len(prec))
        try:
            acns = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["acns_var_name"],
                                        date_ini, date_end)
        except KeyError:
            acns = np.repeat(cnt.acns, len(prec))
        try:
            avg0 = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["avg0_var_name"],
                                        date_ini, date_end)
        except KeyError:
            avg0 = np.repeat(cnt.avg0, len(prec))
        try:
            avgs = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["avgs_var_name"],
                                        date_ini, date_end)
        except KeyError:
            avgs = np.repeat(cnt.avgs, len(prec))
        try:
            cvai = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["cvai_var_name"],
                                        date_ini, date_end)
        except KeyError:
            cvai = np.repeat(cnt.cvai, len(prec))
        try:
            hbas = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["hbas_var_name"],
                                        date_ini, date_end)
        except KeyError:
            hbas = np.repeat(cnt.hbas, len(prec))
        try:
            gsnf = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["gsnf_var_name"],
                                        date_ini, date_end)
        except KeyError:
            gsnf = np.repeat(cnt.gsnf, len(prec))
        try:
            kext = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["kext_var_name"],
                                        date_ini, date_end)
        except KeyError:
            kext = np.repeat(cnt.kext, len(prec))
        try:
            leaf = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["leaf_var_name"],
                                        date_ini, date_end)
        except KeyError:
            leaf = np.repeat(cnt.leaf, len(prec))
        try:
            svai = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["svai_var_name"],
                                        date_ini, date_end)
        except KeyError:
            svai = np.repeat(cnt.svai, len(prec))
        try:
            tunl = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["tunl_var_name"],
                                        date_ini, date_end)
        except KeyError:
            tunl = np.repeat(cnt.tunl, len(prec))
        try:
            wcan = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["wcan_var_name"],
                                        date_ini, date_end)
        except KeyError:
            wcan = np.repeat(cnt.wcan, len(prec))

        date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
        del_t = ifn.generate_dates(date_ini, date_end)

        forcing_df = pd.DataFrame({"year": del_t,
                                  "month": del_t,
                                   "day": del_t,
                                   "hours": del_t,
                                   "SW": short_w,
                                   "LW": long_wave,
                                   "Prec": prec,
                                   "Ta": temp,
                                   "RH": rel_humidity,
                                   "Ua": wind,
                                   "Ps": press,
                                   "VAI": VAI,
                                   "vegh": vegh,
                                   "fsky": fsky,
                                   "Taf": Taf,
                                   "SWEsca": SWEsca,
                                   "subgrid_cv": subgrid_cv,
                                   "asmn": asmn,
                                   "asmx": asmx,
                                   "eta0": eta0,
                                   "hfsn": hfsn,
                                   "kfix": kfix,
                                   "rcld": rcld,
                                   "rfix": rfix,
                                   "rgr0": rgr0,
                                   "rhof": rhof,
                                   "rhow": rhow,
                                   "rmlt": rmlt,
                                   "Salb": Salb,
                                   "snda": snda,
                                   "Talb": Talb,
                                   "tcld": tcld,
                                   "tmlt": tmlt,
                                   "trho": trho,
                                   "Wirr": Wirr,
                                   "z0sn": z0sn,
                                   "alb0": alb0,
                                   "fcly": fcly,
                                   "fsnd": fsnd,
                                   "gsat": gsat,
                                   "z0sf": z0sf,
                                   "acn0": acn0,
                                   "acns": acns,
                                   "avg0": avg0,
                                   "avgs": avgs,
                                   "cvai": cvai,
                                   "gsnf": gsnf,
                                   "hbas": hbas,
                                   "kext": kext,
                                   "leaf": leaf,
                                   "svai": svai,
                                   "tunl": tunl,
                                   "wcan": wcan})

        forcing_df["year"] = forcing_df["year"].dt.year
        forcing_df["month"] = forcing_df["month"].dt.month
        forcing_df["day"] = forcing_df["day"].dt.day
        forcing_df["hours"] = forcing_df["hours"].dt.hour

        forcing_df = unit_conversion(forcing_df)
        # write intermediate file to avoid re-reading the nc files
        ifn.io_write(final_directory, forcing_df)

        if len(del_t) != len(forcing_df.index):
            raise Exception("date_end - date_ini longuer than forcing")

    return forcing_df


def unit_conversion(forcing_df):

    forcing_offset = cnt.forcing_offset
    forcing_multiplier = cnt.forcing_multiplier

    forcing_df.SW = forcing_df.SW * forcing_multiplier["SW"]
    forcing_df.LW = forcing_df.LW * forcing_multiplier["LW"]
    forcing_df.Prec = forcing_df.Prec * forcing_multiplier["Prec"]
    forcing_df.Ta = forcing_df.Ta * forcing_multiplier["Ta"]
    forcing_df.RH = forcing_df.RH * forcing_multiplier["RH"]
    forcing_df.Ua = forcing_df.Ua * forcing_multiplier["Ua"]
    forcing_df.Ps = forcing_df.Ps * forcing_multiplier["Ps"]

    forcing_df.SW = forcing_df.SW + forcing_offset["SW"]
    forcing_df.LW = forcing_df.LW + forcing_offset["LW"]
    forcing_df.Prec = forcing_df.Prec + forcing_offset["Prec"]
    forcing_df.Ta = forcing_df.Ta + forcing_offset["Ta"]
    forcing_df.RH = forcing_df.RH + forcing_offset["RH"]
    forcing_df.Ua = forcing_df.Ua + forcing_offset["Ua"]
    forcing_df.Ps = forcing_df.Ps + forcing_offset["Ps"]

    # Save some space
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        forcing_df = pdc.downcast(forcing_df,
                                  numpy_dtypes_only=True)

    return forcing_df

