#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions to interact with FSM.

Author: Esteban Alonso González - alonsoe@cesbio.cnes.fr
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
import numpy as np
import modules.filters as flt
import modules.internal_fns as ifn
if cfg.DAsord:
    from modules.user_optional_fns import snd_ord
# TODO: homogenize documentation format

if cfg.DAsord:
    model_columns = ("year", "month", "day", "hour", "snd",
                     "SWE", "Tsrf", "fSCA", "alb", 'H', 'LE',
                     tuple(cfg.DAord_names))
else:
    model_columns = ("year", "month", "day", "hour", "snd",
                     "SWE", "Tsrf", "fSCA", "alb", 'H', 'LE')


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
    filedata = filedata.replace('pycv', str(params['subgrid_cv']))

    # Canopy parameters
    filedata = filedata.replace('pyvegh', str(params['vegh']))
    filedata = filedata.replace('pyVAI', str(params['VAI']))
    filedata = filedata.replace('pyfsky', str(params['fsky']))
    filedata = filedata.replace('pyhbas', str(params['hbas']))

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

    # Fortran compiler
    # filedata = filedata.replace('pyFC', cfg.FC)

    # Parameterizations
    filedata = filedata.replace('pyALBEDO', str(cfg.ALBEDO))
    filedata = filedata.replace('pyCONDCT', str(cfg.CONDCT))
    filedata = filedata.replace('pyDENSITY', str(cfg.DENSITY))
    filedata = filedata.replace('pyEXCHNG', str(cfg.EXCHNG))
    filedata = filedata.replace('pyHYDROL', str(cfg.HYDROL))
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


def model_compile_PBS(pbs_task_id):
    # Compile FSM in the first PBS task
    fsm_path = cfg.fsm_src_path
    file_name = os.path.join(fsm_path, "FSM2")

    if pbs_task_id == 0:
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
    order = fsm_exe_dir + " < nlst"
    # TODO: investigate pexpect. seems fast, but it do not wait for output
    # https://stackoverflow.com/questions/69720755/run-a-program-from-python-several-times-whitout-initialize-different-shells
    fsm_run_comand = subprocess.call(order, shell=True, cwd=fsm_path)
    if fsm_run_comand != 0:
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

    data = np.fromfile(state_dir, dtype=dt)
    state = pd.DataFrame(data)
    # Save some memory
    state = state.astype({'year': 'int16',
                          'month': 'int8',
                          'day': 'int8',
                          'hour': 'int8',
                          'snd': 'float32',
                          'SWE': 'float32',
                          'Tsrf': 'float32',
                          'fSCA': 'float32',
                          'alb': 'float32',
                          'H': 'float32',
                          'LE': 'float32'})

    # add optional variables
    if cfg.DAsord:
        state = snd_ord(state)

    if (state.isnull().values.any()):
        raise Exception('NaN found in FSM2 output.\n Checklist:\n'
                        u'\u2022 Check main forcing, its units and internal '
                        'unit conversion in constants.py\n'
                        u'\u2022 Wrong perturbation_strategy?\n'
                        u'\u2022 Check sd_errors/mean_errors in constants.py\n'
                        u'\u2022 Change FORTRAN compiler\n'
                        u'\u2022 Change da_algorithm\n')

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

    return temp_forz_def


def model_forcing_wrt(forcing_df, temp_dest, step=0):

    temp_forz_def = forcing_df.copy()
    temp_forz_def = stable_forcing(temp_forz_def)

    if cfg.precipitation_phase == "Harder":

        Rf, Sf = met.pp_psychrometric(temp_forz_def["Ta"],
                                      temp_forz_def["RH"],
                                      temp_forz_def["Prec"])

    elif cfg.precipitation_phase == "temp_thld":

        Rf, Sf = met.pp_temp_thld_log(temp_forz_def["Ta"],
                                      temp_forz_def["Prec"])

    elif cfg.precipitation_phase == "Liston":

        Rf, Sf = met.linear_liston(temp_forz_def["Ta"],
                                   temp_forz_def["Prec"])

    else:

        raise Exception("Precipitation phase partitioning not implemented")

    temp_forz_def.insert(6, "Sf", Sf)
    temp_forz_def.insert(7, "Rf", Rf)

    file_name = os.path.join(temp_dest, "input.txt")

    params = {"VAI": temp_forz_def.iloc[0]["VAI"],
              "vegh": temp_forz_def.iloc[0]["vegh"],
              "fsky": temp_forz_def.iloc[0]["fsky"],
              "hbas": temp_forz_def.iloc[0]["hbas"],
              "Taf": temp_forz_def.iloc[0]["Taf"],
              "SWEsca": temp_forz_def.iloc[0]["SWEsca"],
              "subgrid_cv": temp_forz_def.iloc[0]["subgrid_cv"]}

    write_nlst(temp_dest, params, step)

    del temp_forz_def["Prec"]

    del temp_forz_def["VAI"]
    del temp_forz_def["vegh"]
    del temp_forz_def["fsky"]
    del temp_forz_def["hbas"]
    del temp_forz_def["Taf"]
    del temp_forz_def["SWEsca"]
    del temp_forz_def["subgrid_cv"]

    # TODO: Explore export to binary
    # https://stackoverflow.com/questions/44074122/reading-in-fortran-binaries-written-with-python
    temp_forz_def.to_csv(file_name, sep="\t",
                         header=False,
                         index=False)


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
              time_dict, step, MCMC=False):

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
    pesos = Ensemble.wgth

    for n, name_col in enumerate(list(list_state[0].columns)):
        # create matrix of colums
        col_arr = [list_state[x].iloc[:, n].to_numpy()
                   for x in range(len(list_state))]
        col_arr = np.vstack(col_arr)

        average_sim = np.average(col_arr, axis=0, weights=pesos)
        sd_sim = flt.weighted_std(col_arr, axis=0, weights=pesos)

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


def forcing_table(lat_idx, lon_idx):

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
    if os.path.exists(final_directory) and cfg.restart_forcing:

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

        press = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                     frocing_var_names["Press_var_name"],
                                     date_ini, date_end)

        # Search for parameters or use the default settings
        try:  # vegetation parametrs
            vegh = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["vegh_var_name"],
                                        date_ini, date_end)
            VAI = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                       param_var_names["VAI_var_name"],
                                       date_ini, date_end)
            fsky = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["fsky_var_name"],
                                        date_ini, date_end)
            hbas = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["hbas_var_name"],
                                        date_ini, date_end)
        except KeyError:
            VAI = np.repeat(cnt.VAI, len(prec))
            vegh = np.repeat(cnt.vegh, len(prec))
            fsky = np.repeat(cnt.fsky, len(prec))
            hbas = np.repeat(cnt.hbas, len(prec))

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
                                   "hbas": hbas,
                                   "fsky": fsky,
                                   "Taf": Taf,
                                   "SWEsca": SWEsca,
                                   "subgrid_cv": subgrid_cv})

        forcing_df["year"] = forcing_df["year"].dt.year
        forcing_df["month"] = forcing_df["month"].dt.month
        forcing_df["day"] = forcing_df["day"].dt.day
        forcing_df["hours"] = forcing_df["hours"].dt.hour

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

    forcing_df = forcing_df.astype({'year': 'int16',
                                    'month': 'int8',
                                    'day': 'int8',
                                    'hours': 'int8',
                                    'SW': 'float32',
                                    'LW': 'float32',
                                    'Prec': 'float32',
                                    'Ta': 'float32',
                                    'RH': 'float32',
                                    'Ua': 'float32',
                                    'Ps': 'float32'})

    return (forcing_df)
