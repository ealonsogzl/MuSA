#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions to interact with a simple degree index snowpack model.

Author: Esteban Alonso González - alonsoe@cesbio.cnes.fr
"""

import os
import constants as cnt
import numpy as np
import numba as nb
import datetime as dt
import pandas as pd
import modules.met_tools as met
import config as cfg
import copy
import modules.filters as flt
import modules.internal_fns as ifn
if cfg.DAsord:
    from modules.user_optional_fns import snd_ord


if cfg.DAsord:
    model_columns = ("year", "month", "day", "hour",
                     "SWE", "snd", tuple(cfg.DAord_names))
else:
    model_columns = ("year", "month", "day", "hour", "SWE", "snd")


def prepare_forz(forcing_sbst):

    Temp = forcing_sbst['Ta'].values
    _, Sf = met.pp_psychrometric(forcing_sbst["Ta"],
                                 forcing_sbst["RH"],
                                 forcing_sbst["Prec"])

    return Temp-cnt.KELVING_CONVER, Sf*3600


@nb.njit(fastmath=True)
def dIm(Temp, Sf, cSWE):

    # Initialize SWE
    SWE = np.zeros(len(Temp))

    for timestep in range(len(Temp)):

        if Temp[timestep] > 0:
            melt = cnt.DMF*(Temp[timestep])
        else:
            melt = 0

        cSWE = cSWE + Sf[timestep] - melt
        cSWE = np.maximum(cSWE, 0)

        SWE[timestep] = cSWE

    return SWE, SWE/cnt.FIX_density/1000


def model_run(forcing_sbst, init=None):

    Temp, Sf = prepare_forz(forcing_sbst)

    if init is None:
        cSWE = 0.
    else:
        cSWE = init

    SWE, HS = dIm(Temp, Sf, cSWE)
    init = SWE[-1]

    Results = pd.DataFrame({'year': forcing_sbst['year'],
                           'month': forcing_sbst['month'],
                            'day': forcing_sbst['day'],
                            'hours': forcing_sbst['hours'],
                            'SWE': SWE,
                            'snd': HS})
    # add optional variables
    if cfg.DAsord:
        Results = snd_ord(Results)

    return Results, init


def model_copy(y_id, x_id):
    return None


def write_nlst(temp_dest, step):
    return None


def model_compile():
    return None


def model_compile_PBS(pbs_task_id):
    return None


def model_read_output(fsm_path, read_dump=True):
    return None


def model_remove(fsm_path):
    return None


def write_input(fsm_path, fsm_input_data):
    return None


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


def model_forcing_wrt(forcing_df, temp_dest):

    return None


def write_dump(dump, fsm_path):
    return None


def get_var_state_position(var):

    state_columns = model_columns

    return state_columns.index(var)


def storeDA(Result_df, step_results, observations_sbst, time_dict, step):

    vars_to_perturbate = cfg.vars_to_perturbate
    var_to_assim = cfg.var_to_assim

    rowIndex = Result_df.index[time_dict["Assimilaiton_steps"][step]:
                               time_dict["Assimilaiton_steps"][step + 1]]

    if len(var_to_assim) > 1:
        for i, var in enumerate(var_to_assim):
            Result_df.loc[rowIndex, var] = observations_sbst[:, i]
    else:
        var = var_to_assim[0]
        Result_df.loc[rowIndex, var] = observations_sbst

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
        col_names = ["Date", "SWE", "snd"]

        # Create results dataframe
        Results = pd.DataFrame(np.nan, index=range(len(del_t)),
                               columns=col_names)

        Results["Date"] = [x.strftime('%d/%m/%Y-%H:%S') for x in del_t]

        Results["SWE"] = [np.nan for x in del_t]
        Results["snd"] = [np.nan for x in del_t]

        Results = Results.astype({'SWE': 'float32',
                                  'snd': 'float32'})

        return Results


def forcing_table(lat_idx, lon_idx):

    nc_forcing_path = cfg.nc_forcing_path
    frocing_var_names = cfg.frocing_var_names
    date_ini = cfg.date_ini
    date_end = cfg.date_end
    intermediate_path = cfg.intermediate_path

    # Path to intermediate file
    final_directory = os.path.join(intermediate_path,
                                   (str(lat_idx) + "_" +
                                    str(lon_idx) + ".pkl"))

    # try to read the forcing from a dumped file
    if os.path.exists(final_directory) and cfg.restart_forcing:

        forcing_df = pd.read_pickle(final_directory)

    else:

        prec = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                    frocing_var_names["Precip_var_name"],
                                    date_ini, date_end)

        temp = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                    frocing_var_names["Temp_var_name"],
                                    date_ini, date_end)

        rel_humidity = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                            frocing_var_names["RH_var_name"],
                                            date_ini, date_end)

        date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
        del_t = ifn.generate_dates(date_ini, date_end)

        forcing_df = pd.DataFrame({"year": del_t,
                                  "month": del_t,
                                   "day": del_t,
                                   "hours": del_t,
                                   "Prec": prec,
                                   "Ta": temp,
                                   "RH": rel_humidity})

        forcing_df["year"] = forcing_df["year"].dt.year
        forcing_df["month"] = forcing_df["month"].dt.month
        forcing_df["day"] = forcing_df["day"].dt.day
        forcing_df["hours"] = forcing_df["hours"].dt.hour

        # write intermediate file to avoid re-reading the nc files
        forcing_df.to_pickle(final_directory)

        if len(del_t) != len(forcing_df.index):
            raise Exception("date_end - date_ini longuer than forcing")

    return forcing_df


def unit_conversion(forcing_df):

    forcing_offset = cnt.forcing_offset
    forcing_multiplier = cnt.forcing_multiplier

    forcing_df.Prec = forcing_df.Prec * forcing_multiplier["Prec"]
    forcing_df.Ta = forcing_df.Ta * forcing_multiplier["Ta"]
    forcing_df.RH = forcing_df.RH * forcing_multiplier["RH"]

    forcing_df.Prec = forcing_df.Prec + forcing_offset["Prec"]
    forcing_df.Ta = forcing_df.Ta + forcing_offset["Ta"]
    forcing_df.RH = forcing_df.RH + forcing_offset["RH"]

    forcing_df = forcing_df.astype({'year': 'int16',
                                    'month': 'int8',
                                    'day': 'int8',
                                    'hours': 'int8',
                                    'Prec': 'float32',
                                    'Ta': 'float32',
                                    'RH': 'float32'})

    return(forcing_df)
